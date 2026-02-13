import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LANGCHAIN_VERBOSE", "false")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from backend.settings import RELATION_EXTRACTOR_CONFIG  # noqa: E402
from LLMRelationExtracter import (  # noqa: E402
    correct_all_categories,
    deduplicate_results,
    extract_frequent_models,
    filter_empty_products,
    load_pages_with_context,
)
from LLMRelationExtracter.relation_extractor import (  # noqa: E402
    RELATION_SCHEMA,
    RelationExtractor,
)
from LLMRelationExtracter_v2 import (  # noqa: E402
    extract_relations_multistage,
)
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402


DEFAULT_SYSTEM_PROMPT = (
    "You are a product information extraction expert. "
    "Only extract fields with evidence; if absent leave empty strings or empty arrays."
)


def build_relation_chain() -> tuple[ChatPromptTemplate, ChatOpenAI, RelationExtractor]:
    system_prompt = RELATION_EXTRACTOR_CONFIG.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                "Extract product relations from the text. "
                "If multiple product models are present (e.g., table rows), output one result per model; "
                "do NOT merge specs across models.\n\n{text}",
            ),
        ]
    )

    llm = ChatOpenAI(
        api_key=RELATION_EXTRACTOR_CONFIG["api_key"],
        base_url=RELATION_EXTRACTOR_CONFIG["base_url"],
        model=RELATION_EXTRACTOR_CONFIG["model"],
        temperature=0,
        timeout=RELATION_EXTRACTOR_CONFIG.get("timeout", 300),
        model_kwargs={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "relation_schema",
                    "schema": RELATION_SCHEMA,
                    "strict": True,
                },
            }
        },
    )

    return prompt, llm, RelationExtractor()


def setup_relation_logger() -> logging.Logger:
    log_path = Path(RELATION_EXTRACTOR_CONFIG["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(
        logging,
        str(RELATION_EXTRACTOR_CONFIG.get("log_level", "INFO")).upper(),
        logging.INFO,
    )
    logger = logging.getLogger("relation_extract_batch")
    logger.setLevel(level)
    logger.propagate = False

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


async def extract_page(
    text: str,
    metadata: dict,
    chain: ChatPromptTemplate,
    llm: ChatOpenAI,
    post_processor: RelationExtractor,
    semaphore: asyncio.Semaphore,
    error_writer,
) -> list[dict]:
    max_retries = int(RELATION_EXTRACTOR_CONFIG.get("max_retries", 3))
    retry_delay = float(RELATION_EXTRACTOR_CONFIG.get("retry_delay", 1.0))

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await (chain | llm).ainvoke({"text": text})
            content = getattr(response, "content", "") or ""
            data = json.loads(content)
            results = data.get("results", [])
            return [post_processor._post_process_item(item) for item in results]
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                error_writer.write(
                    json.dumps(
                        {
                            "stage": "relation_extract",
                            "metadata": metadata,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raise
            await asyncio.sleep(retry_delay)

    return []


def extract_relations_for_csv(
    csv_path: Path,
    output_path: Path,
    error_log_path: Path,
    max_concurrent: int,
    use_sliding_window: bool = True,
    window_size: int = 1,
) -> None:
    logger = setup_relation_logger()
    start_time = time.time()
    logger.info("Relation extraction start: %s", csv_path)

    # Extract frequent models for context
    known_models = None
    if use_sliding_window:
        logger.info("Extracting frequent product models...")
        known_models = extract_frequent_models(str(csv_path), min_frequency=3)
        logger.info("Found %d frequent models", len(known_models))

    # Load pages with or without context window
    if use_sliding_window:
        pages = list(load_pages_with_context(
            str(csv_path),
            window_size=window_size,
            known_models=known_models,
        ))
    else:
        pages = list(load_pages_with_context(str(csv_path), window_size=0, known_models=None))

    if not pages:
        logger.warning("No pages found for %s", csv_path)
        output_path.write_text("[]", encoding="utf-8")
        return

    prompt, llm, post_processor = build_relation_chain()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run() -> list[dict]:
        tasks = []
        metadatas = []
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with error_log_path.open("w", encoding="utf-8") as error_writer, tqdm(
            total=len(pages),
            desc=f"Extract {csv_path.name}",
            unit="page",
        ) as pbar:
            for text, metadata in pages:
                metadatas.append(metadata)
                tasks.append(
                    asyncio.create_task(
                        extract_page(
                            text,
                            metadata,
                            prompt,
                            llm,
                            post_processor,
                            semaphore,
                            error_writer,
                        )
                    )
                )

            results = [None] * len(tasks)

            async def run_one(index: int, task: asyncio.Task):
                try:
                    return index, await task
                except Exception as exc:  # noqa: BLE001
                    return index, exc
                finally:
                    pbar.update(1)

            wrapped = [
                asyncio.create_task(run_one(index, task))
                for index, task in enumerate(tasks)
            ]
            for wrapped_task in wrapped:
                index, result = await wrapped_task
                results[index] = result

        completed = []
        for metadata, result in zip(metadatas, results):
            if isinstance(result, Exception):
                completed.append({"error": str(result)})
                logger.error("Relation extraction failed: %s", metadata)
            else:
                completed.append({"results": result})
                logger.debug(
                    "Relation extraction success: %s (items=%s)",
                    metadata,
                    len(result),
                )
        return completed

    completed = asyncio.run(run())

    # Deduplicate results if using sliding window
    if use_sliding_window:
        logger.info("Deduplicating results...")
        original_count = sum(len(r.get("results", [])) for r in completed)
        deduplicated = deduplicate_results(completed)
        logger.info(
            "Deduplicated: %d -> %d products",
            original_count,
            len(deduplicated),
        )

        # Correct categories
        logger.info("Correcting product categories...")
        corrected = correct_all_categories(deduplicated)
        logger.info("Category correction complete")

        # Filter empty products
        logger.info("Filtering empty products...")
        filtered = filter_empty_products(corrected)
        logger.info(
            "Filtered: %d -> %d products (removed %d empty)",
            len(corrected),
            len(filtered),
            len(corrected) - len(filtered),
        )

        # Wrap filtered results back into the expected format
        completed = [{"results": filtered}]
    else:
        # Even without sliding window, still correct categories and filter
        logger.info("Correcting product categories...")
        all_products = []
        for batch in completed:
            all_products.extend(batch.get("results", []))
        corrected = correct_all_categories(all_products)
        logger.info("Category correction complete")

        logger.info("Filtering empty products...")
        filtered = filter_empty_products(corrected)
        logger.info(
            "Filtered: %d -> %d products (removed %d empty)",
            len(corrected),
            len(filtered),
            len(corrected) - len(filtered),
        )
        completed = [{"results": filtered}]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Relation extraction complete: %s (pages=%s, elapsed=%.2fs)",
        csv_path,
        len(pages),
        time.time() - start_time,
    )


def extract_relations_for_csv_v2(
    csv_path: Path,
    output_path: Path,
    error_log_path: Path,
    max_concurrent: int,
    use_sliding_window: bool = True,
    window_size: int = 1,
) -> None:
    """
    Wrapper to run the staged (brand->series->product) pipeline.
    I/O shape matches v1: [{"results": [...] }].
    """
    logger = setup_relation_logger()
    start_time = time.time()
    logger.info("Relation extraction v2 start: %s", csv_path)
    extract_relations_multistage(
        csv_path,
        output_path,
        error_log_path,
        max_concurrent=max_concurrent,
        window_size=window_size,
        use_sliding_window=use_sliding_window,
    )
    logger.info(
        "Relation extraction v2 complete: %s (elapsed=%.2fs)",
        csv_path,
        time.time() - start_time,
    )


def append_error(error_log_path: Path, stage: str, message: str) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with error_log_path.open("a", encoding="utf-8") as error_writer:
        error_writer.write(
            json.dumps(
                {
                    "stage": stage,
                    "error": message,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run relation extraction."
    )
    parser.add_argument(
        "--input-dir",
        default=str(ROOT_DIR / "results"),
        help="Directory with CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "results"),
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*_all_data.csv",
        help="CSV filename pattern.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Max concurrent LLM calls per CSV.",
    )
    parser.add_argument(
        "--use-sliding-window",
        action="store_true",
        default=True,
        help="Use sliding window context for extraction (default: True).",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_false",
        dest="use_sliding_window",
        help="Disable sliding window context.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Sliding window size (pages before/after current page, default: 1).",
    )
    parser.add_argument(
        "--relation-version",
        choices=["v1", "v2"],
        default="v2",
        help="Choose relation extractor pipeline (v1=page-first, v2=brand->series->product staged).",
    )
    parser.add_argument(
        "--combined-output",
        default="relation_entities_all.json",
        help="Combined relation output filename.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_paths = sorted(input_dir.glob(args.pattern))
    if not csv_paths:
        print(f"No CSV files found in {input_dir}")
        return

    RELATION_EXTRACTOR_CONFIG["max_concurrent"] = args.max_concurrent
    RELATION_EXTRACTOR_CONFIG["global_concurrency"] = args.max_concurrent

    combined_products = []

    for csv_path in csv_paths:
        relation_output = output_dir / f"{csv_path.stem}_relation_results.json"
        error_log = output_dir / f"{csv_path.stem}_errors.log"
        relation_exists = relation_output.exists()
        relation_runner = (
            extract_relations_for_csv_v2
            if args.relation_version == "v2"
            else extract_relations_for_csv
        )

        print(f"Processing {csv_path}...")

        if relation_exists:
            print(f"Skip relation extraction for {csv_path} (exists)")
        else:
            try:
                relation_runner(
                    csv_path,
                    relation_output,
                    error_log,
                    args.max_concurrent,
                    use_sliding_window=args.use_sliding_window,
                    window_size=args.window_size,
                )
            except Exception as exc:  # noqa: BLE001
                append_error(error_log, "relation_extract", str(exc))
                print(f"Relation extraction failed for {csv_path}: {exc}")
                continue

        try:
            relation_data = json.loads(relation_output.read_text(encoding="utf-8"))
            if isinstance(relation_data, list):
                for batch in relation_data:
                    if isinstance(batch, dict):
                        combined_products.extend(batch.get("results", []) or [])
        except Exception as exc:  # noqa: BLE001
            append_error(error_log, "combined_output", str(exc))
            print(f"Failed to append relation data for {csv_path}: {exc}")

    combined_path = output_dir / args.combined_output
    combined_payload = [{"results": combined_products}]
    combined_path.write_text(
        json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Combined relation output saved to {combined_path}")


if __name__ == "__main__":
    main()
