"""
Quick Markdown tester for the staged (v2) relation extractor.

Usage:
  python LLMRelationExtracter_v2/test_md_extract.py

Adjust INPUT_PATH and MODE below.
"""

import asyncio
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Tuple

# Project root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMRelationExtracter_v2.md_task_utils import (  # noqa: E402
    discover_tasks_for_dir,
    has_markdown_files,
    prepare_task_input_dir,
)
from LLMRelationExtracter_v2.runtime_config import (  # noqa: E402
    build_staged_runtime_config,
)
from LLMRelationExtracter_v2.staged_extractor import (  # noqa: E402
    StagedRelationExtractor,
    extract_relations_multistage_with_extractor,
)

# ---- configure here -------------------------------------------------------
# single_dir: split all markdown files under INPUT_PATH into logical doc tasks
# by id/part convention, then run task by task.
# children_dirs: same split strategy, but applied to each first-level subdir.
MODE = "single_dir"  # single_dir | children_dirs

# Example single-dir:
# INPUT_PATH = Path(r"D:\AgentCompetitionAnalysisPlatform\data\20260213_GMV ES")
# Example children-dirs:
# INPUT_PATH = Path(r"D:\AgentCompetitionAnalysisPlatform\data")
INPUT_PATH = Path(r"D:\AgentCompetitionAnalysisPlatform\data\brand_markdown_output\gree")

# If None, outputs are written beside each target directory.
OUTPUT_DIR = None  # set to Path(r"...") to override

MAX_CONCURRENT = 100
LLM_GLOBAL_CONCURRENCY = 10
LLM_GLOBAL_RPM = 0.0
MAX_RETRIES = 8
LLM_CALL_HARD_TIMEOUT = 180.0
WINDOW_SIZE = 1
USE_SLIDING_WINDOW = True
DROP_ID_ONLY = False
MIN_TEXT_CHARS = 0
MAX_DOCS = 0  # 0 means no limit.
KEEP_TASK_INPUTS = False
# --------------------------------------------------------------------------

def _resolve_targets(input_path: Path, mode: str) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_path}")

    if mode == "single_dir":
        if not has_markdown_files(input_path):
            raise FileNotFoundError(f"No .md found under: {input_path}")
        return [input_path]

    if mode == "children_dirs":
        targets = [
            child
            for child in sorted(input_path.iterdir())
            if child.is_dir() and has_markdown_files(child)
        ]
        if not targets:
            raise FileNotFoundError(
                f"No first-level subdirectory with .md found under: {input_path}"
            )
        return targets

    raise ValueError(f"Unsupported MODE: {mode}. Use 'single_dir' or 'children_dirs'.")


def _build_target_output_root(target_dir: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        return target_dir.parent / f"{target_dir.name}_md_test_split"
    return output_dir / target_dir.name


def _count_products(output_path: Path) -> Tuple[int, Optional[dict]]:
    data = json.loads(output_path.read_text(encoding="utf-8"))
    products: List[Dict] = []
    if isinstance(data, list):
        for bucket in data:
            if isinstance(bucket, dict):
                products.extend(bucket.get("results", []) or [])
    return len(products), (products[0] if products else None)


async def arun_md_extract_with_extractor(
    *,
    extractor: StagedRelationExtractor,
    targets: List[Path],
    output_dir: Optional[Path],
    window_size: int = 1,
    use_sliding_window: bool = True,
    drop_id_only: bool = False,
    min_text_chars: int = 0,
    max_docs: int = 0,
    keep_task_inputs: bool = False,
) -> Dict:
    total_products = 0
    total_tasks = 0
    target_summaries: List[Dict] = []
    for target_idx, target_dir in enumerate(targets, start=1):
        target_output_root = _build_target_output_root(target_dir, output_dir)
        target_output_root.mkdir(parents=True, exist_ok=True)
        tasks, stats, _ = discover_tasks_for_dir(
            target_dir,
            drop_id_only=bool(drop_id_only),
            min_text_chars=max(0, int(min_text_chars)),
            max_docs=max(0, int(max_docs)),
        )
        if not tasks:
            print(f"\n[v2 md test] ({target_idx}/{len(targets)}) Target: {target_dir}")
            print("[v2 md test] No tasks found after filtering.")
            target_summaries.append(
                {
                    "target_dir": str(target_dir),
                    "output_root": str(target_output_root),
                    "tasks": 0,
                    "products": 0,
                    "stats": stats,
                }
            )
            continue

        print(f"\n[v2 md test] ({target_idx}/{len(targets)}) Target: {target_dir}")
        print(f"[v2 md test] Output root: {target_output_root}")
        print(
            "[v2 md test] Discovery: "
            f"md_total={stats['md_files_total']}, "
            f"part_group_tasks={stats['tasks_part_group']}, "
            f"single_file_tasks={stats['tasks_single_file']}, "
            f"cleaned_id_only={stats['md_files_cleaned_id_only']}, "
            f"cleaned_low_text={stats['md_files_cleaned_low_text']}"
        )

        target_products = 0
        for task_idx, task in enumerate(tasks, start=1):
            total_tasks += 1
            doc_output_dir = target_output_root / task.doc_key
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = doc_output_dir / "relation_results.json"
            error_log_path = doc_output_dir / "errors.log"
            task_input_dir = prepare_task_input_dir(
                task,
                output_root=target_output_root,
            )
            src_preview = ", ".join(str(path.name) for path in task.md_files[:2])
            if len(task.md_files) > 2:
                src_preview += f", ...(+{len(task.md_files) - 2})"
            print(
                f"[v2 md test]   task({task_idx}/{len(tasks)}): "
                f"doc_key={task.doc_key}, kind={task.kind}, parts={len(task.md_files)}, src={src_preview}"
            )
            print(f"[v2 md test]   output={output_path}")
            print(f"[v2 md test]   errors={error_log_path}")

            try:
                await extract_relations_multistage_with_extractor(
                    extractor,
                    csv_path=task_input_dir,
                    output_path=output_path,
                    error_log_path=error_log_path,
                    window_size=window_size,
                    use_sliding_window=use_sliding_window,
                    show_progress=False,
                )
            finally:
                if not keep_task_inputs and task_input_dir.exists():
                    shutil.rmtree(task_input_dir, ignore_errors=True)

            product_count, sample_product = _count_products(output_path)
            total_products += product_count
            target_products += product_count
            print(f"[v2 md test]   products={product_count}")
            if sample_product:
                print(
                    "[v2 md test]   sample: "
                    f"{json.dumps(sample_product, ensure_ascii=False)[:260]}..."
                )
        target_summaries.append(
            {
                "target_dir": str(target_dir),
                "output_root": str(target_output_root),
                "tasks": len(tasks),
                "products": target_products,
                "stats": stats,
            }
        )

    print(
        "\n[v2 md test] Done. "
        f"targets={len(targets)}, tasks={total_tasks}, total_products={total_products}"
    )
    return {
        "targets": len(targets),
        "tasks": total_tasks,
        "products": total_products,
        "target_summaries": target_summaries,
    }


def run_md_extract(
    *,
    input_path: Path,
    mode: str = "single_dir",
    output_dir: Optional[Path] = None,
    max_concurrent: int = 100,
    llm_global_concurrency: int = 10,
    llm_global_rpm: float = 0.0,
    max_retries: int = 8,
    llm_call_hard_timeout: float = 180.0,
    window_size: int = 1,
    use_sliding_window: bool = True,
    drop_id_only: bool = False,
    min_text_chars: int = 0,
    max_docs: int = 0,
    keep_task_inputs: bool = False,
) -> Dict:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    targets = _resolve_targets(input_path, mode)

    print(f"[v2 md test] Mode: {mode}")
    print(f"[v2 md test] Input: {input_path}")
    print(f"[v2 md test] Targets: {len(targets)}")
    print(
        "[v2 md test] Options: "
        f"max_concurrent={max_concurrent}, llm_global_concurrency={llm_global_concurrency}, "
        f"llm_global_rpm={llm_global_rpm}, "
        f"window_size={window_size}, sliding={use_sliding_window}, "
        f"max_retries={max_retries}, llm_call_hard_timeout={llm_call_hard_timeout}, "
        f"drop_id_only={drop_id_only}, min_text_chars={min_text_chars}"
    )

    cfg = build_staged_runtime_config(
        max_concurrent=max_concurrent,
        llm_global_concurrency=llm_global_concurrency,
        llm_global_rpm=llm_global_rpm,
        max_retries=max_retries,
        llm_call_hard_timeout=llm_call_hard_timeout,
    )

    async def _run_all_targets() -> Dict:
        extractor = StagedRelationExtractor(cfg)
        try:
            return await arun_md_extract_with_extractor(
                extractor=extractor,
                targets=targets,
                output_dir=output_dir,
                window_size=window_size,
                use_sliding_window=use_sliding_window,
                drop_id_only=drop_id_only,
                min_text_chars=min_text_chars,
                max_docs=max_docs,
                keep_task_inputs=keep_task_inputs,
            )
        finally:
            await extractor.aclose()

    return asyncio.run(_run_all_targets())


def main() -> None:
    run_md_extract(
        input_path=INPUT_PATH,
        mode=MODE,
        output_dir=OUTPUT_DIR,
        max_concurrent=MAX_CONCURRENT,
        llm_global_concurrency=LLM_GLOBAL_CONCURRENCY,
        llm_global_rpm=LLM_GLOBAL_RPM,
        max_retries=MAX_RETRIES,
        llm_call_hard_timeout=LLM_CALL_HARD_TIMEOUT,
        window_size=WINDOW_SIZE,
        use_sliding_window=USE_SLIDING_WINDOW,
        drop_id_only=DROP_ID_ONLY,
        min_text_chars=MIN_TEXT_CHARS,
        max_docs=MAX_DOCS,
        keep_task_inputs=KEEP_TASK_INPUTS,
    )


if __name__ == "__main__":
    main()
