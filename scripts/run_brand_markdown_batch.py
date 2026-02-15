"""
Batch runner for v2 relation extraction on brand markdown documents.

Task strategy:
  - Group split markdown parts into one logical document task:
      part_<start>_<end>_<doc_id>.md
  - Keep id-only markdown names by default.
  - Clean low-information markdown files by text threshold.
  - Run document tasks sequentially; request pacing is controlled by global RPM.

Outputs:
  results/<batch_name>/<brand>/<doc_key>/

Batch-level merged files:
  - all_products_raw_flat.json
  - all_products_raw_with_source.json
  - batch_summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform.startswith("win"):
    # Selector policy is more stable for many short-lived asyncio loops in threads.
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from LLMRelationExtracter_v2 import (  # noqa: E402
    StagedRelationExtractor,
    extract_relations_multistage_with_extractor,
)
from LLMRelationExtracter_v2.md_task_utils import (  # noqa: E402
    MarkdownTask,
    discover_tasks_for_brand_root,
    prepare_task_input_dir,
)
from LLMRelationExtracter_v2.runtime_config import build_staged_runtime_config  # noqa: E402
from backend.settings import RELATION_EXTRACTOR_CONFIG  # noqa: E402

_BATCH_ERROR_LOG_LOCK = threading.Lock()


class JsonArrayWriter:
    """Stream JSON array items to file to avoid large in-memory lists."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fp = None
        self._is_first = True

    def __enter__(self) -> "JsonArrayWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")
        self._fp.write("[\n")
        return self

    def write(self, item: Dict) -> None:
        if self._fp is None:
            raise RuntimeError("Writer is not opened.")
        if not self._is_first:
            self._fp.write(",\n")
        self._fp.write(json.dumps(item, ensure_ascii=False, indent=2))
        self._is_first = False

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._fp is not None:
            self._fp.write("\n]\n")
            self._fp.close()
            self._fp = None


def _to_relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def _append_batch_error_record(
    batch_error_log: Path,
    task: MarkdownTask,
    payload: Dict,
) -> None:
    batch_error_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "brand": task.brand or "unknown_brand",
        "doc_key": task.doc_key,
        "raw_doc_id": task.raw_doc_id,
        "kind": task.kind,
        "source_md_paths": [_to_relative_display(path) for path in task.md_files],
        "payload": payload,
    }
    with _BATCH_ERROR_LOG_LOCK:
        with batch_error_log.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _flush_task_error_log_to_batch(
    task_error_log: Path,
    batch_error_log: Path,
    task: MarkdownTask,
) -> int:
    if not task_error_log.exists():
        return 0
    flushed = 0
    try:
        with task_error_log.open("r", encoding="utf-8", errors="ignore") as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line:
                    continue
                entry: Dict
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        entry = parsed
                    else:
                        entry = {"raw_line": line}
                except Exception:
                    entry = {"raw_line": line}
                _append_batch_error_record(
                    batch_error_log,
                    task,
                    {"source": "extractor", "entry": entry},
                )
                flushed += 1
    except Exception as exc:  # noqa: BLE001
        _append_batch_error_record(
            batch_error_log,
            task,
            {"source": "batch_runner", "merge_error": str(exc)},
        )
    return flushed


def _load_products_raw(stage_products_path: Path, relation_output_path: Path) -> List[Dict]:
    if stage_products_path.exists():
        try:
            data = json.loads(stage_products_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except Exception:
            pass

    if relation_output_path.exists():
        try:
            data = json.loads(relation_output_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                merged: List[Dict] = []
                for bucket in data:
                    if isinstance(bucket, dict):
                        rows = bucket.get("results", [])
                        if isinstance(rows, list):
                            merged.extend(item for item in rows if isinstance(item, dict))
                return merged
        except Exception:
            pass
    return []


def _load_stage_brand_count(stage_dir: Path) -> int:
    brands_path = stage_dir / "brands.json"
    if not brands_path.exists():
        return 0
    try:
        data = json.loads(brands_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    except Exception:
        return 0
    return 0


def _error_log_contains_connection_error(error_log_path: Path) -> bool:
    if not error_log_path.exists():
        return False
    try:
        text = error_log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "connection error." in text.lower()


def _run_one_task(
    task: MarkdownTask,
    *,
    extractor: StagedRelationExtractor,
    loop: asyncio.AbstractEventLoop,
    batch_root: Path,
    batch_error_log: Path,
    skip_existing: bool,
    keep_task_inputs: bool,
    window_size: int,
    use_sliding_window: bool,
) -> Dict:
    brand = task.brand or "unknown_brand"
    doc_output_dir = batch_root / brand / task.doc_key
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    legacy_doc_error_log = doc_output_dir / "errors.log"
    if legacy_doc_error_log.exists():
        legacy_doc_error_log.unlink()

    relation_output = doc_output_dir / "relation_results.json"
    task_error_log = batch_root / "_task_error_tmp" / brand / f"{task.doc_key}.jsonl"
    task_error_log.parent.mkdir(parents=True, exist_ok=True)
    if task_error_log.exists():
        task_error_log.unlink()
    stage_dir = doc_output_dir / f"{task.doc_key}_v2_stage"
    stage_products = stage_dir / "products_raw.json"

    task_input_dir: Optional[Path] = None
    try:
        if skip_existing and stage_products.exists():
            products = _load_products_raw(stage_products, relation_output)
            return {
                "status": "skipped",
                "task": task,
                "products": products,
                "relation_output": relation_output,
                "error_log": batch_error_log,
                "stage_products": stage_products,
                "input_dir": None,
                "error": None,
            }

        task_input_dir = prepare_task_input_dir(
            task,
            output_root=batch_root,
            scope_parts=(brand,),
        )
        loop.run_until_complete(
            extract_relations_multistage_with_extractor(
                extractor=extractor,
                csv_path=task_input_dir,
                output_path=relation_output,
                error_log_path=task_error_log,
                window_size=window_size,
                use_sliding_window=use_sliding_window,
                show_progress=False,
            )
        )

        products = _load_products_raw(stage_products, relation_output)
        brand_count = _load_stage_brand_count(stage_dir)
        has_conn_error = _error_log_contains_connection_error(task_error_log)
        _flush_task_error_log_to_batch(task_error_log, batch_error_log, task)

        # Treat transient transport failures as failed tasks, not successful 0-product tasks.
        if has_conn_error and brand_count <= 0 and not products:
            return {
                "status": "failed",
                "task": task,
                "products": [],
                "relation_output": relation_output,
                "error_log": batch_error_log,
                "stage_products": stage_products,
                "input_dir": task_input_dir,
                "error": "Connection error during extraction (no brands/products produced)",
            }

        return {
            "status": "success",
            "task": task,
            "products": products,
            "relation_output": relation_output,
            "error_log": batch_error_log,
            "stage_products": stage_products,
            "input_dir": task_input_dir,
            "error": None,
        }
    except Exception as exc:
        error_text = "".join(traceback.format_exception(exc))
        _append_batch_error_record(
            batch_error_log,
            task,
            {
                "source": "batch_runner",
                "error": str(exc),
                "traceback": error_text,
            },
        )
        _flush_task_error_log_to_batch(task_error_log, batch_error_log, task)
        return {
            "status": "failed",
            "task": task,
            "products": [],
            "relation_output": relation_output,
            "error_log": batch_error_log,
            "stage_products": stage_products,
            "input_dir": task_input_dir,
            "error": str(exc),
        }
    finally:
        if task_error_log.exists():
            task_error_log.unlink()
        if task_input_dir and task_input_dir.exists() and not keep_task_inputs:
            shutil.rmtree(task_input_dir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract all markdown docs under brand_markdown_output with v2 pipeline."
    )
    parser.add_argument(
        "--input-root",
        default=str(ROOT / "data" / "brand_markdown_output"),
        help="Root directory: one brand per subdirectory, docs are .md files.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results"),
        help="Base results directory.",
    )
    parser.add_argument(
        "--batch-name",
        default="brand_markdown_output_v2",
        help="Batch output folder name under --results-dir.",
    )
    parser.add_argument(
        "--include-brands",
        nargs="*",
        default=None,
        help="Optional allow-list of brand directory names.",
    )
    parser.add_argument(
        "--exclude-brands",
        nargs="*",
        default=[".obsidian", "aux"],
        help="Optional deny-list of brand directory names.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Limit total logical document tasks for quick runs (0 means no limit).",
    )
    parser.add_argument(
        "--doc-concurrency",
        type=int,
        default=0,
        help="Deprecated and ignored. Document tasks run sequentially.",
    )
    parser.add_argument(
        "--llm-global-concurrency",
        type=int,
        default=10,
        help="Compatibility value. Used as default RPM when --llm-global-rpm is not provided.",
    )
    parser.add_argument(
        "--llm-global-rpm",
        type=float,
        default=None,
        help="Global LLM request rate limit (requests per minute), evenly distributed.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent LLM calls inside each extraction task. Default follows --llm-global-concurrency.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Sliding window size passed to extractor.",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Disable sliding window context.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated and ignored. Progress bars are disabled; call-level counters are printed.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Max retry count per LLM call. Default follows backend/settings.py.",
    )
    parser.add_argument(
        "--llm-call-hard-timeout",
        type=float,
        default=None,
        help="Hard timeout (seconds) for each single LLM call. Default follows backend/settings.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip extraction when stage products file already exists.",
    )
    parser.add_argument(
        "--drop-id-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop non-part markdown files whose stem looks like random id (default: false).",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=20,
        help="Skip non-part markdown files whose meaningful chars (excluding image links) are below this threshold.",
    )
    parser.add_argument(
        "--keep-task-inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep task-local merged markdown directories under <batch>/_task_inputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered tasks; do not run extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    results_root = Path(args.results_dir).expanduser().resolve()
    batch_root = results_root / args.batch_name
    batch_root.mkdir(parents=True, exist_ok=True)

    tasks, discovery_stats, cleaned_samples = discover_tasks_for_brand_root(
        input_root=input_root,
        include_brands=args.include_brands,
        exclude_brands=args.exclude_brands or [],
        drop_id_only=bool(args.drop_id_only),
        min_text_chars=max(0, int(args.min_text_chars)),
        max_docs=max(0, int(args.max_docs)),
    )
    cleaned_samples = [
        {
            **item,
            "md_path": _to_relative_display(Path(str(item.get("md_path", "")))),
        }
        for item in cleaned_samples
    ]
    if not tasks:
        raise FileNotFoundError(f"No markdown docs found under: {input_root}")

    llm_global_concurrency = max(1, int(args.llm_global_concurrency))
    effective_llm_global_rpm = (
        max(1.0, float(args.llm_global_rpm))
        if args.llm_global_rpm is not None
        else float(llm_global_concurrency)
    )
    safe_max_concurrent = (
        max(1, int(args.max_concurrent))
        if args.max_concurrent is not None
        else llm_global_concurrency
    )
    default_max_retries = max(0, int(RELATION_EXTRACTOR_CONFIG.get("max_retries", 8)))
    default_llm_call_hard_timeout = float(RELATION_EXTRACTOR_CONFIG.get("llm_call_hard_timeout", 180.0))
    effective_max_retries = (
        max(0, int(args.max_retries))
        if args.max_retries is not None
        else default_max_retries
    )
    effective_llm_call_hard_timeout = (
        max(1.0, float(args.llm_call_hard_timeout))
        if args.llm_call_hard_timeout is not None
        else default_llm_call_hard_timeout
    )

    print(f"[batch-md] Input root: {_to_relative_display(input_root)}")
    print(f"[batch-md] Batch root: {_to_relative_display(batch_root)}")
    print(f"[batch-md] Tasks: {len(tasks)}")
    print(
        "[batch-md] Options: "
        "scheduler=sequential, "
        f"max_concurrent={safe_max_concurrent}, "
        f"llm_global_concurrency(arg)={llm_global_concurrency}, "
        f"llm_global_rpm={effective_llm_global_rpm}, "
        f"window_size={args.window_size}, sliding_window={not args.no_sliding_window}, "
        f"skip_existing={args.skip_existing}, drop_id_only={args.drop_id_only}, "
        f"min_text_chars={args.min_text_chars}, "
        f"max_retries={effective_max_retries}, "
        f"llm_call_hard_timeout={effective_llm_call_hard_timeout}, "
        f"progress_bar=disabled"
    )
    if bool(args.show_progress):
        print("[batch-md] Note: --show-progress is deprecated and ignored.")
    if int(args.doc_concurrency) != 0:
        print("[batch-md] Note: --doc-concurrency is deprecated and ignored (sequential scheduler).")
    print(
        "[batch-md] Note: --llm-global-concurrency is compatibility-only; "
        "semaphore concurrency limiter is disabled."
    )
    print(
        "[batch-md] Process-wide LLM pacing: "
        f"rpm={effective_llm_global_rpm} (uniformly distributed)"
    )
    print("[batch-md] Console will print cumulative LLM call completions.")
    print(
        "[batch-md] Discovery: "
        f"md_files_total={discovery_stats['md_files_total']}, "
        f"cleaned_id_only={discovery_stats['md_files_cleaned_id_only']}, "
        f"cleaned_low_text={discovery_stats['md_files_cleaned_low_text']}, "
        f"part_group_tasks={discovery_stats['tasks_part_group']}, "
        f"single_file_tasks={discovery_stats['tasks_single_file']}"
    )

    if args.dry_run:
        preview = tasks[:20]
        for idx, task in enumerate(preview, start=1):
            brand = task.brand or "unknown_brand"
            source_preview = ", ".join(
                _to_relative_display(path) for path in list(task.md_files)[:2]
            )
            if len(task.md_files) > 2:
                source_preview += f", ...(+{len(task.md_files) - 2})"
            print(
                f"  {idx:>4}. brand={brand}, doc={task.doc_key}, kind={task.kind}, "
                f"parts={len(task.md_files)}, src={source_preview}"
            )
        if len(tasks) > len(preview):
            print(f"  ... ({len(tasks) - len(preview)} more)")
        if cleaned_samples:
            print(f"[batch-md] Cleaned samples: {min(20, len(cleaned_samples))}")
            for idx, item in enumerate(cleaned_samples[:20], start=1):
                print(f"  - {idx:>2}. {item['brand']} | {item['md_path']}")
        return

    started_at = datetime.now()
    flat_output = batch_root / "all_products_raw_flat.json"
    with_source_output = batch_root / "all_products_raw_with_source.json"
    summary_output = batch_root / "batch_summary.json"
    cleaned_output = batch_root / "cleaned_md_files.json"
    batch_error_log = batch_root / "batch_errors.jsonl"
    batch_error_log.write_text("", encoding="utf-8")
    cleaned_output.write_text(
        json.dumps(cleaned_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fail_items: List[Dict] = []
    docs_ok = 0
    docs_failed = 0
    docs_skipped = 0
    docs_executed = 0
    products_total = 0
    brand_doc_count: Dict[str, int] = {}
    brand_product_count: Dict[str, int] = {}
    brand_fail_count: Dict[str, int] = {}
    brand_skip_count: Dict[str, int] = {}

    for task in tasks:
        brand = task.brand or "unknown_brand"
        brand_doc_count[brand] = brand_doc_count.get(brand, 0) + 1

    extractor_cfg = build_staged_runtime_config(
        max_concurrent=safe_max_concurrent,
        llm_global_concurrency=llm_global_concurrency,
        llm_global_rpm=effective_llm_global_rpm,
        max_retries=effective_max_retries,
        llm_call_hard_timeout=effective_llm_call_hard_timeout,
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    extractor: Optional[StagedRelationExtractor] = None
    try:
        extractor = StagedRelationExtractor(extractor_cfg)
        print(
            "[batch-md] Extractor mode: process-singleton "
            "(one extractor/llm_client/openai client reused in this run)"
        )
        with JsonArrayWriter(flat_output) as flat_writer, JsonArrayWriter(with_source_output) as src_writer:
            for completed, task in enumerate(tasks, start=1):
                brand = task.brand or "unknown_brand"
                result = _run_one_task(
                    task,
                    extractor=extractor,
                    loop=loop,
                    batch_root=batch_root,
                    batch_error_log=batch_error_log,
                    skip_existing=bool(args.skip_existing),
                    keep_task_inputs=bool(args.keep_task_inputs),
                    window_size=int(args.window_size),
                    use_sliding_window=not args.no_sliding_window,
                )
                status = result["status"]
                products = result["products"] or []

                if status == "failed":
                    docs_failed += 1
                    docs_executed += 1
                    brand_fail_count[brand] = brand_fail_count.get(brand, 0) + 1
                    fail_items.append(
                        {
                            "brand": brand,
                            "doc_key": task.doc_key,
                            "raw_doc_id": task.raw_doc_id,
                            "kind": task.kind,
                            "source_md_paths": [
                                _to_relative_display(path) for path in task.md_files
                            ],
                            "error_log": _to_relative_display(result["error_log"]),
                            "error": result["error"],
                        }
                    )
                    print(
                        f"[batch-md] ({completed}/{len(tasks)}) "
                        f"{brand}/{task.doc_key} failed: {result['error']}"
                    )
                    continue

                if status == "skipped":
                    docs_skipped += 1
                    brand_skip_count[brand] = brand_skip_count.get(brand, 0) + 1
                else:
                    docs_executed += 1

                docs_ok += 1
                products_total += len(products)
                brand_product_count[brand] = (
                    brand_product_count.get(brand, 0) + len(products)
                )

                source_md_paths = [_to_relative_display(path) for path in task.md_files]
                for product in products:
                    flat_writer.write(product)
                    src_writer.write(
                        {
                            "source_brand": brand,
                            "source_doc_key": task.doc_key,
                            "source_doc_id": task.raw_doc_id,
                            "source_kind": task.kind,
                            "source_md_paths": source_md_paths,
                            "product_raw": product,
                        }
                    )

                print(
                    f"[batch-md] ({completed}/{len(tasks)}) {brand}/{task.doc_key} "
                    f"{status}: products_raw={len(products)}"
                )
    finally:
        try:
            if extractor is not None:
                loop.run_until_complete(extractor.aclose())
        finally:
            loop.close()
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass

    finished_at = datetime.now()
    summary = {
        "input_root": _to_relative_display(input_root),
        "results_dir": _to_relative_display(results_root),
        "batch_root": _to_relative_display(batch_root),
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round((finished_at - started_at).total_seconds(), 2),
        "tasks_total": len(tasks),
        "tasks_succeeded": docs_ok,
        "tasks_failed": docs_failed,
        "tasks_skipped": docs_skipped,
        "tasks_executed": docs_executed,
        "products_raw_total": products_total,
        "discovery": discovery_stats,
        "by_brand_doc_count": brand_doc_count,
        "by_brand_skip_count": brand_skip_count,
        "by_brand_fail_count": brand_fail_count,
        "by_brand_product_count": brand_product_count,
        "outputs": {
            "all_products_raw_flat": _to_relative_display(flat_output),
            "all_products_raw_with_source": _to_relative_display(with_source_output),
            "batch_errors": _to_relative_display(batch_error_log),
            "cleaned_md_files": _to_relative_display(cleaned_output),
            "batch_summary": _to_relative_display(summary_output),
        },
        "runtime_config": {
            "scheduler_mode": "sequential",
            "threadpool_workers": 0,
            "extractor_reuse": "single_instance",
            "max_concurrent": safe_max_concurrent,
            "llm_global_concurrency": llm_global_concurrency,
            "llm_global_rpm": effective_llm_global_rpm,
            "max_retries": effective_max_retries,
            "retry_delay": float(RELATION_EXTRACTOR_CONFIG.get("retry_delay", 2.0)),
            "retry_backoff_factor": float(RELATION_EXTRACTOR_CONFIG.get("retry_backoff_factor", 2.5)),
            "retry_max_delay": float(RELATION_EXTRACTOR_CONFIG.get("retry_max_delay", 120.0)),
            "llm_call_hard_timeout": effective_llm_call_hard_timeout,
        },
        "failures": fail_items,
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[batch-md] Done.")
    print(
        "[batch-md] Summary: "
        f"tasks={len(tasks)}, ok={docs_ok}, failed={docs_failed}, skipped={docs_skipped}, "
        f"executed={docs_executed}, products_raw_total={products_total}"
    )
    print(f"[batch-md] all_products_raw_flat: {_to_relative_display(flat_output)}")
    print(f"[batch-md] all_products_raw_with_source: {_to_relative_display(with_source_output)}")
    print(f"[batch-md] batch_errors: {_to_relative_display(batch_error_log)}")
    print(f"[batch-md] batch_summary: {_to_relative_display(summary_output)}")


if __name__ == "__main__":
    main()
