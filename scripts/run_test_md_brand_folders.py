"""
Run test_md_extract logic for every brand folder under data/brand_markdown_output.

Usage:
  python scripts/run_test_md_brand_folders.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMRelationExtracter_v2.md_task_utils import has_markdown_files  # noqa: E402
from LLMRelationExtracter_v2.runtime_config import (  # noqa: E402
    build_staged_runtime_config,
)
from LLMRelationExtracter_v2.staged_extractor import StagedRelationExtractor  # noqa: E402
from LLMRelationExtracter_v2.test_md_extract import (  # noqa: E402
    arun_md_extract_with_extractor,
)

# ---- configure here -------------------------------------------------------
INPUT_ROOT = ROOT_DIR / "data" / "brand_markdown_output"
OUTPUT_ROOT = ROOT_DIR / "results" / "brand_markdown_output_test_md"

INCLUDE_BRANDS: List[str] = []  # empty means all brands
EXCLUDE_BRANDS: List[str] = [".obsidian", "aux"]

MAX_CONCURRENT = 100
LLM_GLOBAL_CONCURRENCY = 10
MAX_RETRIES = 8
LLM_CALL_HARD_TIMEOUT = 180.0
WINDOW_SIZE = 1
USE_SLIDING_WINDOW = True
DROP_ID_ONLY = False
MIN_TEXT_CHARS = 0
MAX_DOCS_PER_BRAND = 0  # 0 means no limit.
KEEP_TASK_INPUTS = False
STOP_ON_ERROR = False
# --------------------------------------------------------------------------


def _discover_brand_dirs(input_root: Path) -> List[Path]:
    include_set = {name.strip() for name in INCLUDE_BRANDS if name.strip()}
    exclude_set = {name.strip() for name in EXCLUDE_BRANDS if name.strip()}
    use_include = bool(include_set)

    brand_dirs: List[Path] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if use_include and child.name not in include_set:
            continue
        if child.name in exclude_set:
            continue
        if not has_markdown_files(child):
            continue
        brand_dirs.append(child.resolve())
    return brand_dirs


def _build_extractor_config() -> Dict:
    return build_staged_runtime_config(
        max_concurrent=MAX_CONCURRENT,
        llm_global_concurrency=LLM_GLOBAL_CONCURRENCY,
        max_retries=MAX_RETRIES,
        llm_call_hard_timeout=LLM_CALL_HARD_TIMEOUT,
    )


async def _run_batch_async(brand_dirs: List[Path], output_root: Path) -> Dict:
    extractor = StagedRelationExtractor(_build_extractor_config())
    try:
        total_tasks = 0
        total_products = 0
        ok_count = 0
        fail_count = 0
        brand_results: List[Dict] = []
        failures: List[Dict] = []

        for idx, brand_dir in enumerate(brand_dirs, start=1):
            brand = brand_dir.name
            print(f"\n[test-md-batch] ({idx}/{len(brand_dirs)}) brand={brand}")
            try:
                summary = await arun_md_extract_with_extractor(
                    extractor=extractor,
                    targets=[brand_dir],
                    output_dir=output_root,
                    window_size=WINDOW_SIZE,
                    use_sliding_window=USE_SLIDING_WINDOW,
                    drop_id_only=DROP_ID_ONLY,
                    min_text_chars=MIN_TEXT_CHARS,
                    max_docs=MAX_DOCS_PER_BRAND,
                    keep_task_inputs=KEEP_TASK_INPUTS,
                )
                brand_result = {
                    "brand": brand,
                    "status": "ok",
                    "targets": int(summary.get("targets", 0)),
                    "tasks": int(summary.get("tasks", 0)),
                    "products": int(summary.get("products", 0)),
                    "target_summaries": summary.get("target_summaries", []),
                }
                brand_results.append(brand_result)
                ok_count += 1
                total_tasks += brand_result["tasks"]
                total_products += brand_result["products"]
                print(
                    f"[test-md-batch] brand={brand} done: "
                    f"tasks={brand_result['tasks']}, products={brand_result['products']}"
                )
            except Exception as exc:  # noqa: BLE001
                fail_count += 1
                err = {
                    "brand": brand,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": "".join(traceback.format_exception(exc)),
                }
                failures.append(err)
                brand_results.append(err)
                print(f"[test-md-batch] brand={brand} failed: {exc}")
                if STOP_ON_ERROR:
                    break

        return {
            "brand_total": len(brand_dirs),
            "brand_ok": ok_count,
            "brand_failed": fail_count,
            "tasks_total": total_tasks,
            "products_total": total_products,
            "brands": brand_results,
            "failures": failures,
        }
    finally:
        await extractor.aclose()


def main() -> None:
    input_root = INPUT_ROOT.expanduser().resolve()
    output_root = OUTPUT_ROOT.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    brand_dirs = _discover_brand_dirs(input_root)
    if not brand_dirs:
        raise FileNotFoundError(f"No brand folder with markdown files found under: {input_root}")

    print(f"[test-md-batch] Input root: {input_root}")
    print(f"[test-md-batch] Output root: {output_root}")
    print(f"[test-md-batch] Brands: {len(brand_dirs)}")
    print(
        "[test-md-batch] Options: "
        f"max_concurrent={MAX_CONCURRENT}, llm_global_concurrency={LLM_GLOBAL_CONCURRENCY}, "
        f"max_retries={MAX_RETRIES}, llm_call_hard_timeout={LLM_CALL_HARD_TIMEOUT}, "
        f"window_size={WINDOW_SIZE}, sliding={USE_SLIDING_WINDOW}, "
        f"drop_id_only={DROP_ID_ONLY}, min_text_chars={MIN_TEXT_CHARS}, "
        f"max_docs_per_brand={MAX_DOCS_PER_BRAND}"
    )

    started_at = datetime.now()
    batch_result = asyncio.run(_run_batch_async(brand_dirs, output_root))

    finished_at = datetime.now()
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        "brand_total": int(batch_result.get("brand_total", 0)),
        "brand_ok": int(batch_result.get("brand_ok", 0)),
        "brand_failed": int(batch_result.get("brand_failed", 0)),
        "tasks_total": int(batch_result.get("tasks_total", 0)),
        "products_total": int(batch_result.get("products_total", 0)),
        "runtime_config": {
            "max_concurrent": MAX_CONCURRENT,
            "llm_global_concurrency": LLM_GLOBAL_CONCURRENCY,
            "max_retries": MAX_RETRIES,
            "llm_call_hard_timeout": LLM_CALL_HARD_TIMEOUT,
            "window_size": WINDOW_SIZE,
            "use_sliding_window": USE_SLIDING_WINDOW,
            "drop_id_only": DROP_ID_ONLY,
            "min_text_chars": MIN_TEXT_CHARS,
            "max_docs_per_brand": MAX_DOCS_PER_BRAND,
            "keep_task_inputs": KEEP_TASK_INPUTS,
            "stop_on_error": STOP_ON_ERROR,
            "extractor_scope": "single_instance_for_all_brands",
        },
        "brands": batch_result.get("brands", []),
        "failures": batch_result.get("failures", []),
    }

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n[test-md-batch] Finished.")
    print(
        "[test-md-batch] "
        f"brand_ok={summary['brand_ok']}, brand_failed={summary['brand_failed']}"
    )
    print(
        "[test-md-batch] "
        f"tasks_total={summary['tasks_total']}, products_total={summary['products_total']}"
    )
    print(f"[test-md-batch] summary={summary_path}")


if __name__ == "__main__":
    main()
