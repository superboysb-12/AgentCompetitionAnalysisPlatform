"""
Batch runner for v2 relation extraction on brand markdown documents.

Task strategy:
  - Group split markdown parts into one logical document task:
      part_<start>_<end>_<doc_id>.md
  - Keep id-only markdown names by default.
  - Clean low-information markdown files by text threshold.
  - Run multiple document tasks concurrently.

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
import os
import json
import re
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform.startswith("win"):
    # Selector policy is more stable for many short-lived asyncio loops in threads.
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from LLMRelationExtracter_v2 import extract_relations_multistage  # noqa: E402

_PART_STEM_RE = re.compile(
    r"^part_(?P<start>\d+)_(?P<end>\d+)_(?P<doc_id>.+?)(?:_result)?$",
    re.IGNORECASE,
)
_ID_ONLY_STEM_RE = re.compile(r"^[0-9a-z]{20,}$")
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_MEANINGFUL_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


@dataclass(frozen=True)
class Task:
    brand: str
    doc_key: str
    raw_doc_id: str
    kind: str
    brand_dir: Path
    md_files: Tuple[Path, ...]


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


def _normalize_doc_key(md_path: Path, brand_dir: Path) -> str:
    relative_no_suffix = md_path.relative_to(brand_dir).with_suffix("")
    raw_key = "__".join(relative_no_suffix.parts)
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_key).strip("_")
    return cleaned or md_path.stem


def _sanitize_key(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", (text or "").strip()).strip("_")
    return cleaned or "doc"


def _extract_part_doc_id(doc_tail: str) -> str:
    """
    Extract stable doc id from part stem tail.
    Example:
      CN119178214A_空调系统... -> CN119178214A
      WO2024187755A1_除霜控制_ -> WO2024187755A1
      085134vs3s5ekwzj3jq51m -> 085134vs3s5ekwzj3jq51m
    """
    text = (doc_tail or "").strip("_")
    if not text:
        return doc_tail
    prefix_match = re.match(r"^[0-9A-Za-z-]+", text)
    if prefix_match:
        return prefix_match.group(0)
    return text


def _ensure_unique_key(base_key: str, used_keys: set[str]) -> str:
    if base_key not in used_keys:
        used_keys.add(base_key)
        return base_key
    seq = 2
    while True:
        candidate = f"{base_key}__{seq}"
        if candidate not in used_keys:
            used_keys.add(candidate)
            return candidate
        seq += 1


def _discover_tasks(
    input_root: Path,
    include_brands: Optional[Iterable[str]],
    exclude_brands: Iterable[str],
    drop_id_only: bool,
    min_text_chars: int,
    max_docs: int,
) -> tuple[List[Task], Dict, List[Dict]]:
    include_set = {name.strip() for name in (include_brands or []) if name.strip()}
    use_include_filter = bool(include_set)
    exclude_set = {name.strip() for name in exclude_brands if str(name).strip()}

    tasks: List[Task] = []
    cleaned_samples: List[Dict] = []
    stats: Dict = {
        "md_files_total": 0,
        "md_files_cleaned_id_only": 0,
        "md_files_cleaned_low_text": 0,
        "tasks_part_group": 0,
        "tasks_single_file": 0,
        "by_brand": {},
    }

    for brand_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        brand = brand_dir.name
        if use_include_filter and brand not in include_set:
            continue
        if brand in exclude_set:
            continue
        if brand.startswith("."):
            continue

        md_files = sorted(path.resolve() for path in brand_dir.rglob("*.md") if path.is_file())
        if not md_files:
            continue

        brand_stats = {
            "md_files_total": len(md_files),
            "md_files_cleaned_id_only": 0,
            "md_files_cleaned_low_text": 0,
            "tasks_part_group": 0,
            "tasks_single_file": 0,
        }
        stats["by_brand"][brand] = brand_stats
        stats["md_files_total"] += len(md_files)

        part_groups: Dict[str, List[tuple[int, int, Path]]] = {}
        single_files: List[Path] = []
        used_keys: set[str] = set()

        for md_path in md_files:
            stem = md_path.stem
            part_match = _PART_STEM_RE.match(stem)
            if part_match:
                start = int(part_match.group("start"))
                end = int(part_match.group("end"))
                tail = (part_match.group("doc_id") or "").strip("_") or stem
                doc_id = _extract_part_doc_id(tail)
                part_groups.setdefault(doc_id, []).append((start, end, md_path))
                continue

            low_text = False
            if min_text_chars > 0:
                try:
                    raw_text = md_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    raw_text = md_path.read_text(encoding="utf-8-sig")
                except Exception:
                    raw_text = ""

                text_no_img = _IMAGE_MD_RE.sub(" ", raw_text or "")
                meaningful_chars = len(_MEANINGFUL_CHAR_RE.findall(text_no_img))
                if meaningful_chars < min_text_chars:
                    low_text = True

            if low_text:
                brand_stats["md_files_cleaned_low_text"] += 1
                stats["md_files_cleaned_low_text"] += 1
                if len(cleaned_samples) < 200:
                    cleaned_samples.append(
                        {
                            "brand": brand,
                            "md_path": _to_relative_display(md_path),
                            "reason": f"low_text_content(<{min_text_chars})",
                        }
                    )
                continue

            if drop_id_only and _ID_ONLY_STEM_RE.fullmatch(stem):
                brand_stats["md_files_cleaned_id_only"] += 1
                stats["md_files_cleaned_id_only"] += 1
                if len(cleaned_samples) < 200:
                    cleaned_samples.append(
                        {
                            "brand": brand,
                            "md_path": _to_relative_display(md_path),
                            "reason": "id_only_name",
                        }
                    )
                continue

            single_files.append(md_path)

        for doc_id in sorted(part_groups.keys(), key=lambda x: x.lower()):
            ordered_files = tuple(
                path
                for _, _, path in sorted(
                    part_groups[doc_id],
                    key=lambda item: (item[0], item[1], str(item[2])),
                )
            )
            base_key = _sanitize_key(doc_id)
            doc_key = _ensure_unique_key(base_key, used_keys)
            tasks.append(
                Task(
                    brand=brand,
                    doc_key=doc_key,
                    raw_doc_id=doc_id,
                    kind="part_group",
                    brand_dir=brand_dir.resolve(),
                    md_files=ordered_files,
                )
            )
            brand_stats["tasks_part_group"] += 1
            stats["tasks_part_group"] += 1

        for md_path in sorted(single_files, key=lambda path: str(path)):
            relative_no_suffix = md_path.relative_to(brand_dir).with_suffix("")
            raw_key = "__".join(relative_no_suffix.parts)
            base_key = _sanitize_key(raw_key)
            doc_key = _ensure_unique_key(base_key, used_keys)
            tasks.append(
                Task(
                    brand=brand,
                    doc_key=doc_key,
                    raw_doc_id=md_path.stem,
                    kind="single_file",
                    brand_dir=brand_dir.resolve(),
                    md_files=(md_path,),
                )
            )
            brand_stats["tasks_single_file"] += 1
            stats["tasks_single_file"] += 1

    tasks = sorted(tasks, key=lambda item: (item.brand.lower(), item.doc_key.lower()))
    if max_docs > 0:
        tasks = tasks[:max_docs]
    return tasks, stats, cleaned_samples


def _append_error_log(error_log_path: Path, message: str) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with error_log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"\n[{timestamp}] {message}\n")


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


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)


def _prepare_task_input_dir(task: Task, batch_root: Path) -> Path:
    """
    Build a task-local markdown directory so extractor runs directory mode.
    """
    task_input_dir = batch_root / "_task_inputs" / task.brand / task.doc_key
    if task_input_dir.exists():
        shutil.rmtree(task_input_dir)
    task_input_dir.mkdir(parents=True, exist_ok=True)

    for md_path in task.md_files:
        relative = md_path.relative_to(task.brand_dir)
        dst = task_input_dir / relative
        _link_or_copy_file(md_path, dst)
    return task_input_dir


def _run_one_task(
    task: Task,
    *,
    batch_root: Path,
    skip_existing: bool,
    keep_task_inputs: bool,
    max_concurrent: int,
    window_size: int,
    use_sliding_window: bool,
    show_progress: bool,
) -> Dict:
    doc_output_dir = batch_root / task.brand / task.doc_key
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    relation_output = doc_output_dir / "relation_results.json"
    error_log = doc_output_dir / "errors.log"
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
                "error_log": error_log,
                "stage_products": stage_products,
                "input_dir": None,
                "error": None,
            }

        task_input_dir = _prepare_task_input_dir(task, batch_root)
        extract_relations_multistage(
            csv_path=task_input_dir,
            output_path=relation_output,
            error_log_path=error_log,
            max_concurrent=max_concurrent,
            window_size=window_size,
            use_sliding_window=use_sliding_window,
            show_progress=show_progress,
        )

        products = _load_products_raw(stage_products, relation_output)
        brand_count = _load_stage_brand_count(stage_dir)
        has_conn_error = _error_log_contains_connection_error(error_log)

        # Treat transient transport failures as failed tasks, not successful 0-product tasks.
        if has_conn_error and brand_count <= 0 and not products:
            return {
                "status": "failed",
                "task": task,
                "products": [],
                "relation_output": relation_output,
                "error_log": error_log,
                "stage_products": stage_products,
                "input_dir": task_input_dir,
                "error": "Connection error during extraction (no brands/products produced)",
            }

        return {
            "status": "success",
            "task": task,
            "products": products,
            "relation_output": relation_output,
            "error_log": error_log,
            "stage_products": stage_products,
            "input_dir": task_input_dir,
            "error": None,
        }
    except Exception as exc:
        error_text = "".join(traceback.format_exception(exc))
        _append_error_log(error_log, error_text)
        return {
            "status": "failed",
            "task": task,
            "products": [],
            "relation_output": relation_output,
            "error_log": error_log,
            "stage_products": stage_products,
            "input_dir": task_input_dir,
            "error": str(exc),
        }
    finally:
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
        default=2,
        help="How many document tasks run in parallel.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max concurrent LLM calls inside each extraction task.",
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
        default=True,
        help="Show per-document stage progress bars (use --no-show-progress to disable).",
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

    tasks, discovery_stats, cleaned_samples = _discover_tasks(
        input_root=input_root,
        include_brands=args.include_brands,
        exclude_brands=args.exclude_brands or [],
        drop_id_only=bool(args.drop_id_only),
        min_text_chars=max(0, int(args.min_text_chars)),
        max_docs=max(0, int(args.max_docs)),
    )
    if not tasks:
        raise FileNotFoundError(f"No markdown docs found under: {input_root}")

    print(f"[batch-md] Input root: {_to_relative_display(input_root)}")
    print(f"[batch-md] Batch root: {_to_relative_display(batch_root)}")
    print(f"[batch-md] Tasks: {len(tasks)}")
    print(
        "[batch-md] Options: "
        f"doc_concurrency={args.doc_concurrency}, max_concurrent={args.max_concurrent}, "
        f"window_size={args.window_size}, sliding_window={not args.no_sliding_window}, "
        f"skip_existing={args.skip_existing}, drop_id_only={args.drop_id_only}, "
        f"min_text_chars={args.min_text_chars}, show_progress={args.show_progress}"
    )
    print(
        "[batch-md] Approx max in-flight LLM calls: "
        f"{max(1, int(args.doc_concurrency)) * max(1, int(args.max_concurrent))}"
    )
    if args.show_progress and max(1, int(args.doc_concurrency)) > 1:
        print("[batch-md] Note: progress bars from concurrent docs may interleave.")
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
            source_preview = ", ".join(
                _to_relative_display(path) for path in list(task.md_files)[:2]
            )
            if len(task.md_files) > 2:
                source_preview += f", ...(+{len(task.md_files) - 2})"
            print(
                f"  {idx:>4}. brand={task.brand}, doc={task.doc_key}, kind={task.kind}, "
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
    safe_doc_concurrency = max(1, int(args.doc_concurrency))
    safe_max_concurrent = max(1, int(args.max_concurrent))
    flat_output = batch_root / "all_products_raw_flat.json"
    with_source_output = batch_root / "all_products_raw_with_source.json"
    summary_output = batch_root / "batch_summary.json"
    cleaned_output = batch_root / "cleaned_md_files.json"
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
        brand_doc_count[task.brand] = brand_doc_count.get(task.brand, 0) + 1

    with JsonArrayWriter(flat_output) as flat_writer, JsonArrayWriter(with_source_output) as src_writer:
        max_workers = safe_doc_concurrency
        future_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for task in tasks:
                future = pool.submit(
                    _run_one_task,
                    task,
                    batch_root=batch_root,
                    skip_existing=bool(args.skip_existing),
                    keep_task_inputs=bool(args.keep_task_inputs),
                    max_concurrent=safe_max_concurrent,
                    window_size=int(args.window_size),
                    use_sliding_window=not args.no_sliding_window,
                    show_progress=bool(args.show_progress),
                )
                future_map[future] = task

            completed = 0
            for future in as_completed(future_map):
                completed += 1
                result = future.result()
                task = result["task"]
                status = result["status"]
                products = result["products"] or []

                if status == "failed":
                    docs_failed += 1
                    docs_executed += 1
                    brand_fail_count[task.brand] = brand_fail_count.get(task.brand, 0) + 1
                    fail_items.append(
                        {
                            "brand": task.brand,
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
                        f"{task.brand}/{task.doc_key} failed: {result['error']}"
                    )
                    continue

                if status == "skipped":
                    docs_skipped += 1
                    brand_skip_count[task.brand] = brand_skip_count.get(task.brand, 0) + 1
                else:
                    docs_executed += 1

                docs_ok += 1
                products_total += len(products)
                brand_product_count[task.brand] = brand_product_count.get(task.brand, 0) + len(products)

                source_md_paths = [_to_relative_display(path) for path in task.md_files]
                for product in products:
                    flat_writer.write(product)
                    src_writer.write(
                        {
                            "source_brand": task.brand,
                            "source_doc_key": task.doc_key,
                            "source_doc_id": task.raw_doc_id,
                            "source_kind": task.kind,
                            "source_md_paths": source_md_paths,
                            "product_raw": product,
                        }
                    )

                print(
                    f"[batch-md] ({completed}/{len(tasks)}) {task.brand}/{task.doc_key} "
                    f"{status}: products_raw={len(products)}"
                )

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
            "cleaned_md_files": _to_relative_display(cleaned_output),
            "batch_summary": _to_relative_display(summary_output),
        },
        "runtime_config": {
            "doc_concurrency": safe_doc_concurrency,
            "max_concurrent": safe_max_concurrent,
            "max_inflight_llm_calls": safe_doc_concurrency * safe_max_concurrent,
            "retry_until_success": os.getenv("RETRY_UNTIL_SUCCESS", "true").lower() == "true",
            "max_retries": int(os.getenv("MAX_RETRIES", "8")),
            "retry_delay": float(os.getenv("RETRY_DELAY", "2.0")),
            "retry_backoff_factor": float(os.getenv("RETRY_BACKOFF_FACTOR", "2.5")),
            "retry_max_delay": float(os.getenv("RETRY_MAX_DELAY", "120.0")),
            "retry_jitter": float(os.getenv("RETRY_JITTER", "0.3")),
            "retry_non_retryable_errors": os.getenv(
                "RETRY_NON_RETRYABLE_ERRORS", "true"
            ).lower()
            == "true",
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
    print(f"[batch-md] batch_summary: {_to_relative_display(summary_output)}")


if __name__ == "__main__":
    main()
