"""
Batch runner for v2 relation extraction on brand markdown documents.

Task strategy:
  - Group split markdown parts into one logical document task:
      part_<start>_<end>_<doc_id>.md
  - Keep id-only markdown names by default.
  - Clean low-information markdown files by text threshold.
  - Run document tasks sequentially; global LLM RPM cap throttles outgoing calls.

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
import threading
import traceback
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
from backend.settings import RELATION_EXTRACTOR_CONFIG  # noqa: E402

_PART_STEM_RE = re.compile(
    r"^part_(?P<start>\d+)_(?P<end>\d+)_(?P<doc_id>.+?)(?:_result)?$",
    re.IGNORECASE,
)
_ID_ONLY_STEM_RE = re.compile(r"^[0-9a-z]{20,}$")
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_MEANINGFUL_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_BATCH_ERROR_LOG_LOCK = threading.Lock()


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


def _append_batch_error_record(
    batch_error_log: Path,
    task: Task,
    payload: Dict,
) -> None:
    batch_error_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "brand": task.brand,
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
    task: Task,
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
    batch_error_log: Path,
    skip_existing: bool,
    keep_task_inputs: bool,
    max_concurrent: int,
    llm_global_rpm: int,
    window_size: int,
    use_sliding_window: bool,
    max_retries: int,
    llm_call_hard_timeout: float,
) -> Dict:
    doc_output_dir = batch_root / task.brand / task.doc_key
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    legacy_doc_error_log = doc_output_dir / "errors.log"
    if legacy_doc_error_log.exists():
        legacy_doc_error_log.unlink()

    relation_output = doc_output_dir / "relation_results.json"
    task_error_log = batch_root / "_task_error_tmp" / task.brand / f"{task.doc_key}.jsonl"
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

        task_input_dir = _prepare_task_input_dir(task, batch_root)
        extract_relations_multistage(
            csv_path=task_input_dir,
            output_path=relation_output,
            error_log_path=task_error_log,
            max_concurrent=max_concurrent,
            llm_global_concurrency=llm_global_rpm,
            window_size=window_size,
            use_sliding_window=use_sliding_window,
            show_progress=False,
            max_retries=max_retries,
            llm_call_hard_timeout=llm_call_hard_timeout,
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
        help="Global LLM request cap in RPM (requests per minute) shared by all tasks.",
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

    llm_global_rpm = max(1, int(args.llm_global_concurrency))
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
        f"max_concurrent={args.max_concurrent}, "
        f"llm_global_rpm={llm_global_rpm}, "
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
        "[batch-md] Process-wide LLM RPM cap: "
        f"{llm_global_rpm} req/min"
    )
    print(
        f"[batch-md] LLM pacing: one request about every {60.0 / llm_global_rpm:.2f}s"
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
    safe_max_concurrent = max(1, int(args.max_concurrent))
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
        brand_doc_count[task.brand] = brand_doc_count.get(task.brand, 0) + 1

    with JsonArrayWriter(flat_output) as flat_writer, JsonArrayWriter(with_source_output) as src_writer:
        for completed, task in enumerate(tasks, start=1):
            result = _run_one_task(
                task,
                batch_root=batch_root,
                batch_error_log=batch_error_log,
                skip_existing=bool(args.skip_existing),
                keep_task_inputs=bool(args.keep_task_inputs),
                max_concurrent=safe_max_concurrent,
                llm_global_rpm=llm_global_rpm,
                window_size=int(args.window_size),
                use_sliding_window=not args.no_sliding_window,
                max_retries=effective_max_retries,
                llm_call_hard_timeout=effective_llm_call_hard_timeout,
            )
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
            brand_product_count[task.brand] = (
                brand_product_count.get(task.brand, 0) + len(products)
            )

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
            "batch_errors": _to_relative_display(batch_error_log),
            "cleaned_md_files": _to_relative_display(cleaned_output),
            "batch_summary": _to_relative_display(summary_output),
        },
        "runtime_config": {
            "scheduler_mode": "sequential",
            "threadpool_workers": 0,
            "max_concurrent": safe_max_concurrent,
            "llm_global_rpm": llm_global_rpm,
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
