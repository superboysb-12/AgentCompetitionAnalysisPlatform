"""
Shared markdown task discovery and input staging helpers.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PART_STEM_RE = re.compile(
    r"^part_(?P<start>\d+)_(?P<end>\d+)_(?P<doc_id>.+?)(?:_result)?$",
    re.IGNORECASE,
)
_ID_ONLY_STEM_RE = re.compile(r"^[0-9a-z]{20,}$")
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_MEANINGFUL_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


@dataclass(frozen=True)
class MarkdownTask:
    doc_key: str
    raw_doc_id: str
    kind: str
    source_root: Path
    md_files: Tuple[Path, ...]
    brand: Optional[str] = None


def has_markdown_files(directory: Path) -> bool:
    return any(path.is_file() for path in directory.rglob("*.md"))


def _sanitize_key(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", (text or "").strip()).strip("_")
    return cleaned or "doc"


def _extract_part_doc_id(doc_tail: str) -> str:
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


def discover_tasks_for_dir(
    target_dir: Path,
    *,
    drop_id_only: bool,
    min_text_chars: int,
    max_docs: int,
    brand: Optional[str] = None,
    cleaned_sample_limit: int = 200,
) -> tuple[List[MarkdownTask], Dict, List[Dict]]:
    md_files = sorted(path.resolve() for path in target_dir.rglob("*.md") if path.is_file())
    part_groups: Dict[str, List[tuple[int, int, Path]]] = {}
    single_files: List[Path] = []
    used_keys: set[str] = set()
    stats: Dict = {
        "md_files_total": len(md_files),
        "md_files_cleaned_id_only": 0,
        "md_files_cleaned_low_text": 0,
        "tasks_part_group": 0,
        "tasks_single_file": 0,
    }
    cleaned_samples: List[Dict] = []

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
            stats["md_files_cleaned_low_text"] += 1
            if len(cleaned_samples) < cleaned_sample_limit:
                item = {
                    "md_path": str(md_path),
                    "reason": f"low_text_content(<{min_text_chars})",
                }
                if brand:
                    item["brand"] = brand
                cleaned_samples.append(item)
            continue

        if drop_id_only and _ID_ONLY_STEM_RE.fullmatch(stem):
            stats["md_files_cleaned_id_only"] += 1
            if len(cleaned_samples) < cleaned_sample_limit:
                item = {
                    "md_path": str(md_path),
                    "reason": "id_only_name",
                }
                if brand:
                    item["brand"] = brand
                cleaned_samples.append(item)
            continue

        single_files.append(md_path)

    tasks: List[MarkdownTask] = []
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
            MarkdownTask(
                brand=brand,
                doc_key=doc_key,
                raw_doc_id=doc_id,
                kind="part_group",
                source_root=target_dir.resolve(),
                md_files=ordered_files,
            )
        )
        stats["tasks_part_group"] += 1

    for md_path in sorted(single_files, key=lambda path: str(path)):
        relative_no_suffix = md_path.relative_to(target_dir).with_suffix("")
        raw_key = "__".join(relative_no_suffix.parts)
        base_key = _sanitize_key(raw_key)
        doc_key = _ensure_unique_key(base_key, used_keys)
        tasks.append(
            MarkdownTask(
                brand=brand,
                doc_key=doc_key,
                raw_doc_id=md_path.stem,
                kind="single_file",
                source_root=target_dir.resolve(),
                md_files=(md_path,),
            )
        )
        stats["tasks_single_file"] += 1

    tasks = sorted(tasks, key=lambda item: item.doc_key.lower())
    if max_docs > 0:
        tasks = tasks[:max_docs]
    return tasks, stats, cleaned_samples


def discover_tasks_for_brand_root(
    *,
    input_root: Path,
    include_brands: Optional[Iterable[str]],
    exclude_brands: Iterable[str],
    drop_id_only: bool,
    min_text_chars: int,
    max_docs: int,
    cleaned_sample_limit: int = 200,
) -> tuple[List[MarkdownTask], Dict, List[Dict]]:
    include_set = {name.strip() for name in (include_brands or []) if name.strip()}
    use_include_filter = bool(include_set)
    exclude_set = {name.strip() for name in exclude_brands if str(name).strip()}

    tasks: List[MarkdownTask] = []
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
        if not has_markdown_files(brand_dir):
            continue

        brand_tasks, brand_stats, brand_cleaned = discover_tasks_for_dir(
            brand_dir.resolve(),
            drop_id_only=drop_id_only,
            min_text_chars=min_text_chars,
            max_docs=0,
            brand=brand,
            cleaned_sample_limit=cleaned_sample_limit,
        )

        stats["by_brand"][brand] = brand_stats
        stats["md_files_total"] += int(brand_stats.get("md_files_total", 0))
        stats["md_files_cleaned_id_only"] += int(
            brand_stats.get("md_files_cleaned_id_only", 0)
        )
        stats["md_files_cleaned_low_text"] += int(
            brand_stats.get("md_files_cleaned_low_text", 0)
        )
        stats["tasks_part_group"] += int(brand_stats.get("tasks_part_group", 0))
        stats["tasks_single_file"] += int(brand_stats.get("tasks_single_file", 0))
        tasks.extend(brand_tasks)

        if len(cleaned_samples) < cleaned_sample_limit:
            remain = cleaned_sample_limit - len(cleaned_samples)
            cleaned_samples.extend(brand_cleaned[:remain])

    tasks = sorted(
        tasks,
        key=lambda item: ((item.brand or "").lower(), item.doc_key.lower()),
    )
    if max_docs > 0:
        tasks = tasks[:max_docs]
    return tasks, stats, cleaned_samples


def link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)


def prepare_task_input_dir(
    task: MarkdownTask,
    *,
    output_root: Path,
    scope_parts: Sequence[str] = (),
) -> Path:
    task_input_dir = output_root / "_task_inputs" / Path(*scope_parts) / task.doc_key
    if task_input_dir.exists():
        shutil.rmtree(task_input_dir)
    task_input_dir.mkdir(parents=True, exist_ok=True)

    for md_path in task.md_files:
        relative = md_path.relative_to(task.source_root)
        dst = task_input_dir / relative
        link_or_copy_file(md_path, dst)
    return task_input_dir

