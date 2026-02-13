"""
Markdown processor - load Markdown, split into logical pages, and
yield context windows in the same shape as csv_processor.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from backend.settings import RELATION_EXTRACTOR_CONFIG


_PART_MD_STEM_RE = re.compile(
    r"^part_(?P<start>\d+)_(?P<end>\d+)_(?P<sample>.+)_result$",
    re.IGNORECASE,
)
_TABLE_BLOCK_RE = re.compile(r"(?is)<table\b.*?</table>")
_TABLE_TOKEN_RE = re.compile(r"^__TABLE_BLOCK_(\d+)__$")
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_PAGE_MARKER_RE = re.compile(r"^\s*(\d{1,3})\s*/\s*(\d{1,3})\s*$")


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_table: List[List[str]] = []
        self.current_row: List[str] = []
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs: list) -> None:  # noqa: ARG002
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self.current_row:
            self.current_table.append(self.current_row)
        elif tag in {"td", "th"}:
            self.in_cell = False

    def handle_data(self, data: str) -> None:
        if not self.in_cell:
            return
        text = re.sub(r"\s+", " ", data or "").strip()
        if text:
            self.current_row.append(text)

    def get_table(self) -> List[List[str]]:
        return self.current_table


def _parse_table_from_html(html: str) -> List[List[str]]:
    parser = _TableHTMLParser()
    parser.feed(html)
    return parser.get_table()


def _strip_html_tags(text: str) -> str:
    stripped = re.sub(r"(?is)<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", stripped).strip()


def _derive_md_identity(md_path: Path) -> Tuple[str, int, Optional[int]]:
    stem = md_path.stem
    match = _PART_MD_STEM_RE.match(stem)
    if not match:
        sample_id = stem.replace("_result", "")
        return sample_id, 0, None

    start = int(match.group("start"))
    end = int(match.group("end"))
    sample_id = match.group("sample")
    expected_pages = max(0, end - start)
    return sample_id, start, expected_pages if expected_pages > 0 else None


def _new_row(
    *,
    sample: str,
    file_name: str,
    row_type: str,
    content: str = "",
    table_data: str = "",
    image_path: str = "",
) -> Dict:
    return {
        "sample": sample,
        "file": file_name,
        "page": "",
        "type": row_type,
        "bbox": "",
        "content": content,
        "table_data": table_data,
        "image_path": image_path,
    }


def _extract_markdown_segments(
    markdown_text: str,
    sample: str,
    file_name: str,
) -> Tuple[List[List[Dict]], List[Tuple[int, int]]]:
    table_blocks: List[str] = []

    def _replace_table(match: re.Match) -> str:
        index = len(table_blocks)
        table_blocks.append(match.group(0))
        return f"\n__TABLE_BLOCK_{index}__\n"

    normalized = (markdown_text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = _TABLE_BLOCK_RE.sub(_replace_table, normalized)

    marker_pairs: List[Tuple[int, int]] = []
    segments: List[List[Dict]] = [[]]

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        marker_match = _PAGE_MARKER_RE.match(line)
        if marker_match:
            marker_pairs.append((int(marker_match.group(1)), int(marker_match.group(2))))
            if segments[-1]:
                segments.append([])
            continue

        table_token = _TABLE_TOKEN_RE.match(line)
        if table_token:
            index = int(table_token.group(1))
            html = table_blocks[index] if 0 <= index < len(table_blocks) else ""
            parsed_rows = _parse_table_from_html(html)
            if parsed_rows:
                formatted_rows: List[str] = []
                for row in parsed_rows:
                    cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
                    cells = [cell for cell in cells if cell]
                    if cells:
                        formatted_rows.append(" | ".join(cells))
                table_data = "\n".join(formatted_rows).strip()
                if table_data:
                    segments[-1].append(
                        _new_row(
                            sample=sample,
                            file_name=file_name,
                            row_type="table",
                            table_data=table_data,
                        )
                    )
                    continue
            fallback = _strip_html_tags(html)
            if fallback:
                segments[-1].append(
                    _new_row(
                        sample=sample,
                        file_name=file_name,
                        row_type="table",
                        content=fallback,
                    )
                )
            continue

        image_paths = _IMAGE_MD_RE.findall(line)
        if image_paths:
            for image_path in image_paths:
                if image_path:
                    segments[-1].append(
                        _new_row(
                            sample=sample,
                            file_name=file_name,
                            row_type="image",
                            image_path=image_path,
                        )
                    )
            line = _IMAGE_MD_RE.sub("", line).strip()
            if not line:
                continue

        if line.startswith("#"):
            title = re.sub(r"^#+\s*", "", line).strip()
            if title:
                segments[-1].append(
                    _new_row(
                        sample=sample,
                        file_name=file_name,
                        row_type="title",
                        content=title,
                    )
                )
            continue

        if line.startswith(("* ", "- ", "+ ")):
            content = line[2:].strip()
            if content:
                segments[-1].append(
                    _new_row(
                        sample=sample,
                        file_name=file_name,
                        row_type="list",
                        content=content,
                    )
                )
            continue

        segments[-1].append(
            _new_row(
                sample=sample,
                file_name=file_name,
                row_type="text",
                content=line,
            )
        )

    segments = [segment for segment in segments if segment]
    if not segments:
        segments = [[]]
    return segments, marker_pairs


def _segment_char_size(segment: List[Dict]) -> int:
    total = 0
    for row in segment:
        total += len(str(row.get("content", "")))
        total += len(str(row.get("table_data", "")))
    return max(total, len(segment))


def _split_segment_half(segment: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    if len(segment) <= 1:
        return segment, []

    target = _segment_char_size(segment) / 2.0
    running = 0
    split_idx = 0
    for idx, row in enumerate(segment):
        running += len(str(row.get("content", ""))) + len(str(row.get("table_data", ""))) + 1
        if running >= target:
            split_idx = idx + 1
            break
    if split_idx <= 0 or split_idx >= len(segment):
        split_idx = len(segment) // 2
    return segment[:split_idx], segment[split_idx:]


def _reshape_segments_to_pages(segments: List[List[Dict]], target_pages: int) -> List[List[Dict]]:
    if target_pages <= 0:
        return segments or [[]]

    working = [list(segment) for segment in (segments or [[]])]
    if not working:
        working = [[]]

    while len(working) > target_pages:
        idx = min(range(len(working)), key=lambda i: _segment_char_size(working[i]))
        if idx == 0 and len(working) > 1:
            working[1] = working[0] + working[1]
            del working[0]
        else:
            working[idx - 1].extend(working[idx])
            del working[idx]

    while len(working) < target_pages:
        idx = max(range(len(working)), key=lambda i: _segment_char_size(working[i]))
        left, right = _split_segment_half(working[idx])
        if not right:
            working.insert(idx + 1, [])
        else:
            working[idx] = left
            working.insert(idx + 1, right)

    if len(working) > target_pages:
        overflow = working[target_pages:]
        working = working[:target_pages]
        for extra in overflow:
            working[-1].extend(extra)
    elif len(working) < target_pages:
        working.extend([[] for _ in range(target_pages - len(working))])

    return working


def _format_page_content(rows: List[Dict]) -> str:
    ignored = set(RELATION_EXTRACTOR_CONFIG.get("ignored_types", []))
    filtered_rows = [row for row in rows if str(row.get("type", "")).lower() not in ignored]

    prefix_map = {
        "title": "[title]",
        "text": "[text]",
        "list": "[text]",
        "table": "[table]",
    }

    text_parts: List[str] = []
    for row in filtered_rows:
        row_type = str(row.get("type", "")).lower()
        table_data = str(row.get("table_data", "")).strip()
        content = str(row.get("content", "")).strip()

        if row_type == "table" and table_data:
            table_lines = table_data.replace(" \\n ", "\n").replace("\\n", "\n")
            lines = [
                f"[table_row] {line.strip()}"
                for line in table_lines.splitlines()
                if line.strip()
            ]
            if lines:
                text_parts.append("\n".join(lines))
            continue

        if content:
            text_parts.append(f"{prefix_map.get(row_type, '[block]')} {content}")
            continue

        if table_data:
            table_lines = table_data.replace(" \\n ", "\n").replace("\\n", "\n")
            text_parts.append(f"{prefix_map.get(row_type, '[block]')}\n{table_lines}")

    return "\n\n".join(text_parts)


def _safe_page_order(value, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _build_windowed_pages(
    page_entries: List[Dict],
    window_size: int = 1,
    known_models: Optional[List[str]] = None,
) -> Iterator[Tuple[str, Dict]]:
    if not page_entries:
        return

    ordered = sorted(
        page_entries,
        key=lambda item: (
            _safe_page_order(item.get("page_order"), 0),
            str(item.get("page", "")),
            str(item.get("file", "")),
        ),
    )

    for idx, page_entry in enumerate(ordered):
        rows = list(page_entry.get("rows") or [])
        start_idx = max(0, idx - window_size)
        end_idx = min(len(ordered), idx + window_size + 1)
        window_entries = ordered[start_idx:end_idx]
        window_page_labels = [str(entry.get("page", "")) for entry in window_entries]

        window_texts: List[str] = []
        if known_models:
            models_text = "Known product models in this document:\n" + ", ".join(known_models[:20])
            window_texts.append(models_text)

        current_page_label = str(page_entry.get("page", ""))
        for window_entry in window_entries:
            window_page_label = str(window_entry.get("page", ""))
            window_rows = list(window_entry.get("rows") or [])
            is_current = window_page_label == current_page_label

            page_marker = f"\n{'=' * 60}\n"
            if is_current:
                page_marker += f">>> CURRENT PAGE {window_page_label} (FOCUS ON THIS PAGE) <<<\n"
            else:
                page_marker += f"Context Page {window_page_label}\n"
            page_marker += f"{'=' * 60}\n"

            window_texts.append(page_marker)
            window_texts.append(_format_page_content(window_rows))

        full_text = "\n\n".join(window_texts)
        metadata = {
            "sample": str(page_entry.get("sample", "")),
            "file": str(page_entry.get("file", "")),
            "doc_group": str(page_entry.get("doc_group", "")),
            "page": current_page_label,
            "page_order": _safe_page_order(page_entry.get("page_order"), idx),
            "row_count": len(rows),
            "window_pages": window_page_labels,
            "rows": rows,
        }
        yield full_text, metadata


def _load_single_md_page_entries(path: Path) -> List[Dict]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = path.read_text(encoding="utf-8-sig")

    sample_id, page_offset, expected_pages = _derive_md_identity(path)
    segments, marker_pairs = _extract_markdown_segments(
        raw_text,
        sample=sample_id,
        file_name=path.name,
    )

    if expected_pages is None:
        if marker_pairs:
            expected_pages = max((pair[1] for pair in marker_pairs), default=0)
        if not expected_pages:
            expected_pages = max(1, len(segments))

    page_segments = _reshape_segments_to_pages(segments, expected_pages)
    entries: List[Dict] = []
    for idx, segment in enumerate(page_segments):
        page_number = page_offset + idx
        page_label = f"{sample_id}_result_page_{page_number}"

        rows: List[Dict] = []
        for row in segment:
            cloned = dict(row)
            cloned["page"] = page_label
            rows.append(cloned)

        entries.append(
            {
                "sample": sample_id,
                "file": path.name,
                "doc_group": sample_id,
                "page": page_label,
                "page_order": page_number,
                "rows": rows,
            }
        )

    return entries


def load_pages_with_context_from_md(
    md_path: str,
    window_size: int = 1,
    known_models: Optional[List[str]] = None,
) -> Iterator[Tuple[str, Dict]]:
    """
    Load pages from markdown with sliding-window context.

    Returns the same tuple shape as csv_processor.load_pages_with_context:
    (text, metadata).
    """
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")

    entries = _load_single_md_page_entries(path)
    yield from _build_windowed_pages(
        entries,
        window_size=window_size,
        known_models=known_models,
    )


def load_pages_with_context_from_md_directory(
    md_dir_path: str,
    window_size: int = 1,
    known_models: Optional[List[str]] = None,
) -> Iterator[Tuple[str, Dict]]:
    """
    Load pages by aggregating all markdown parts under one directory as one document.
    """
    dir_path = Path(md_dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {md_dir_path}")

    md_files = sorted(p for p in dir_path.rglob("*_result.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No *_result.md found under directory: {md_dir_path}")

    entries: List[Dict] = []
    for md_file in md_files:
        for page_entry in _load_single_md_page_entries(md_file):
            cloned = dict(page_entry)
            # Force all parts in this folder into one logical document group.
            cloned["doc_group"] = dir_path.name
            entries.append(cloned)

    yield from _build_windowed_pages(
        entries,
        window_size=window_size,
        known_models=known_models,
    )
