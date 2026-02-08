"""
CSV processor - load CSV, group by page, sort, and concatenate text.
"""

import ast
import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from backend.settings import RELATION_EXTRACTOR_CONFIG


def load_pages_from_csv(csv_path: str) -> Iterator[Tuple[str, Dict]]:
    """
    Legacy function for backward compatibility.
    Loads pages without context window.
    """
    yield from load_pages_with_context(csv_path, window_size=0, known_models=None)


def load_pages_with_context(
    csv_path: str,
    window_size: int = 1,
    known_models: Optional[List[str]] = None,
) -> Iterator[Tuple[str, Dict]]:
    """
    Load pages from CSV with sliding window context and yield concatenated text.

    Args:
        csv_path: Path to CSV file
        window_size: Number of pages before/after to include as context
                     0 = single page only
                     1 = current + prev + next (3 pages total)
        known_models: List of known product models to inject as context

    Yields:
        (text, metadata) tuples where text includes context window
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    pages: Dict[str, List[Dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            page_num = str(row.get("page", "0"))
            pages.setdefault(page_num, []).append(row)

    page_keys = sorted(pages.keys())

    for idx, page_num in enumerate(page_keys):
        rows = pages[page_num]

        # Determine window range
        start_idx = max(0, idx - window_size)
        end_idx = min(len(page_keys), idx + window_size + 1)
        window_page_keys = page_keys[start_idx:end_idx]

        # Build context text from window pages
        window_texts = []

        # Add known models header if provided
        if known_models:
            models_text = "Known product models in this document:\n" + ", ".join(known_models[:20])
            window_texts.append(models_text)

        # Process each page in window
        for win_idx, win_page_num in enumerate(window_page_keys):
            win_rows = pages[win_page_num]
            is_current_page = (win_page_num == page_num)

            page_marker = f"\n{'='*60}\n"
            if is_current_page:
                page_marker += f">>> CURRENT PAGE {win_page_num} (FOCUS ON THIS PAGE) <<<\n"
            else:
                page_marker += f"Context Page {win_page_num}\n"
            page_marker += f"{'='*60}\n"

            window_texts.append(page_marker)
            window_texts.append(_format_page_content(win_rows))

        full_text = "\n\n".join(window_texts)

        metadata = {
            "sample": rows[0].get("sample", ""),
            "file": rows[0].get("file", ""),
            "page": int(page_num) if str(page_num).isdigit() else page_num,
            "row_count": len(rows),
            "window_pages": [int(p) if p.isdigit() else p for p in window_page_keys],
        }

        yield full_text, metadata


def _format_page_content(rows: List[Dict]) -> str:
    """Format a single page's content with sorting and type tags."""
def _format_page_content(rows: List[Dict]) -> str:
    """Format a single page's content with sorting and type tags."""
    filtered_rows = [
        row
        for row in rows
        if row.get("type") not in RELATION_EXTRACTOR_CONFIG["ignored_types"]
    ]

    def get_xy_center(row: Dict) -> Tuple[float, float]:
        try:
            bbox = ast.literal_eval(row.get("bbox", "[]"))
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
                y_center = (float(bbox[1]) + float(bbox[3])) / 2.0
                return y_center, x_center
        except Exception:
            pass
        return 0.0, 0.0

    sorted_rows = sorted(filtered_rows, key=get_xy_center)

    def format_row(row: Dict) -> str:
        row_type = str(row.get("type", "")).lower()
        prefix_map = {
            "title": "[title]",
            "text": "[text]",
            "table": "[table]",
        }
        prefix = prefix_map.get(row_type, "[block]")

        table_data = str(row.get("table_data", "")).strip()
        content = str(row.get("content", "")).strip()

        if row_type == "table" and table_data:
            table_lines = table_data.replace(" \\n ", "\n").replace("\\n", "\n")
            lines = [
                f"[table_row] {line.strip()}"
                for line in table_lines.splitlines()
                if line.strip()
            ]
            return "\n".join(lines)

        if content:
            return f"{prefix} {content}"

        if table_data:
            table_lines = table_data.replace(" \\n ", "\n").replace("\\n", "\n")
            return f"{prefix}\n{table_lines}"

        return ""

    text_parts: List[str] = []
    for row in sorted_rows:
        formatted = format_row(row)
        if formatted:
            text_parts.append(formatted)

    return "\n\n".join(text_parts)
