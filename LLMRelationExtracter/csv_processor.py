"""
CSV processor - load CSV, group by page, sort, and concatenate text.
"""

import ast
import csv
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from backend_v2.settings import RELATION_EXTRACTOR_CONFIG


def load_pages_from_csv(csv_path: str) -> Iterator[Tuple[str, Dict]]:
    """
    Load pages from CSV and yield concatenated text with metadata.
    Improvements:
    - Sort blocks by y then x to reduce列串扰.
    - Preserve block type tags to help LLM区分标题/文本/表格.
    - For表格优先使用 table_data，保留行分隔符，避免压扁结构.
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

    for page_num, rows in pages.items():
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

            # Prefer structured table_data when present
            if row_type == "table" and table_data:
                # table_data in CSV stores escaped newlines as " \\n " or "\n"
                table_lines = table_data.replace(" \\n ", "\n").replace("\\n", "\n")
                lines = [
                    f"[table_row] {line.strip()}"
                    for line in table_lines.splitlines()
                    if line.strip()
                ]
                return "\n".join(lines)

            if content:
                return f"{prefix} {content}"

            # Fallback: if no content but table_data exists
            if table_data:
                table_lines = table_data.replace(" \\n ", "\n").replace("\\n", "\n")
                return f"{prefix}\n{table_lines}"

            return ""

        text_parts: List[str] = []
        for row in sorted_rows:
            formatted = format_row(row)
            if formatted:
                text_parts.append(formatted)

        text = "\n\n".join(text_parts)

        metadata = {
            "sample": rows[0].get("sample", ""),
            "file": rows[0].get("file", ""),
            "page": int(page_num) if str(page_num).isdigit() else page_num,
            "row_count": len(sorted_rows),
        }

        yield text, metadata
