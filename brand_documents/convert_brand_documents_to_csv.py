import argparse
import csv
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional


class TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_table = []
        self.current_row = []
        self.in_td = False
        self.in_th = False

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_td = True
            self.in_th = tag == "th"

    def handle_endtag(self, tag) -> None:
        if tag == "tr" and self.current_row:
            self.current_table.append(self.current_row)
        elif tag in ("td", "th"):
            self.in_td = False
            self.in_th = False

    def handle_data(self, data) -> None:
        if self.in_td or self.in_th:
            self.current_row.append(data.strip())

    def get_table(self):
        return self.current_table


def looks_like_numeric_range_title(text: str) -> bool:
    """
    Filter OCR noise titles like:
    "2.2/2.5/2.8/3.2/3.6" or "14/22.4/28/33.5"
    """
    raw = str(text or "").strip()
    if not raw:
        return False
    compact = re.sub(r"\s+", "", raw).lower()
    compact = (
        compact.replace("kw", "")
        .replace("hp", "")
        .replace("匹", "")
        .replace("℃", "")
    )
    if re.search(r"[a-z一-龥]", compact):
        return False
    if not re.fullmatch(r"[0-9./~+\-x×*()]+", compact):
        return False
    nums = re.findall(r"\d+(?:\.\d+)?", compact)
    if len(nums) < 2:
        return False
    return any(sep in compact for sep in ("/", "~", "-", "x", "×"))


def extract_text_from_html(html_content: str) -> str:
    text = re.sub(r"<[^>]+>", "", html_content)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_table_from_html(html_content: str):
    parser = TableHTMLParser()
    parser.feed(html_content)
    return parser.get_table()


def normalize_table_rows(table_data) -> list:
    normalized = []
    for row in table_data or []:
        if not isinstance(row, (list, tuple)):
            continue
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        if any(cells):
            normalized.append(cells)
    return normalized


def get_source_file_name(document: dict) -> str:
    source_url = document.get("source_url", "")
    if source_url:
        return source_url.rstrip("/").split("/")[-1]
    doc_id = document.get("id", "")
    return f"{doc_id}.json" if doc_id else "unknown.json"


def extract_text_from_lines(lines) -> str:
    texts = []
    for line in lines:
        for span in line.get("spans", []):
            if span.get("type") == "text":
                content = span.get("content", "")
                if content:
                    texts.append(content)
    return " ".join(texts).strip()


def parse_content(document: dict) -> Optional[dict]:
    content_raw = document.get("content", "")
    if not content_raw:
        return None
    if isinstance(content_raw, dict):
        return content_raw

    try:
        content = json.loads(content_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(content, dict):
        return None
    return content


def parse_page_idx(page_info: dict) -> int:
    try:
        return int(page_info.get("page_idx", 0))
    except Exception:
        return 0


def process_page(
    page_info: dict,
    sample_name: str,
    source_file: str,
    page_label: str,
) -> list:
    results = []
    for block in page_info.get("para_blocks", []):
        block_type = block.get("type", "")
        bbox = block.get("bbox", [])

        result = {
            "sample": sample_name,
            "file": source_file,
            "page": page_label,
            "type": block_type,
            "bbox": str(bbox),
            "content": "",
            "table_data": "",
            "image_path": "",
        }

        if block_type == "table":
            table_text_fallback = ""
            for inner_block in block.get("blocks", []):
                for line in inner_block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("type") == "table":
                            html = span.get("html", "")
                            if html:
                                table_data = normalize_table_rows(parse_table_from_html(html))
                                if table_data:
                                    table_str = "\n".join([" | ".join(row) for row in table_data])
                                    result["table_data"] = table_str
                                    # Avoid duplicating full table as flattened content.
                                    result["content"] = ""
                                else:
                                    table_text_fallback = extract_text_from_html(html)
                            result["image_path"] = span.get("image_path", "")
            if not result["table_data"] and table_text_fallback:
                result["content"] = table_text_fallback

        elif block_type in ("text", "title", "list"):
            result["content"] = extract_text_from_lines(block.get("lines", []))
            if block_type == "title" and looks_like_numeric_range_title(result["content"]):
                continue

        elif block_type == "image":
            for inner_block in block.get("blocks", []):
                for line in inner_block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("type") == "image":
                            result["image_path"] = span.get("image_path", "")
                            result["content"] = "[image]"

        if result["content"] or result["table_data"]:
            results.append(result)

    for block in page_info.get("discarded_blocks", []):
        block_type = block.get("type", "")
        if block_type not in ("header", "footer"):
            continue
        result = {
            "sample": sample_name,
            "file": source_file,
            "page": page_label,
            "type": f"discarded_{block_type}",
            "bbox": str(block.get("bbox", [])),
            "content": "",
            "table_data": "",
            "image_path": "",
        }
        result["content"] = extract_text_from_lines(block.get("lines", []))
        if result["content"]:
            results.append(result)

    return results


def collect_page_units(documents: list) -> list:
    """
    Flatten all pages in JSON original order.
    Each unit represents one parsed page with its local page_idx.
    """
    units = []
    for document in documents:
        content = parse_content(document)
        if not content:
            continue

        source_file = get_source_file_name(document)
        for page_info in content.get("pdf_info", []):
            units.append(
                {
                    "source_file": source_file,
                    "page_idx": parse_page_idx(page_info),
                    "page_info": page_info,
                }
            )
    return units


def split_units_by_page_reset(units: list) -> list:
    """
    Split into logical documents by page_idx reset:
    - when page_idx jumps from >0 back to 0, start a new CSV.
    - consecutive 0 page_idx stay in the same logical document.
    """
    segments = []
    current = []
    prev_page_idx = None

    for unit in units:
        page_idx = unit.get("page_idx", 0)
        if current and page_idx == 0 and isinstance(prev_page_idx, int) and prev_page_idx > 0:
            segments.append(current)
            current = []
        current.append(unit)
        prev_page_idx = page_idx

    if current:
        segments.append(current)

    return segments


def build_rows_for_segment(segment: list, segment_name: str) -> list:
    """
    Build rows for one logical document segment.
    Rule:
    - consecutive page_idx == 0 are merged into one logical page label.
    """
    rows = []
    logical_page_idx = 0
    prev_page_idx = None

    for unit in segment:
        page_idx = unit.get("page_idx", 0)
        if prev_page_idx is not None and not (page_idx == 0 and prev_page_idx == 0):
            logical_page_idx += 1

        page_label = f"{segment_name}_page_{logical_page_idx}"
        rows.extend(
            process_page(
                page_info=unit["page_info"],
                sample_name=segment_name,
                source_file=unit["source_file"],
                page_label=page_label,
            )
        )
        prev_page_idx = page_idx

    return rows


def save_to_csv(data: list, output_file: Path) -> None:
    if not data:
        print(f"No data to save: {output_file}")
        return

    fieldnames = [
        "sample",
        "file",
        "page",
        "type",
        "bbox",
        "content",
        "table_data",
        "image_path",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            row_copy = dict(row)
            row_copy["table_data"] = row_copy["table_data"].replace("\n", " \\n ")
            writer.writerow(row_copy)
    print(f"Saved {len(data)} rows to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert brand documents JSON to CSV.")
    parser.add_argument(
        "--input-dir",
        default=r"D:\AgentCompetitionAnalysisPlatform\brand_documents",
        help="Directory with brand JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\AgentCompetitionAnalysisPlatform\results",
        help="Directory to write CSV outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        print(f"No JSON files found in {input_dir}")
        return

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        units = collect_page_units(data.get("documents", []))
        if not units:
            print(f"No page units found in {json_path}")
            continue

        segments = split_units_by_page_reset(units)
        print(f"{json_path.name}: split into {len(segments)} logical documents")

        for i, segment in enumerate(segments, 1):
            segment_name = f"{json_path.stem}_doc_{i:03d}"
            rows = build_rows_for_segment(segment, segment_name)
            output_file = output_dir / f"{segment_name}_all_data.csv"
            save_to_csv(rows, output_file)


if __name__ == "__main__":
    main()
