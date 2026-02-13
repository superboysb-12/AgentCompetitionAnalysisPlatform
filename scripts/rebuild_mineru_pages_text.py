import argparse
import csv
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_table: List[List[str]] = []
        self.current_row: List[str] = []
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_table_html(html: str) -> List[List[str]]:
    parser = TableHTMLParser()
    parser.feed(html)
    return parser.get_table()


def extract_text_from_lines(lines: Iterable[dict]) -> str:
    parts: List[str] = []
    for line in lines or []:
        for span in line.get("spans", []) or []:
            if span.get("type") not in {"text", "inline_equation"}:
                continue
            content = str(span.get("content", ""))
            if content:
                parts.append(content)
    return normalize_whitespace("".join(parts))


def extract_table_text(block: dict) -> str:
    rows: List[List[str]] = []
    html_fallback = ""
    for inner in block.get("blocks", []) or []:
        for line in inner.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                if span.get("type") != "table":
                    continue
                html = span.get("html", "")
                if html:
                    table_rows = parse_table_html(html)
                    if table_rows:
                        rows.extend(table_rows)
                    else:
                        html_fallback = normalize_whitespace(unescape(re.sub(r"<[^>]+>", "", html)))
    if rows:
        formatted = []
        for row in rows:
            cells = [normalize_whitespace(str(cell)) for cell in row]
            cells = [c for c in cells if c]
            if cells:
                formatted.append(" | ".join(cells))
        if formatted:
            return "\n".join(formatted)

    line_fallback = extract_text_from_lines(block.get("lines", []))
    if line_fallback:
        return line_fallback
    return html_fallback


def safe_stem(name: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", name or "").strip("_")
    return cleaned or "unknown"


def sort_blocks_for_reading(blocks: List[dict]) -> List[dict]:
    def key(block: dict) -> Tuple[int, float, float]:
        idx = block.get("index")
        if not isinstance(idx, int):
            idx = 10**9
        bbox = block.get("bbox", []) or []
        y = float(bbox[1]) if len(bbox) > 1 else 10**9
        x = float(bbox[0]) if len(bbox) > 0 else 10**9
        return idx, y, x

    return sorted(blocks or [], key=key)


def block_to_text(block: dict, keep_image_placeholder: bool) -> str:
    block_type = str(block.get("type", ""))
    if block_type == "table":
        return extract_table_text(block)
    if block_type == "image":
        if not keep_image_placeholder:
            return ""
        return "[image]"
    return extract_text_from_lines(block.get("lines", []))


def compose_page_text(
    page_info: dict,
    include_discarded: bool,
    keep_image_placeholder: bool,
) -> str:
    lines: List[str] = []
    for block in sort_blocks_for_reading(page_info.get("para_blocks", []) or []):
        text = block_to_text(block, keep_image_placeholder=keep_image_placeholder)
        if text:
            lines.append(text)

    if include_discarded:
        for block in sort_blocks_for_reading(page_info.get("discarded_blocks", []) or []):
            text = block_to_text(block, keep_image_placeholder=keep_image_placeholder)
            if text:
                lines.append(text)

    return "\n".join(lines).strip()


def parse_document_content(document: dict) -> Optional[Dict]:
    raw = document.get("content", "")
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def iter_documents(root: Dict) -> List[dict]:
    docs = root.get("documents")
    if isinstance(docs, list):
        return docs
    if "pdf_info" in root:
        return [{"id": "single_document", "content": root}]
    return []


def parse_page_idx(page_info: dict) -> int:
    try:
        return int(page_info.get("page_idx", 0))
    except Exception:
        return 0


def collect_page_units(root: Dict) -> List[dict]:
    units: List[dict] = []
    for unit_order, document in enumerate(iter_documents(root)):
        parsed = parse_document_content(document)
        if not parsed:
            continue
        pages = parsed.get("pdf_info", [])
        if not isinstance(pages, list):
            continue
        for page in pages:
            units.append(
                {
                    "unit_order": unit_order,
                    "doc_id": str(document.get("id", "") or ""),
                    "source_url": str(document.get("source_url", "") or ""),
                    "page_idx": parse_page_idx(page),
                    "page_info": page,
                }
            )
    return units


def split_units_by_page_reset(units: List[dict]) -> List[List[dict]]:
    """
    Split logical documents in original JSON order:
    - when page_idx jumps from >0 back to 0, start a new logical document.
    - consecutive 0 values remain in current document.
    """
    segments: List[List[dict]] = []
    current: List[dict] = []
    prev_page_idx: Optional[int] = None

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


def group_consecutive_zero_pages(segment: List[dict]) -> List[List[dict]]:
    """
    Group pages for output txt files:
    - consecutive page_idx == 0 are merged into one output page.
    - all other transitions create a new output page.
    """
    grouped: List[List[dict]] = []
    current: List[dict] = []
    prev_page_idx: Optional[int] = None

    for unit in segment:
        page_idx = unit.get("page_idx", 0)
        if current and not (page_idx == 0 and prev_page_idx == 0):
            grouped.append(current)
            current = []
        current.append(unit)
        prev_page_idx = page_idx

    if current:
        grouped.append(current)
    return grouped


def rebuild_pages(
    input_json: Path,
    output_dir: Path,
    include_discarded: bool,
    keep_image_placeholder: bool,
) -> Tuple[int, int]:
    root = json.loads(input_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[dict] = []
    logical_doc_count = 0
    page_count = 0

    units = collect_page_units(root)
    segments = split_units_by_page_reset(units)
    base_name = safe_stem(input_json.stem)

    for seg_idx, segment in enumerate(segments, 1):
        doc_label = f"{base_name}_doc_{seg_idx:03d}"
        doc_dir = output_dir / doc_label
        doc_dir.mkdir(parents=True, exist_ok=True)
        logical_doc_count += 1

        logical_pages = group_consecutive_zero_pages(segment)
        for logical_page_idx, page_group in enumerate(logical_pages):
            page_texts = []
            raw_page_idxs = []
            doc_ids = []
            source_urls = []
            for unit in page_group:
                text = compose_page_text(
                    unit["page_info"],
                    include_discarded=include_discarded,
                    keep_image_placeholder=keep_image_placeholder,
                )
                if text:
                    page_texts.append(text)
                raw_page_idxs.append(str(unit.get("page_idx", 0)))
                doc_ids.append(unit.get("doc_id", ""))
                source_urls.append(unit.get("source_url", ""))

            page_text = "\n".join(page_texts).strip()
            out_file = doc_dir / f"page_{logical_page_idx:04d}.txt"
            out_file.write_text(page_text, encoding="utf-8")
            page_count += 1

            unique_doc_ids = [x for x in dict.fromkeys(doc_ids) if x]
            unique_source_urls = [x for x in dict.fromkeys(source_urls) if x]
            index_rows.append(
                {
                    "doc_label": doc_label,
                    "logical_page_idx": logical_page_idx,
                    "raw_page_idx_seq": ",".join(raw_page_idxs),
                    "unit_count": len(page_group),
                    "doc_ids": "|".join(unique_doc_ids),
                    "source_urls": "|".join(unique_source_urls),
                    "output_file": str(out_file.as_posix()),
                }
            )

    if index_rows:
        with (output_dir / "_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "doc_label",
                    "logical_page_idx",
                    "raw_page_idx_seq",
                    "unit_count",
                    "doc_ids",
                    "source_urls",
                    "output_file",
                ],
            )
            writer.writeheader()
            writer.writerows(index_rows)

    return logical_doc_count, page_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild page-level plain text from MinerU parsed JSON output."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to MinerU output JSON (supports both top-level documents and single pdf_info JSON).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write page text files.",
    )
    parser.add_argument(
        "--include-discarded",
        action="store_true",
        help="Also append discarded blocks (e.g. header/footer) to each page text.",
    )
    parser.add_argument(
        "--drop-image-placeholder",
        action="store_true",
        help="Do not keep [image] placeholders in page text.",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)

    docs, pages = rebuild_pages(
        input_json=input_json,
        output_dir=output_dir,
        include_discarded=args.include_discarded,
        keep_image_placeholder=not args.drop_image_placeholder,
    )
    print(f"Rebuilt {pages} pages from {docs} documents into {output_dir}")


if __name__ == "__main__":
    main()
