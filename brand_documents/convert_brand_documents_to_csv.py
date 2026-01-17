import argparse
import csv
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


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


def extract_text_from_html(html_content: str) -> str:
    text = re.sub(r"<[^>]+>", "", html_content)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_table_from_html(html_content: str):
    parser = TableHTMLParser()
    parser.feed(html_content)
    return parser.get_table()


def get_source_file_name(document: dict) -> str:
    source_url = document.get("source_url", "")
    if source_url:
        return source_url.rstrip("/").split("/")[-1]
    doc_id = document.get("id", "")
    return f"{doc_id}.json" if doc_id else "unknown.json"


def derive_sample_name(source_stem: str, doc_id: str) -> str:
    base = source_stem or doc_id
    if base.endswith("_result"):
        return base[: -len("_result")]
    return base


def extract_text_from_lines(lines) -> str:
    texts = []
    for line in lines:
        for span in line.get("spans", []):
            if span.get("type") == "text":
                content = span.get("content", "")
                if content:
                    texts.append(content)
    return " ".join(texts).strip()


def process_document(document: dict) -> list:
    content_raw = document.get("content", "")
    if not content_raw:
        return []

    try:
        content = json.loads(content_raw)
    except json.JSONDecodeError:
        return []

    source_file = get_source_file_name(document)
    source_stem = Path(source_file).stem
    doc_id = document.get("id", "")
    sample_name = derive_sample_name(source_stem, doc_id)

    results = []
    for page_info in content.get("pdf_info", []):
        page_idx = page_info.get("page_idx", 0)
        page_label = f"{source_stem}_page_{page_idx}"

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
                for inner_block in block.get("blocks", []):
                    for line in inner_block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("type") == "table":
                                html = span.get("html", "")
                                if html:
                                    result["content"] = extract_text_from_html(html)
                                    table_data = parse_table_from_html(html)
                                    if table_data:
                                        table_str = "\n".join(
                                            [" | ".join(row) for row in table_data]
                                        )
                                        result["table_data"] = table_str
                                result["image_path"] = span.get("image_path", "")

            elif block_type in ("text", "title", "list"):
                result["content"] = extract_text_from_lines(block.get("lines", []))

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
        default=r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\brand_documents\brand_documents",
        help="Directory with brand JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\brand_documents\output",
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

        rows = []
        for document in data.get("documents", []):
            rows.extend(process_document(document))

        output_file = output_dir / f"{json_path.stem}_all_data.csv"
        save_to_csv(rows, output_file)


if __name__ == "__main__":
    main()
