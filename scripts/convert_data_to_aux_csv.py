import argparse
import csv
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Tuple


class TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_table: List[List[str]] = []
        self.current_row: List[str] = []
        self.in_td = False
        self.in_th = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_td = True
            self.in_th = tag == "th"

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self.current_row:
            self.current_table.append(self.current_row)
        elif tag in ("td", "th"):
            self.in_td = False
            self.in_th = False

    def handle_data(self, data: str) -> None:
        if self.in_td or self.in_th:
            text = data.strip()
            if text:
                self.current_row.append(text)

    def get_table(self) -> List[List[str]]:
        return self.current_table


def extract_text_from_html(html_content: str) -> str:
    text = re.sub(r"<[^>]+>", "", html_content)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_table_from_html(html_content: str) -> List[List[str]]:
    parser = TableHTMLParser()
    parser.feed(html_content)
    return parser.get_table()


def extract_text_from_lines(lines: Iterable[dict]) -> str:
    texts: List[str] = []
    for line in lines or []:
        for span in line.get("spans", []):
            if span.get("type") == "text":
                content = span.get("content", "")
                if content:
                    texts.append(content)
    return " ".join(texts).strip()


def parse_file_metadata(path: Path) -> Tuple[str, str, int]:
    """
    Derive sample id, file label, and page offset from filename.
    Expected patterns:
    - part_{start}_{end}_{sample}_result.json
    - {sample}_result.json
    """
    stem = path.stem
    base = stem[:-len("_result")] if stem.endswith("_result") else stem

    part_match = re.match(r"part_(\d+)_\d+_(.+)", base)
    if part_match:
        start_offset = int(part_match.group(1))
        sample_id = part_match.group(2)
        file_label = f"{sample_id}_result.json"
    else:
        start_offset = 0
        sample_id = base
        file_label = f"{sample_id}_result.json" if stem.endswith("_result") else path.name

    return sample_id, file_label, start_offset


def process_json_file(path: Path) -> List[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"Skip {path} (read error: {exc})")
        return []

    sample_id, file_label, start_offset = parse_file_metadata(path)
    results: List[dict] = []

    for page_info in data.get("pdf_info", []):
        page_idx = page_info.get("page_idx", 0)
        try:
            page_number = int(page_idx)
        except Exception:
            page_number = 0
        page_number += start_offset
        page_label = f"{sample_id}_result_page_{page_number}"

        for block in page_info.get("para_blocks", []):
            block_type = block.get("type", "")
            row = {
                "sample": sample_id,
                "file": file_label,
                "page": page_label,
                "type": block_type,
                "bbox": str(block.get("bbox", [])),
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
                                    row["content"] = extract_text_from_html(html)
                                    table_data = parse_table_from_html(html)
                                    if table_data:
                                        row["table_data"] = "\n".join(
                                            [" | ".join(r) for r in table_data]
                                        )
                                if not row["image_path"]:
                                    row["image_path"] = span.get("image_path", "")

            elif block_type in {"text", "title", "list"}:
                row["content"] = extract_text_from_lines(block.get("lines", []))

            elif block_type == "image":
                for inner_block in block.get("blocks", []):
                    for line in inner_block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("type") == "image":
                                row["image_path"] = span.get("image_path", "")
                                row["content"] = "[image]"

            else:
                row["content"] = extract_text_from_lines(block.get("lines", []))

            if row["content"] or row["table_data"] or row["image_path"]:
                results.append(row)

        for block in page_info.get("discarded_blocks", []):
            discard_type = block.get("type", "")
            if discard_type not in {"header", "footer"}:
                continue

            content = extract_text_from_lines(block.get("lines", []))
            if not content:
                continue

            results.append(
                {
                    "sample": sample_id,
                    "file": file_label,
                    "page": page_label,
                    "type": f"discarded_{discard_type}",
                    "bbox": str(block.get("bbox", [])),
                    "content": content,
                    "table_data": "",
                    "image_path": "",
                }
            )

    return results


def write_csv(rows: List[dict], output_file: Path) -> None:
    if not rows:
        print("No rows to write.")
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
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            row_copy["table_data"] = row_copy["table_data"].replace("\n", " \\n ")
            writer.writerow(row_copy)

    print(f"Wrote {len(rows)} rows to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert parsed PDF JSON blocks to aux-style CSV."
    )
    parser.add_argument(
        "--input-dir",
        default=r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\data",
        help="Directory containing *_result.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for result JSON files.",
    )
    parser.add_argument(
        "--output-file",
        default=(
            r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform"
            r"\results\aux_documents_all_data_generated.csv"
        ),
        help="Destination CSV path (will not overwrite the existing aux_documents_all_data.csv by default).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    json_paths = sorted(input_dir.glob(args.pattern))

    if not json_paths:
        print(f"No JSON files found in {input_dir}")
        return

    all_rows: List[dict] = []
    for path in json_paths:
        all_rows.extend(process_json_file(path))

    write_csv(all_rows, output_file)


if __name__ == "__main__":
    main()
