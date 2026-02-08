"""
Quick single-CSV tester for the staged (v2) relation extractor.

Usage:
  python LLMRelationExtracter_v2/test_csv_extract.py

Adjust CSV_PATH below to point at your CSV. Output JSON is written next to it.
"""

from pathlib import Path
import json
import sys

# Project root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMRelationExtracter_v2 import extract_relations_multistage  # noqa: E402

# ---- configure here -------------------------------------------------------
# Set to your CSV file path
CSV_PATH = Path(r"D:\AgentCompetitionAnalysisPlatform\results\gree_documents_all_data.csv")
# Defaults: results saved beside CSV
OUTPUT_PATH = None  # set to Path(...) to override
ERROR_LOG_PATH = None  # set to Path(...) to override
MAX_CONCURRENT = 10
WINDOW_SIZE = 1
USE_SLIDING_WINDOW = True
SHOW_PROGRESS = True
# --------------------------------------------------------------------------


def main() -> None:
    csv_path = Path(CSV_PATH).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path = (
        Path(OUTPUT_PATH).expanduser().resolve()
        if OUTPUT_PATH
        else csv_path.with_name("relation_results.json")
    )
    error_log = (
        Path(ERROR_LOG_PATH).expanduser().resolve()
        if ERROR_LOG_PATH
        else csv_path.with_name("relation_errors.log")
    )

    print(f"[v2 test] CSV: {csv_path}")
    print(f"[v2 test] Output: {output_path}")
    print(f"[v2 test] Errors: {error_log}")
    print(
        f"[v2 test] Concurrency={MAX_CONCURRENT}, window_size={WINDOW_SIZE}, sliding={USE_SLIDING_WINDOW}"
    )

    extract_relations_multistage(
        csv_path=csv_path,
        output_path=output_path,
        error_log_path=error_log,
        max_concurrent=MAX_CONCURRENT,
        window_size=WINDOW_SIZE,
        use_sliding_window=USE_SLIDING_WINDOW,
        show_progress=SHOW_PROGRESS,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    product_count = sum(len(batch.get("results", [])) for batch in data)
    print(f"[v2 test] Done. Products: {product_count}")
    print(f"[v2 test] Sample product: {json.dumps(data[0].get('results', [])[0], ensure_ascii=False, indent=2) if product_count else 'N/A'}")


if __name__ == "__main__":
    main()
