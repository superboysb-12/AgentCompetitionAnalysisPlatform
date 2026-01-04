"""
Quick test script for running the relation extractor on a CSV file.

Usage:
    python LLMRelationExtracter_v2/test_csv_extract.py

Edit CSV_PATH below if needed.
"""

import asyncio
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Ensure project root is on sys.path so package imports work when run directly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMRelationExtracter_v2 import RelationExtractor, load_pages_from_csv  # noqa: E402

# Adjust to your CSV path if needed
CSV_PATH = Path(r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\解析前后数据样例\output\all_data.csv")
# Output file for aggregated results
OUTPUT_PATH = Path("relation_results.json")


async def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    extractor = RelationExtractor()

    tasks = [
        asyncio.create_task(extractor.extract_relations_async(text, metadata))
        for text, metadata in load_pages_from_csv(str(CSV_PATH))
    ]

    completed = []

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            res = await task
            count = len(res) if res else 0
            completed.append({"results": res})
            print(f"[task] extracted {count} items")
            if res:
                print(res[:1])
        except Exception as err:  # noqa: BLE001
            completed.append({"error": str(err)})
            print(f"[task] failed: {err}")

    # Save aggregated results to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(completed, fp, ensure_ascii=False, indent=2)
    print(f"Saved results to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
