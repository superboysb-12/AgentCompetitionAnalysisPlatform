"""
Quick test script for running the relation extractor on a CSV file.

Usage:
    python LLMRelationExtracter/test_csv_extract.py

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

from LLMRelationExtracter import RelationExtractor  # noqa: E402
from LLMRelationExtracter.category_corrector import correct_all_categories  # noqa: E402
from LLMRelationExtracter.csv_processor import load_pages_with_context  # noqa: E402
from LLMRelationExtracter.deduplicator import deduplicate_results  # noqa: E402
from LLMRelationExtracter.model_extractor import extract_frequent_models  # noqa: E402
from LLMRelationExtracter.product_filter import filter_empty_products  # noqa: E402

# Adjust to your CSV path if needed
CSV_PATH = Path(r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\解析前后数据样例\output\all_data.csv")
# Output file for aggregated results
OUTPUT_PATH = Path("relation_results.json")
OUTPUT_PATH_DEDUP = Path("relation_results_deduplicated.json")


async def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print("Step 1: Extracting frequent product models...")
    known_models = extract_frequent_models(str(CSV_PATH), min_frequency=3)
    print(f"Found {len(known_models)} frequent models: {known_models[:10]}...")

    print("\nStep 2: Loading pages with sliding window context...")
    extractor = RelationExtractor()

    tasks = [
        asyncio.create_task(extractor.extract_relations_async(text, metadata))
        for text, metadata in load_pages_with_context(
            str(CSV_PATH),
            window_size=1,
            known_models=known_models,
        )
    ]

    print(f"Created {len(tasks)} extraction tasks")

    completed = []

    print("\nStep 3: Running extraction with context windows...")
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        res = await task
        count = len(res) if res else 0
        completed.append({"results": res})
        if res:
            print(f"[task] extracted {count} items, first: {res[0].get('product_model', 'N/A')}")

    # Save raw results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(completed, fp, ensure_ascii=False, indent=2)
    print(f"\nStep 4: Saved raw results to {OUTPUT_PATH.resolve()}")

    # Deduplicate results
    print("\nStep 5: Deduplicating results...")
    deduplicated = deduplicate_results(completed)
    print(f"Deduplicated: {len(sum([r.get('results', []) for r in completed], []))} -> {len(deduplicated)} products")

    # Correct categories
    print("\nStep 6: Correcting product categories...")
    corrected = correct_all_categories(deduplicated)
    category_counts = {}
    for product in corrected:
        cat = product.get("category", "")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print(f"Category distribution: {category_counts}")

    # Filter empty products
    print("\nStep 7: Filtering empty products...")
    filtered = filter_empty_products(corrected)
    print(f"Filtered: {len(corrected)} -> {len(filtered)} products (removed {len(corrected) - len(filtered)} empty products)")

    with OUTPUT_PATH_DEDUP.open("w", encoding="utf-8") as fp:
        json.dump(filtered, fp, ensure_ascii=False, indent=2)
    print(f"Saved final results to {OUTPUT_PATH_DEDUP.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
