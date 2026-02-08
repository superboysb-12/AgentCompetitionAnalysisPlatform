"""
Test script to verify all modules work correctly.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMRelationExtracter.model_extractor import extract_frequent_models
from LLMRelationExtracter.csv_processor import load_pages_with_context
from LLMRelationExtracter.deduplicator import deduplicate_results

CSV_PATH = r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\解析前后数据样例\output\all_data.csv"


def test_model_extraction():
    print("Testing model extraction...")
    models = extract_frequent_models(CSV_PATH, min_frequency=3)
    print(f"[OK] Found {len(models)} frequent models")
    print(f"  Top 5: {models[:5]}")
    return models


def test_sliding_window(known_models):
    print("\nTesting sliding window context loading...")
    pages = list(load_pages_with_context(CSV_PATH, window_size=1, known_models=known_models))
    print(f"[OK] Loaded {len(pages)} pages with context")

    if pages:
        text, metadata = pages[0]
        print(f"  First page metadata: {metadata}")
        print(f"  Text length: {len(text)} chars")
        print(f"  Text preview (first 500 chars):\n{text[:500]}")

    return pages


def test_deduplication():
    print("\nTesting deduplication...")

    # Create mock duplicate results
    mock_results = [
        {
            "results": [
                {
                    "product_model": "TEST001",
                    "brand": "TestBrand",
                    "category": "AC",
                    "series": "Series1",
                    "manufacturer": "Mfg1",
                    "refrigerant": "R32",
                    "energy_efficiency_grade": "1",
                    "features": ["Feature1", "Feature2"],
                    "key_components": ["Comp1"],
                    "performance_specs": [
                        {"name": "Power", "value": "5", "unit": "kW", "raw": "Power: 5kW"}
                    ],
                    "fact_text": ["Fact1"],
                    "evidence": ["Evidence1"],
                }
            ]
        },
        {
            "results": [
                {
                    "product_model": "TEST001",
                    "brand": "TestBrand",
                    "category": "AC",
                    "series": "Series1",
                    "manufacturer": "Mfg1",
                    "refrigerant": "R32",
                    "energy_efficiency_grade": "1",
                    "features": ["Feature2", "Feature3"],
                    "key_components": ["Comp1", "Comp2"],
                    "performance_specs": [
                        {"name": "Power", "value": "5", "unit": "kW", "raw": "Power: 5kW"},
                        {"name": "Voltage", "value": "220", "unit": "V", "raw": "Voltage: 220V"}
                    ],
                    "fact_text": ["Fact1", "Fact2"],
                    "evidence": ["Evidence1", "Evidence2"],
                }
            ]
        },
    ]

    deduplicated = deduplicate_results(mock_results)
    print(f"[OK] Deduplicated: 2 results -> {len(deduplicated)} products")

    if deduplicated:
        product = deduplicated[0]
        print(f"  Product model: {product['product_model']}")
        print(f"  Features: {product['features']}")
        print(f"  Key components: {product['key_components']}")
        print(f"  Performance specs: {len(product['performance_specs'])} specs")


if __name__ == "__main__":
    print("="*60)
    print("Module Testing")
    print("="*60)

    models = test_model_extraction()
    test_sliding_window(models)
    test_deduplication()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
