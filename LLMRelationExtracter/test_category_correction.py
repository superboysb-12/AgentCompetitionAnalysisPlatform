"""
Test script for category correction functionality.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMRelationExtracter.category_corrector import (
    analyze_document_context,
    correct_all_categories,
    correct_product_category,
    detect_category_by_keywords,
)


def test_keyword_detection():
    """测试关键字检测功能"""
    print("="*60)
    print("Test 1: Keyword Detection")
    print("="*60)

    test_cases = [
        ("风管式室内机", "室内机"),
        ("壁挂式空调", "室内机"),
        ("室外机主机", "室外机"),
        ("压缩机组", "室外机"),
        ("MDV-D280W", None),
        ("吊顶式新风机", "室内机"),
        ("落地式空调", "室内机"),
    ]

    for text, expected in test_cases:
        result = detect_category_by_keywords(text)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} '{text}' -> {result} (expected: {expected})")


def test_single_product_correction():
    """测试单个产品的category修正"""
    print("\n" + "="*60)
    print("Test 2: Single Product Correction")
    print("="*60)

    test_products = [
        {
            "product_model": "MDV-D280W",
            "series": "风管机",
            "category": "",
        },
        {
            "product_model": "RUXYQ22CA",
            "series": "多联机组",
            "category": "",
        },
        {
            "product_model": "FXS-20MVJU",
            "series": "壁挂式室内机",
            "category": "空调",
        },
    ]

    for product in test_products:
        corrected = correct_product_category(product.copy())
        print(f"\nProduct: {product['product_model']}")
        print(f"  Series: {product['series']}")
        print(f"  Original category: '{product['category']}'")
        print(f"  Corrected category: '{corrected}'")


def test_document_context_analysis():
    """测试文档上下文分析"""
    print("\n" + "="*60)
    print("Test 3: Document Context Analysis")
    print("="*60)

    mock_products = [
        {"category": "室外机", "series": "多联机组"},
        {"category": "室外机", "series": "多联机组"},
        {"category": "室内机", "series": "风管机"},
        {"category": "室外机", "series": "多联机组"},
        {"category": "室内机", "series": "壁挂机"},
    ]

    context = analyze_document_context(mock_products)
    print(f"Main type: {context['main_type']}")
    print(f"Main series: {context['main_series']}")
    print(f"Category counts: {context['category_counts']}")
    print(f"Series counts: {context['series_counts']}")


def test_full_correction_flow():
    """测试完整的修正流程"""
    print("\n" + "="*60)
    print("Test 4: Full Correction Flow")
    print("="*60)

    mock_products = [
        {
            "product_model": "MDV-D280W",
            "series": "风管机",
            "category": "",
            "brand": "美的",
        },
        {
            "product_model": "MDV-D320W",
            "series": "风管机",
            "category": "",
            "brand": "美的",
        },
        {
            "product_model": "RUXYQ22CA",
            "series": "多联机组",
            "category": "",
            "brand": "美的",
        },
        {
            "product_model": "FXS-20MVJU",
            "series": "壁挂式",
            "category": "空调",
            "brand": "大金",
        },
    ]

    print("\nBefore correction:")
    for p in mock_products:
        print(f"  {p['product_model']} ({p['series']}) -> category: '{p['category']}'")

    corrected = correct_all_categories(mock_products)

    print("\nAfter correction:")
    for p in corrected:
        print(f"  {p['product_model']} ({p['series']}) -> category: '{p['category']}'")


if __name__ == "__main__":
    test_keyword_detection()
    test_single_product_correction()
    test_document_context_analysis()
    test_full_correction_flow()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
