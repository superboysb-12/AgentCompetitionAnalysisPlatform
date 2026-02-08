"""
Product filter - filter out invalid or empty products.
"""

from typing import Dict, List


def is_valid_product(product: Dict) -> bool:
    """
    判断产品是否有效。

    有效产品至少需要满足以下条件之一：
    - 有product_model
    - 有brand
    - 有category
    - 有series

    Args:
        product: 产品字典

    Returns:
        True if valid, False otherwise
    """
    # 关键字段列表
    key_fields = ["brand", "category", "series", "product_model"]

    # 检查是否至少有一个关键字段非空
    for field in key_fields:
        value = product.get(field, "").strip()
        if value:
            return True

    return False


def filter_empty_products(products: List[Dict]) -> List[Dict]:
    """
    过滤掉全空的产品。

    面向过程的过滤流程：
    1. 遍历所有产品
    2. 检查每个产品是否有效
    3. 只保留有效产品

    Args:
        products: 产品列表

    Returns:
        过滤后的产品列表
    """
    valid_products = []
    filtered_count = 0

    for product in products:
        if is_valid_product(product):
            valid_products.append(product)
        else:
            filtered_count += 1

    return valid_products


def get_product_completeness_score(product: Dict) -> int:
    """
    计算产品信息的完整度分数。

    用于排序或筛选产品。

    Args:
        product: 产品字典

    Returns:
        完整度分数（0-100）
    """
    score = 0

    # 关键字段（每个10分）
    key_fields = ["brand", "category", "series", "product_model"]
    for field in key_fields:
        if product.get(field, "").strip():
            score += 10

    # 次要字段（每个5分）
    secondary_fields = ["manufacturer", "refrigerant", "energy_efficiency_grade"]
    for field in secondary_fields:
        if product.get(field, "").strip():
            score += 5

    # 列表字段（每个有内容加5分）
    list_fields = ["features", "key_components", "performance_specs"]
    for field in list_fields:
        if product.get(field, []):
            score += 5

    return min(score, 100)
