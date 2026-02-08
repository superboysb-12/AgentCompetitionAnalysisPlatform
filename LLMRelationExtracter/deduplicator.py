"""
Result deduplicator - merge duplicate product extractions from sliding windows.
"""

from typing import Dict, List


def deduplicate_results(all_results: List[Dict]) -> List[Dict]:
    """
    去重并合并产品抽取结果。

    面向过程的去重流程：
    1. 收集所有产品
    2. 按product_model分组
    3. 合并同一型号的多个抽取结果

    Args:
        all_results: List of extraction result dicts, each containing a "results" key

    Returns:
        Deduplicated list of product dicts
    """
    # 阶段1：收集所有产品
    all_products = collect_all_products(all_results)

    # 阶段2：按product_model分组
    grouped_products = group_products_by_model(all_products)

    # 阶段3：合并每组产品
    merged_products = merge_grouped_products(grouped_products)

    return merged_products


def collect_all_products(all_results: List[Dict]) -> List[Dict]:
    """
    从所有结果中收集产品列表。

    Args:
        all_results: 包含"results"键的结果列表

    Returns:
        所有产品的扁平列表
    """
    all_products = []
    for result_batch in all_results:
        products = result_batch.get("results", [])
        all_products.extend(products)
    return all_products


def group_products_by_model(products: List[Dict]) -> Dict[str, List[Dict]]:
    """
    按product_model分组产品。

    Args:
        products: 产品列表

    Returns:
        以product_model为key的产品分组字典
    """
    grouped = {}
    for product in products:
        model = product.get("product_model", "").strip()
        if not model:
            # 没有型号的产品单独处理，使用唯一ID
            model = f"_no_model_{id(product)}"

        if model not in grouped:
            grouped[model] = []
        grouped[model].append(product)

    return grouped


def merge_grouped_products(grouped_products: Dict[str, List[Dict]]) -> List[Dict]:
    """
    合并每组产品。

    Args:
        grouped_products: 按型号分组的产品字典

    Returns:
        合并后的产品列表
    """
    merged_products = []
    for model, products in grouped_products.items():
        if len(products) == 1:
            # 只有一个产品，直接使用
            merged_products.append(products[0])
        else:
            # 多个产品，合并它们
            merged = merge_multiple_products(products)
            merged_products.append(merged)

    return merged_products


def merge_multiple_products(products: List[Dict]) -> Dict:
    """
    合并多个相同型号的产品。

    合并策略：
    - 字符串字段：保留最长的非空值
    - 列表字段：合并并去重
    - 性能参数：合并并去重

    Args:
        products: 相同型号的产品列表

    Returns:
        合并后的产品字典
    """
    if not products:
        return {}

    if len(products) == 1:
        return products[0]

    # 从第一个产品开始，逐个合并
    merged = products[0].copy()
    for product in products[1:]:
        merged = merge_two_products(merged, product)

    return merged


def merge_two_products(product1: Dict, product2: Dict) -> Dict:
    """
    合并两个产品字典。

    合并策略：
    - 字符串字段：保留最长的非空值
    - 列表字段：合并并去重
    - 性能参数：合并并去重

    Args:
        product1: 第一个产品
        product2: 第二个产品

    Returns:
        合并后的产品字典
    """
    merged = {}

    # 字符串字段列表
    string_fields = [
        "brand",
        "category",
        "series",
        "product_model",
        "manufacturer",
        "refrigerant",
        "energy_efficiency_grade",
    ]

    # 合并字符串字段
    for field in string_fields:
        merged[field] = merge_string_field(
            product1.get(field, ""),
            product2.get(field, "")
        )

    # 列表字段
    list_fields = ["features", "key_components", "fact_text", "evidence"]

    # 合并列表字段
    for field in list_fields:
        merged[field] = merge_list_field(
            product1.get(field, []),
            product2.get(field, [])
        )

    # 合并性能参数
    merged["performance_specs"] = merge_performance_specs(
        product1.get("performance_specs", []),
        product2.get("performance_specs", [])
    )

    return merged


def merge_string_field(value1: str, value2: str) -> str:
    """
    合并字符串字段，保留最长的非空值。

    Args:
        value1: 第一个值
        value2: 第二个值

    Returns:
        合并后的值
    """
    val1 = value1.strip() if value1 else ""
    val2 = value2.strip() if value2 else ""

    if not val1:
        return val2
    if not val2:
        return val1

    # 都非空，返回更长的
    return val1 if len(val1) >= len(val2) else val2


def merge_list_field(list1: List[str], list2: List[str]) -> List[str]:
    """
    合并列表字段，去重并保持顺序。

    Args:
        list1: 第一个列表
        list2: 第二个列表

    Returns:
        合并后的列表
    """
    seen = set()
    result = []

    for item in list1 + list2:
        item_stripped = item.strip() if isinstance(item, str) else str(item).strip()
        if item_stripped and item_stripped not in seen:
            seen.add(item_stripped)
            result.append(item_stripped)

    return result


def merge_performance_specs(specs1: List[Dict], specs2: List[Dict]) -> List[Dict]:
    """
    合并性能参数列表。

    按(name, value, unit)去重，保留raw更长的那个。

    Args:
        specs1: 第一个性能参数列表
        specs2: 第二个性能参数列表

    Returns:
        合并后的性能参数列表
    """
    spec_map = {}

    for spec in specs1 + specs2:
        name = spec.get("name", "").strip()
        value = spec.get("value", "").strip()
        unit = spec.get("unit", "").strip()
        raw = spec.get("raw", "").strip()

        if not name or not raw:
            continue

        key = (name, value, unit)

        if key not in spec_map:
            spec_map[key] = spec
        else:
            # 保留raw更长的
            existing_raw = spec_map[key].get("raw", "")
            if len(raw) > len(existing_raw):
                spec_map[key] = spec

    return list(spec_map.values())
