"""
Category corrector - correct product category based on keywords and context.
"""

from typing import Dict, List, Optional


# 室内机关键字
INDOOR_KEYWORDS = [
    "风管", "室内", "吊顶", "嵌入", "落地", "壁挂", "挂壁",
    "新风", "全热", "除湿", "内机", "厨房", "管道", "吊落"
]

# 室外机关键字
OUTDOOR_KEYWORDS = [
    "室外机", "外机", "主机", "压缩机", "冷凝器", "室外", "户外"
]


def analyze_document_context(all_products: List[Dict]) -> Dict:
    """
    分析整个文档的主要产品系列和类型。

    通过统计所有产品的category和series，判断文档的主要产品类型。

    Args:
        all_products: 所有产品列表

    Returns:
        文档上下文信息，包含主要产品类型和系列
    """
    if not all_products:
        return {
            "main_type": None,
            "main_series": None,
            "category_counts": {},
            "series_counts": {},
        }

    category_counts = {}
    series_counts = {}

    for product in all_products:
        category = product.get("category", "").strip()
        series = product.get("series", "").strip()

        if category:
            category_counts[category] = category_counts.get(category, 0) + 1

        if series:
            series_counts[series] = series_counts.get(series, 0) + 1

    # 找出最常见的category和series
    main_type = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
    main_series = max(series_counts.items(), key=lambda x: x[1])[0] if series_counts else None

    return {
        "main_type": main_type,
        "main_series": main_series,
        "category_counts": category_counts,
        "series_counts": series_counts,
    }


def detect_category_by_keywords(text: str) -> Optional[str]:
    """
    根据关键字检测产品类型。

    Args:
        text: 要检测的文本（通常是product_model或series）

    Returns:
        检测到的类型（'室内机'、'室外机'）或None
    """
    if not text:
        return None

    text_lower = text.lower()

    # 检查室内机关键字
    for keyword in INDOOR_KEYWORDS:
        if keyword in text_lower:
            return "室内机"

    # 检查室外机关键字
    for keyword in OUTDOOR_KEYWORDS:
        if keyword in text_lower:
            return "室外机"

    return None


def correct_product_category(
    product: Dict,
    document_context: Optional[Dict] = None
) -> str:
    """
    修正单个产品的category字段。

    按照优先级规则：
    1. 如果已有明确的category且合理，保留
    2. 根据product_model和series中的关键字判断
    3. 根据文档主要类型判断

    Args:
        product: 产品字典
        document_context: 文档上下文信息（可选）

    Returns:
        修正后的category
    """
    current_category = product.get("category", "").strip()
    product_model = product.get("product_model", "").strip()
    series = product.get("series", "").strip()

    # 优先级1：如果已有明确的室内机/室外机标注，保留
    if current_category in ["室内机", "室外机", "内机", "外机"]:
        # 统一格式
        if current_category in ["内机", "室内机"]:
            return "室内机"
        if current_category in ["外机", "室外机"]:
            return "室外机"

    # 优先级2：根据product_model关键字判断
    detected_from_model = detect_category_by_keywords(product_model)
    if detected_from_model:
        return detected_from_model

    # 优先级3：根据series关键字判断
    detected_from_series = detect_category_by_keywords(series)
    if detected_from_series:
        return detected_from_series

    # 优先级4：根据文档主要类型判断
    if document_context and document_context.get("main_type"):
        main_type = document_context["main_type"]
        main_series = document_context.get("main_series")

        # 如果产品系列与主要系列相同，使用主要类型
        if main_series and series == main_series:
            return main_type

        # 如果文档主要是室外机，但当前产品没有明确类型，可能是适配的内机
        # 这种情况保留原category或使用通用类别

    # 如果都无法判断，保留原category
    return current_category if current_category else ""


def correct_all_categories(products: List[Dict]) -> List[Dict]:
    """
    修正所有产品的category字段。

    这是一个面向过程的函数，完成整个category修正流程：
    1. 分析文档上下文
    2. 逐个修正产品category

    Args:
        products: 产品列表

    Returns:
        修正后的产品列表
    """
    if not products:
        return products

    # 阶段1：分析文档上下文
    document_context = analyze_document_context(products)

    # 阶段2：修正每个产品的category
    corrected_products = []
    for product in products:
        corrected_category = correct_product_category(product, document_context)
        product["category"] = corrected_category
        corrected_products.append(product)

    return corrected_products
