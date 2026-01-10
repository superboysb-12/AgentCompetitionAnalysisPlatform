"""
数据加载模块
负责加载和预处理关系抽取结果
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def load_relation_results(
    filepath: str,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载关系抽取结果

    原始格式: [{"results": [...]}, {"results": [...]}]
    转换为: 扁平化的实体列表

    Args:
        filepath: JSON文件路径
        logger: 日志记录器

    Returns:
        tuple: (实体列表, 原始数据)
    """
    if logger:
        logger.info(f"加载数据: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有产品并添加batch标记
    all_entities = []
    for batch_idx, batch_data in enumerate(data):
        for result in batch_data.get('results', []):
            result['_batch'] = batch_idx
            all_entities.append(result)

    if logger:
        logger.info(f"  原始实体数: {len(all_entities)}")
        logger.info(f"  原始batch数: {len(data)}")

    return all_entities, data


def filter_valid_entities(
    entities: List[Dict],
    require_fields: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """
    过滤有效实体

    Args:
        entities: 实体列表
        require_fields: 必需字段列表（至少有一个不为空）
        logger: 日志记录器

    Returns:
        过滤后的实体列表
    """
    if require_fields is None:
        require_fields = ['product_model', 'category']

    valid_entities = []
    invalid_count = 0

    for entity in entities:
        # 检查是否至少有一个必需字段
        is_valid = any(
            bool(entity.get(field)) for field in require_fields
        )

        if is_valid:
            valid_entities.append(entity)
        else:
            invalid_count += 1

    if logger:
        logger.info(f"  有效实体: {len(valid_entities)}")
        if invalid_count > 0:
            logger.info(f"  过滤掉无效实体: {invalid_count}")

    return valid_entities


def load_and_prepare_data(
    filepath: str,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载并准备数据（组合函数）

    Args:
        filepath: JSON文件路径
        logger: 日志记录器

    Returns:
        tuple: (预处理后的实体列表, 原始数据)
    """
    # 加载
    all_entities, original_data = load_relation_results(filepath, logger)

    # 过滤
    entities = filter_valid_entities(all_entities, logger=logger)

    return entities, original_data


def split_entities_by_brand(
    entities: List[Dict],
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    分离有品牌和无品牌的实体

    Args:
        entities: 实体列表
        logger: 日志记录器

    Returns:
        tuple: (有品牌实体列表, 无品牌实体列表)
    """
    with_brand = []
    without_brand = []

    for entity in entities:
        brand = entity.get('brand', '')
        if brand and brand.strip():
            with_brand.append(entity)
        else:
            without_brand.append(entity)

    if logger:
        total = len(entities)
        logger.info(f"  有品牌: {len(with_brand)} ({len(with_brand)/total*100:.1f}%)")
        logger.info(f"  无品牌: {len(without_brand)} ({len(without_brand)/total*100:.1f}%)")

    return with_brand, without_brand


def get_unique_brands(
    entities: List[Dict],
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    提取所有唯一的品牌名称

    Args:
        entities: 实体列表
        logger: 日志记录器

    Returns:
        唯一品牌名称列表（按频率排序）
    """
    from collections import Counter

    brands = [e.get('brand', '') for e in entities if e.get('brand')]
    brand_counter = Counter(brands)

    # 按频率排序
    sorted_brands = [brand for brand, _ in brand_counter.most_common()]

    if logger:
        logger.info(f"  唯一品牌数: {len(sorted_brands)}")
        logger.info(f"  品牌分布: 前5个品牌: {sorted_brands[:5]}")

    return sorted_brands


def filter_rare_brands(
    entities: List[Dict],
    min_frequency_ratio: float = 0.01,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    过滤掉稀有品牌（占比小于指定阈值）

    Args:
        entities: 实体列表
        min_frequency_ratio: 最小频率比例（默认0.01，即1%）
        logger: 日志记录器

    Returns:
        tuple: (保留的实体列表, 被过滤的实体列表)
    """
    from collections import Counter

    # 统计品牌频率
    total = len(entities)
    brand_counter = Counter([e.get('brand', '') for e in entities if e.get('brand')])

    # 计算阈值
    min_count = max(1, int(total * min_frequency_ratio))

    # 识别稀有品牌
    rare_brands = {brand for brand, count in brand_counter.items() if count < min_count}

    # 分离实体
    kept_entities = []
    filtered_entities = []

    for entity in entities:
        brand = entity.get('brand', '')
        if brand and brand in rare_brands:
            filtered_entities.append(entity)
        else:
            kept_entities.append(entity)

    if logger:
        logger.info(f"  稀有品牌过滤统计:")
        logger.info(f"    阈值: {min_frequency_ratio*100:.1f}% (最少{min_count}个)")
        logger.info(f"    稀有品牌数: {len(rare_brands)}")
        logger.info(f"    被过滤实体: {len(filtered_entities)}")
        logger.info(f"    保留实体: {len(kept_entities)}")
        if rare_brands and len(rare_brands) <= 20:
            logger.info(f"    被过滤品牌: {', '.join(sorted(rare_brands))}")

    return kept_entities, filtered_entities


def get_statistics(
    entities: List[Dict],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    获取数据集统计信息

    Args:
        entities: 实体列表
        logger: 日志记录器

    Returns:
        统计信息字典
    """
    stats = {
        "总实体数": len(entities),
        "有品牌": sum(1 for e in entities if e.get('brand')),
        "无品牌": sum(1 for e in entities if not e.get('brand')),
        "有型号": sum(1 for e in entities if e.get('product_model')),
        "有类别": sum(1 for e in entities if e.get('category')),
        "有制造商": sum(1 for e in entities if e.get('manufacturer')),
    }

    # 批次统计
    batches = set(e.get('_batch', 0) for e in entities)
    stats["批次数量"] = len(batches)

    if logger:
        from utils import log_stats
        log_stats(logger, stats, "数据集统计")

    return stats


if __name__ == "__main__":
    # 测试数据加载
    from logger import get_logger

    logger = get_logger()

    # 这里需要实际的测试数据文件
    test_file = "../LLMRelationExtracter_v2/relation_results.json"

    if Path(test_file).exists():
        entities, original_data = load_and_prepare_data(test_file, logger)
        stats = get_statistics(entities, logger)
    else:
        logger.warning(f"测试文件不存在: {test_file}")
