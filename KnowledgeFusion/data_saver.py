"""
数据保存模块
负责保存各种格式的结果和日志
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


def ensure_output_dir(output_dir: str) -> Path:
    """
    确保输出目录存在

    Args:
        output_dir: 输出目录路径

    Returns:
        Path对象
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_entities_flat(
    entities: List[Dict],
    filepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    保存实体列表为扁平JSON格式

    Args:
        entities: 实体列表
        filepath: 输出文件路径
        logger: 日志记录器
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    if logger:
        logger.info(f"已保存扁平格式: {filepath}")


def save_entities_original_format(
    entities: List[Dict],
    original_data: List[Dict],
    filepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    按原始格式保存实体列表

    原始格式: [{"results": [...]}, {"results": [...]}]

    策略：合并的实体放到其主要batch（第一个来源batch）中

    Args:
        entities: 融合后的实体列表
        original_data: 原始数据（用于获取batch结构）
        filepath: 输出文件路径
        logger: 日志记录器
    """
    # 为每个实体分配主要batch
    entity_to_main_batch = {}

    for entity in entities:
        # 获取该实体来自的所有batch
        batches = entity.get('_merged_from_batches', [])
        if not batches:
            # 如果没有merged_from_batches，使用_batch字段
            batches = [entity.get('_batch', 0)]

        # 主要batch是第一个来源batch（通常是频率最高或最早的）
        main_batch = min(batches) if batches else 0
        entity_to_main_batch[id(entity)] = main_batch

    # 按batch分组
    batch_to_entities = {}
    for entity in entities:
        main_batch = entity_to_main_batch[id(entity)]
        if main_batch not in batch_to_entities:
            batch_to_entities[main_batch] = []
        batch_to_entities[main_batch].append(entity)

    # 构建原始格式的结果
    result = []
    for batch_idx in range(len(original_data)):
        # 清理实体（移除内部字段）
        entities_in_batch = batch_to_entities.get(batch_idx, [])
        cleaned_entities = []

        for entity in entities_in_batch:
            cleaned = entity.copy()
            # 移除内部字段（以_开头的）
            for key in list(cleaned.keys()):
                if key.startswith('_'):
                    del cleaned[key]
            cleaned_entities.append(cleaned)

        result.append({
            "results": cleaned_entities
        })

    # 保存
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if logger:
        logger.info(f"已保存原始格式: {filepath} (避免了重复实体)")


def save_log(
    log_data: Any,
    filepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    保存日志数据

    Args:
        log_data: 日志数据（字典或列表）
        filepath: 输出文件路径
        logger: 日志记录器
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    if logger:
        logger.info(f"已保存日志: {filepath}")


def save_combined_log(
    all_logs: Dict[str, Any],
    filepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    保存合并的日志文件

    Args:
        all_logs: 所有日志的字典
        filepath: 输出文件路径
        logger: 日志记录器
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)

    if logger:
        logger.info(f"已保存合并日志: {filepath}")


def save_all_results(
    entities: List[Dict],
    original_data: List[Dict],
    logs: Dict[str, Any],
    output_dir: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    保存所有结果（组合函数）

    Args:
        entities: 融合后的实体列表
        original_data: 原始数据
        logs: 日志字典（包含各种日志）
        output_dir: 输出目录
        logger: 日志记录器

    Returns:
        保存的文件路径字典
    """
    output_path = ensure_output_dir(output_dir)
    saved_files = {}

    # 1. 扁平格式
    flat_path = output_path / "fused_entities_flat.json"
    save_entities_flat(entities, str(flat_path), logger)
    saved_files['flat'] = str(flat_path)

    # 2. 原始格式
    original_path = output_path / "fused_entities_original_format.json"
    save_entities_original_format(entities, original_data, str(original_path), logger)
    saved_files['original_format'] = str(original_path)

    # 3. 各种日志
    for log_type, log_data in logs.items():
        if log_data:
            log_path = output_path / f"{log_type}_log.json"
            save_log(log_data, str(log_path), logger)
            saved_files[log_type] = str(log_path)

    if logger:
        logger.info(f"所有结果已保存到目录: {output_dir}")

    return saved_files


def save_alias_fusion_log(
    alias_map: Dict[str, List[str]],
    llm_results: List[Dict],
    filepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    保存指代融合日志

    Args:
        alias_map: 别名映射字典 {canonical_name: [alias1, alias2, ...]}
        llm_results: LLM判断结果列表
        filepath: 输出文件路径
        logger: 日志记录器
    """
    log_data = {
        "alias_map": alias_map,
        "llm_judgments": llm_results,
        "statistics": {
            "total_brands": len(alias_map) + sum(len(v) for v in alias_map.values()),
            "canonical_names": len(alias_map),
            "total_aliases": sum(len(v) for v in alias_map.values())
        }
    }

    save_log(log_data, filepath, logger)


def save_brand_inference_log(
    inference_results: List[Dict],
    filepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    保存品牌推断日志

    Args:
        inference_results: 推断结果列表
        filepath: 输出文件路径
        logger: 日志记录器
    """
    # 转换为可序列化的格式
    log_data = []
    for result in inference_results:
        entity = result['entity']

        log_entry = {
            'product_model': entity.get('product_model', ''),
            'category': entity.get('category', ''),
            'manufacturer': entity.get('manufacturer', ''),
            'inferred_brand': result.get('inferred_brand'),
            'confidence': result.get('confidence'),
            'confidence_level': result.get('confidence_level'),
            'inference_method': result.get('inference_method', 'knn'),
            'neighbors': [
                {
                    'brand': n['brand'],
                    'score': n['score'],
                    'product_model': n['entity'].get('product_model', ''),
                    'manufacturer': n['entity'].get('manufacturer', '')
                }
                for n in result.get('neighbors', [])
            ]
        }

        # 添加LLM相关信息
        if result.get('inference_method') == 'llm':
            log_entry['llm_reasoning'] = result.get('llm_reasoning', '')
            if result.get('original_knn_inference'):
                log_entry['original_knn_inference'] = result['original_knn_inference']

        log_data.append(log_entry)

    save_log(log_data, filepath, logger)


def save_fusion_log(
    fusion_log: List[Dict],
    filepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    保存知识融合日志

    Args:
        fusion_log: 融合日志列表
        filepath: 输出文件路径
        logger: 日志记录器
    """
    # 日志已经是可以序列化的格式
    save_log(fusion_log, filepath, logger)


if __name__ == "__main__":
    # 测试数据保存
    from logger import get_logger

    logger = get_logger()

    # 创建测试数据
    test_entities = [
        {
            "brand": "美的",
            "product_model": "MDV-D15Q4",
            "_batch": 0,
            "_merged_from_batches": [0, 1]
        },
        {
            "brand": "大金",
            "product_model": "FXS-20",
            "_batch": 1
        }
    ]

    test_original_data = [
        {"results": []},
        {"results": []}
    ]

    test_logs = {
        "fusion": [{"test": "log"}],
        "inference": [{"test": "inference"}]
    }

    # 保存测试
    saved_files = save_all_results(
        test_entities,
        test_original_data,
        test_logs,
        "output",
        logger
    )

    logger.info(f"保存的文件: {saved_files}")
