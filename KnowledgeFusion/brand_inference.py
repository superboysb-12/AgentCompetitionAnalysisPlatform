"""
品牌推断模块 - 基于k-NN的迁移学习 + LLM辅助推断
使用有品牌实体为无品牌实体推断品牌

重构版本：函数化设计，清晰的数据流
"""

import asyncio
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from levenshtein import calculate_entity_similarity
from utils import run_async, extract_json


# ==================== 性能优化函数 ====================

def compute_fingerprint(entity: Dict) -> Dict:
    """
    计算快速匹配指纹（用于粗分组和快速过滤）

    Args:
        entity: 实体字典

    Returns:
        指纹字典
    """
    return {
        'model_first_chars': entity.get('product_model', '')[:5].lower(),
        'manufacturer_first_char': entity.get('manufacturer', '')[:1].lower(),
        'category': entity.get('category', ''),
    }


def coarse_group_key(entity: Dict) -> str:
    """
    计算粗分组键（用于快速过滤候选集）

    Args:
        entity: 实体字典

    Returns:
        分组键字符串
    """
    fp = compute_fingerprint(entity)

    # 优先级：manufacturer > category > product_model首字母
    if fp['manufacturer_first_char']:
        return f"mfg_{fp['manufacturer_first_char']}"
    elif fp['category']:
        return f"cat_{fp['category'][:3]}"
    elif fp['model_first_chars']:
        return f"model_{fp['model_first_chars'][:2]}"
    else:
        return 'unknown'


def quick_filter_score(fp1: Dict, fp2: Dict) -> float:
    """
    快速过滤评分（简单的字符串包含匹配）

    Args:
        fp1: 实体1的指纹
        fp2: 实体2的指纹

    Returns:
        评分 (0.0-1.0)
    """
    score = 0.0

    # product_model前缀匹配
    if fp1['model_first_chars'] and fp2['model_first_chars']:
        if fp1['model_first_chars'] in fp2['model_first_chars'] or \
           fp2['model_first_chars'] in fp1['model_first_chars']:
            score += 0.5

    # manufacturer完全匹配
    if fp1['manufacturer_first_char'] and fp2['manufacturer_first_char']:
        if fp1['manufacturer_first_char'] == fp2['manufacturer_first_char']:
            score += 0.3

    # category完全匹配
    if fp1['category'] and fp2['category']:
        if fp1['category'] == fp2['category']:
            score += 0.2

    return score


def filter_candidates(
    entity_without: Dict,
    entities_with: List[Dict],
    filter_threshold: float = 0.3
) -> List[Dict]:
    """
    快速过滤候选实体（减少Levenshtein计算量）

    Args:
        entity_without: 无品牌实体
        entities_with: 有品牌实体列表
        filter_threshold: 过滤阈值

    Returns:
        过滤后的候选实体列表
    """
    fp_without = compute_fingerprint(entity_without)
    candidates = []

    for entity_with in entities_with:
        fp_with = compute_fingerprint(entity_with)

        # 计算快速过滤分数
        score = quick_filter_score(fp_without, fp_with)

        if score >= filter_threshold:
            candidates.append(entity_with)

    return candidates


def build_coarse_group_index(entities_with: List[Dict]) -> Dict[str, List[Dict]]:
    """
    构建粗分组索引

    Args:
        entities_with: 有品牌实体列表

    Returns:
        分组索引字典
    """
    index = defaultdict(list)
    for entity in entities_with:
        key = coarse_group_key(entity)
        index[key].append(entity)
    return dict(index)


def get_coarse_group_candidates(
    entity_without: Dict,
    coarse_index: Dict[str, List[Dict]],
    k: int
) -> List[Dict]:
    """
    获取粗分组内的候选实体

    Args:
        entity_without: 无品牌实体
        coarse_index: 粗分组索引
        k: k近邻数

    Returns:
        候选实体列表
    """
    group_key = coarse_group_key(entity_without)
    candidates = coarse_index.get(group_key, [])

    # 如果同组候选太少，返回所有
    if len(candidates) < k * 2:
        # 返回所有实体（需要外部传入完整列表）
        return None

    return candidates


# ==================== k-NN推断函数 ====================

def infer_brand_for_entity(
    entity_without_brand: Dict,
    entities_with_brand: List[Dict],
    k: int,
    coarse_index: Optional[Dict[str, List[Dict]]] = None,
    use_optimization: bool = True
) -> Tuple[Optional[str], float, List[Dict]]:
    """
    为单个无品牌实体推断品牌（k-NN算法）

    Args:
        entity_without_brand: 无品牌实体
        entities_with_brand: 有品牌实体列表（候选集）
        k: k近邻数
        coarse_index: 粗分组索引（可选）
        use_optimization: 是否使用性能优化

    Returns:
        tuple: (推断的品牌, 置信度, k个近邻详情)
    """
    # Step 1: 粗分组过滤
    candidates = entities_with_brand
    if use_optimization and coarse_index:
        coarse_candidates = get_coarse_group_candidates(
            entity_without_brand,
            coarse_index,
            k
        )
        if coarse_candidates is not None:
            candidates = coarse_candidates

    # Step 2: 快速过滤
    if use_optimization:
        candidates = filter_candidates(
            entity_without_brand,
            candidates,
            filter_threshold=0.3
        )

    # 如果过滤后候选太少，使用全部
    if len(candidates) < k:
        candidates = entities_with_brand

    # Step 3: 计算Levenshtein相似度
    similarities = []
    for entity_with_brand in candidates:
        total_score, details = calculate_entity_similarity(
            entity_without_brand,
            entity_with_brand
        )

        if total_score > 0:
            similarities.append({
                'score': total_score,
                'brand': entity_with_brand.get('brand', ''),
                'details': details,
                'entity': entity_with_brand
            })

    if not similarities:
        return None, 0.0, []

    # Step 4: 取top-k
    similarities.sort(key=lambda x: x['score'], reverse=True)
    top_k = similarities[:k]

    # Step 5: 加权投票（相似度平方作为权重）
    brand_scores = defaultdict(float)
    for item in top_k:
        brand = item['brand']
        weight = item['score'] ** 2
        brand_scores[brand] += weight

    if not brand_scores:
        return None, 0.0, top_k[:3]

    # 找得分最高的品牌
    best_brand = max(brand_scores.items(), key=lambda x: x[1])
    inferred_brand = best_brand[0]
    total_weight = sum(brand_scores.values())
    confidence = best_brand[1] / total_weight if total_weight > 0 else 0.0

    return inferred_brand, confidence, top_k[:3]


def batch_infer_brands_knn(
    entities_without_brand: List[Dict],
    entities_with_brand: List[Dict],
    k: int,
    use_optimization: bool = True,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """
    批量k-NN品牌推断

    Args:
        entities_without_brand: 无品牌实体列表
        entities_with_brand: 有品牌实体列表
        k: k近邻数
        use_optimization: 是否使用性能优化
        logger: 日志记录器

    Returns:
        推断结果列表
    """
    results = []

    # 构建粗分组索引
    coarse_index = None
    if use_optimization:
        coarse_index = build_coarse_group_index(entities_with_brand)
        if logger:
            logger.info(f"  构建粗分组索引: {len(coarse_index)} 个组")

    # 批量推断
    for entity in tqdm(entities_without_brand, desc="k-NN品牌推断"):
        inferred_brand, confidence, neighbors = infer_brand_for_entity(
            entity,
            entities_with_brand,
            k,
            coarse_index,
            use_optimization
        )

        result = {
            'entity': entity,
            'inferred_brand': inferred_brand,
            'confidence': confidence,
            'neighbors': neighbors,
            'inference_method': 'knn'
        }
        results.append(result)

    return results


# ==================== LLM辅助推断函数 ====================

def extract_representative_entities(
    entities_with_brand: List[Dict],
    max_per_brand: int = 3
) -> Dict[str, List[Dict]]:
    """
    提取每个品牌的代表性实体（作为LLM上下文）

    Args:
        entities_with_brand: 有品牌的实体列表
        max_per_brand: 每个品牌最多提取几个代表性实体

    Returns:
        {brand_name: [代表性实体列表]}
    """
    from collections import defaultdict

    brand_entities = defaultdict(list)

    # 按品牌分组
    for entity in entities_with_brand:
        brand = entity.get('brand', '')
        if brand:
            brand_entities[brand].append(entity)

    # 为每个品牌选择代表性实体
    representative = {}
    for brand, entities in brand_entities.items():
        # 优先选择信息完整的实体（有manufacturer、product_model等）
        scored_entities = []
        for entity in entities:
            score = 0
            if entity.get('manufacturer'):
                score += 2
            if entity.get('product_model'):
                score += 2
            if entity.get('category'):
                score += 1
            if entity.get('series'):
                score += 1
            scored_entities.append((score, entity))

        # 按分数排序，取前N个
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        representative[brand] = [e for _, e in scored_entities[:max_per_brand]]

    return representative


def build_brand_inference_prompt(
    entity: Dict,
    neighbors: List[Dict],
    known_brands: List[str],
    representative_entities: Optional[Dict[str, List[Dict]]] = None
) -> str:
    """
    构建品牌推断的Prompt

    Args:
        entity: 待推断实体
        neighbors: k-NN邻居列表
        known_brands: 已知品牌列表
        representative_entities: 每个品牌的代表性实体 {brand: [entities]}

    Returns:
        Prompt字符串
    """
    # 格式化实体信息
    entity_info = []
    if entity.get('product_model'):
        entity_info.append(f"产品型号: {entity['product_model']}")
    if entity.get('category'):
        entity_info.append(f"产品类别: {entity['category']}")
    if entity.get('manufacturer'):
        entity_info.append(f"制造商: {entity['manufacturer']}")
    if entity.get('series'):
        entity_info.append(f"系列: {entity['series']}")

    # 格式化k-NN邻居信息
    neighbor_info = []
    for i, neighbor in enumerate(neighbors[:3], 1):
        n = neighbor['entity']
        neighbor_info.append(f"""
邻居{i}:
  - 品牌: {neighbor['brand']}
  - 相似度: {neighbor['score']:.3f}
  - 产品型号: {n.get('product_model', '未知')}
  - 制造商: {n.get('manufacturer', '未知')}
  - 产品类别: {n.get('category', '未知')}
        """)

    # 已知品牌列表
    brands_list = "、".join(sorted(list(known_brands))[:20])

    # 格式化代表性实体信息（如果提供）
    representative_info = ""
    if representative_entities:
        representative_info = "\n## 各品牌代表性实体（参考）\n\n"
        for brand, entities in list(representative_entities.items())[:10]:  # 最多显示10个品牌
            if entities:
                representative_info += f"**{brand}** 品牌典型产品:\n"
                for idx, e in enumerate(entities[:2], 1):  # 每个品牌显示2个实体
                    representative_info += f"  样例{idx}:\n"
                    if e.get('product_model'):
                        representative_info += f"    - 型号: {e['product_model']}\n"
                    if e.get('manufacturer'):
                        representative_info += f"    - 制造商: {e['manufacturer']}\n"
                    if e.get('category'):
                        representative_info += f"    - 类别: {e['category']}\n"
                representative_info += "\n"

    # 格式化实体信息（不能在f-string中使用join）
    entity_info_str = '\n'.join(entity_info)
    neighbor_info_str = ''.join(neighbor_info)

    prompt = f"""你是一个专业的产品品牌识别专家。请根据以下信息判断产品的品牌。

## 已知品牌列表
{brands_list}
{representative_info}
## 待识别产品信息
{entity_info_str}

## k-NN相似产品参考（重要上下文）
{neighbor_info_str}

## 判断标准
请基于以下标准综合判断：

1. **产品型号特征**：型号前缀、命名规则往往包含品牌特征
2. **制造商信息**：制造商和品牌通常有关联
3. **产品类别**：某些品牌专注于特定类别
4. **相似产品**：参考k-NN最相似产品的品牌（相似度高时参考价值大）
5. **品牌代表性实体**：参考各品牌的典型产品特征

## 输出格式
请以JSON格式输出，严格遵循以下结构：
{{
    "brand": "品牌名称（必须是已知品牌列表中的一个）",
    "confidence": 0.0-1.0,
    "reasoning": "详细的判断理由，说明为什么选择这个品牌"
}}

请给出你的判断："""

    return prompt


async def llm_infer_brand_async(
    entity: Dict,
    neighbors: List[Dict],
    known_brands: List[str],
    llm: ChatOpenAI,
    representative_entities: Optional[Dict[str, List[Dict]]] = None
) -> Dict:
    """
    使用LLM异步推断单个实体的品牌

    Args:
        entity: 实体字典
        neighbors: k-NN邻居列表
        known_brands: 已知品牌列表
        llm: LangChain LLM实例
        representative_entities: 每个品牌的代表性实体（可选）

    Returns:
        推断结果字典
    """
    try:
        # 构建prompt
        prompt = build_brand_inference_prompt(entity, neighbors, known_brands, representative_entities)

        # 创建chain
        chain = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的产品品牌识别专家。"),
            ("human", "{prompt}")
        ]) | llm

        # 调用LLM
        response = await chain.ainvoke({"prompt": prompt})
        result_text = response.content.strip()

        # 解析JSON响应
        result = extract_json(result_text)

        if result:
            inferred_brand = result.get('brand', '')
            if inferred_brand and inferred_brand in known_brands:
                return {
                    'entity': entity,
                    'llm_inferred_brand': inferred_brand,
                    'llm_confidence': result.get('confidence', 0.7),
                    'llm_reasoning': result.get('reasoning', '')
                }

        # 解析失败
        return {
            'entity': entity,
            'llm_inferred_brand': None,
            'llm_confidence': 0.0,
            'llm_error': '解析失败或品牌不在列表中'
        }

    except Exception as e:
        return {
            'entity': entity,
            'llm_inferred_brand': None,
            'llm_confidence': 0.0,
            'llm_error': str(e)
        }


async def batch_llm_infer_brands_async(
    results: List[Dict],
    known_brands: List[str],
    llm: ChatOpenAI,
    representative_entities: Optional[Dict[str, List[Dict]]] = None,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """
    批量使用LLM推断品牌（异步并发）

    Args:
        results: k-NN推断结果列表
        known_brands: 已知品牌列表
        llm: LangChain LLM实例
        representative_entities: 每个品牌的代表性实体（可选）
        logger: 日志记录器

    Returns:
        LLM推断结果列表
    """
    tasks = []
    for result in results:
        task = asyncio.create_task(
            llm_infer_brand_async(
                result['entity'],
                result['neighbors'],
                known_brands,
                llm,
                representative_entities
            )
        )
        tasks.append(task)

    # 并发执行
    llm_results = []
    pending = set(tasks)

    with tqdm(total=len(tasks), desc="LLM品牌推断", unit="个") as pbar:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    result = task.result()  # 使用 task.result() 而不是 await task
                    llm_results.append(result)
                except Exception as e:
                    if logger:
                        logger.warning(f"LLM推断失败: {e}")
                    llm_results.append({
                        'entity': {},
                        'llm_inferred_brand': None,
                        'llm_confidence': 0.0,
                        'llm_error': str(e)
                    })

                pbar.update(1)

    return llm_results


# ==================== 主函数 ====================

def perform_brand_inference(
    entities: List[Dict],
    llm_config: Dict,
    inference_config: Dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    执行品牌推断（主函数）

    流程：
    1. 分离有品牌和无品牌实体
    2. 提取各品牌的代表性实体（用于LLM上下文）
    3. 对无品牌实体进行k-NN推断
    4. 对所有k-NN结果使用LLM进行二次判断（使用代表性实体上下文）
    5. 应用推断结果（推断失败的设置为UNKNOWN）

    Args:
        entities: 实体列表
        llm_config: LLM配置
        inference_config: 推断配置
        logger: 日志记录器

    Returns:
        tuple: (推断后的实体列表, 推断日志列表)
    """
    # 配置参数
    k = inference_config.get('k', 5)
    high_threshold = inference_config.get('high_confidence_threshold', 0.75)
    medium_threshold = inference_config.get('medium_confidence_threshold', 0.6)
    use_optimization = inference_config.get('use_optimization', True)
    use_llm_backup = inference_config.get('use_llm_backup', True)

    # 步骤1: 分离实体
    if logger:
        logger.info("步骤1: 分离有品牌和无品牌实体")

    entities_with_brand = [e for e in entities if e.get('brand', '').strip()]
    entities_without_brand = [e for e in entities if not e.get('brand', '').strip()]

    if not entities_without_brand:
        if logger:
            logger.info("  所有实体都有品牌，跳过推断")
        return entities, []

    if logger:
        logger.info(f"  有品牌: {len(entities_with_brand)}")
        logger.info(f"  无品牌: {len(entities_without_brand)}")

    # 步骤1.5: 提取代表性实体（用于LLM上下文）
    if logger:
        logger.info("步骤1.5: 提取各品牌代表性实体")

    representative_entities = extract_representative_entities(
        entities_with_brand,
        max_per_brand=3
    )

    if logger:
        logger.info(f"  已提取 {len(representative_entities)} 个品牌的代表性实体")

    # 步骤2: k-NN推断
    if logger:
        logger.info("步骤2: k-NN品牌推断")

    knn_results = batch_infer_brands_knn(
        entities_without_brand,
        entities_with_brand,
        k,
        use_optimization,
        logger
    )

    # 步骤3: LLM辅助推断（所有k-NN结果）
    llm_results = []
    if use_llm_backup:
        # 所有k-NN结果都用LLM进行二次判断
        if logger:
            logger.info(f"步骤3: LLM二次判断 (所有 {len(knn_results)} 个实体)")

        # 初始化LLM
        llm = ChatOpenAI(
            api_key=llm_config.get('api_key'),
            base_url=llm_config.get('base_url'),
            model=llm_config.get('model', 'gpt-4o-mini'),
            temperature=0.1,
            timeout=llm_config.get('timeout', 300)
        )

        known_brands = list(set(e.get('brand', '') for e in entities_with_brand))

        # 异步批量LLM推断（传递代表性实体）
        llm_results = run_async(
            batch_llm_infer_brands_async(
                knn_results,  # 所有结果
                known_brands,
                llm,
                representative_entities,
                logger
            )
        )

        # 合并结果：用LLM结果覆盖k-NN结果
        llm_result_map = {id(r['entity']): r for r in llm_results}
        for knn_result in knn_results:
            llm_result = llm_result_map.get(id(knn_result['entity']))
            if llm_result and llm_result.get('llm_inferred_brand'):
                # 保存k-NN原始推断
                knn_result['original_knn_inference'] = {
                    'brand': knn_result['inferred_brand'],
                    'confidence': knn_result['confidence']
                }
                # 更新为LLM推断
                knn_result['inferred_brand'] = llm_result['llm_inferred_brand']
                knn_result['confidence'] = llm_result['llm_confidence']
                knn_result['inference_method'] = 'llm'
                knn_result['llm_reasoning'] = llm_result.get('llm_reasoning', '')
            elif not llm_result or not llm_result.get('llm_inferred_brand'):
                # LLM推断失败，保留k-NN结果
                if knn_result.get('inferred_brand'):
                    knn_result['inference_method'] = 'knn_only'
                    if logger:
                        logger.debug(f"  LLM推断失败，保留k-NN结果: {knn_result['inferred_brand']}")

    # 步骤4: 应用推断结果
    if logger:
        logger.info("步骤4: 应用推断结果")

    # 确定置信度等级
    for result in knn_results:
        if result['confidence'] >= high_threshold:
            result['confidence_level'] = '高'
        elif result['confidence'] >= medium_threshold:
            result['confidence_level'] = '中'
        elif result['confidence'] > 0:
            result['confidence_level'] = '低'
        else:
            result['confidence_level'] = '失败'

    # 应用到实体：所有推断结果都应用（因为都经过了LLM二次判断）
    final_entities = []
    applied_count = 0
    for entity in entities:
        if entity.get('brand', '').strip():
            # 有品牌的实体直接保留
            final_entities.append(entity)
        else:
            # 无品牌的实体应用推断
            result = next((r for r in knn_results if r['entity'] is entity), None)
            if result:
                entity_copy = entity.copy()
                if result['inferred_brand']:
                    # 应用推断品牌
                    entity_copy['brand'] = result['inferred_brand']
                    entity_copy['_brand_inferred'] = True
                    entity_copy['_brand_confidence'] = result['confidence']
                    entity_copy['_brand_confidence_level'] = result['confidence_level']
                    entity_copy['_brand_inference_method'] = result.get('inference_method', 'knn')
                    applied_count += 1
                else:
                    # 推断失败，设置为UNKNOWN
                    entity_copy['brand'] = 'UNKNOWN'
                    entity_copy['_brand_inferred'] = False
                    entity_copy['_brand_inference_failed'] = True

                final_entities.append(entity_copy)
            else:
                # 没有找到对应的推断结果（不应该发生）
                entity_copy = entity.copy()
                entity_copy['brand'] = 'UNKNOWN'
                entity_copy['_brand_no_inference'] = True
                final_entities.append(entity_copy)

    # 统计
    if logger:
        high_count = sum(1 for r in knn_results if r['confidence_level'] == '高')
        llm_count = sum(1 for r in knn_results if r['inference_method'] == 'llm')
        knn_only_count = sum(1 for r in knn_results if r.get('inference_method') == 'knn_only')
        logger.info(f"  k-NN高置信度: {high_count}")
        logger.info(f"  LLM二次判断成功: {llm_count}")
        logger.info(f"  LLM失败保留k-NN: {knn_only_count}")
        logger.info(f"  应用品牌推断: {applied_count}/{len(entities_without_brand)}")

    # 最终统计
    final_with_brand = sum(1 for e in final_entities if e.get('brand', '').strip() and e.get('brand') != 'UNKNOWN')
    final_unknown = sum(1 for e in final_entities if e.get('brand') == 'UNKNOWN')

    if logger:
        logger.info(f"\n最终结果:")
        logger.info(f"  总实体数: {len(final_entities)}")
        logger.info(f"  有品牌: {final_with_brand} ({final_with_brand/len(final_entities)*100:.1f}%)")
        logger.info(f"  UNKNOWN: {final_unknown} ({final_unknown/len(final_entities)*100:.1f}%)")

        if final_unknown > 0:
            logger.warning(f"  ⚠️  有 {final_unknown} 个实体推断失败，设置为UNKNOWN")

    return final_entities, knn_results


if __name__ == "__main__":
    # 测试品牌推断
    from logger import get_logger

    logger = get_logger()
    logger.info("品牌推断模块测试")
    logger.info("使用 main.py 进行完整测试")
