"""
指代融合模块 - 处理品牌别名的统一
使用LLM判断不同品牌名称是否为同一指代
"""

import asyncio
import logging
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils import run_async, extract_json, UnionFind


def extract_unique_brands_with_frequency(
    entities: List[Dict],
    logger: Optional[logging.Logger] = None
) -> List[Tuple[str, int]]:
    """
    提取所有唯一品牌名称及其频率

    Args:
        entities: 实体列表
        logger: 日志记录器

    Returns:
        品牌列表，每个元素为 (brand_name, frequency)
    """
    brands = [e.get('brand', '') for e in entities if e.get('brand')]
    brand_counter = Counter(brands)

    # 按频率排序（高频在前）
    sorted_brands = list(brand_counter.most_common())

    if logger:
        logger.info(f"  发现 {len(sorted_brands)} 个唯一品牌")
        logger.info(f"  前5个高频品牌: {[b[0] for b in sorted_brands[:5]]}")

    return sorted_brands


def quick_rule_judge_alias(brand1: str, brand2: str, freq1: int, freq2: int) -> Optional[Dict]:
    """
    使用快速规则判断品牌是否为别名（避免调用LLM）

    Args:
        brand1: 品牌1
        brand2: 品牌2
        freq1: 品牌1的频率
        freq2: 品牌2的频率

    Returns:
        如果规则能够判断，返回判断结果字典；否则返回None（需要LLM判断）
    """
    b1 = brand1.strip().lower()
    b2 = brand2.strip().lower()

    # 规则1: 完全相同（忽略大小写）
    if b1 == b2:
        return {
            'brand1': brand1,
            'brand2': brand2,
            'is_alias': True,
            'canonical': brand1 if freq1 >= freq2 else brand2,
            'confidence': 1.0,
            'reasoning': '品牌名称完全相同（忽略大小写）',
            'method': 'rule_exact_match'
        }

    # 规则2: 明显不同的品牌（长度差异很大且没有包含关系）
    len1, len2 = len(brand1), len(brand2)
    if abs(len1 - len2) > max(len1, len2) * 0.7:  # 长度差异>70%
        if brand1 not in brand2 and brand2 not in brand1:
            return {
                'brand1': brand1,
                'brand2': brand2,
                'is_alias': False,
                'canonical': None,
                'confidence': 0.9,
                'reasoning': '品牌名称长度差异过大且无包含关系',
                'method': 'rule_length_diff'
            }

    # 规则3: 简单的包含关系（去掉常见后缀）
    suffixes = ['集团', '公司', '有限公司', '股份有限公司', '电器', '工业', '株式会社']
    b1_clean = b1
    b2_clean = b2
    for suffix in suffixes:
        b1_clean = b1_clean.replace(suffix, '')
        b2_clean = b2_clean.replace(suffix, '')

    # 去掉空格和特殊字符
    import re
    b1_clean = re.sub(r'[^\w]', '', b1_clean)
    b2_clean = re.sub(r'[^\w]', '', b2_clean)

    # 如果清理后相同，则是别名
    if b1_clean and b2_clean and b1_clean == b2_clean:
        canonical = brand1 if freq1 >= freq2 else brand2
        return {
            'brand1': brand1,
            'brand2': brand2,
            'is_alias': True,
            'canonical': canonical,
            'confidence': 0.92,
            'reasoning': f'去除常见后缀后核心名称相同（{b1_clean}）',
            'method': 'rule_core_match'
        }

    # 规则4: 一个包含另一个且长度差异合理
    if brand1 in brand2 or brand2 in brand1:
        shorter = brand1 if len1 < len2 else brand2
        longer = brand2 if len1 < len2 else brand1
        shorter_freq = freq1 if len1 < len2 else freq2
        longer_freq = freq2 if len1 < len2 else freq1

        # 长的版本包含短的版本，且长度差异不是特别大
        if len(longer) - len(shorter) <= 10:  # 最多多10个字符
            canonical = shorter if shorter_freq >= longer_freq else longer
            return {
                'brand1': brand1,
                'brand2': brand2,
                'is_alias': True,
                'canonical': canonical,
                'confidence': 0.85,
                'reasoning': f'"{shorter}"是"{longer}"的简称或全称关系',
                'method': 'rule_contains'
            }

    # 规则5: 明显不同的品牌（没有共同字符或共同字符太少）
    common_chars = set(b1_clean) & set(b2_clean)
    if not common_chars or len(common_chars) < 2:
        return {
            'brand1': brand1,
            'brand2': brand2,
            'is_alias': False,
            'canonical': None,
            'confidence': 0.88,
            'reasoning': '品牌名称没有足够的共同字符',
            'method': 'rule_no_common'
        }

    # 无法通过规则判断，返回None
    return None


def build_brand_pairs(
    brands: List[Tuple[str, int]],
    logger: Optional[logging.Logger] = None
) -> List[Tuple[str, str]]:
    """
    生成所有品牌对（用于LLM判断）

    Args:
        brands: 品牌列表 [(name1, freq1), (name2, freq2), ...]
        logger: 日志记录器

    Returns:
        品牌对列表 [(brand1, brand2), ...]
    """
    brand_names = [b[0] for b in brands]
    pairs = list(combinations(brand_names, 2))

    if logger:
        logger.info(f"  生成 {len(pairs)} 个品牌对用于判断")

    return pairs


async def llm_judge_alias_async(
    brand1: str,
    brand2: str,
    freq1: int,
    freq2: int,
    llm: ChatOpenAI,
    prompt_template: ChatPromptTemplate
) -> Dict:
    """
    使用LLM判断两个品牌是否为同一指代

    Args:
        brand1: 品牌1
        brand2: 品牌2
        freq1: 品牌1的频率
        freq2: 品牌2的频率
        llm: LangChain LLM实例
        prompt_template: Prompt模板

    Returns:
        判断结果字典
    """
    # 构建prompt
    prompt_input = {
        "brand1": brand1,
        "brand2": brand2,
        "freq1": freq1,
        "freq2": freq2
    }

    try:
        # 调用LLM - 使用chain
        chain = prompt_template | llm
        response = await chain.ainvoke(prompt_input)
        result_text = response.content.strip()

        # 解析JSON
        result = extract_json(result_text)

        if result:
            return {
                'brand1': brand1,
                'brand2': brand2,
                'is_alias': result.get('is_alias', False),
                'canonical': result.get('canonical', brand1),
                'reasoning': result.get('reasoning', result_text),
                'confidence': result.get('confidence', 0.5)
            }
        else:
            # JSON解析失败，尝试文本解析
            text_lower = result_text.lower()
            is_alias = '是' in result_text or 'yes' in text_lower or 'true' in text_lower

            return {
                'brand1': brand1,
                'brand2': brand2,
                'is_alias': is_alias,
                'canonical': brand1 if is_alias else None,
                'reasoning': result_text,
                'confidence': 0.6 if is_alias else 0.5
            }

    except Exception as e:
        # 异常情况 - 添加详细错误信息
        import traceback
        error_detail = traceback.format_exc()
        return {
            'brand1': brand1,
            'brand2': brand2,
            'is_alias': False,
            'canonical': None,
            'reasoning': f'LLM调用异常: {str(e)}',
            'confidence': 0.0,
            'error': str(e),
            'error_detail': error_detail
        }


async def batch_llm_judge_aliases_async(
    brand_pairs: List[Tuple[str, str]],
    brand_freq_map: Dict[str, int],
    llm: ChatOpenAI,
    prompt_template: ChatPromptTemplate,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """
    批量使用LLM判断品牌对是否为别名（异步并发）

    Args:
        brand_pairs: 品牌对列表
        brand_freq_map: 品牌频率映射 {brand_name: frequency}
        llm: LangChain LLM实例
        prompt_template: Prompt模板
        logger: 日志记录器

    Returns:
        判断结果列表
    """
    # 创建所有异步任务，并使用列表保持顺序
    tasks_with_brands = []
    for brand1, brand2 in brand_pairs:
        freq1 = brand_freq_map.get(brand1, 0)
        freq2 = brand_freq_map.get(brand2, 0)
        task = asyncio.create_task(
            llm_judge_alias_async(brand1, brand2, freq1, freq2, llm, prompt_template)
        )
        tasks_with_brands.append((task, brand1, brand2))

    # 使用asyncio.wait逐个处理完成的任务
    results = []
    pending = {item[0] for item in tasks_with_brands}
    task_to_brands = {item[0]: (item[1], item[2]) for item in tasks_with_brands}

    from tqdm import tqdm

    with tqdm(total=len(pending), desc="LLM指代判断", unit="对") as pbar:
        while pending:
            # 等待任意一个任务完成
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                # 获取对应的品牌对
                brand1, brand2 = task_to_brands.get(task, ('unknown', 'unknown'))

                try:
                    result = task.result()  # 使用 task.result() 而不是 await task
                    results.append(result)
                except Exception as e:
                    if logger:
                        logger.warning(f"LLM判断失败 ({brand1} vs {brand2}): {e}")
                    results.append({
                        'brand1': brand1,
                        'brand2': brand2,
                        'is_alias': False,
                        'canonical': None,
                        'reasoning': f'任务异常: {str(e)}',
                        'confidence': 0.0,
                        'error': str(e)
                    })

                pbar.update(1)

    return results


def build_alias_map(
    brands: List[Tuple[str, int]],
    llm_results: List[Dict],
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[str]]:
    """
    根据LLM判断结果构建别名映射

    策略：使用并查集合并别名，选择最高频的品牌作为规范名称

    Args:
        brands: 品牌列表 [(name, freq), ...]
        llm_results: LLM判断结果列表
        logger: 日志记录器

    Returns:
        别名映射字典 {canonical_name: [alias1, alias2, ...]}
    """
    # 构建品牌名称到索引的映射
    brand_list = [b[0] for b in brands]
    brand_to_idx = {name: idx for idx, name in enumerate(brand_list)}

    # 初始化并查集
    uf = UnionFind(len(brand_list))

    # 根据LLM判断结果合并
    merge_count = 0
    for result in llm_results:
        if result.get('is_alias') and result.get('confidence', 0) > 0.6:
            brand1 = result['brand1']
            brand2 = result['brand2']

            if brand1 in brand_to_idx and brand2 in brand_to_idx:
                idx1 = brand_to_idx[brand1]
                idx2 = brand_to_idx[brand2]
                uf.union(idx1, idx2)
                merge_count += 1

    if logger:
        logger.info(f"  LLM判断合并了 {merge_count} 对品牌")

    # 构建别名映射（选择最高频的作为规范名称）
    groups = uf.get_groups()

    alias_map = {}
    for root_idx, members in groups.items():
        if len(members) == 1:
            # 单个品牌，不需要别名映射
            continue

        # 找出组内最高频的品牌作为规范名称
        member_brands = [(brand_list[idx], brands[idx][1]) for idx in members]
        member_brands.sort(key=lambda x: x[1], reverse=True)  # 按频率降序

        canonical_name = member_brands[0][0]  # 最高频的
        aliases = [name for name, _ in member_brands[1:]]  # 其他都是别名

        alias_map[canonical_name] = aliases

    if logger:
        logger.info(f"  构建了 {len(alias_map)} 个别名组")
        for canonical, aliases in alias_map.items():
            logger.info(f"    {canonical} <- {', '.join(aliases)}")

    return alias_map


def apply_alias_map(
    entities: List[Dict],
    alias_map: Dict[str, List[str]],
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """
    应用别名映射到实体列表，统一品牌名称

    Args:
        entities: 实体列表
        alias_map: 别名映射 {canonical: [alias1, alias2, ...]}
        logger: 日志记录器

    Returns:
        应用别名映射后的实体列表
    """
    # 构建反向映射：alias -> canonical
    reverse_map = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            reverse_map[alias] = canonical

    # 应用映射
    changed_count = 0
    for entity in entities:
        brand = entity.get('brand', '')
        if brand in reverse_map:
            old_brand = brand
            entity['brand'] = reverse_map[brand]
            entity['_brand_alias_merged'] = True
            entity['_original_brand'] = old_brand
            changed_count += 1

    if logger:
        total = len(entities)
        logger.info(f"  应用了别名映射: {changed_count}/{total} 个实体的品牌名称被统一")

    return entities


def create_alias_fusion_prompt_template() -> ChatPromptTemplate:
    """
    创建指代融合的Prompt模板

    Returns:
        ChatPromptTemplate实例
    """
    system_message = """你是一个专业的品牌识别和实体对齐专家，擅长判断不同品牌名称是否指代同一家公司或品牌。

## 判断标准

### 应判定为同一品牌（is_alias=true）的情况：

1. **全称与简称**
   - "美的集团" vs "美的" → 同一品牌
   - "海尔智家股份有限公司" vs "海尔" → 同一品牌

2. **公司全名与简称**
   - "广东美的暖通设备有限公司" vs "美的" → 同一品牌（子公司）
   - "青岛海尔空调器有限总公司" vs "海尔" → 同一品牌

3. **商标一致**
   - "Midea" vs "美的" → 同一品牌（中英文）
   - "GREE" vs "格力" → 同一品牌

4. **明显的所属关系**
   - "美的集团" vs "广东美的" → 同一品牌体系
   - "大金工业株式会社" vs "大金" → 同一品牌

5. **仅有后缀差异**
   - "格力电器" vs "格力" → 同一品牌
   - "海信集团" vs "海信" → 同一品牌

### 应判定为不同品牌（is_alias=false）的情况：

1. **完全不同的品牌**
   - "美的" vs "格力" → 不同品牌
   - "大金" vs "三菱" → 不同品牌

2. **容易混淆但实际不同**
   - "三菱电机" vs "三菱重工" → 不同品牌（不同公司）
   - "奥克斯" vs "奥克莱" → 不同品牌

3. **仅有部分字符相同**
   - "美" vs "美的" → 不同（太短，可能是缩写或不完整）
   - "金" vs "大金" → 不同

## 规范名称选择标准

当判定为同一品牌时，选择canonical名称的优先级：
1. **频率最高的名称**（数据集中出现最多的）
2. **更完整的名称**（包含更多信息）
3. **不包含地域前缀**（"美的" 优于 "广东美的"）
4. **不包含组织类型后缀**（"美的" 优于 "美的集团"）
5. **不包含业务范围**（"美的" 优于 "美的暖通设备"）

## 注意事项

- 当两个名称差异很大时，即使有部分字符相同，也要谨慎判断
- 不确定时，confidence应该设置为较低值（<0.7）
- reasoning中要清晰说明判断依据"""

    human_message = """请判断以下两个品牌名称是否指代同一个品牌。

## 品牌信息

**品牌A**: {brand1}
- 在数据集中出现频率: {freq1} 次

**品牌B**: {brand2}
- 在数据集中出现频率: {freq2} 次

## 分析要点

1. 检查是否存在全称-简称关系
2. 检查是否存在公司-子公司关系
3. 检查核心商标名称是否一致
4. 检查是否仅有后缀或前缀差异
5. 排除完全不同的品牌

## 输出格式

请以JSON格式输出，严格遵循以下结构：

```json
{{
    "is_alias": true/false,
    "canonical": "规范名称（选择频率更高或更简洁的名称）",
    "confidence": 0.0-1.0,
    "reasoning": "详细的判断理由，说明：(1)是否为同一品牌 (2)为什么选择该canonical名称 (3)关键判断依据"
}}
```

## 判断示例

**示例1**: "美的集团" vs "美的"
- is_alias: true
- canonical: "美的"（更简洁，去掉了组织类型后缀）
- confidence: 0.95
- reasoning: "两者核心商标'美的'一致，'美的集团'是'美的'的全称，应统一为更简洁的'美的'"

**示例2**: "美的" vs "格力"
- is_alias: false
- canonical: null
- confidence: 0.99
- reasoning: "两个完全不同的空调品牌，没有任何关联关系"

请给出你的判断："""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def perform_alias_fusion(
    entities: List[Dict],
    llm_config: Dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], Dict[str, List[str]], List[Dict]]:
    """
    执行指代融合（主函数）

    流程：
    1. 提取所有唯一品牌
    2. 生成所有品牌对
    3. 使用LLM判断是否为别名
    4. 构建别名映射
    5. 应用到实体

    Args:
        entities: 实体列表
        llm_config: LLM配置字典
        logger: 日志记录器

    Returns:
        tuple: (融合后的实体列表, 别名映射字典, LLM判断结果列表)
    """
    # 步骤1: 提取品牌
    if logger:
        logger.info("步骤1: 提取唯一品牌名称")
    brands = extract_unique_brands_with_frequency(entities, logger)

    if len(brands) < 2:
        if logger:
            logger.info("  品牌数量少于2个，跳过指代融合")
        return entities, {}, []

    # 步骤2: 生成品牌对
    if logger:
        logger.info("步骤2: 生成品牌对")
    brand_pairs = build_brand_pairs(brands, logger)

    # 构建品牌频率映射
    brand_freq_map = {brand: freq for brand, freq in brands}

    # 步骤3: 规则+LLM判断
    if logger:
        logger.info("步骤3: 判断品牌对（规则+LLM）")

    # 3.1: 先使用规则快速判断
    rule_results = []
    need_llm_pairs = []

    for brand1, brand2 in brand_pairs:
        freq1 = brand_freq_map.get(brand1, 0)
        freq2 = brand_freq_map.get(brand2, 0)

        rule_result = quick_rule_judge_alias(brand1, brand2, freq1, freq2)

        if rule_result is not None:
            # 规则能判断
            rule_results.append(rule_result)
        else:
            # 需要LLM判断
            need_llm_pairs.append((brand1, brand2))

    if logger:
        logger.info(f"  规则判断: {len(rule_results)} 对")
        logger.info(f"  需要LLM: {len(need_llm_pairs)} 对")

    # 3.2: 对需要的品牌对使用LLM判断
    llm_results = []
    if need_llm_pairs:
        # 初始化LLM
        llm = ChatOpenAI(
            api_key=llm_config.get('api_key'),
            base_url=llm_config.get('base_url'),
            model=llm_config.get('model', 'gpt-4o-mini'),
            temperature=0.1,
            timeout=llm_config.get('timeout', 300)
        )

        prompt_template = create_alias_fusion_prompt_template()

        # 异步批量判断
        llm_results = run_async(
            batch_llm_judge_aliases_async(need_llm_pairs, brand_freq_map, llm, prompt_template, logger)
        )

    # 合并规则和LLM结果
    all_results = rule_results + llm_results

    # 步骤4: 构建别名映射
    if logger:
        logger.info("步骤4: 构建别名映射")
    alias_map = build_alias_map(brands, all_results, logger)

    # 步骤5: 应用别名映射
    if logger:
        logger.info("步骤5: 应用别名映射到实体")
    fused_entities = apply_alias_map(entities, alias_map, logger)

    return fused_entities, alias_map, all_results


if __name__ == "__main__":
    # 测试指代融合
    from logger import get_logger

    logger = get_logger()

    # 测试数据
    test_entities = [
        {"brand": "美的", "product_model": "A1"},
        {"brand": "美的集团", "product_model": "A2"},
        {"brand": "广东美的", "product_model": "A3"},
        {"brand": "大金", "product_model": "B1"},
        {"brand": "大金空调", "product_model": "B2"},
    ]

    test_llm_config = {
        "api_key": "test",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "timeout": 300
    }

    logger.info("指代融合模块测试")
    logger.info("注意: 需要有效的API密钥才能实际运行")
