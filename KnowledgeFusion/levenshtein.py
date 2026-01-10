"""
Levenshtein距离计算模块
用于计算字符串相似度
"""

def levenshtein_similarity(str1, str2):
    """
    计算两个字符串的Levenshtein相似度

    Args:
        str1: 字符串1
        str2: 字符串2

    Returns:
        float: 相似度 (0.0 - 1.0)，1.0表示完全相同
    """
    if not str1 and not str2:
        return 1.0

    if not str1 or not str2:
        return 0.0

    # 标准化：去除空格、转小写
    str1 = str(str1).strip().lower()
    str2 = str(str2).strip().lower()

    # 如果完全相同
    if str1 == str2:
        return 1.0

    # 计算Levenshtein距离
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # 删除
                    dp[i][j - 1] + 1,      # 插入
                    dp[i - 1][j - 1] + 1   # 替换
                )

    distance = dp[m][n]
    max_len = max(m, n)

    # 转换为相似度
    similarity = 1.0 - (distance / max_len)

    return similarity


def calculate_entity_similarity(entity1, entity2, weights=None):
    """
    计算两个实体的综合相似度

    Args:
        entity1: 实体1（字典）
        entity2: 实体2（字典）
        weights: 各字段权重，默认为 {
            'brand': 0.3,
            'product_model': 0.4,
            'manufacturer': 0.2,
            'series': 0.1
        }

    Returns:
        tuple: (总相似度, 各字段详细分数)
    """
    if weights is None:
        weights = {
            'brand': 0.3,
            'product_model': 0.4,
            'manufacturer': 0.2,
            'series': 0.1
        }

    # ⭐ 关键修复：product_model是核心字段，必须两个都有值才能进行融合判断
    model1 = str(entity1.get('product_model', '')).strip()
    model2 = str(entity2.get('product_model', '')).strip()

    # 场景1: 两个都为空 → 无法判断，返回0
    if not model1 and not model2:
        details = {
            'product_model': {
                'value1': model1,
                'value2': model2,
                'similarity': 0.0,
                'weight': weights.get('product_model', 0.4),
                'note': '两者都缺失关键字段product_model，无法判断是否为同一产品'
            }
        }
        return 0.0, details

    # 场景2: 只有一个为空 → 不是同一产品，返回0
    # （如果真是同一产品，它们的product_model应该都有值且相同）
    if not model1 or not model2:
        details = {
            'product_model': {
                'value1': model1,
                'value2': model2,
                'similarity': 0.0,
                'weight': weights.get('product_model', 0.4),
                'note': '一方缺失product_model，不能确定是同一产品'
            }
        }
        return 0.0, details

    details = {}
    total_score = 0.0
    total_weight = 0.0

    for field, weight in weights.items():
        val1 = entity1.get(field, '')
        val2 = entity2.get(field, '')

        # 如果两个都有值
        if val1 and val2:
            sim = levenshtein_similarity(val1, val2)
            details[field] = {
                'value1': val1,
                'value2': val2,
                'similarity': sim,
                'weight': weight
            }
            total_score += sim * weight
            total_weight += weight

        # 如果只有一个有值，不惩罚（因为数据不完整）
        else:
            details[field] = {
                'value1': val1,
                'value2': val2,
                'similarity': None,
                'weight': weight,
                'note': '一方缺失'
            }

    # 归一化
    if total_weight > 0:
        total_score = total_score / total_weight

    return total_score, details


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("Levenshtein相似度测试")
    print("=" * 60)

    test_cases = [
        ("美的", "美的集团"),
        ("美的", "广东美的暖通设备有限公司"),
        ("大金", "大金空调"),
        ("MDV-D15Q4/BP3N1-E(M)", "MDV-D15Q4/BP3N1-E"),
        ("美的", "大金"),
    ]

    for s1, s2 in test_cases:
        sim = levenshtein_similarity(s1, s2)
        print(f"'{s1}' vs '{s2}': {sim:.3f}")
