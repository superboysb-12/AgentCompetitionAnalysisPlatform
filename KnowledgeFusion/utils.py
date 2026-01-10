"""
工具函数模块
提供异步处理、JSON解析、并查集等通用工具
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple


# ==================== 异步处理工具 ====================

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    获取或创建事件循环

    解决 RuntimeError: There is no current event loop 的问题

    Returns:
        事件循环对象
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def run_async(coro):
    """
    同步运行异步协程

    Args:
        coro: 异步协程对象

    Returns:
        协程的返回值
    """
    loop = get_or_create_event_loop()
    return loop.run_until_complete(coro)


# ==================== JSON解析工具 ====================

def extract_json(text: str) -> Optional[Dict]:
    """
    从文本中提取JSON对象

    支持以下格式：
    1. 纯JSON
    2. ```json ... ```
    3. ``` ... ```

    Args:
        text: 包含JSON的文本

    Returns:
        解析后的字典，失败返回None
    """
    if not text:
        return None

    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从代码块中提取
    patterns = [
        ("```json", "```"),  # ```json ... ```
        ("```", "```"),      # ``` ... ```
    ]

    for start_marker, end_marker in patterns:
        if start_marker in text:
            start = text.find(start_marker) + len(start_marker)
            end = text.find(end_marker, start)
            if end > start:
                try:
                    json_text = text[start:end].strip()
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    continue

    return None


def parse_bool_response(text: str, default: bool = False) -> bool:
    """
    从文本中解析布尔值

    支持多种表达方式：
    - "应该合并", "should_merge": true → True
    - "不应合并", "keep_separate" → False

    Args:
        text: 文本响应
        default: 默认值

    Returns:
        解析的布尔值
    """
    text_lower = text.lower()

    positive_keywords = ["应该合并", "should merge", "是", "yes", "true", '"should_merge": true']
    negative_keywords = ["不应合并", "keep separate", "否", "no", "false", '"should_merge": false']

    for keyword in positive_keywords:
        if keyword in text_lower:
            return True

    for keyword in negative_keywords:
        if keyword in text_lower:
            return False

    return default


# ==================== 并查集工具 ====================

class UnionFind:
    """
    并查集（不相交集合）数据结构
    用于知识融合中的实体合并
    """

    def __init__(self, size: int):
        """
        初始化并查集

        Args:
            size: 元素数量
        """
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        """
        查找元素所属的集合根节点（带路径压缩）

        Args:
            x: 元素索引

        Returns:
            根节点索引
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x: int, y: int):
        """
        合并两个元素所在的集合（按秩合并）

        Args:
            x: 元素1索引
            y: 元素2索引
        """
        px, py = self.find(x), self.find(y)

        if px == py:
            return  # 已经在同一集合

        # 按秩合并
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

    def get_groups(self) -> Dict[int, List[int]]:
        """
        获取所有分组

        Returns:
            字典：{根节点: [成员索引列表]}
        """
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups


# ==================== 实体格式化工具 ====================

def format_entity(entity: Dict, fields: Optional[List[str]] = None) -> str:
    """
    格式化实体信息为可读文本

    Args:
        entity: 实体字典
        fields: 要显示的字段列表（默认显示所有）

    Returns:
        格式化的文本
    """
    if fields is None:
        fields = ['brand', 'product_model', 'category', 'series', 'manufacturer']

    lines = []
    for field in fields:
        value = entity.get(field, '')
        if value:
            # 处理列表字段
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value[:5])
                if len(value) > 5:
                    value_str += f" ... (共{len(value)}项)"
                lines.append(f"- {field}: {value_str}")
            else:
                lines.append(f"- {field}: {value}")

    return "\n".join(lines) if lines else "无详细信息"


def format_similarity(details: Dict) -> str:
    """
    格式化相似度信息为可读文本

    Args:
        details: 相似度详情字典

    Returns:
        格式化的文本
    """
    lines = []
    for field, info in details.items():
        if isinstance(info, dict):
            sim = info.get('similarity')
            if sim is not None:
                lines.append(f"- {field}: {sim:.3f}")
            else:
                note = info.get('note', '缺失')
                lines.append(f"- {field}: {note}")

    return "\n".join(lines) if lines else "无相似度数据"


# ==================== 统计工具 ====================

def calculate_stats(
    total: int,
    processed: int,
    successful: int,
    failed: int
) -> Dict[str, Any]:
    """
    计算统计信息

    Args:
        total: 总数
        processed: 已处理数
        successful: 成功数
        failed: 失败数

    Returns:
        统计信息字典
    """
    return {
        "总数": total,
        "已处理": processed,
        "成功": successful,
        "失败": failed,
        "成功率": f"{successful/processed*100:.1f}%" if processed > 0 else "N/A",
        "处理率": f"{processed/total*100:.1f}%" if total > 0 else "N/A",
    }


# ==================== 测试 ====================

if __name__ == "__main__":
    # 测试JSON提取
    print("测试 JSON 提取:")
    test_cases = [
        '{"key": "value"}',
        '```json\n{"key": "value"}\n```',
        '```\n{"key": "value"}\n```',
        'Some text {"key": "value"} more text',
    ]

    for text in test_cases:
        result = extract_json(text)
        print(f"  输入: {text[:50]}...")
        print(f"  结果: {result}\n")

    # 测试并查集
    print("测试并查集:")
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(1, 2)

    groups = uf.get_groups()
    print(f"  分组: {groups}")

    # 测试实体格式化
    print("\n测试实体格式化:")
    entity = {
        'brand': '美的',
        'product_model': 'MDV-D15Q4/BP3N1-E',
        'category': '空调',
        'features': ['变频', '静音', '节能']
    }
    print(format_entity(entity))

    print("\n工具函数模块测试完成")
