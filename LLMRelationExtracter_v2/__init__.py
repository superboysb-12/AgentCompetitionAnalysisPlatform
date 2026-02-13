"""
LLMRelationExtracter_v2
-----------------------

Brand → Series → Product 分阶段抽取管线，保持与 v1 相同的输入/输出约定：
- 输入：CSV，或 Markdown（会先转换为同结构的 page/rows 视图）
- 输出：[{ "results": [product...] } ] 结构，字段与 v1 一致。
"""

from .staged_extractor import (
    StagedRelationExtractor,
    extract_relations_multistage,
    load_pages_with_context_v2,
)

__all__ = [
    "StagedRelationExtractor",
    "extract_relations_multistage",
    "load_pages_with_context_v2",
]
