"""
知识图谱三元组抽取系统

这是一个基于OpenAI API的知识图谱三元组联合抽取系统，
专门用于从文本中抽取结构化的知识图谱三元组。

主要模块:
- KnowledgeGraphExtractor: 核心抽取器
- KnowledgeGraphBuilder: 批处理构建器
- FewShotManager: Few-shot示例管理器

使用示例:
    from triple_extractor import KnowledgeGraphExtractor, KnowledgeGraphBuilder

    # 单文本抽取
    extractor = KnowledgeGraphExtractor('config.yaml')
    result = extractor.extract_from_text("你的文本")

    # 批量处理
    builder = KnowledgeGraphBuilder('config.yaml')
    result = builder.build_knowledge_graph('input.json')
"""

from .kg_extractor import KnowledgeGraphExtractor, Triplet, ExtractionResult
from .kg_builder import KnowledgeGraphBuilder
from .few_shot_manager import FewShotManager, FewShotExample

__version__ = "1.0.0"
__author__ = "Knowledge Graph Team"

__all__ = [
    'KnowledgeGraphExtractor',
    'KnowledgeGraphBuilder',
    'FewShotManager',
    'Triplet',
    'ExtractionResult',
    'FewShotExample'
]