"""
Embedding 基类
定义 Embedding 模型的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbeddings(ABC):
    """
    Embedding 基类
    所有 Embedding 模型必须继承此类并实现 embed_documents 和 embed_query 方法
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文档文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        将查询文本转换为向量

        Args:
            text: 查询文本

        Returns:
            List[float]: 向量
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度
        """
        pass
