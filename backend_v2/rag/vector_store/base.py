"""
向量存储基类
定义向量数据库的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseVectorStore(ABC):
    """
    向量存储基类
    所有向量数据库必须继承此类并实现相关方法
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        添加文档和对应的向量

        Args:
            documents: 文档列表，每个文档包含元数据
            embeddings: 向量列表
            ids: 文档ID列表，如果为None则自动生成

        Returns:
            List[str]: 文档ID列表
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        向量相似度搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回的结果数量
            filter: 元数据过滤条件

        Returns:
            List[Tuple[Dict[str, Any], float]]: (文档, 相似度分数) 列表
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        删除文档

        Args:
            ids: 文档ID列表

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        清空所有文档

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        获取文档数量

        Returns:
            int: 文档数量
        """
        pass
