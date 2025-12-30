"""
RAG 服务封装
提供统一的 RAG 服务接口
"""

from typing import List, Dict, Any, Optional
import logging
from rag import RAGIndexer, RAGRetriever

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 服务

    封装索引构建和检索功能，提供统一的对外接口
    """

    def __init__(
        self,
        mysql_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        chroma_config: Dict[str, Any],
        rag_config: Dict[str, Any],
    ):
        """
        初始化 RAG 服务

        Args:
            mysql_config: MySQL 配置
            embedding_config: Embedding 模型配置
            chroma_config: Chroma 向量数据库配置
            rag_config: RAG 服务配置
        """
        self.mysql_config = mysql_config
        self.embedding_config = embedding_config
        self.chroma_config = chroma_config
        self.rag_config = rag_config

        # 初始化索引构建器
        self.indexer = RAGIndexer(
            mysql_config=mysql_config,
            embedding_config=embedding_config,
            chroma_config=chroma_config,
        )

        # 初始化检索器
        self.retriever = RAGRetriever(
            mysql_config=mysql_config,
            embedding_config=embedding_config,
            chroma_config=chroma_config,
            rag_config=rag_config,
        )

        logger.info("RAG 服务初始化完成")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        return_full_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索

        Args:
            query: 查询文本
            top_k: 返回的结果数量
            score_threshold: 相似度阈值
            return_full_content: 是否返回完整内容

        Returns:
            List[Dict]: 搜索结果列表
        """
        return self.retriever.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            return_full_content=return_full_content,
        )

    def build_index(
        self,
        batch_size: int = 100,
        last_indexed_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        构建或更新索引

        Args:
            batch_size: 批量处理大小
            last_indexed_id: 上次索引的最大ID

        Returns:
            Dict: 构建结果统计
        """
        return self.indexer.build_index(
            batch_size=batch_size,
            last_indexed_id=last_indexed_id,
        )

    def get_index_status(self) -> Dict[str, Any]:
        """
        获取索引状态

        Returns:
            Dict: 索引状态信息
        """
        return self.indexer.get_index_status()

    def clear_index(self) -> bool:
        """
        清空索引

        Returns:
            bool: 是否成功
        """
        return self.indexer.clear_index()
