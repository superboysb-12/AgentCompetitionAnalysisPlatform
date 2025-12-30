"""
Chroma 向量数据库封装
提供向量的存储、检索、删除等功能
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma 向量数据库

    基于 ChromaDB 实现的向量存储，支持持久化存储
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "crawl_results",
        distance_metric: str = "cosine",
        anonymized_telemetry: bool = False,
    ):
        """
        初始化 Chroma 向量存储

        Args:
            persist_directory: 持久化存储目录
            collection_name: 集合名称
            distance_metric: 距离度量 (cosine, l2, ip)
            anonymized_telemetry: 是否启用匿名遥测
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # 确保存储目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # 延迟加载客户端
        self._client = None
        self._collection = None

        # 配置遥测
        if not anonymized_telemetry:
            import os
            os.environ["ANONYMIZED_TELEMETRY"] = "False"

        logger.info(f"Chroma 向量存储配置完成: {persist_directory}/{collection_name}")

    def _get_client(self):
        """获取或创建 Chroma 客户端"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                logger.info(f"正在初始化 Chroma 客户端...")

                # 创建持久化客户端
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    )
                )

                logger.info(f"✓ Chroma 客户端初始化成功")

            except ImportError:
                error_msg = "chromadb 库未安装，请运行: pip install chromadb"
                logger.error(error_msg)
                raise ImportError(error_msg)
            except Exception as e:
                logger.error(f"✗ Chroma 客户端初始化失败: {e}")
                raise

        return self._client

    def _get_collection(self):
        """获取或创建集合"""
        if self._collection is None:
            client = self._get_client()

            try:
                # 尝试获取现有集合
                self._collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=None,  # 我们自己管理 embeddings
                )
                logger.info(f"✓ 已连接到现有集合: {self.collection_name}")

            except Exception:
                # 集合不存在，创建新集合
                logger.info(f"集合不存在，创建新集合: {self.collection_name}")

                self._collection = client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric},
                    embedding_function=None,
                )
                logger.info(f"✓ 集合创建成功: {self.collection_name}")

        return self._collection

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
            ids: 文档ID列表，如果为None则使用文档中的'id'字段

        Returns:
            List[str]: 文档ID列表
        """
        if not documents or not embeddings:
            return []

        if len(documents) != len(embeddings):
            raise ValueError(f"文档数量({len(documents)})与向量数量({len(embeddings)})不匹配")

        collection = self._get_collection()

        try:
            # 准备数据
            doc_ids = ids if ids else [str(doc.get('id', i)) for i, doc in enumerate(documents)]
            metadatas = []
            texts = []

            for doc in documents:
                # 提取文本内容（用于显示，不用于检索）
                text = doc.get('content', '')[:500]  # 限制长度
                texts.append(text if text else " ")  # Chroma 不接受空字符串

                # 提取元数据（Chroma 要求所有值必须是字符串、数字或布尔值）
                metadata = {
                    'db_id': str(doc.get('id', '')),
                    'url': str(doc.get('url', ''))[:2000],  # 限制长度
                    'title': str(doc.get('title', ''))[:500],
                    'crawled_at': str(doc.get('crawled_at', '')),
                }
                metadatas.append(metadata)

            # 批量添加
            collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
            )

            logger.info(f"✓ 成功添加 {len(doc_ids)} 个文档到向量数据库")
            return doc_ids

        except Exception as e:
            logger.error(f"✗ 添加文档失败: {e}")
            raise

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
            filter: 元数据过滤条件（暂不支持）

        Returns:
            List[Tuple[Dict[str, Any], float]]: (文档元数据, 相似度分数) 列表
        """
        collection = self._get_collection()

        try:
            # 执行查询
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter,  # 元数据过滤
                include=['metadatas', 'distances']
            )

            # 解析结果
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]

                    # 将距离转换为相似度分数 (cosine: 1 - distance)
                    if self.distance_metric == "cosine":
                        score = 1 - distance
                    else:
                        score = 1 / (1 + distance)  # 简单转换

                    search_results.append((metadata, score))

            logger.debug(f"检索到 {len(search_results)} 个相关文档")
            return search_results

        except Exception as e:
            logger.error(f"✗ 向量搜索失败: {e}")
            raise

    def delete(self, ids: List[str]) -> bool:
        """
        删除文档

        Args:
            ids: 文档ID列表

        Returns:
            bool: 是否成功
        """
        if not ids:
            return True

        collection = self._get_collection()

        try:
            collection.delete(ids=ids)
            logger.info(f"✓ 成功删除 {len(ids)} 个文档")
            return True

        except Exception as e:
            logger.error(f"✗ 删除文档失败: {e}")
            return False

    def clear(self) -> bool:
        """
        清空所有文档

        Returns:
            bool: 是否成功
        """
        try:
            client = self._get_client()

            # 删除并重建集合
            try:
                client.delete_collection(name=self.collection_name)
                logger.info(f"✓ 已删除集合: {self.collection_name}")
            except Exception:
                pass  # 集合可能不存在

            # 重建集合
            self._collection = client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
                embedding_function=None,
            )

            logger.info(f"✓ 集合已清空并重建: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"✗ 清空集合失败: {e}")
            return False

    def count(self) -> int:
        """
        获取文档数量

        Returns:
            int: 文档数量
        """
        try:
            collection = self._get_collection()
            return collection.count()

        except Exception as e:
            logger.error(f"✗ 获取文档数量失败: {e}")
            return 0
