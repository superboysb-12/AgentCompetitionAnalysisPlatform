"""
RAG 检索器
实现语义搜索功能
"""

from typing import List, Dict, Any, Optional
import logging
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from .embeddings import BGEM3Embeddings
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    RAG 检索器

    提供语义搜索功能，根据查询文本检索最相关的文档
    """

    def __init__(
        self,
        mysql_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        chroma_config: Dict[str, Any],
        rag_config: Dict[str, Any],
    ):
        """
        初始化检索器

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

        # 使用单例管理器获取共享的 Embedding 和向量存储实例
        from .singleton import RAGSingletonManager

        self.embeddings = RAGSingletonManager.get_embeddings(embedding_config)
        self.vector_store = RAGSingletonManager.get_vector_store(chroma_config)

        # 初始化 MySQL 连接
        self._engine = None
        self._Session = None

        logger.info("RAG 检索器初始化完成（使用共享实例）")

    def _get_db_session(self):
        """获取数据库会话"""
        if self._engine is None:
            connection_string = (
                f"mysql+pymysql://{self.mysql_config['user']}:{self.mysql_config['password']}"
                f"@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}"
                f"?charset={self.mysql_config.get('charset', 'utf8mb4')}"
            )

            self._engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                echo=False
            )
            self._Session = sessionmaker(bind=self._engine)

        return self._Session()

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
            top_k: 返回的结果数量，默认使用配置中的值
            score_threshold: 相似度阈值，低于此值的结果会被过滤
            return_full_content: 是否返回完整内容（包括 raw_content）

        Returns:
            List[Dict]: 搜索结果列表
                [
                    {
                        'id': 文档ID,
                        'url': URL,
                        'title': 标题,
                        'content': 内容（可能被截断）,
                        'score': 相似度分数,
                        'crawled_at': 爬取时间,
                        'raw_content': 原始内容（仅在 return_full_content=True 时返回）
                    },
                    ...
                ]
        """
        if not query or not query.strip():
            logger.warning("查询文本为空")
            return []

        # 使用配置的默认值
        top_k = top_k or self.rag_config['top_k']
        score_threshold = score_threshold if score_threshold is not None else self.rag_config['score_threshold']
        max_content_length = self.rag_config['max_content_length']

        try:
            # 1. 将查询文本转换为向量
            logger.info(f"查询: {query[:100]}...")
            query_embedding = self.embeddings.embed_query(query)

            # 2. 向量搜索
            logger.info(f"在向量数据库中搜索 Top-{top_k} 相关文档...")
            vector_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            if not vector_results:
                logger.info("未找到相关文档")
                return []

            # 3. 过滤低分结果
            filtered_results = [
                (metadata, score)
                for metadata, score in vector_results
                if score >= score_threshold
            ]

            if not filtered_results:
                logger.info(f"所有结果的相似度低于阈值 {score_threshold}")
                return []

            logger.info(f"找到 {len(filtered_results)} 个相关文档（分数 >= {score_threshold}）")

            # 4. 从 MySQL 获取完整的文档信息
            doc_ids = [int(metadata['db_id']) for metadata, _ in filtered_results]
            session = self._get_db_session()

            try:
                # 导入模型
                from crawl.mysql.models import CrawlResultModel

                # 查询完整文档
                query = select(CrawlResultModel).where(
                    CrawlResultModel.id.in_(doc_ids)
                )
                db_results = session.execute(query).scalars().all()

                # 构建结果字典（用于快速查找）
                db_results_dict = {result.id: result for result in db_results}

                # 5. 组装最终结果
                final_results = []
                for metadata, score in filtered_results:
                    doc_id = int(metadata['db_id'])
                    db_result = db_results_dict.get(doc_id)

                    if not db_result:
                        logger.warning(f"文档 {doc_id} 在 MySQL 中不存在，跳过")
                        continue

                    # 截断内容
                    content = db_result.content or ""
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."

                    result = {
                        'id': db_result.id,
                        'url': db_result.url,
                        'title': db_result.title,
                        'content': content,
                        'score': round(score, 4),
                        'crawled_at': db_result.crawled_at.isoformat() if db_result.crawled_at else None,
                    }

                    # 可选：返回完整内容
                    if return_full_content:
                        result['full_content'] = db_result.content
                        result['raw_content'] = db_result.raw_content

                    final_results.append(result)

                logger.info(f"✓ 返回 {len(final_results)} 个结果")
                return final_results

            finally:
                session.close()

        except Exception as e:
            logger.error(f"✗ 搜索失败: {e}")
            import traceback
            traceback.print_exc()
            raise
