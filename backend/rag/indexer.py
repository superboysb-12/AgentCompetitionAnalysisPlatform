"""
RAG 索引构建器
从 MySQL 读取爬虫结果，生成向量并存储到 Chroma
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from .embeddings import BGEM3Embeddings
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGIndexer:
    """
    RAG 索引构建器

    负责从 MySQL 读取爬虫结果，批量生成 embeddings，并写入向量数据库
    """

    def __init__(
        self,
        mysql_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        chroma_config: Dict[str, Any],
    ):
        """
        初始化索引构建器

        Args:
            mysql_config: MySQL 配置
            embedding_config: Embedding 模型配置
            chroma_config: Chroma 向量数据库配置
        """
        self.mysql_config = mysql_config
        self.embedding_config = embedding_config
        self.chroma_config = chroma_config

        # 使用单例管理器获取共享的 Embedding 和向量存储实例
        from .singleton import RAGSingletonManager

        self.embeddings = RAGSingletonManager.get_embeddings(embedding_config)
        self.vector_store = RAGSingletonManager.get_vector_store(chroma_config)

        # 初始化 MySQL 连接
        self._engine = None
        self._Session = None

        logger.info("RAG 索引构建器初始化完成（使用共享实例）")

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

    def build_index(
        self,
        batch_size: int = 100,
        last_indexed_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        构建或更新索引

        Args:
            batch_size: 批量处理大小
            last_indexed_id: 上次索引的最大ID，用于增量更新

        Returns:
            Dict: 构建结果统计
                {
                    'total_processed': 处理的文档数量,
                    'total_indexed': 成功索引的文档数量,
                    'last_id': 最后处理的文档ID,
                    'start_time': 开始时间,
                    'end_time': 结束时间,
                    'duration_seconds': 耗时（秒）
                }
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("开始构建 RAG 索引...")

        session = self._get_db_session()
        total_processed = 0
        total_indexed = 0
        last_id = last_indexed_id or 0

        try:
            # 导入模型
            from crawl.mysql.models import CrawlResultModel

            # 先统计需要处理的文档总数
            count_query = select(func.count()).select_from(CrawlResultModel).where(
                CrawlResultModel.id > last_id
            )
            total_count = session.execute(count_query).scalar() or 0

            if total_count == 0:
                logger.info("没有需要索引的新文档")
                return {
                    'total_processed': 0,
                    'total_indexed': 0,
                    'last_id': last_id,
                    'start_time': start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': 0,
                }

            logger.info(f"找到 {total_count} 个待索引文档")

            # 查询需要索引的文档
            query = select(CrawlResultModel).where(
                CrawlResultModel.id > last_id
            ).order_by(CrawlResultModel.id)

            # 创建进度条
            pbar = tqdm(
                total=total_count,
                desc="📊 索引构建进度",
                unit="docs",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            # 分批处理
            offset = 0
            batch_num = 0
            while True:
                # 获取一批数据
                batch_query = query.offset(offset).limit(batch_size)
                results = session.execute(batch_query).scalars().all()

                if not results:
                    break  # 没有更多数据

                batch_num += 1
                pbar.set_description(f"📊 批次 {batch_num} (共 {len(results)} 条)")

                # 准备文档数据
                documents = []
                texts = []

                for result in results:
                    # 构建用于 embedding 的文本（标题 + 内容）
                    title = result.title or ""
                    content = result.content or ""
                    text = f"{title}\n{content}".strip()

                    if not text:
                        logger.debug(f"文档 {result.id} 内容为空，跳过")
                        pbar.update(1)
                        continue

                    texts.append(text)
                    documents.append({
                        'id': result.id,
                        'url': result.url,
                        'title': title,
                        'content': content,
                        'crawled_at': result.crawled_at.isoformat() if result.crawled_at else None,
                    })

                    last_id = result.id  # 更新最后处理的ID

                if not texts:
                    offset += batch_size
                    continue

                # 生成 embeddings
                pbar.set_description(f"🧠 生成向量 (批次 {batch_num})")
                embeddings = self.embeddings.embed_documents(texts)

                # 存储到向量数据库
                pbar.set_description(f"💾 存储向量 (批次 {batch_num})")
                doc_ids = self.vector_store.add_documents(
                    documents=documents,
                    embeddings=embeddings,
                )

                total_processed += len(results)
                total_indexed += len(doc_ids)

                # 更新进度条
                pbar.update(len(doc_ids))
                pbar.set_postfix({
                    '已索引': total_indexed,
                    '成功率': f"{total_indexed/total_processed*100:.1f}%" if total_processed > 0 else "0%"
                })

                offset += batch_size

            # 关闭进度条
            pbar.close()

            # 构建完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = {
                'total_processed': total_processed,
                'total_indexed': total_indexed,
                'last_id': last_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
            }

            logger.info("=" * 60)
            logger.info(f"✓ 索引构建完成！")
            logger.info(f"  处理文档: {total_processed}")
            logger.info(f"  成功索引: {total_indexed}")
            logger.info(f"  最后ID: {last_id}")
            logger.info(f"  耗时: {duration:.2f} 秒")
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error(f"✗ 索引构建失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            session.close()

    def get_index_status(self) -> Dict[str, Any]:
        """
        获取索引状态

        Returns:
            Dict: 索引状态信息
                {
                    'total_documents': 向量数据库中的文档数量,
                    'vector_dimension': 向量维度,
                }
        """
        try:
            total_docs = self.vector_store.count()

            return {
                'total_documents': total_docs,
                'vector_dimension': self.embeddings.dimension,
            }

        except Exception as e:
            logger.error(f"✗ 获取索引状态失败: {e}")
            return {
                'total_documents': 0,
                'vector_dimension': 0,
                'error': str(e)
            }

    def clear_index(self) -> bool:
        """
        清空索引

        Returns:
            bool: 是否成功
        """
        logger.warning("⚠ 正在清空 RAG 索引...")
        success = self.vector_store.clear()

        if success:
            logger.info("✓ 索引已清空")
        else:
            logger.error("✗ 清空索引失败")

        return success
