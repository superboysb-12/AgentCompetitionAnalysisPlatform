"""
MySQL 存储实现
提供 MySQL 数据库存储功能，支持 URL 去重和批量检查
"""

from typing import List, Dict, Any, Set
from datetime import datetime
from crawl.core.types import CrawlResult
from .models import CrawlResultModel, Base
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class MySQLStorage:
    """
    MySQL 存储类
    将爬取结果保存到 MySQL 数据库，支持 URL 去重检查
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 MySQL 存储器

        Args:
            config: 全局配置字典，包含 MySQL 连接信息
        """
        self.config = config
        mysql_config = config.get('mysql', {})

        # 构建连接字符串
        connection_string = (
            f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}"
            f"@{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
            f"?charset={mysql_config.get('charset', 'utf8mb4')}"
        )

        # 创建引擎和会话工厂
        self.engine = create_engine(
            connection_string,
            pool_size=mysql_config.get('pool_size', 5),
            max_overflow=mysql_config.get('max_overflow', 10),
            pool_pre_ping=mysql_config.get('pool_pre_ping', True),
            pool_recycle=mysql_config.get('pool_recycle', 3600),
            echo=mysql_config.get('echo', False)
        )
        self.Session = sessionmaker(bind=self.engine)

        # 用于异步执行的线程池
        self.executor = ThreadPoolExecutor(max_workers=5)

        logger.info(f"MySQL 存储初始化成功: {mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}")

    async def save(self, task_name: str, results: List[CrawlResult]) -> int:
        """
        保存爬取结果到 MySQL 数据库（实时保存，支持单条或批量）

        Args:
            task_name: 任务名称（保留接口兼容性，实际不使用）
            results: 爬取结果列表

        Returns:
            int: 成功保存的记录数量
        """
        if not results:
            return 0

        def _save_sync():
            session = self.Session()
            try:
                saved_count = 0
                for result in results:
                    try:
                        # 从 content 字典中提取 title 和 content
                        title = None
                        content_text = None

                        if result.content:
                            # 提取 title
                            if 'title' in result.content:
                                title_data = result.content['title']
                                if isinstance(title_data, list) and title_data:
                                    title = title_data[0]
                                elif isinstance(title_data, str):
                                    title = title_data

                            # 提取 content
                            if 'content' in result.content:
                                content_data = result.content['content']
                                if isinstance(content_data, list):
                                    content_text = '\n'.join(content_data)
                                elif isinstance(content_data, str):
                                    content_text = content_data

                        # 创建数据库记录
                        db_result = CrawlResultModel(
                            url=result.url,
                            original_url=result.original_url,
                            title=title,
                            content=content_text,
                            raw_content=result.content,
                            crawled_at=datetime.fromisoformat(result.timestamp) if result.timestamp else datetime.now(),
                            new_tab=result.new_tab,
                            strategy_used=result.strategy_used.value if result.strategy_used else None,
                            button_info=result.button_info
                        )

                        session.add(db_result)
                        session.commit()
                        saved_count += 1
                        logger.info(f"✓ 成功保存到数据库: {result.url}")

                    except IntegrityError as e:
                        # URL 重复，跳过
                        session.rollback()
                        logger.warning(f"⊗ URL 已存在，跳过: {result.url}")
                    except Exception as e:
                        session.rollback()
                        logger.error(f"✗ 保存失败: {result.url}, 错误: {e}")

                logger.info(f"批量保存完成，成功: {saved_count}/{len(results)}")
                return saved_count

            finally:
                session.close()

        # 在线程池中执行同步数据库操作
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(self.executor, _save_sync)
        return count

    async def url_exists(self, url: str) -> bool:
        """
        检查 URL 是否已存在于数据库中

        Args:
            url: 待检查的 URL

        Returns:
            bool: 存在返回 True，不存在返回 False
        """
        def _check_sync():
            session = self.Session()
            try:
                stmt = select(CrawlResultModel).where(CrawlResultModel.url == url)
                result = session.execute(stmt).first()
                return result is not None
            finally:
                session.close()

        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(self.executor, _check_sync)
        return exists

    async def batch_check_urls(self, urls: List[str]) -> Set[str]:
        """
        批量检查 URL 是否存在，返回已存在的 URL 集合

        Args:
            urls: URL 列表

        Returns:
            Set[str]: 已存在的 URL 集合
        """
        if not urls:
            return set()

        def _batch_check_sync():
            session = self.Session()
            try:
                stmt = select(CrawlResultModel.url).where(CrawlResultModel.url.in_(urls))
                results = session.execute(stmt).scalars().all()
                return set(results)
            finally:
                session.close()

        loop = asyncio.get_event_loop()
        existing_urls = await loop.run_in_executor(self.executor, _batch_check_sync)
        logger.info(f"批量检查完成，已存在 URL 数量: {len(existing_urls)}/{len(urls)}")
        return existing_urls

    async def get_by_original_url(self, original_url: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据 original_url 查询爬取结果（用于数据分析）

        Args:
            original_url: 原始 URL
            limit: 返回结果数量限制

        Returns:
            List[Dict]: 结果列表
        """
        def _query_sync():
            session = self.Session()
            try:
                stmt = (
                    select(CrawlResultModel)
                    .where(CrawlResultModel.original_url == original_url)
                    .order_by(CrawlResultModel.crawled_at.desc())
                    .limit(limit)
                )
                results = session.execute(stmt).scalars().all()
                return [
                    {
                        'id': r.id,
                        'url': r.url,
                        'title': r.title,
                        'content': r.content,
                        'crawled_at': r.crawled_at.isoformat() if r.crawled_at else None
                    }
                    for r in results
                ]
            finally:
                session.close()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _query_sync)

    def close(self):
        """关闭连接池和线程池"""
        self.executor.shutdown(wait=True)
        self.engine.dispose()
        logger.info("MySQL 存储已关闭")
