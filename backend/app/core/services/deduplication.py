"""
去重服务
提供基于URL和内容的去重功能
"""
from typing import Optional
from sqlalchemy.orm import Session
from app.core.models.result import URLDeduplication, CrawlResult
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DeduplicationService:
    """去重服务"""

    def __init__(self, db: Session):
        self.db = db

    def is_url_duplicate(self, url: str) -> bool:
        """
        检查URL是否已存在

        Args:
            url: 待检查的URL

        Returns:
            bool: True表示重复，False表示不重复
        """
        url_hash = CrawlResult.compute_url_hash(url)
        existing = self.db.query(URLDeduplication).filter(
            URLDeduplication.url_hash == url_hash
        ).first()

        return existing is not None

    def mark_url_seen(self, url: str) -> URLDeduplication:
        """
        标记URL已被访问（新增或更新）

        Args:
            url: URL地址

        Returns:
            URLDeduplication: 去重记录
        """
        url_hash = CrawlResult.compute_url_hash(url)

        # 查找现有记录
        existing = self.db.query(URLDeduplication).filter(
            URLDeduplication.url_hash == url_hash
        ).first()

        if existing:
            # 更新现有记录
            existing.last_seen = datetime.utcnow()
            existing.occurrence_count += 1
            self.db.commit()
            logger.debug(f"URL已存在，更新计数: {url[:50]}...")
            return existing
        else:
            # 创建新记录
            new_record = URLDeduplication(
                url_hash=url_hash,
                url=url,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                occurrence_count=1
            )
            self.db.add(new_record)
            self.db.commit()
            logger.debug(f"新URL记录: {url[:50]}...")
            return new_record

    def is_content_duplicate(self, content: str, threshold: float = 0.95) -> bool:
        """
        检查内容是否重复

        Args:
            content: 内容文本
            threshold: 相似度阈值（暂时使用精确匹配）

        Returns:
            bool: True表示重复
        """
        if not content:
            return False

        content_hash = CrawlResult.compute_content_hash(content)

        # 检查是否存在相同哈希的内容
        existing = self.db.query(CrawlResult).filter(
            CrawlResult.content_hash == content_hash
        ).first()

        return existing is not None

    def check_and_mark(self, url: str) -> tuple[bool, URLDeduplication]:
        """
        检查URL并标记

        Args:
            url: URL地址

        Returns:
            tuple: (是否重复, 去重记录)
        """
        is_dup = self.is_url_duplicate(url)
        record = self.mark_url_seen(url)
        return is_dup, record

    def get_dedup_stats(self) -> dict:
        """
        获取去重统计信息

        Returns:
            dict: 统计信息
        """
        total_urls = self.db.query(URLDeduplication).count()
        total_occurrences = self.db.query(
            URLDeduplication
        ).with_entities(
            URLDeduplication.occurrence_count
        ).all()

        total_count = sum(count[0] for count in total_occurrences)

        return {
            "unique_urls": total_urls,
            "total_occurrences": total_count,
            "duplicate_rate": (total_count - total_urls) / total_count if total_count > 0 else 0
        }
