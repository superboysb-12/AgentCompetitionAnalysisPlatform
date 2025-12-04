"""
爬取结果和去重模型
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Boolean,
    Index, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship
from app.database import Base
import hashlib


class CrawlResult(Base):
    """爬取结果表"""
    __tablename__ = "crawl_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # 关联任务
    task_id = Column(Integer, ForeignKey("crawl_tasks.id"), nullable=False, index=True, comment="关联任务ID")

    # URL信息
    url = Column(String(2048), nullable=False, comment="爬取URL")  # 移除index=True，URL太长无法索引
    url_hash = Column(String(64), nullable=False, index=True, comment="URL哈希(SHA256)")
    original_url = Column(String(2048), nullable=True, comment="原始URL")

    # 内容信息
    title = Column(Text, nullable=True, comment="标题")
    content = Column(Text, nullable=True, comment="正文内容")
    content_hash = Column(String(64), nullable=True, index=True, comment="内容哈希(SHA256)")

    # 爬取元数据
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, comment="爬取时间")
    new_tab = Column(Boolean, default=False, comment="是否新标签页")
    strategy_used = Column(String(50), nullable=True, comment="点击策略")

    # 按钮信息(JSON存储)
    button_text = Column(Text, nullable=True, comment="按钮文本")
    button_selector = Column(String(500), nullable=True, comment="按钮选择器")
    button_href = Column(String(2048), nullable=True, comment="按钮链接")

    # 额外数据
    extracted_data = Column(Text, nullable=True, comment="提取的额外数据(JSON)")

    # 去重标记
    is_duplicate = Column(Boolean, default=False, index=True, comment="是否重复")

    # 任务关系
    # task = relationship("CrawlTask", back_populates="results")

    # 索引优化
    __table_args__ = (
        Index('idx_url_hash_task', 'url_hash', 'task_id'),
        Index('idx_content_hash', 'content_hash'),
        Index('idx_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<CrawlResult(id={self.id}, url={self.url[:50]}...)>"

    @staticmethod
    def compute_url_hash(url: str) -> str:
        """计算URL的SHA256哈希"""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """计算内容的SHA256哈希"""
        if not content:
            return ""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class URLDeduplication(Base):
    """URL去重表（快速查询优化）"""
    __tablename__ = "url_deduplication"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # URL哈希（唯一索引）
    url_hash = Column(String(64), nullable=False, unique=True, comment="URL哈希")

    # 原始URL（用于调试）
    url = Column(String(2048), nullable=False, comment="原始URL")

    # 首次发现时间
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False, comment="首次发现时间")

    # 最后更新时间
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="最后发现时间")

    # 出现次数
    occurrence_count = Column(Integer, default=1, comment="出现次数")

    # 唯一约束
    __table_args__ = (
        UniqueConstraint('url_hash', name='uix_url_hash'),
    )

    def __repr__(self):
        return f"<URLDeduplication(url_hash={self.url_hash[:16]}..., count={self.occurrence_count})>"
