"""
SQLAlchemy ORM Model 定义
定义爬虫结果和任务记录的数据库表结构
"""

from sqlalchemy import Column, BigInteger, String, Text, JSON, DateTime, Boolean, Index, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class CrawlTaskModel(Base):
    """
    爬取任务记录表
    记录每次爬取任务的配置文件、状态、错误信息等
    """
    __tablename__ = 'crawl_tasks'

    # 主键
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')

    # 任务信息
    task_id = Column(String(255), unique=True, nullable=False, index=True, comment='任务ID（来自Redis）')
    config_path = Column(String(512), nullable=False, comment='配置文件路径')
    config_name = Column(String(255), comment='配置文件名')

    # 任务状态
    status = Column(String(50), default='pending', comment='任务状态：pending/running/completed/failed')

    # 时间记录
    started_at = Column(DateTime, comment='任务开始时间')
    completed_at = Column(DateTime, comment='任务完成时间')
    duration_seconds = Column(Integer, comment='任务耗时（秒）')

    # 结果统计
    results_count = Column(Integer, default=0, comment='爬取到的结果数量')

    # 错误信息
    error_message = Column(Text, comment='错误信息')
    error_traceback = Column(Text, comment='错误堆栈')

    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), comment='记录创建时间')
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment='记录更新时间')

    # 索引
    __table_args__ = (
        Index('idx_config_path', 'config_path'),
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<CrawlTask(id={self.id}, task_id='{self.task_id}', config='{self.config_name}', status='{self.status}')>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'task_id': self.task_id,
            'config_path': self.config_path,
            'config_name': self.config_name,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'results_count': self.results_count,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class CrawlResultModel(Base):
    """
    爬取结果数据表 Model
    用于存储网页爬取的所有数据，包括 RAG 核心字段和元数据
    """
    __tablename__ = 'crawl_results'

    # 主键
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')

    # URL 字段（去重核心）
    url = Column(String(2048), unique=True, nullable=False, index=True, comment='实际爬取的URL')
    original_url = Column(String(2048), comment='原始起始URL（列表页）')  # 索引在 __table_args__ 中定义

    # RAG 核心字段
    title = Column(Text, comment='页面标题')
    content = Column(Text, comment='正文内容')

    # 完整数据保留（JSON 格式）
    raw_content = Column(JSON, comment='完整的提取内容（links, images 等）')

    # 时间字段
    crawled_at = Column(DateTime, index=True, comment='爬取时间')
    created_at = Column(DateTime, server_default=func.now(), comment='记录创建时间')
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment='记录更新时间')

    # 爬取元数据
    new_tab = Column(Boolean, default=False, comment='是否在新标签页打开')
    strategy_used = Column(String(50), comment='使用的点击策略')
    button_info = Column(JSON, comment='按钮信息（text, selector, href）')

    # 定义索引
    __table_args__ = (
        Index('idx_original_url', 'original_url', mysql_length=255),
        Index('idx_crawled_at', 'crawled_at'),
        Index('idx_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<CrawlResult(id={self.id}, url='{self.url[:50]}...', title='{self.title[:30] if self.title else None}')>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'url': self.url,
            'original_url': self.original_url,
            'title': self.title,
            'content': self.content,
            'raw_content': self.raw_content,
            'crawled_at': self.crawled_at.isoformat() if self.crawled_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'new_tab': self.new_tab,
            'strategy_used': self.strategy_used,
            'button_info': self.button_info
        }
