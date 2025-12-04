"""
爬虫任务模型
记录每次调度的爬取任务
"""
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Text, Enum as SQLEnum
from app.database import Base


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消


class CrawlTask(Base):
    """爬虫任务表"""
    __tablename__ = "crawl_tasks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # 任务标识
    task_name = Column(String(100), nullable=False, index=True, comment="任务名称")
    config_file = Column(String(255), nullable=False, comment="配置文件路径")

    # 任务状态
    status = Column(
        SQLEnum(TaskStatus),
        default=TaskStatus.PENDING,
        nullable=False,
        index=True,
        comment="任务状态"
    )

    # 时间信息
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="创建时间")
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")

    # 执行信息
    celery_task_id = Column(String(255), nullable=True, index=True, comment="Celery任务ID")

    # 结果统计
    total_urls = Column(Integer, default=0, comment="总URL数")
    success_count = Column(Integer, default=0, comment="成功数量")
    failed_count = Column(Integer, default=0, comment="失败数量")
    duplicate_count = Column(Integer, default=0, comment="去重数量")

    # 错误信息
    error_message = Column(Text, nullable=True, comment="错误信息")

    # 元数据
    metadata_json = Column(Text, nullable=True, comment="额外元数据(JSON)")

    def __repr__(self):
        return f"<CrawlTask(id={self.id}, name={self.task_name}, status={self.status})>"
