"""数据库模型初始化"""
from .task import CrawlTask, TaskStatus
from .result import CrawlResult, URLDeduplication
from .extraction import ExtractionRecord, ExtractionStatus

__all__ = [
    "CrawlTask",
    "TaskStatus",
    "CrawlResult",
    "URLDeduplication",
    "ExtractionRecord",
    "ExtractionStatus",
]
