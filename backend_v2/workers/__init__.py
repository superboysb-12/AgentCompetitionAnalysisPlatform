"""
Worker 模块
提供基础 Worker 类和具体 Worker 实现
"""

from .base import BaseWorker
from .crawler_worker import CrawlerWorker
from .placeholder_worker import PlaceholderWorker

__all__ = ['BaseWorker', 'CrawlerWorker', 'PlaceholderWorker']
