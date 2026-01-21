"""
MySQL 存储模块
包含 MySQL 数据库相关的所有逻辑
"""

from .models import Base, CrawlResultModel
from .storage import MySQLStorage
from .init_db import ensure_tables_exist, init_database

__all__ = [
    'Base',
    'CrawlResultModel',
    'MySQLStorage',
    'ensure_tables_exist',
    'init_database',
]
