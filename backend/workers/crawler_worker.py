"""
爬虫 Worker
订阅 crawler:task 频道，执行爬虫任务
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
from .base import BaseWorker

logger = logging.getLogger(__name__)


class CrawlerWorker(BaseWorker):
    """
    爬虫 Worker
    负责执行 SmartCrawler 爬取任务
    """

    CHANNEL = "crawler:task"

    def __init__(self):
        """初始化爬虫 Worker"""
        super().__init__(self.CHANNEL)

        # 确保 crawl 模块可以被导入
        backend_dir = Path(__file__).resolve().parent.parent
        crawl_dir = backend_dir / "crawl"
        if str(crawl_dir) not in sys.path:
            sys.path.insert(0, str(crawl_dir))

        logger.info("CrawlerWorker 已初始化，可调用 SmartCrawler")

    def process_task(self, task_id: str, message: Dict[str, Any]) -> None:
        """
        处理爬虫任务

        Args:
            task_id: 任务 ID
            message: 消息字典，包含 config_path 字段
        """
        import traceback
        from datetime import datetime
        from pathlib import Path as PathLib
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        config_path = message.get('config_path')
        if not config_path:
            raise ValueError("消息缺少 config_path 字段")

        logger.info(f"任务 {task_id} 配置文件: {config_path}")

        # 验证配置文件是否存在
        if not Path(config_path).exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 初始化数据库连接（用于记录任务）
        backend_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(backend_dir))
        from settings import MYSQL_CONFIG

        connection_string = (
            f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
            f"?charset={MYSQL_CONFIG.get('charset', 'utf8mb4')}"
        )
        engine = create_engine(connection_string)
        Session = sessionmaker(bind=engine)

        # 导入 Model（需要在 crawl 目录的 sys.path 中）
        crawl_dir = backend_dir / "crawl"
        if str(crawl_dir) not in sys.path:
            sys.path.insert(0, str(crawl_dir))

        # 记录任务开始
        session = Session()
        start_time = datetime.now()

        try:
            # 导入 models（避免触发 __init__.py）
            import mysql.models as models
            CrawlTaskModel = models.CrawlTaskModel

            # 创建任务记录
            task_record = CrawlTaskModel(
                task_id=task_id,
                config_path=config_path,
                config_name=PathLib(config_path).name,
                status='running',
                started_at=start_time
            )
            session.add(task_record)
            session.commit()

            # 执行爬虫
            from crawler import SmartCrawler

            logger.info(f"开始执行爬虫任务: {task_id}")
            crawler = SmartCrawler(config_path)

            # 在新的事件循环中运行异步爬虫，获取保存的结果数量
            results_count = asyncio.run(crawler.start())

            # 任务成功完成
            end_time = datetime.now()
            duration = int((end_time - start_time).total_seconds())

            task_record.status = 'completed'
            task_record.completed_at = end_time
            task_record.duration_seconds = duration
            task_record.results_count = results_count

            session.commit()

            logger.info(f"爬虫任务执行成功: {task_id}, 耗时: {duration}秒, 保存记录: {results_count}条")

        except Exception as e:
            # 任务失败
            end_time = datetime.now()
            duration = int((end_time - start_time).total_seconds())
            error_msg = str(e)
            error_trace = traceback.format_exc()

            logger.error(f"爬虫任务执行失败: {task_id}, 错误: {error_msg}")

            # 更新任务记录
            try:
                task_record.status = 'failed'
                task_record.completed_at = end_time
                task_record.duration_seconds = duration
                task_record.error_message = error_msg
                task_record.error_traceback = error_trace
                session.commit()
            except:
                # 如果更新失败，至少记录日志
                logger.error(f"无法更新任务记录: {task_id}")

            raise

        finally:
            session.close()
            engine.dispose()
