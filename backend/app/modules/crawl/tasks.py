"""
Celery爬虫任务定义
"""
from celery import Task
from typing import Dict, Any, List
import asyncio
import logging

from celery_app import celery_app
from app.database import SessionLocal
from .services.crawler_service import CrawlerService
from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """带数据库会话的任务基类"""

    def __call__(self, *args, **kwargs):
        """重写调用方法，确保每个任务都有数据库会话"""
        with SessionLocal() as db:
            self.db = db
            return self.run(*args, **kwargs)


@celery_app.task(bind=True, base=DatabaseTask, name="app.tasks.crawl_tasks.crawl_single_config")
def crawl_single_config(self, config_path: str, task_name: str) -> Dict[str, Any]:
    """
    爬取单个配置文件

    Args:
        config_path: 配置文件路径
        task_name: 任务名称

    Returns:
        dict: 执行结果
    """
    logger.info(f"Celery任务开始: {task_name}, 任务ID: {self.request.id}")

    try:
        # 创建爬虫服务
        crawler_service = CrawlerService(self.db)

        # 由于爬虫是异步的，需要运行在事件循环中
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                crawler_service.execute_crawl_task(
                    config_path=config_path,
                    task_name=task_name,
                    celery_task_id=self.request.id
                )
            )
        finally:
            loop.close()

        logger.info(f"Celery任务完成: {task_name}")
        return result

    except Exception as e:
        logger.error(f"Celery任务失败: {task_name}, 错误: {e}")
        raise


@celery_app.task(bind=True, base=DatabaseTask, name="app.tasks.crawl_tasks.crawl_all_configs")
def crawl_all_configs(self, config_dir: str = None) -> List[Dict[str, Any]]:
    """
    爬取所有配置文件（用于定时任务）

    Args:
        config_dir: 配置文件目录，默认从settings读取

    Returns:
        list: 所有任务的执行结果
    """
    if config_dir is None:
        config_dir = settings.crawler_config_dir

    logger.info(f"批量爬取任务开始: {config_dir}, 任务ID: {self.request.id}")

    try:
        # 创建爬虫服务
        crawler_service = CrawlerService(self.db)

        # 运行异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(
                crawler_service.execute_all_configs(config_dir=config_dir)
            )
        finally:
            loop.close()

        logger.info(f"批量爬取任务完成，共{len(results)}个配置文件")
        return results

    except Exception as e:
        logger.error(f"批量爬取任务失败: {e}")
        raise


@celery_app.task(name="app.tasks.crawl_tasks.health_check")
def health_check() -> Dict[str, str]:
    """
    健康检查任务

    Returns:
        dict: 健康状态
    """
    return {
        "status": "healthy",
        "message": "Celery worker is running"
    }
