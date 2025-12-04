"""
爬虫服务层
协调爬虫执行、结果存储和去重
"""
from typing import List, Dict, Any
from pathlib import Path
from sqlalchemy.orm import Session
import logging

from .crawler_wrapper import CrawlerWrapper
from app.core.services.storage import MySQLStorageService
from app.core.models.task import TaskStatus
from app.config import settings

logger = logging.getLogger(__name__)


class CrawlerService:
    """爬虫服务"""

    def __init__(self, db: Session):
        self.db = db
        self.storage_service = MySQLStorageService(db)

    async def execute_crawl_task(
        self,
        config_path: str,
        task_name: str,
        celery_task_id: str = None
    ) -> Dict[str, Any]:
        """
        执行单个爬取任务

        Args:
            config_path: 配置文件路径
            task_name: 任务名称
            celery_task_id: Celery任务ID

        Returns:
            dict: 执行结果统计
        """
        # 创建任务记录
        task = self.storage_service.create_task(
            task_name=task_name,
            config_file=config_path,
            celery_task_id=celery_task_id
        )

        try:
            # 更新状态为运行中
            self.storage_service.update_task_status(task.id, TaskStatus.RUNNING)

            # 初始化爬虫
            crawler = CrawlerWrapper(config_path)
            await crawler.initialize()

            # 执行爬取
            logger.info(f"开始爬取任务: {task_name}")
            results = await crawler.run()

            # 保存结果到数据库（带去重）
            save_stats = self.storage_service.save_results(
                task_id=task.id,
                results=results,
                enable_dedup=True
            )

            # 关闭爬虫
            await crawler.close()

            # 更新任务状态为完成
            self.storage_service.update_task_status(task.id, TaskStatus.COMPLETED)

            logger.info(f"任务完成: {task_name}, 统计: {save_stats}")

            return {
                "task_id": task.id,
                "task_name": task_name,
                "status": "completed",
                "statistics": save_stats
            }

        except Exception as e:
            logger.error(f"任务执行失败: {task_name}, 错误: {e}")

            # 更新任务状态为失败
            self.storage_service.update_task_status(
                task.id,
                TaskStatus.FAILED,
                error_message=str(e)
            )

            return {
                "task_id": task.id,
                "task_name": task_name,
                "status": "failed",
                "error": str(e)
            }

    async def execute_all_configs(self, config_dir: str = None) -> List[Dict[str, Any]]:
        """
        执行目录下所有配置文件

        Args:
            config_dir: 配置文件目录，默认从settings读取

        Returns:
            List[dict]: 所有任务的执行结果
        """
        if config_dir is None:
            config_dir = settings.crawler_config_dir

        config_path = Path(config_dir)
        if not config_path.exists():
            raise ValueError(f"配置目录不存在: {config_dir}")

        # 查找所有YAML配置文件
        config_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))

        if not config_files:
            logger.warning(f"配置目录下没有找到YAML文件: {config_dir}")
            return []

        logger.info(f"找到{len(config_files)}个配置文件")

        results = []
        for config_file in config_files:
            task_name = config_file.stem  # 文件名（不含扩展名）

            try:
                result = await self.execute_crawl_task(
                    config_path=str(config_file),
                    task_name=task_name
                )
                results.append(result)
            except Exception as e:
                logger.error(f"配置文件执行失败: {config_file}, 错误: {e}")
                results.append({
                    "task_name": task_name,
                    "status": "failed",
                    "error": str(e)
                })

        return results

    def get_task_statistics(self, task_id: int) -> Dict[str, Any]:
        """
        获取任务统计信息

        Args:
            task_id: 任务ID

        Returns:
            dict: 统计信息
        """
        task = self.storage_service.get_task_by_id(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")

        results = self.storage_service.get_task_results(task_id, include_duplicates=False)

        return {
            "task_id": task.id,
            "task_name": task.task_name,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "total_urls": task.total_urls,
            "success_count": task.success_count,
            "duplicate_count": task.duplicate_count,
            "unique_results": len(results)
        }
