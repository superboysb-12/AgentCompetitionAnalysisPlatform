"""
爬虫模块适配器
将现有的爬虫功能适配到模块系统
"""
from typing import Optional, Dict, Any
from fastapi import FastAPI, APIRouter
from celery import Celery

from app.modules.base import BaseModule


class CrawlModule(BaseModule):
    """爬虫模块"""

    module_name = "crawl"
    module_version = "1.0.0"
    module_description = "Web crawling and content extraction module"

    def register_routes(self, app: FastAPI) -> Optional[APIRouter]:
        """注册爬虫API路由"""
        # 导入模块内的路由
        from . import api

        # 返回路由器，让模块加载器自动注册
        router = api.router

        # 设置路由前缀和标签
        router.prefix = f"/api/v1/tasks"
        router.tags = ["任务管理", "crawl"]

        return router

    def register_tasks(self, celery_app: Celery) -> None:
        """注册爬虫Celery任务"""
        # 爬虫任务已经在 app.tasks.crawl_tasks 中定义
        # Celery会自动发现并注册，这里无需额外操作
        pass

    def on_startup(self) -> None:
        """爬虫模块启动"""
        import logging
        from pathlib import Path
        from app.config import settings

        logger = logging.getLogger(__name__)

        # 验证爬虫配置目录是否存在
        try:
            config_dir = Path(settings.crawler_config_dir)
            if not config_dir.exists():
                logger.warning(f"爬虫配置目录不存在: {config_dir}")
            else:
                # 统计配置文件数量
                config_files = list(config_dir.glob("*.yaml"))
                logger.info(f"✓ 爬虫模块启动成功，发现 {len(config_files)} 个配置文件")
        except Exception as e:
            logger.warning(f"爬虫模块初始化警告: {e}")

    def on_shutdown(self) -> None:
        """爬虫模块关闭"""
        # 清理资源（如果需要）
        pass

    def get_config_schema(self) -> Dict[str, Any]:
        """返回爬虫模块配置Schema"""
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否启用爬虫模块"
                },
                "config_dir": {
                    "type": "string",
                    "default": "crawler_configs",
                    "description": "爬虫配置目录"
                },
                "max_concurrent_tasks": {
                    "type": "integer",
                    "default": 5,
                    "description": "最大并发爬取任务数"
                }
            }
        }

    def get_dependencies(self) -> list[str]:
        """爬虫模块依赖"""
        # 爬虫模块不依赖其他模块
        return []
