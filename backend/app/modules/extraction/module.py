"""
知识抽取模块适配器
将现有的抽取功能适配到模块系统
"""
from typing import Optional, Dict, Any
from fastapi import FastAPI, APIRouter
from celery import Celery

from app.modules.base import BaseModule


class ExtractionModule(BaseModule):
    """知识抽取模块"""

    module_name = "extraction"
    module_version = "1.0.0"
    module_description = "Knowledge graph extraction and Neo4j import module"

    def register_routes(self, app: FastAPI) -> Optional[APIRouter]:
        """注册抽取API路由"""
        # 导入模块内的路由
        from . import api

        # 返回路由器，让模块加载器自动注册
        router = api.router

        # 设置路由前缀和标签
        router.prefix = f"/api/v1/extraction"
        router.tags = ["知识图谱抽取", "extraction"]

        return router

    def register_tasks(self, celery_app: Celery) -> None:
        """注册抽取Celery任务"""
        # 抽取任务已经在 app.tasks.extraction_tasks 中定义
        # Celery会自动发现并注册，这里无需额外操作
        pass

    def on_startup(self) -> None:
        """抽取模块启动"""
        from .services.kg_extractor import KnowledgeGraphExtractor
        from .services.neo4j_service import Neo4jService
        import logging

        logger = logging.getLogger(__name__)

        # 验证LLM配置
        try:
            from app.config import settings
            # 使用模块配置目录
            kg_config_path = "app/modules/extraction/config/config.yaml"

            # 可以在这里做一些验证，例如检查配置文件是否存在
            from pathlib import Path
            if not Path(kg_config_path).exists():
                logger.warning(f"知识图谱配置文件不存在: {kg_config_path}")

        except Exception as e:
            logger.warning(f"抽取模块初始化警告: {e}")

        # 验证Neo4j连接（可选）
        try:
            with Neo4jService() as neo4j:
                neo4j.test_connection()
            logger.info("✓ Neo4j连接正常")
        except Exception as e:
            logger.warning(f"Neo4j连接失败: {e}")

    def on_shutdown(self) -> None:
        """抽取模块关闭"""
        # 清理资源（如果需要）
        pass

    def get_config_schema(self) -> Dict[str, Any]:
        """返回抽取模块配置Schema"""
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否启用抽取模块"
                },
                "kg_config_path": {
                    "type": "string",
                    "default": "kg_config/config.yaml",
                    "description": "知识图谱抽取配置文件路径"
                },
                "max_workers": {
                    "type": "integer",
                    "default": 10,
                    "description": "并行抽取最大线程数"
                },
                "max_retries": {
                    "type": "integer",
                    "default": 3,
                    "description": "抽取失败最大重试次数"
                },
                "neo4j_enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否启用Neo4j导入"
                }
            }
        }

    def get_dependencies(self) -> list[str]:
        """抽取模块依赖"""
        # 抽取模块依赖爬虫模块（需要爬取的数据）
        return ["crawl"]
