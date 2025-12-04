"""
模块基类定义
所有新模块都应该继承这个基类
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from fastapi import FastAPI, APIRouter
from celery import Celery


class BaseModule(ABC):
    """模块基类

    新模块只需要继承这个类并实现需要的方法即可
    """

    # 模块元信息
    module_name: str = "base"
    module_version: str = "1.0.0"
    module_description: str = "Base module"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化模块

        Args:
            config: 模块配置（可选）
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)

    @abstractmethod
    def register_routes(self, app: FastAPI) -> Optional[APIRouter]:
        """注册API路由

        Args:
            app: FastAPI应用实例

        Returns:
            APIRouter: 返回路由器（如果有）

        Example:
            router = APIRouter(prefix="/api/v1/your_module", tags=["your_module"])

            @router.get("/status")
            async def get_status():
                return {"status": "ok"}

            return router
        """
        pass

    @abstractmethod
    def register_tasks(self, celery_app: Celery) -> None:
        """注册Celery任务

        Args:
            celery_app: Celery应用实例

        Example:
            @celery_app.task(name="your_module.your_task")
            def your_task():
                return "done"
        """
        pass

    def on_startup(self) -> None:
        """模块启动时的钩子

        可选实现，用于初始化资源、连接数据库等
        """
        pass

    def on_shutdown(self) -> None:
        """模块关闭时的钩子

        可选实现，用于清理资源、关闭连接等
        """
        pass

    def get_config_schema(self) -> Dict[str, Any]:
        """返回模块配置的JSON Schema

        可选实现，用于配置验证和文档生成

        Returns:
            dict: JSON Schema格式的配置定义
        """
        return {}

    def get_dependencies(self) -> list[str]:
        """返回模块依赖的其他模块名称

        可选实现，用于依赖检查

        Returns:
            list: 依赖的模块名称列表
        """
        return []
