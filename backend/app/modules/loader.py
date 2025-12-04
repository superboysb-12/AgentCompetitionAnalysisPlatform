"""
模块加载器
自动发现并加载所有模块
"""
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from celery import Celery

from .base import BaseModule

logger = logging.getLogger(__name__)


class ModuleLoader:
    """模块加载器"""

    def __init__(self):
        self.modules: Dict[str, BaseModule] = {}
        self.modules_dir = Path(__file__).parent

    def discover_modules(self) -> List[str]:
        """自动发现所有模块

        Returns:
            list: 模块名称列表
        """
        module_names = []

        # 遍历modules目录下的所有子目录
        for item in self.modules_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                # 检查是否存在module.py文件
                module_file = item / "module.py"
                if module_file.exists():
                    module_names.append(item.name)

        logger.info(f"发现 {len(module_names)} 个模块: {module_names}")
        return module_names

    def load_module(self, module_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModule:
        """加载单个模块

        Args:
            module_name: 模块名称
            config: 模块配置

        Returns:
            BaseModule: 模块实例
        """
        try:
            # 动态导入模块
            module_path = f"app.modules.{module_name}.module"
            module_lib = importlib.import_module(module_path)

            # 获取模块类（约定：类名为 ModuleName + "Module"）
            # 例如：crawl -> CrawlModule, extraction -> ExtractionModule
            class_name = f"{module_name.capitalize()}Module"
            module_class = getattr(module_lib, class_name)

            # 实例化模块
            module_instance = module_class(config)

            logger.info(f"✓ 加载模块: {module_name} (v{module_instance.module_version})")
            return module_instance

        except Exception as e:
            logger.error(f"✗ 加载模块失败: {module_name}, 错误: {e}")
            raise

    def load_all_modules(self, modules_config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """加载所有模块

        Args:
            modules_config: 模块配置字典，格式:
                {
                    "crawl": {"enabled": True, ...},
                    "extraction": {"enabled": True, ...}
                }
        """
        modules_config = modules_config or {}

        # 自动发现模块
        module_names = self.discover_modules()

        # 加载每个模块
        for module_name in module_names:
            module_config = modules_config.get(module_name, {})

            # 检查模块是否启用
            if not module_config.get('enabled', True):
                logger.info(f"⊘ 跳过禁用的模块: {module_name}")
                continue

            try:
                module = self.load_module(module_name, module_config)
                self.modules[module_name] = module
            except Exception as e:
                logger.error(f"加载模块 {module_name} 失败: {e}")
                # 继续加载其他模块

        logger.info(f"模块加载完成，已加载 {len(self.modules)}/{len(module_names)} 个模块")

    def register_all(self, app: FastAPI, celery_app: Celery) -> None:
        """注册所有模块的路由和任务

        Args:
            app: FastAPI应用实例
            celery_app: Celery应用实例
        """
        for module_name, module in self.modules.items():
            logger.info(f"注册模块: {module_name}")

            try:
                # 注册路由
                router = module.register_routes(app)
                if router:
                    app.include_router(router)
                    logger.info(f"  ✓ 注册��由: {router.prefix if hasattr(router, 'prefix') else 'N/A'}")

                # 注册任务
                module.register_tasks(celery_app)
                logger.info(f"  ✓ 注册Celery任务")

            except Exception as e:
                logger.error(f"  ✗ 注册模块失败: {module_name}, 错误: {e}")
                raise

    def startup_all(self) -> None:
        """调用所有模块的startup钩子"""
        for module_name, module in self.modules.items():
            try:
                module.on_startup()
                logger.info(f"✓ 启动模块: {module_name}")
            except Exception as e:
                logger.error(f"✗ 启动模块失败: {module_name}, 错误: {e}")

    def shutdown_all(self) -> None:
        """调用所有模块的shutdown钩子"""
        for module_name, module in self.modules.items():
            try:
                module.on_shutdown()
                logger.info(f"✓ 关闭模块: {module_name}")
            except Exception as e:
                logger.error(f"✗ 关闭模块失败: {module_name}, 错误: {e}")

    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """获取模块实例

        Args:
            module_name: 模块名称

        Returns:
            BaseModule: 模块实例，如果不存在返回None
        """
        return self.modules.get(module_name)

    def list_modules(self) -> List[Dict[str, Any]]:
        """列出所有已加载的模块

        Returns:
            list: 模块信息列表
        """
        return [
            {
                "name": module.module_name,
                "version": module.module_version,
                "description": module.module_description,
                "enabled": module.enabled,
                "dependencies": module.get_dependencies()
            }
            for module in self.modules.values()
        ]


# 全局模块加载器实例
module_loader = ModuleLoader()
