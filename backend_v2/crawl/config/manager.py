"""
配置管理模块
负责加载、验证和管理爬虫任务的配置文件
支持从 YAML 文件读取任务配置，从 backend_v2/config.py 读取数据库配置
"""

from typing import Dict, Any
from pathlib import Path
import yaml
import logging
import sys
import os

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    配置管理器类
    加载YAML配置文件，设置默认值，并验证配置的正确性
    """

    def __init__(self, config_path: str):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件的路径（YAML格式）
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_backend_config()  # 加载后端配置（MySQL 等）
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        从YAML文件加载配置

        Returns:
            Dict[str, Any]: 加载并设置了默认值的配置字典

        Raises:
            Exception: 当配置文件加载失败时抛出异常
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            config = self._set_defaults(config)
            return config

        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise

    def _set_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        为配置设置默认值，包括全局设置和任务级设置

        Args:
            config: 原始配置字典

        Returns:
            Dict[str, Any]: 设置了默认值后的配置字典
        """
        settings = config.setdefault('settings', {})
        settings.setdefault('browser_type', 'chromium')
        settings.setdefault('headless', False)
        settings.setdefault('output_dir', 'results')
        settings.setdefault('concurrent_limit', 1)
        settings.setdefault('global_timeout', 300)

        for task in config.get('tasks', []):
            browser = task.setdefault('browser', {})
            browser.setdefault('timeout', 180)
            browser.setdefault('js_wait_time', 8)
            browser.setdefault('max_retries', 3)
            browser.setdefault('button_interval', 10)  # 按钮间等待时间（秒）

            discovery = task.setdefault('button_discovery', {})
            discovery.setdefault('max_buttons', 5)
            discovery.setdefault('selectors', [])

            task.setdefault('content_extraction', {})
            task.setdefault('wait_conditions', {})

        return config

    def _load_backend_config(self) -> None:
        """
        从 backend_v2/config.py 加载后端配置（MySQL、Redis 等）

        将 backend_v2 添加到 Python 路径，导入配置模块
        """
        try:
            # 获取 backend_v2 目录的路径
            crawl_dir = Path(__file__).resolve().parent.parent
            backend_dir = crawl_dir.parent

            # 将 backend_v2 添加到 Python 路径
            if str(backend_dir) not in sys.path:
                sys.path.insert(0, str(backend_dir))

            # 导入配置
            import settings as backend_config

            # 将 MySQL 配置合并到 settings 中
            settings = self.config.setdefault('settings', {})
            settings['mysql'] = backend_config.MYSQL_CONFIG
            settings['storage_type'] = backend_config.CRAWLER_CONFIG.get('storage_type', 'mysql')
            settings['output_dir'] = backend_config.CRAWLER_CONFIG.get('output_dir', 'results')

            logger.info(f"成功加载后端配置，存储类型: {settings['storage_type']}")

        except Exception as e:
            logger.error(f"加载后端配置失败: {e}")
            logger.warning("将使用默认配置")
            # 设置默认的 MySQL 配置
            settings = self.config.setdefault('settings', {})
            settings.setdefault('storage_type', 'json')
            settings.setdefault('mysql', {
                'host': 'localhost',
                'port': 3306,
                'database': 'agent_competition_db',
                'user': 'root',
                'password': 'your_password',
                'charset': 'utf8mb4',
                'pool_size': 5,
                'max_overflow': 10,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
                'echo': False
            })

    def _validate_config(self) -> None:
        """
        验证配置的有效性

        检查必须的字段是否存在，包括任务列表、任务名称和起始URL

        Raises:
            ValueError: 当配置不满足要求时抛出异常
        """
        if not self.config.get('tasks'):
            raise ValueError("配置文件必须包含至少一个任务")

        for task in self.config['tasks']:
            if not task.get('name'):
                raise ValueError("每个任务必须有名称")
            if not task.get('start_url'):
                raise ValueError("每个任务必须有起始URL")

    def get_config(self) -> Dict[str, Any]:
        """
        获取完整的配置字典

        Returns:
            Dict[str, Any]: 完整的配置信息
        """
        return self.config

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """
        根据任务名称获取特定任务的配置

        Args:
            task_name: 任务名称

        Returns:
            Dict[str, Any]: 任务的配置信息

        Raises:
            ValueError: 当找不到指定任务时抛出异常
        """
        for task in self.config['tasks']:
            if task['name'] == task_name:
                return task
        raise ValueError(f"未找到任务: {task_name}")

    def get_global_settings(self) -> Dict[str, Any]:
        """
        获取全局设置

        Returns:
            Dict[str, Any]: 全局设置信息
        """
        return self.config.get('settings', {})
