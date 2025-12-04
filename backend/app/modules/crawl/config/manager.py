from typing import Dict, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigManager:

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            config = self._set_defaults(config)
            return config

        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise

    def _set_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
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

            discovery = task.setdefault('button_discovery', {})
            discovery.setdefault('max_buttons', 5)
            discovery.setdefault('selectors', [])

            task.setdefault('content_extraction', {})
            task.setdefault('wait_conditions', {})

        return config

    def _validate_config(self) -> None:
        if not self.config.get('tasks'):
            raise ValueError("配置文件必须包含至少一个任务")

        for task in self.config['tasks']:
            if not task.get('name'):
                raise ValueError("每个任务必须有名称")
            if not task.get('start_url'):
                raise ValueError("每个任务必须有起始URL")

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        for task in self.config['tasks']:
            if task['name'] == task_name:
                return task
        raise ValueError(f"未找到任务: {task_name}")

    def get_global_settings(self) -> Dict[str, Any]:
        return self.config.get('settings', {})
