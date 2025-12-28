"""
预留 Worker（第二个子进程）
订阅 placeholder:task 频道，暂时不处理任务
"""

import logging
from typing import Dict, Any
from .base import BaseWorker

logger = logging.getLogger(__name__)


class PlaceholderWorker(BaseWorker):
    """
    预留 Worker
    暂时只订阅频道，不做实际处理
    """

    CHANNEL = "placeholder:task"

    def __init__(self):
        """初始化预留 Worker"""
        super().__init__(self.CHANNEL)
        logger.info("PlaceholderWorker 已初始化（预留功能）")

    def process_task(self, task_id: str, message: Dict[str, Any]) -> None:
        """
        处理任务（暂时为空实现）

        Args:
            task_id: 任务 ID
            message: 消息字典
        """
        logger.info(f"PlaceholderWorker 收到任务: {task_id}, 消息: {message}")
        logger.info("暂未实现具体处理逻辑")
