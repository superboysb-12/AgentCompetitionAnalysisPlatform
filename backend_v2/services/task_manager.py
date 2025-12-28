"""
任务管理模块
负责任务状态的创建、查询、更新等操作（基于 Redis）
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from .redis_client import RedisClient

logger = logging.getLogger(__name__)


class TaskManager:
    """
    任务管理器
    使用 Redis Hash 存储任务状态
    """

    TASK_KEY_PREFIX = "task:"

    def __init__(self, redis_client: RedisClient):
        """
        初始化任务管理器

        Args:
            redis_client: Redis 客户端实例
        """
        self.redis = redis_client

    def create_task(self, task_type: str, config: Dict[str, Any]) -> str:
        """
        创建新任务

        Args:
            task_type: 任务类型（如 'crawl'）
            config: 任务配置字典

        Returns:
            str: 任务 ID
        """
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        task_data = {
            'task_id': task_id,
            'task_type': task_type,
            'status': 'pending',
            'config': str(config),  # 转为字符串存储
            'created_at': now,
            'updated_at': now
        }

        key = f"{self.TASK_KEY_PREFIX}{task_id}"
        self.redis.hset_dict(key, task_data)
        logger.info(f"任务已创建: {task_id}, 类型: {task_type}")

        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, str]]:
        """
        获取任务信息

        Args:
            task_id: 任务 ID

        Returns:
            Optional[Dict]: 任务信息字典，不存在返回 None
        """
        key = f"{self.TASK_KEY_PREFIX}{task_id}"
        if not self.redis.exists(key):
            return None

        return self.redis.hget_all(key)

    def update_task_status(self, task_id: str, status: str, error: Optional[str] = None) -> None:
        """
        更新任务状态

        Args:
            task_id: 任务 ID
            status: 新状态 (pending|running|completed|failed)
            error: 错误信息（可选）
        """
        key = f"{self.TASK_KEY_PREFIX}{task_id}"

        update_data = {
            'status': status,
            'updated_at': datetime.now().isoformat()
        }

        if error:
            update_data['error'] = error

        self.redis.hset_dict(key, update_data)
        logger.info(f"任务状态已更新: {task_id} -> {status}")

    def list_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, str]]:
        """
        列出所有任务

        Args:
            task_type: 过滤任务类型（可选）

        Returns:
            List[Dict]: 任务列表
        """
        pattern = f"{self.TASK_KEY_PREFIX}*"
        task_keys = self.redis.keys(pattern)

        tasks = []
        for key in task_keys:
            task = self.redis.hget_all(key)
            if task and (task_type is None or task.get('task_type') == task_type):
                tasks.append(task)

        # 按创建时间倒序排序
        tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return tasks

    def delete_task(self, task_id: str) -> None:
        """
        删除任务

        Args:
            task_id: 任务 ID
        """
        key = f"{self.TASK_KEY_PREFIX}{task_id}"
        self.redis.delete(key)
        logger.info(f"任务已删除: {task_id}")
