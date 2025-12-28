"""
基础 Worker 类
定义 Worker 的通用接口和行为
"""

import logging
import signal
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any
from services import RedisClient, TaskManager

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    基础 Worker 类
    所有 Worker 必须继承此类并实现 process_task 方法
    """

    def __init__(self, channel: str):
        """
        初始化 Worker

        Args:
            channel: Redis 订阅频道名称
        """
        self.channel = channel
        self.running = True

        # 初始化 Redis 客户端（每个 Worker 独立连接）
        self.redis_client = RedisClient()
        self.task_manager = TaskManager(self.redis_client)

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"{self.__class__.__name__} 初始化完成，订阅频道: {channel}")

    def _signal_handler(self, signum, frame):
        """
        信号处理器（优雅关闭）

        Args:
            signum: 信号编号
            frame: 当前栈帧
        """
        logger.info(f"{self.__class__.__name__} 收到信号 {signum}，准备关闭...")
        self.running = False
        sys.exit(0)

    def start(self) -> None:
        """
        启动 Worker，订阅 Redis 频道
        """
        logger.info(f"{self.__class__.__name__} 开始监听频道: {self.channel}")

        try:
            self.redis_client.subscribe(self.channel, self._handle_message)
        except Exception as e:
            logger.error(f"{self.__class__.__name__} 运行出错: {e}")
        finally:
            self.close()

    def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        处理接收到的消息

        Args:
            message: 消息字典
        """
        task_id = message.get('task_id')
        if not task_id:
            logger.warning(f"消息缺少 task_id: {message}")
            return

        logger.info(f"开始处理任务: {task_id}")

        try:
            # 更新状态为运行中
            self.task_manager.update_task_status(task_id, 'running')

            # 调用具体的任务处理方法
            self.process_task(task_id, message)

            # 更新状态为完成
            self.task_manager.update_task_status(task_id, 'completed')
            logger.info(f"任务完成: {task_id}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"任务失败: {task_id}, 错误: {error_msg}")
            self.task_manager.update_task_status(task_id, 'failed', error=error_msg)

    @abstractmethod
    def process_task(self, task_id: str, message: Dict[str, Any]) -> None:
        """
        处理具体任务（子类必须实现）

        Args:
            task_id: 任务 ID
            message: 任务消息字典
        """
        pass

    def close(self) -> None:
        """关闭 Worker 资源"""
        logger.info(f"{self.__class__.__name__} 正在关闭...")
        self.redis_client.close()
        logger.info(f"{self.__class__.__name__} 已关闭")
