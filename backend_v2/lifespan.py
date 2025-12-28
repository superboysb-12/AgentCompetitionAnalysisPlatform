"""
Lifespan 管理模块
负责 FastAPI 应用的启动和关闭生命周期管理
在启动时创建子进程，关闭时终止子进程
"""

import logging
import multiprocessing as mp
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI
from services import RedisClient, TaskManager

logger = logging.getLogger(__name__)


def _run_crawler_worker():
    """
    在子进程中运行 CrawlerWorker
    """
    # 配置子进程日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s - %(message)s'
    )

    from workers import CrawlerWorker

    worker = CrawlerWorker()
    worker.start()


def _run_placeholder_worker():
    """
    在子进程中运行 PlaceholderWorker
    """
    # 配置子进程日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s - %(message)s'
    )

    from workers import PlaceholderWorker

    worker = PlaceholderWorker()
    worker.start()


class LifespanManager:
    """
    生命周期管理器
    管理子进程的启动和关闭
    """

    def __init__(self):
        self.processes: List[mp.Process] = []
        self.redis_client: RedisClient = None
        self.task_manager: TaskManager = None

    def startup(self):
        """
        应用启动时执行
        """
        logger.info("========== 应用启动中 ==========")

        # 初始化 Redis 客户端和任务管理器
        self.redis_client = RedisClient()
        self.task_manager = TaskManager(self.redis_client)
        logger.info("Redis 客户端已初始化")

        # 启动子进程1: CrawlerWorker
        crawler_process = mp.Process(target=_run_crawler_worker, name="CrawlerWorker")
        crawler_process.start()
        self.processes.append(crawler_process)
        logger.info(f"子进程1 CrawlerWorker 已启动，PID: {crawler_process.pid}")

        # 启动子进程2: PlaceholderWorker
        placeholder_process = mp.Process(target=_run_placeholder_worker, name="PlaceholderWorker")
        placeholder_process.start()
        self.processes.append(placeholder_process)
        logger.info(f"子进程2 PlaceholderWorker 已启动，PID: {placeholder_process.pid}")

        logger.info("========== 应用启动完成 ==========")

    def shutdown(self):
        """
        应用关闭时执行
        """
        logger.info("========== 应用关闭中 ==========")

        # 终止所有子进程
        for process in self.processes:
            if process.is_alive():
                logger.info(f"正在终止子进程: {process.name} (PID: {process.pid})")
                process.terminate()
                process.join(timeout=5)

                if process.is_alive():
                    logger.warning(f"子进程 {process.name} 未能正常关闭，强制杀死")
                    process.kill()
                    process.join()

                logger.info(f"子进程 {process.name} 已关闭")

        # 关闭 Redis 连接
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis 客户端已关闭")

        logger.info("========== 应用已关闭 ==========")


# 全局实例
_lifespan_manager = LifespanManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan 上下文管理器

    Args:
        app: FastAPI 应用实例
    """
    # 启动时执行
    _lifespan_manager.startup()

    # 将 redis_client 和 task_manager 注入到 app.state 中
    app.state.redis_client = _lifespan_manager.redis_client
    app.state.task_manager = _lifespan_manager.task_manager

    yield

    # 关闭时执行
    _lifespan_manager.shutdown()
