"""
RAG Worker（索引构建子进程）
订阅 rag:task 频道，支持手动触发和定时构建索引
"""

import logging
from typing import Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from .base import BaseWorker
from services import RAGService
from settings import MYSQL_CONFIG, EMBEDDING_CONFIG, CHROMA_CONFIG, RAG_CONFIG, SCHEDULER_CONFIG

logger = logging.getLogger(__name__)


class RAGWorker(BaseWorker):
    """
    RAG Worker

    负责定时构建和更新 RAG 索引，支持手动触发
    """

    CHANNEL = "rag:task"

    def __init__(self):
        """初始化 RAG Worker"""
        super().__init__(self.CHANNEL)

        # 初始化 RAG 服务
        self.rag_service = RAGService(
            mysql_config=MYSQL_CONFIG,
            embedding_config=EMBEDDING_CONFIG,
            chroma_config=CHROMA_CONFIG,
            rag_config=RAG_CONFIG,
        )

        # 初始化定时任务调度器
        self.scheduler = BackgroundScheduler(timezone=SCHEDULER_CONFIG['timezone'])

        # 记录上次索引的最大ID（用于增量更新）
        self.last_indexed_id = None

        # 设置定时任务
        self._setup_scheduler()

        logger.info("RAG Worker 已初始化")

    def _setup_scheduler(self):
        """设置定时任务"""
        # 优先使用 Cron 表达式
        cron_expr = SCHEDULER_CONFIG.get('rag_index_cron')

        if cron_expr:
            # 使用 Cron 表达式
            try:
                # 解析 Cron 表达式（格式：分 时 日 月 周）
                parts = cron_expr.split()
                trigger = CronTrigger(
                    minute=parts[0] if len(parts) > 0 else '0',
                    hour=parts[1] if len(parts) > 1 else '*',
                    day=parts[2] if len(parts) > 2 else '*',
                    month=parts[3] if len(parts) > 3 else '*',
                    day_of_week=parts[4] if len(parts) > 4 else '*',
                    timezone=SCHEDULER_CONFIG['timezone']
                )
                self.scheduler.add_job(
                    self._scheduled_build_index,
                    trigger=trigger,
                    id='rag_index_build_cron',
                    name='RAG 索引定时构建 (Cron)',
                    replace_existing=True,
                )
                logger.info(f"✓ 定时任务已设置 (Cron): {cron_expr}")
            except Exception as e:
                logger.error(f"✗ Cron 表达式解析失败: {e}，使用间隔触发")
                self._setup_interval_trigger()
        else:
            # 使用间隔触发
            self._setup_interval_trigger()

        # 启动调度器
        self.scheduler.start()
        logger.info("✓ 定时任务调度器已启动")

    def _setup_interval_trigger(self):
        """设置间隔触发器"""
        interval_hours = SCHEDULER_CONFIG.get('rag_index_interval_hours', 6)

        self.scheduler.add_job(
            self._scheduled_build_index,
            trigger=IntervalTrigger(hours=interval_hours, timezone=SCHEDULER_CONFIG['timezone']),
            id='rag_index_build_interval',
            name=f'RAG 索引定时构建 (每{interval_hours}小时)',
            replace_existing=True,
        )
        logger.info(f"✓ 定时任务已设置 (间隔): 每 {interval_hours} 小时")

    def _scheduled_build_index(self):
        """定时任务：构建索引"""
        logger.info("=" * 60)
        logger.info("定时任务触发：开始构建 RAG 索引")
        logger.info("=" * 60)

        try:
            result = self.rag_service.build_index(
                batch_size=100,
                last_indexed_id=self.last_indexed_id,
            )

            # 更新最后索引的ID
            if result.get('last_id'):
                self.last_indexed_id = result['last_id']

            logger.info(f"✓ 定时索引构建完成: {result}")

        except Exception as e:
            logger.error(f"✗ 定时索引构建失败: {e}")
            import traceback
            traceback.print_exc()

    def process_task(self, task_id: str, message: Dict[str, Any]) -> None:
        """
        处理任务（手动触发）

        支持的任务类型：
        - build_index: 构建索引
        - clear_index: 清空索引
        - get_status: 获取索引状态

        Args:
            task_id: 任务 ID
            message: 消息字典
                {
                    'task_id': 任务ID,
                    'action': 'build_index' | 'clear_index' | 'get_status',
                    'params': {...}  # 可选参数
                }
        """
        action = message.get('action', 'build_index')
        params = message.get('params', {})

        logger.info(f"RAG Worker 收到任务: {task_id}, 操作: {action}")

        try:
            if action == 'build_index':
                # 手动触发索引构建
                batch_size = params.get('batch_size', 100)
                last_indexed_id = params.get('last_indexed_id', self.last_indexed_id)

                result = self.rag_service.build_index(
                    batch_size=batch_size,
                    last_indexed_id=last_indexed_id,
                )

                # 更新最后索引的ID
                if result.get('last_id'):
                    self.last_indexed_id = result['last_id']

                logger.info(f"✓ 索引构建完成: {result}")

            elif action == 'clear_index':
                # 清空索引
                success = self.rag_service.clear_index()

                if success:
                    self.last_indexed_id = None  # 重置
                    logger.info(f"✓ 索引已清空")
                else:
                    logger.error(f"✗ 清空索引失败")

            elif action == 'get_status':
                # 获取索引状态
                status = self.rag_service.get_index_status()
                logger.info(f"索引状态: {status}")

            else:
                logger.warning(f"未知操作: {action}")

        except Exception as e:
            logger.error(f"✗ 任务执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close(self) -> None:
        """关闭 Worker 资源"""
        # 关闭调度器
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("定时任务调度器已关闭")

        # 调用父类关闭方法
        super().close()
