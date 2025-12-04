"""
APScheduler定时调度器
管理周期性爬虫任务
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from datetime import datetime
import logging

from app.config import settings
from app.modules.crawl.tasks import crawl_all_configs
from app.modules.extraction.tasks import batch_extract_all_pending_parallel

logger = logging.getLogger(__name__)


class CrawlScheduler:
    """爬虫调度器"""

    def __init__(self):
        """初始化调度器"""
        # 配置jobstores和executors
        jobstores = {
            'default': MemoryJobStore()
        }

        executors = {
            'default': ThreadPoolExecutor(max_workers=3)
        }

        job_defaults = {
            'coalesce': True,  # 合并多个等待执行的同一任务
            'max_instances': 1,  # 同一任务最多同时运行1个实例
            'misfire_grace_time': 3600  # 错过时间窗口1小时内仍执行
        }

        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=settings.scheduler_timezone
        )

        self._is_running = False

    def add_weekly_crawl_job(self):
        """
        添加每周日半夜的定时爬取任务
        默认时间：每周日 00:00
        """
        # 解析cron表达式
        # 格式: minute hour day month day_of_week
        # 例如: "0 0 * * 0" 表示每周日午夜
        cron_parts = settings.weekly_crawl_cron.split()

        if len(cron_parts) != 5:
            logger.error(f"无效的cron表达式: {settings.weekly_crawl_cron}")
            return

        trigger = CronTrigger(
            minute=cron_parts[0],
            hour=cron_parts[1],
            day=cron_parts[2],
            month=cron_parts[3],
            day_of_week=cron_parts[4],
            timezone=settings.scheduler_timezone
        )

        self.scheduler.add_job(
            func=self._execute_weekly_crawl,
            trigger=trigger,
            id="weekly_crawl_all_configs",
            name="每周定时爬取所有配置",
            replace_existing=True
        )

        logger.info(f"已添加定时任务: 每周日半夜爬取 (cron: {settings.weekly_crawl_cron})")

    def add_hourly_extraction_job(self):
        """
        添加每小时执行一次的知识抽取任务（并行模式）
        默认：每小时的第10分钟执行
        """
        trigger = CronTrigger(
            minute='10',
            hour='*',
            timezone=settings.scheduler_timezone
        )

        self.scheduler.add_job(
            func=self._execute_batch_extraction,
            trigger=trigger,
            id="hourly_batch_extraction_parallel",
            name="每小时并行批量抽取未处理记录",
            replace_existing=True
        )

        logger.info(f"已添加定时任务: 每小时并行批量抽取知识图谱")

    def _execute_batch_extraction(self):
        """执行批量知识抽取任务（并行模式）"""
        logger.info(f"定时任务触发: 开始并行批量抽取 [{datetime.now()}]")

        try:
            # 调用Celery异步并行任务
            task = batch_extract_all_pending_parallel.delay(max_workers=settings.kg_max_workers)

            logger.info(f"Celery并行抽取任务已提交: {task.id}")

            return {
                "status": "submitted",
                "celery_task_id": task.id,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"定时抽取任务执行失败: {e}")
            raise

    def _execute_weekly_crawl(self):
        """执行周期性爬取任务"""
        logger.info(f"定时任务触发: 开始批量爬取 [{datetime.now()}]")

        try:
            # 调用Celery异步任务
            task = crawl_all_configs.delay(settings.crawler_config_dir)

            logger.info(f"Celery任务已提交: {task.id}")

            # 注意：这里不等待任务完成，任务在Celery worker中异步执行
            return {
                "status": "submitted",
                "celery_task_id": task.id,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"定时任务执行失败: {e}")
            raise

    def add_custom_job(self, func, trigger, job_id: str, **kwargs):
        """
        添加自定义定时任务

        Args:
            func: 任务函数
            trigger: 触发器
            job_id: 任务ID
            **kwargs: 其他参数
        """
        self.scheduler.add_job(
            func=func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            **kwargs
        )

        logger.info(f"已添加自定义任务: {job_id}")

    def start(self):
        """启动调度器"""
        if not settings.scheduler_enabled:
            logger.warning("调度器已禁用（SCHEDULER_ENABLED=False）")
            return

        if self._is_running:
            logger.warning("调度器已在运行")
            return

        # 添加默认的周期性任务
        self.add_weekly_crawl_job()
        self.add_hourly_extraction_job()  # 添加知识抽取定时任务

        # 启动调度器
        self.scheduler.start()
        self._is_running = True

        logger.info("调度器已启动")
        self._print_jobs()

    def shutdown(self, wait: bool = True):
        """
        关闭调度器

        Args:
            wait: 是否等待正在执行的任务完成
        """
        if not self._is_running:
            logger.warning("调度器未运行")
            return

        self.scheduler.shutdown(wait=wait)
        self._is_running = False

        logger.info("调度器已关闭")

    def _print_jobs(self):
        """打印所有已调度的任务"""
        jobs = self.scheduler.get_jobs()

        if not jobs:
            logger.info("当前没有调度任务")
            return

        logger.info(f"当前调度任务列表 ({len(jobs)}个):")
        for job in jobs:
            logger.info(f"  - [{job.id}] {job.name} | 下次执行: {job.next_run_time}")

    def get_jobs_info(self) -> list:
        """
        获取所有任务信息

        Returns:
            list: 任务信息列表
        """
        jobs = self.scheduler.get_jobs()

        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            }
            for job in jobs
        ]

    def is_running(self) -> bool:
        """检查调度器是否运行"""
        return self._is_running


# 创建全局调度器实例
crawler_scheduler = CrawlScheduler()
