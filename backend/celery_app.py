"""
Celery应用配置
跨平台任务队列支持（Windows/Linux）

🆕 支持模块化任务注册：
- 静态任务：通过 include 参数自动发现
- 动态任务：模块通过 register_tasks() 方法注册
"""
from celery import Celery
from app.config import settings

# 创建Celery应用
celery_app = Celery(
    "crawler_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    # 🆕 静态任务自动发现（保留现有任务兼容性）
    include=[
        "app.modules.crawl.tasks",
        "app.modules.extraction.tasks"
    ]
)

# Celery配置
celery_app.conf.update(
    # 任务序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.scheduler_timezone,
    enable_utc=True,

    # 任务结果配置
    result_expires=3600 * 24 * 7,  # 结果保留7天
    result_backend_transport_options={"master_name": "mymaster"},

    # 任务路由
    task_routes={
        "app.tasks.crawl_tasks.*": {"queue": "crawl_queue"},
        "app.tasks.extraction_tasks.*": {"queue": "extraction_queue"},
    },

    # Worker配置
    worker_prefetch_multiplier=1,  # 一次只预取一个任务
    worker_max_tasks_per_child=50,  # 每个worker最多执行50个任务后重启

    # 任务超时
    task_time_limit=3600 * 2,  # 硬超时：2小时
    task_soft_time_limit=3600 * 1.5,  # 软超时：1.5小时

    # 重试配置
    task_acks_late=True,  # 任务执行完成后才确认
    task_reject_on_worker_lost=True,  # Worker丢失时拒绝任务

    # 日志
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
)

# Windows兼容性配置
if settings.app_env == "development":
    # 开发环境使用eventlet或gevent（Windows兼容）
    celery_app.conf.update(
        worker_pool="solo",  # Windows单进程模式
    )
