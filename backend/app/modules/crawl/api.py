"""
任务管理API路由
提供手动触发爬虫、查询任务状态等功能
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path
import logging

from app.database import get_db
from app.core.schemas.task import (
    TriggerTaskRequest,
    TriggerTaskResponse,
    TaskListResponse,
    TaskDetailResponse,
)
from .tasks import crawl_single_config, crawl_all_configs
from app.core.models.task import CrawlTask, TaskStatus
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/trigger", response_model=TriggerTaskResponse)
async def trigger_crawl_task(
    request: TriggerTaskRequest,
    db: Session = Depends(get_db)
):
    """
    立即触发爬虫任务

    - **config_path**: 配置文件路径（为空则执行所有配置）
    - **task_name**: 任务名称（可选，自动提取）

    返回Celery任务ID，可用于后续查询任务状态
    """
    try:
        # 如果没有指定config_path，执行所有配置
        if not request.config_path:
            logger.info("触发批量爬取任务（所有配置）")

            # 提交Celery任务
            celery_task = crawl_all_configs.delay(settings.crawler_config_dir)

            return TriggerTaskResponse(
                status="submitted",
                message="批量爬取任务已提交到队列",
                celery_task_id=celery_task.id,
                config_path=settings.crawler_config_dir
            )

        # 执行单个配置文件
        config_path = Path(request.config_path)

        # 如果是相对路径，基于配置目录解析
        if not config_path.is_absolute():
            config_dir = Path(settings.crawler_config_dir)
            config_path = config_dir / config_path

        # 检查配置文件是否存在
        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"配置文件不存在: {config_path}"
            )

        # 提取任务名称
        task_name = request.task_name or config_path.stem

        logger.info(f"触发单个爬取任务: {task_name} ({config_path})")

        # 提交Celery任务
        celery_task = crawl_single_config.delay(
            config_path=str(config_path),
            task_name=task_name
        )

        return TriggerTaskResponse(
            status="submitted",
            message=f"爬取任务 '{task_name}' 已提交到队列",
            celery_task_id=celery_task.id,
            config_path=str(config_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"触发任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-all", response_model=TriggerTaskResponse)
async def trigger_all_crawl_tasks(db: Session = Depends(get_db)):
    """
    一键触发所有爬虫任务

    自动读取 crawler_configs 目录下的所有 .yaml 配置文件，
    并逐一触发爬虫任务。

    不需要任何参数，发送一次请求即可爬取所有配置。
    """
    try:
        logger.info("触发一键爬取所有配置文件")

        # 提交Celery任务（批量爬取所有配置）
        celery_task = crawl_all_configs.delay(settings.crawler_config_dir)

        return TriggerTaskResponse(
            status="submitted",
            message="一键爬取任务已提交到队列，将爬取所有配置文件",
            celery_task_id=celery_task.id,
            config_path=settings.crawler_config_dir
        )

    except Exception as e:
        logger.error(f"触发一键爬取任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="按状态筛选"),
    limit: int = Query(10, ge=1, le=100, description="返回数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db)
):
    """
    查询任务列表

    - **status**: 按状态筛选（pending/running/completed/failed）
    - **limit**: 返回数量（1-100）
    - **offset**: 偏移量（分页）
    """
    try:
        # 构建查询
        query = db.query(CrawlTask)

        # 状态筛选
        if status:
            try:
                status_enum = TaskStatus(status)
                query = query.filter(CrawlTask.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的状态值: {status}"
                )

        # 总数
        total = query.count()

        # 分页查询
        tasks = query.order_by(CrawlTask.created_at.desc()).offset(offset).limit(limit).all()

        # 转换为字典
        task_list = [
            {
                "task_id": task.id,
                "task_name": task.task_name,
                "config_file": task.config_file,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "success_count": task.success_count,
                "duplicate_count": task.duplicate_count,
            }
            for task in tasks
        ]

        return TaskListResponse(total=total, tasks=task_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task_detail(
    task_id: int,
    db: Session = Depends(get_db)
):
    """
    查询任务详情

    - **task_id**: 任务ID
    """
    try:
        task = db.query(CrawlTask).filter(CrawlTask.id == task_id).first()

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        return TaskDetailResponse(
            task_id=task.id,
            task_name=task.task_name,
            config_file=task.config_file,
            status=task.status.value,
            celery_task_id=task.celery_task_id,
            created_at=task.created_at.isoformat(),
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            total_urls=task.total_urls,
            success_count=task.success_count,
            failed_count=task.failed_count,
            duplicate_count=task.duplicate_count,
            error_message=task.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs/list")
async def list_available_configs():
    """
    列出所有可用的配置文件

    返回配置目录下的所有YAML文件
    """
    try:
        config_dir = Path(settings.crawler_config_dir)

        if not config_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"配置目录不存在: {config_dir}"
            )

        # 查找所有YAML文件
        yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        configs = [
            {
                "name": f.stem,
                "path": f.name,  # 只返回文件名，不包含目录
                "absolute_path": str(f),
                "size": f.stat().st_size,
            }
            for f in yaml_files
        ]

        return {
            "total": len(configs),
            "config_dir": str(config_dir),
            "configs": configs
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出配置文件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
