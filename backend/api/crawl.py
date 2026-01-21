"""
爬虫相关 API 路由
提供任务提交、状态查询、结果列表等接口
"""

from fastapi import APIRouter, HTTPException, Query, Request
from pathlib import Path
from typing import List, Optional
import sys

from api.schemas import (
    TaskSubmitRequest,
    TaskSubmitResponse,
    TaskStatusResponse,
    CrawlResultResponse,
    CrawlResultListResponse,
    BatchTaskSubmitRequest,
    BatchTaskSubmitResponse,
    CrawlTaskResponse,
    CrawlTaskListResponse
)

router = APIRouter(prefix="/api/crawl", tags=["Crawl"])


@router.post("/task", response_model=TaskSubmitResponse)
async def submit_task(
    request: Request,
    task_request: TaskSubmitRequest
):
    """
    提交爬虫任务

    - 创建任务记录（Redis）
    - 发布任务消息到 crawler:task 频道
    - 返回任务 ID

    Args:
        request: FastAPI Request 对象（包含 app.state）
        task_request: 任务请求体

    Returns:
        TaskSubmitResponse: 任务提交响应
    """
    task_manager = request.app.state.task_manager
    redis_client = request.app.state.redis_client

    # 验证配置文件是否存在
    config_path = task_request.config_path
    if not Path(config_path).exists():
        raise HTTPException(status_code=400, detail=f"配置文件不存在: {config_path}")

    # 创建任务
    task_id = task_manager.create_task(
        task_type='crawl',
        config={'config_path': config_path}
    )

    # 发布任务到 Redis 频道
    message = {
        'task_id': task_id,
        'config_path': config_path
    }
    redis_client.publish('crawler:task', message)

    return TaskSubmitResponse(
        task_id=task_id,
        status='pending',
        message='任务已提交'
    )


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    request: Request,
    task_id: str
):
    """
    查询任务状态

    Args:
        request: FastAPI Request 对象
        task_id: 任务 ID

    Returns:
        TaskStatusResponse: 任务状态响应
    """
    task_manager = request.app.state.task_manager

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return TaskStatusResponse(**task)


@router.get("/results", response_model=CrawlResultListResponse)
async def get_results(
    limit: int = Query(default=10, ge=1, le=100, description="每页数量"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
    url: Optional[str] = Query(default=None, description="URL 过滤")
):
    """
    获取爬取结果列表（从 MySQL）

    Args:
        limit: 每页数量
        offset: 偏移量
        url: URL 过滤（可选）

    Returns:
        CrawlResultListResponse: 结果列表响应
    """
    # 导入 MySQL storage
    backend_dir = Path(__file__).resolve().parent.parent
    crawl_dir = backend_dir / "crawl"
    if str(crawl_dir) not in sys.path:
        sys.path.insert(0, str(crawl_dir))

    try:
        from mysql.storage import MySQLStorage
        from mysql.models import CrawlResultModel
        from settings import MYSQL_CONFIG
        from sqlalchemy import create_engine, select, func
        from sqlalchemy.orm import sessionmaker

        # 创建数据库连接
        connection_string = (
            f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
            f"?charset={MYSQL_CONFIG.get('charset', 'utf8mb4')}"
        )
        engine = create_engine(connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # 构建查询
            query = select(CrawlResultModel)

            if url:
                query = query.where(CrawlResultModel.url.like(f"%{url}%"))

            # 计算总数
            total_query = select(func.count()).select_from(query.subquery())
            total = session.execute(total_query).scalar()

            # 分页查询
            query = query.order_by(CrawlResultModel.created_at.desc())
            query = query.offset(offset).limit(limit)

            results = session.execute(query).scalars().all()

            # 转换为响应格式
            result_list = [
                CrawlResultResponse(
                    id=r.id,
                    url=r.url,
                    original_url=r.original_url,
                    title=r.title,
                    content=r.content,
                    raw_content=r.raw_content,
                    crawled_at=r.crawled_at.isoformat() if r.crawled_at else None,
                    created_at=r.created_at.isoformat() if r.created_at else None,
                    updated_at=r.updated_at.isoformat() if r.updated_at else None,
                    new_tab=r.new_tab,
                    strategy_used=r.strategy_used,
                    button_info=r.button_info
                )
                for r in results
            ]

            return CrawlResultListResponse(
                total=total,
                results=result_list
            )

        finally:
            session.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/batch-task", response_model=BatchTaskSubmitResponse)
async def submit_batch_task(
    request: Request,
    batch_request: BatchTaskSubmitRequest
):
    """
    批量提交爬虫任务

    扫描指定目录下的所有配置文件，批量提交任务

    Args:
        request: FastAPI Request 对象
        batch_request: 批量任务请求体

    Returns:
        BatchTaskSubmitResponse: 批量任务提交响应
    """
    import glob
    from pathlib import Path

    task_manager = request.app.state.task_manager
    redis_client = request.app.state.redis_client

    # 获取配置目录（支持相对路径和绝对路径）
    config_dir = Path(batch_request.config_dir)
    if not config_dir.is_absolute():
        # 相对路径，相对于 backend_v2 目录
        base_dir = Path(__file__).resolve().parent.parent
        config_dir = base_dir / batch_request.config_dir

    # 检查目录是否存在
    if not config_dir.exists():
        raise HTTPException(status_code=400, detail=f"配置目录不存在: {config_dir}")

    if not config_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"路径不是目录: {config_dir}")

    # 扫描配置文件
    pattern = batch_request.pattern
    config_files = list(config_dir.glob(pattern))

    if not config_files:
        raise HTTPException(
            status_code=404,
            detail=f"目录 {config_dir} 下未找到匹配 '{pattern}' 的配置文件"
        )

    # 批量提交任务
    task_ids = []
    configs = []

    for config_path in config_files:
        # 创建任务
        task_id = task_manager.create_task(
            task_type='crawl',
            config={'config_path': str(config_path)}
        )

        # 发布任务到 Redis 频道
        message = {
            'task_id': task_id,
            'config_path': str(config_path)
        }
        redis_client.publish('crawler:task', message)

        task_ids.append(task_id)
        configs.append(config_path.name)

    return BatchTaskSubmitResponse(
        total=len(task_ids),
        task_ids=task_ids,
        configs=configs,
        message=f"成功提交 {len(task_ids)} 个爬虫任务"
    )


@router.post("/crawl-all", response_model=BatchTaskSubmitResponse)
async def crawl_all(request: Request):
    """
    一键爬取所有配置文件

    使用 config.py 中配置的默认目录和匹配模式，自动扫描并提交所有爬虫任务

    Returns:
        BatchTaskSubmitResponse: 批量任务提交响应
    """
    from pathlib import Path
    from settings import CRAWLER_CONFIG

    task_manager = request.app.state.task_manager
    redis_client = request.app.state.redis_client

    # 从配置中获取目录和模式
    config_dir_str = CRAWLER_CONFIG.get("task_config_dir", "crawl/task_config")
    pattern = CRAWLER_CONFIG.get("config_pattern", "*.yaml")

    # 获取配置目录（相对路径，相对于 backend_v2）
    base_dir = Path(__file__).resolve().parent.parent
    config_dir = base_dir / config_dir_str

    # 检查目录是否存在
    if not config_dir.exists():
        raise HTTPException(status_code=400, detail=f"配置目录不存在: {config_dir}")

    if not config_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"路径不是目录: {config_dir}")

    # 扫描配置文件
    config_files = list(config_dir.glob(pattern))

    if not config_files:
        raise HTTPException(
            status_code=404,
            detail=f"目录 {config_dir} 下未找到匹配 '{pattern}' 的配置文件"
        )

    # 批量提交任务
    task_ids = []
    configs = []

    for config_path in config_files:
        # 转换为相对路径（相对于 backend_v2）
        try:
            relative_path = config_path.relative_to(base_dir)
        except ValueError:
            # 如果无法转换为相对路径，使用绝对路径
            relative_path = config_path

        # 创建任务
        task_id = task_manager.create_task(
            task_type='crawl',
            config={'config_path': str(relative_path)}
        )

        # 发布任务到 Redis 频道
        message = {
            'task_id': task_id,
            'config_path': str(relative_path)
        }
        redis_client.publish('crawler:task', message)

        task_ids.append(task_id)
        configs.append(config_path.name)

    return BatchTaskSubmitResponse(
        total=len(task_ids),
        task_ids=task_ids,
        configs=configs,
        message=f"成功提交 {len(task_ids)} 个爬虫任务（来自 {config_dir_str}）"
    )


@router.get("/tasks", response_model=CrawlTaskListResponse)
async def get_task_history(
    limit: int = Query(default=20, ge=1, le=100, description="每页数量"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
    status: Optional[str] = Query(default=None, description="状态过滤（pending/running/completed/failed）"),
    config_name: Optional[str] = Query(default=None, description="配置文件名过滤")
):
    """
    获取爬取任务历史记录

    可以查看所有任务的执行状态、耗时、错误信息等
    方便排查问题配置文件

    Args:
        limit: 每页数量
        offset: 偏移量
        status: 状态过滤
        config_name: 配置文件名过滤

    Returns:
        CrawlTaskListResponse: 任务历史列表
    """
    # 导入 MySQL storage
    backend_dir = Path(__file__).resolve().parent.parent
    crawl_dir = backend_dir / "crawl"
    if str(crawl_dir) not in sys.path:
        sys.path.insert(0, str(crawl_dir))

    try:
        # 直接导入 models 避免触发 __init__.py
        import mysql.models as models
        CrawlTaskModel = models.CrawlTaskModel

        from settings import MYSQL_CONFIG
        from sqlalchemy import create_engine, select, func
        from sqlalchemy.orm import sessionmaker

        # 创建数据库连接
        connection_string = (
            f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
            f"?charset={MYSQL_CONFIG.get('charset', 'utf8mb4')}"
        )
        engine = create_engine(connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # 构建查询
            query = select(CrawlTaskModel)

            if status:
                query = query.where(CrawlTaskModel.status == status)

            if config_name:
                query = query.where(CrawlTaskModel.config_name.like(f"%{config_name}%"))

            # 计算总数
            total_query = select(func.count()).select_from(query.subquery())
            total = session.execute(total_query).scalar()

            # 分页查询（按创建时间倒序）
            query = query.order_by(CrawlTaskModel.created_at.desc())
            query = query.offset(offset).limit(limit)

            results = session.execute(query).scalars().all()

            # 转换为响应格式
            task_list = [
                CrawlTaskResponse(
                    id=r.id,
                    task_id=r.task_id,
                    config_path=r.config_path,
                    config_name=r.config_name,
                    status=r.status,
                    started_at=r.started_at.isoformat() if r.started_at else None,
                    completed_at=r.completed_at.isoformat() if r.completed_at else None,
                    duration_seconds=r.duration_seconds,
                    results_count=r.results_count or 0,
                    error_message=r.error_message,
                    created_at=r.created_at.isoformat() if r.created_at else None,
                )
                for r in results
            ]

            return CrawlTaskListResponse(
                total=total,
                tasks=task_list
            )

        finally:
            session.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
