"""
知识抽取API路由
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.database import get_db
from app.core.models.extraction import ExtractionRecord, ExtractionStatus
from .services.extraction_service import ExtractionService
from .tasks import (
    extract_single_result,
    retry_failed_extractions,
    batch_extract_all_pending_parallel
)

router = APIRouter()


# ========== Pydantic Models ==========

class ExtractionTriggerRequest(BaseModel):
    """触发抽取请求"""
    result_id: int


class ExtractionTriggerResponse(BaseModel):
    """触发抽取响应"""
    success: bool
    message: str
    extraction_record_id: Optional[int] = None
    celery_task_id: Optional[str] = None


class ExtractionRecordResponse(BaseModel):
    """抽取记录响应"""
    id: int
    result_id: int
    status: str
    triplet_count: int
    error_message: Optional[str]
    neo4j_imported: bool
    neo4j_error_message: Optional[str]
    extraction_started_at: Optional[datetime]
    extraction_completed_at: Optional[datetime]
    created_at: datetime
    retry_count: int
    neo4j_retry_count: int

    class Config:
        from_attributes = True


class ExtractionListResponse(BaseModel):
    """抽取记录列表响应"""
    total: int
    records: List[ExtractionRecordResponse]


# ========== API Endpoints ==========

@router.post("/trigger", response_model=ExtractionTriggerResponse)
def trigger_extraction(
    request: ExtractionTriggerRequest,
    db: Session = Depends(get_db)
):
    """
    触发单个爬取结果的知识抽取（异步）

    - **result_id**: 爬取结果ID
    """
    # 提交Celery异步任务
    task = extract_single_result.delay(request.result_id)

    return ExtractionTriggerResponse(
        success=True,
        message=f"抽取任务已提交",
        celery_task_id=task.id
    )


@router.post("/trigger-batch-all-parallel", response_model=ExtractionTriggerResponse)
def trigger_batch_all_parallel_extraction(
    max_workers: int = Query(default=10, ge=1, le=50, description="最大并行线程数"),
    db: Session = Depends(get_db)
):
    """
    批量并行抽取所有未抽取记录的知识抽取（异步，使用ThreadPoolExecutor并行处理）

    此接口会：
    1. 获取所有未进行信息抽取的爬取结果
    2. 使用ThreadPoolExecutor进行多线程并行处理
    3. 在日志中显示实时进度
    4. 与LLMRelationExtracter项目使用相同的并行方式

    - **max_workers**: 最大并行线程数（1-50），默认10
    """
    # 提交Celery异步任务
    task = batch_extract_all_pending_parallel.delay(max_workers=max_workers)

    return ExtractionTriggerResponse(
        success=True,
        message=f"并行批量抽取任务已提交（max_workers={max_workers}）",
        celery_task_id=task.id
    )


@router.get("/list", response_model=ExtractionListResponse)
def list_extraction_records(
    status: Optional[str] = Query(default=None, description="筛选状态: pending/running/success/failed"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """
    查询抽取记录列表

    - **status**: 筛选状态（可选）
    - **limit**: 每页数量
    - **offset**: 偏移量
    """
    query = db.query(ExtractionRecord)

    # 状态筛选
    if status:
        try:
            status_enum = ExtractionStatus(status)
            query = query.filter(ExtractionRecord.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的状态值: {status}")

    # 总数
    total = query.count()

    # 分页查询
    records = query.order_by(ExtractionRecord.created_at.desc()).offset(offset).limit(limit).all()

    return ExtractionListResponse(
        total=total,
        records=records
    )


@router.get("/{extraction_id}", response_model=ExtractionRecordResponse)
def get_extraction_record(
    extraction_id: int,
    db: Session = Depends(get_db)
):
    """
    查询单个抽取记录详情

    - **extraction_id**: 抽取记录ID
    """
    record = db.query(ExtractionRecord).filter(ExtractionRecord.id == extraction_id).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"抽取记录不存在: {extraction_id}")

    return record


@router.post("/retry-all-failed", response_model=ExtractionTriggerResponse)
def retry_all_failed(db: Session = Depends(get_db)):
    """
    重试所有失败的抽取记录（异步）
    """
    # 提交Celery任务
    task = retry_failed_extractions.delay()

    return ExtractionTriggerResponse(
        success=True,
        message="批量重试任务已提交",
        celery_task_id=task.id
    )


@router.get("/stats/summary")
def get_extraction_stats(db: Session = Depends(get_db)):
    """
    获取抽取统计信息
    """
    total = db.query(ExtractionRecord).count()
    pending = db.query(ExtractionRecord).filter(ExtractionRecord.status == ExtractionStatus.PENDING).count()
    running = db.query(ExtractionRecord).filter(ExtractionRecord.status == ExtractionStatus.RUNNING).count()
    success = db.query(ExtractionRecord).filter(ExtractionRecord.status == ExtractionStatus.SUCCESS).count()
    failed = db.query(ExtractionRecord).filter(ExtractionRecord.status == ExtractionStatus.FAILED).count()

    # Neo4j导入统计
    neo4j_imported = db.query(ExtractionRecord).filter(ExtractionRecord.neo4j_imported == True).count()
    neo4j_failed = db.query(ExtractionRecord).filter(
        ExtractionRecord.status == ExtractionStatus.SUCCESS,
        ExtractionRecord.neo4j_imported == False
    ).count()

    # 三元组总数
    from sqlalchemy import func
    total_triplets = db.query(func.sum(ExtractionRecord.triplet_count)).filter(
        ExtractionRecord.status == ExtractionStatus.SUCCESS
    ).scalar() or 0

    return {
        "total_records": total,
        "status_breakdown": {
            "pending": pending,
            "running": running,
            "success": success,
            "failed": failed
        },
        "neo4j_stats": {
            "imported": neo4j_imported,
            "failed": neo4j_failed
        },
        "total_triplets": total_triplets
    }
