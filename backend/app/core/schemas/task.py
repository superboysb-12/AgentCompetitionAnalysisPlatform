"""
Pydantic Schemas - 任务相关
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TaskStatusEnum(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TriggerTaskRequest(BaseModel):
    """触发任务请求"""
    config_path: Optional[str] = Field(
        None,
        description="配置文件路径（相对或绝对），为空则执行所有配置"
    )
    task_name: Optional[str] = Field(
        None,
        description="任务名称，为空则自动从配置文件名提取"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "config_path": "../crawl/task_config/abi_indirect.yaml",
                "task_name": "abi_indirect"
            }
        }


class TriggerTaskResponse(BaseModel):
    """触发任务响应"""
    status: str = Field(..., description="状态：submitted/error")
    message: str = Field(..., description="响应消息")
    celery_task_id: Optional[str] = Field(None, description="Celery任务ID")
    task_id: Optional[int] = Field(None, description="数据库任务ID")
    config_path: Optional[str] = Field(None, description="配置文件路径")


class TaskStatistics(BaseModel):
    """任务统计信息"""
    task_id: int
    task_name: str
    status: str
    created_at: str
    completed_at: Optional[str]
    total_urls: int
    success_count: int
    duplicate_count: int
    unique_results: int


class TaskListResponse(BaseModel):
    """任务列表响应"""
    total: int
    tasks: list[Dict[str, Any]]


class TaskDetailResponse(BaseModel):
    """任务详情响应"""
    task_id: int
    task_name: str
    config_file: str
    status: str
    celery_task_id: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    total_urls: int
    success_count: int
    failed_count: int
    duplicate_count: int
    error_message: Optional[str]
