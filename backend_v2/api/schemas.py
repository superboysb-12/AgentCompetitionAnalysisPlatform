"""
Pydantic 模型定义
定义 API 请求和响应的数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class TaskSubmitRequest(BaseModel):
    """任务提交请求"""
    config_path: str = Field(..., description="爬虫配置文件路径", example="crawl/task_config/daikin_direct.yaml")


class TaskSubmitResponse(BaseModel):
    """任务提交响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(default="任务已提交", description="响应消息")


class TaskStatusResponse(BaseModel):
    """任务状态查询响应"""
    task_id: str
    task_type: str
    status: str
    config: str
    created_at: str
    updated_at: str
    error: Optional[str] = None


class CrawlResultResponse(BaseModel):
    """爬取结果响应"""
    id: int
    url: str
    original_url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    raw_content: Optional[Dict[str, Any]] = None
    crawled_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    new_tab: Optional[bool] = None
    strategy_used: Optional[str] = None
    button_info: Optional[Dict[str, Any]] = None


class CrawlResultListResponse(BaseModel):
    """爬取结果列表响应"""
    total: int = Field(..., description="总数")
    results: List[CrawlResultResponse] = Field(..., description="结果列表")


class BatchTaskSubmitRequest(BaseModel):
    """批量任务提交请求"""
    config_dir: str = Field(
        default="crawl/task_config",
        description="配置文件目录路径",
        example="crawl/task_config"
    )
    pattern: str = Field(
        default="*.yaml",
        description="文件匹配模式",
        example="*.yaml"
    )


class BatchTaskSubmitResponse(BaseModel):
    """批量任务提交响应"""
    total: int = Field(..., description="提交的任务总数")
    task_ids: List[str] = Field(..., description="任务ID列表")
    configs: List[str] = Field(..., description="配置文件列表")
    message: str = Field(default="批量任务已提交", description="响应消息")


class CrawlTaskResponse(BaseModel):
    """爬取任务记录响应"""
    id: int
    task_id: str
    config_path: str
    config_name: Optional[str] = None
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    results_count: int = 0
    error_message: Optional[str] = None
    created_at: Optional[str] = None


class CrawlTaskListResponse(BaseModel):
    """爬取任务列表响应"""
    total: int = Field(..., description="总数")
    tasks: List[CrawlTaskResponse] = Field(..., description="任务列表")
