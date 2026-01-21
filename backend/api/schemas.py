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


# ============================================
# RAG 相关模型
# ============================================

class RAGSearchRequest(BaseModel):
    """RAG 语义搜索请求"""
    query: str = Field(..., description="查询文本", example="大金空调的技术参数")
    top_k: Optional[int] = Field(default=None, description="返回结果数量", example=5)
    score_threshold: Optional[float] = Field(default=None, description="相似度阈值（0-1）", example=0.5)
    return_full_content: bool = Field(default=False, description="是否返回完整内容")


class RAGSearchResultItem(BaseModel):
    """RAG 搜索结果项"""
    id: int = Field(..., description="文档ID")
    url: str = Field(..., description="文档URL")
    title: Optional[str] = Field(None, description="文档标题")
    content: str = Field(..., description="文档内容（可能被截断）")
    score: float = Field(..., description="相似度分数（0-1）")
    crawled_at: Optional[str] = Field(None, description="爬取时间")
    full_content: Optional[str] = Field(None, description="完整内容（仅当 return_full_content=True）")
    raw_content: Optional[Dict[str, Any]] = Field(None, description="原始内容（仅当 return_full_content=True）")


class RAGSearchResponse(BaseModel):
    """RAG 搜索响应"""
    query: str = Field(..., description="查询文本")
    total: int = Field(..., description="返回结果数量")
    results: List[RAGSearchResultItem] = Field(..., description="搜索结果列表")


class RAGIndexBuildRequest(BaseModel):
    """RAG 索引构建请求"""
    batch_size: int = Field(default=100, description="批量处理大小", example=100)
    last_indexed_id: Optional[int] = Field(default=None, description="上次索引的最大ID（用于增量更新）")


class RAGIndexBuildResponse(BaseModel):
    """RAG 索引构建响应"""
    total_processed: int = Field(..., description="处理的文档数量")
    total_indexed: int = Field(..., description="成功索引的文档数量")
    last_id: int = Field(..., description="最后处理的文档ID")
    start_time: str = Field(..., description="开始时间")
    end_time: str = Field(..., description="结束时间")
    duration_seconds: float = Field(..., description="耗时（秒）")


class RAGIndexStatusResponse(BaseModel):
    """RAG 索引状态响应"""
    total_documents: int = Field(..., description="索引中的文档总数")
    vector_dimension: int = Field(..., description="向量维度")
    error: Optional[str] = Field(None, description="错误信息")


class RAGIndexClearResponse(BaseModel):
    """RAG 索引清空响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")

