"""
RAG API 路由
提供语义搜索和索引管理功能
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from api.schemas import (
    RAGSearchRequest,
    RAGSearchResponse,
    RAGSearchResultItem,
    RAGIndexBuildRequest,
    RAGIndexBuildResponse,
    RAGIndexStatusResponse,
    RAGIndexClearResponse,
)
from services import RAGService
from settings import MYSQL_CONFIG, EMBEDDING_CONFIG, CHROMA_CONFIG, RAG_CONFIG

logger = logging.getLogger(__name__)

# 创建路由
rag_router = APIRouter(prefix="/api/rag", tags=["RAG"])

# 全局 RAG 服务实例（懒加载）
_rag_service: RAGService = None


def get_rag_service() -> RAGService:
    """获取 RAG 服务实例（单例模式）"""
    global _rag_service
    if _rag_service is None:
        logger.info("初始化 RAG 服务...")
        _rag_service = RAGService(
            mysql_config=MYSQL_CONFIG,
            embedding_config=EMBEDDING_CONFIG,
            chroma_config=CHROMA_CONFIG,
            rag_config=RAG_CONFIG,
        )
        logger.info("✓ RAG 服务初始化完成")
    return _rag_service


@rag_router.post("/search", response_model=RAGSearchResponse, summary="语义搜索")
async def search(request: RAGSearchRequest):
    """
    语义搜索接口

    根据查询文本，返回最相关的 Top-K 文档

    **示例请求**:
    ```json
    {
        "query": "大金空调的技术参数",
        "top_k": 5,
        "score_threshold": 0.5,
        "return_full_content": false
    }
    ```

    **返回**:
    - 相关文档列表，按相似度分数降序排列
    """
    try:
        logger.info(f"收到搜索请求: {request.query[:100]}...")

        # 获取 RAG 服务
        rag_service = get_rag_service()

        # 执行搜索
        results = rag_service.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            return_full_content=request.return_full_content,
        )

        # 转换为响应格式
        result_items = [RAGSearchResultItem(**result) for result in results]

        response = RAGSearchResponse(
            query=request.query,
            total=len(result_items),
            results=result_items,
        )

        logger.info(f"✓ 搜索完成，返回 {len(result_items)} 个结果")
        return response

    except Exception as e:
        logger.error(f"✗ 搜索失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@rag_router.get("/index/status", response_model=RAGIndexStatusResponse, summary="查询索引状态")
async def get_index_status():
    """
    查询索引状态

    返回向量数据库中的文档总数和向量维度等信息

    **返回**:
    - 索引状态信息
    """
    try:
        logger.info("查询索引状态...")

        # 获取 RAG 服务
        rag_service = get_rag_service()

        # 获取状态
        status = rag_service.get_index_status()

        logger.info(f"✓ 索引状态: {status}")
        return RAGIndexStatusResponse(**status)

    except Exception as e:
        logger.error(f"✗ 查询索引状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@rag_router.post("/index/build", response_model=RAGIndexBuildResponse, summary="手动构建索引")
async def build_index(request: RAGIndexBuildRequest, bg_request: Request):
    """
    手动触发索引构建

    从 MySQL 读取爬虫结果，生成向量并存储到向量数据库

    **注意**:
    - 这是一个同步操作，可能需要较长时间
    - 对于大量数据，建议使用定时任务自动构建

    **示例请求**:
    ```json
    {
        "batch_size": 100,
        "last_indexed_id": 1000
    }
    ```

    **返回**:
    - 构建结果统计
    """
    try:
        logger.info(f"手动触发索引构建: batch_size={request.batch_size}, last_indexed_id={request.last_indexed_id}")

        # 获取 RAG 服务
        rag_service = get_rag_service()

        # 执行构建
        result = rag_service.build_index(
            batch_size=request.batch_size,
            last_indexed_id=request.last_indexed_id,
        )

        logger.info(f"✓ 索引构建完成: {result}")
        return RAGIndexBuildResponse(**result)

    except Exception as e:
        logger.error(f"✗ 索引构建失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")


@rag_router.delete("/index/clear", response_model=RAGIndexClearResponse, summary="清空索引")
async def clear_index():
    """
    清空索引

    删除向量数据库中的所有文档

    **警告**:
    - 此操作不可逆，请谨慎使用
    - 清空后需要重新构建索引

    **返回**:
    - 操作结果
    """
    try:
        logger.warning("⚠ 收到清空索引请求")

        # 获取 RAG 服务
        rag_service = get_rag_service()

        # 执行清空
        success = rag_service.clear_index()

        if success:
            message = "索引已清空"
            logger.info(f"✓ {message}")
        else:
            message = "清空索引失败"
            logger.error(f"✗ {message}")

        return RAGIndexClearResponse(
            success=success,
            message=message,
        )

    except Exception as e:
        logger.error(f"✗ 清空索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")


@rag_router.get("/", summary="RAG 服务信息")
async def rag_info():
    """
    RAG 服务信息

    返回 RAG 服务的基本信息和配置

    **返回**:
    - 服务信息
    """
    # 获取模型加载状态
    from rag import RAGSingletonManager
    status = RAGSingletonManager.get_status()

    return {
        "service": "RAG (Retrieval-Augmented Generation)",
        "version": "1.0.0",
        "description": "基于 BGE M3 + Chroma 的语义搜索服务",
        "endpoints": {
            "search": "/api/rag/search",
            "index_status": "/api/rag/index/status",
            "build_index": "/api/rag/index/build",
            "clear_index": "/api/rag/index/clear",
            "model_status": "/api/rag/model/status",
        },
        "config": {
            "embedding_model": EMBEDDING_CONFIG['model_name'],
            "vector_store": "Chroma",
            "top_k": RAG_CONFIG['top_k'],
            "score_threshold": RAG_CONFIG['score_threshold'],
        },
        "model_status": status
    }


@rag_router.get("/model/status", summary="查询模型加载状态")
async def get_model_status():
    """
    查询模型加载状态

    返回 Embedding 模型和向量数据库的加载状态

    **返回**:
    - 模型加载状态信息
    """
    from rag import RAGSingletonManager
    status = RAGSingletonManager.get_status()

    return {
        "status": "success",
        "data": status,
        "message": "模型已加载" if status['embeddings_loaded'] else "模型未加载"
    }
