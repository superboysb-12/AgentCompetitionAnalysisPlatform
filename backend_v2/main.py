"""
FastAPI 主程序
负责路由分发，通过 Redis Pub/Sub 通知子进程执行任务
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lifespan import lifespan
from api import crawl_router, rag_router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Agent Competition Analysis Platform",
    description="智能爬虫 + RAG 平台后端 API",
    version="2.0.0",
    lifespan=lifespan
)

# 配置 CORS（跨域资源共享）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(crawl_router)
app.include_router(rag_router)


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "Agent Competition Analysis Platform API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "crawl_api": "/api/crawl",
            "rag_api": "/api/rag"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    # 开发环境运行配置
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发时启用热重载
        log_level="info"
    )
