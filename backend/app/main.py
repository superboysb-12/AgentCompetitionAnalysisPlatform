"""
FastAPI主应用
预留Web API接口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.database import init_db
from app.scheduler import crawler_scheduler
from app.modules import module_loader  # 🆕 导入模块加载器

# 配置日志
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("应用启动...")

    # 初始化数据库表
    try:
        init_db()
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")

    # 🆕 加载并注册所有模块
    try:
        from celery_app import celery_app

        # 加载模块配置
        modules_config = settings.modules_config

        # 加载所有模块
        module_loader.load_all_modules(modules_config)

        # 注册模块路由和任务
        module_loader.register_all(app, celery_app)

        # 调用模块启动钩子
        module_loader.startup_all()

        logger.info(f"✓ 模块系统启动完成，已加载 {len(module_loader.modules)} 个模块")
    except Exception as e:
        logger.error(f"✗ 模块加载失败: {e}")
        raise

    # 启动调度器
    try:
        crawler_scheduler.start()
        logger.info("调度器启动完成")
    except Exception as e:
        logger.error(f"调度器启动失败: {e}")

    yield

    # 关闭时执行
    logger.info("应用关闭...")

    # 🆕 关闭所有模块
    try:
        module_loader.shutdown_all()
        logger.info("模块已关闭")
    except Exception as e:
        logger.error(f"模块关闭失败: {e}")

    # 关闭调度器
    try:
        crawler_scheduler.shutdown()
        logger.info("调度器已关闭")
    except Exception as e:
        logger.error(f"调度器关闭失败: {e}")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    description="智能爬虫竞争分析平台后端",
    version="0.1.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 根路径
@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "running",
        "app_name": settings.app_name,
        "version": "0.1.0",
        "scheduler_enabled": settings.scheduler_enabled,
        "scheduler_running": crawler_scheduler.is_running()
    }


@app.get("/health")
async def health_check():
    """详细健康检查"""
    return {
        "status": "healthy",
        "database": "connected",  # TODO: 实际检查数据库连接
        "scheduler": "running" if crawler_scheduler.is_running() else "stopped",
        "celery": "unknown",  # TODO: 检查Celery worker状态
        "modules": module_loader.list_modules()  # 🆕 显示已加载的模块
    }


# 🆕 模块系统API
@app.get("/modules")
async def list_modules():
    """列出所有已加载的模块"""
    return {
        "total": len(module_loader.modules),
        "modules": module_loader.list_modules()
    }


# ==================== 路由注册 ====================
# ⚠️ 所有路由通过模块系统自动注册
# 模块会在 lifespan 启动时通过 module_loader.register_all() 自动注册
# 如需添加新模块，请在 app/modules/ 下创建新目录，无需修改此文件


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
