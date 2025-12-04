"""
配置管理模块 - 统一配置文件
所有配置都在这个文件中管理，无需额外的 .env 文件
"""
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置 - 统一配置中心

    修改配置方法:
    1. 开发环境: 直接修改此文件中的默认值
    2. 生产环境: 通过环境变量覆盖 (可选)

    环境变量命名规则:
    - 使用大写: DATABASE_URL, REDIS_URL 等
    - 自动映射到对应的配置项
    """

    # ==================== Application ====================
    app_name: str = "AgentCompetitionAnalysisPlatform"
    app_env: str = "development"  # development | production
    debug: bool = True

    # ==================== Database ====================
    database_url: str = Field(
        default="mysql+pymysql://root:869589@localhost:3306/crawler_db",
        description="MySQL数据库连接URL - 修改为你的实际数据库密码"
    )
    db_echo: bool = False  # 是否打印SQL语句

    # ==================== Redis & Celery ====================
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ==================== Crawler ====================
    crawler_config_dir: str = "crawler_configs"  # 爬虫配置文件目录
    crawler_headless: bool = True  # 无头模式运行浏览器
    crawler_max_concurrent: int = 3  # 最大并发爬虫数

    # ==================== Scheduler ====================
    scheduler_timezone: str = "Asia/Shanghai"
    scheduler_enabled: bool = True
    weekly_crawl_cron: str = "0 0 * * 0"  # 每周日午夜 (Cron表达式: 分 时 日 月 周)

    # ==================== Logging ====================
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_dir: str = "logs"

    # ==================== API ====================
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:3000"]

    # ==================== Knowledge Graph Extraction ====================
    kg_config_path: str = "kg_config/config.yaml"  # LLM抽取器配置文件路径
    kg_max_retries: int = 3  # 抽取失败最大重试次数
    kg_batch_size: int = 10  # 批量抽取每批处理数量
    kg_max_workers: int = 10  # 并行抽取最大线程数（用于batch_extract_all_pending_parallel）

    # ==================== Neo4j ====================
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "gjl869589"  # 修改为你的Neo4j密码
    neo4j_database: str = "knowledge"  # Neo4j 4.x+支持多数据库
    neo4j_max_retries: int = 3  # Neo4j导入失败最大重试次数

    # ==================== 模块系统配置 ====================
    modules_config: dict = {
        "crawl": {
            "enabled": True,
            "config_dir": "app/modules/crawl/config",  # 模块内配置目录
            "max_concurrent_tasks": 5,
            "description": "网页爬取模块"
        },
        "extraction": {
            "enabled": True,
            "kg_config_path": "app/modules/extraction/config/config.yaml",  # 模块内配置
            "max_workers": 10,
            "max_retries": 3,
            "neo4j_enabled": True,
            "enable_scheduled_extraction": True,  # 是否启用定时抽取
            "description": "知识图谱抽取模块"
        }
    }

    @property
    def is_production(self) -> bool:
        """判断是否生产环境"""
        return self.app_env.lower() == "production"


# ==================== 全局配置实例 ====================
# 使用方法: from app.config import settings
settings = Settings()
