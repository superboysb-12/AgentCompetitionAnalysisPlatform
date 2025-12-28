"""
后端系统配置文件
统一管理 MySQL、Redis 等配置信息
"""

import os
from pathlib import Path

# ============================================
# 项目路径配置
# ============================================
BASE_DIR = Path(__file__).resolve().parent
CRAWL_DIR = BASE_DIR / "crawl"

# ============================================
# MySQL 数据库配置
# ============================================
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "database": os.getenv("MYSQL_DATABASE", "agent_competition_db"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "869589"),
    "charset": "utf8mb4",
    # 连接池配置
    "pool_size": 5,
    "max_overflow": 10,
    "pool_pre_ping": True,  # 连接池预ping，确保连接有效
    "pool_recycle": 3600,   # 连接回收时间（秒）
    "echo": False  # 生产环境设为 False，开发时可设为 True 查看 SQL
}

# SQLAlchemy 连接字符串
MYSQL_URL = (
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
    f"?charset={MYSQL_CONFIG['charset']}"
)

# ============================================
# Redis 配置
# ============================================
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "password": os.getenv("REDIS_PASSWORD", None),
    "decode_responses": False,  # 索引数据是二进制，不解码
}

# ============================================
# 爬虫配置
# ============================================
CRAWLER_CONFIG = {
    "storage_type": "mysql",  # 存储类型：mysql 或 json
    "output_dir": str(BASE_DIR / "results"),  # JSON 备份目录
    "task_config_dir": "crawl/task_config",  # 任务配置文件目录（相对于 BASE_DIR）
    "config_pattern": "*.yaml",  # 配置文件匹配模式
}

# ============================================
# 向量索引配置
# ============================================
INDEX_CONFIG = {
    "index_type": "faiss",  # faiss 或 annoy
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "dimension": 384,  # 向量维度
}

# ============================================
# RAG 服务配置
# ============================================
RAG_CONFIG = {
    "port": int(os.getenv("RAG_PORT", 8000)),
    "top_k": 5,  # 检索Top-K文档
    "llm_provider": "openai",  # openai 或 anthropic
    "llm_model": "gpt-4",
}

# ============================================
# 定时任务配置
# ============================================
SCHEDULER_CONFIG = {
    "crawler_interval_hours": 24,  # 爬虫执行间隔（小时）
    "index_interval_hours": 6,     # 索引更新间隔（小时）
}
