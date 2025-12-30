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
RAG_DIR = BASE_DIR / "rag"

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
# Embedding 模型配置
# ============================================
# BGE M3 模型资源消耗说明：
# - 模型大小：FP32约2.2GB，FP16约1.1GB
# - 参数量：560M（5.6亿参数）
# - 向量维度：1024维
#
# 显存占用估算（使用GPU时）：
# - batch_size=32, max_length=512: ~2.4-3.1GB 显存（需要4GB+显卡）
# - batch_size=8,  max_length=256: ~2.0GB 显存（适合4GB显卡，紧张）
# - batch_size=4,  max_length=128: ~1.5GB 显存（适合2GB显卡）
#
# 设备选择建议：
# - CPU: 稳定可靠，不占显存，速度较慢（5-10 docs/s），推荐用于离线索引构建
# - CUDA: 速度快（30-50 docs/s），但需要足够显存，推荐用于在线实时检索
#
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),  # BGE M3模型
    "model_cache_dir": str(BASE_DIR / "models" / "embeddings"),  # 模型缓存目录

    # 设备配置（根据你的硬件选择）
    "device": os.getenv("EMBEDDING_DEVICE", "cpu"),  # 运行设备：cpu 或 cuda

    # CPU配置（推荐，稳定）
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", 32)),  # CPU推荐：16-32
    "max_length": int(os.getenv("EMBEDDING_MAX_LENGTH", 512)),  # CPU推荐：512

    # 4GB显卡GPU配置（如需使用，取消注释并注释掉上面的配置）
    # "device": "cuda",
    # "batch_size": 8,   # 4GB显卡推荐：4-8
    # "max_length": 256,  # 4GB显卡推荐：256

    # 其他配置
    "normalize_embeddings": True,  # 是否归一化向量
}

# ============================================
# Chroma 向量数据库配置
# ============================================
CHROMA_CONFIG = {
    "persist_directory": str(BASE_DIR / "data" / "chroma"),  # 持久化存储目录
    "collection_name": os.getenv("CHROMA_COLLECTION", "crawl_results"),  # 集合名称
    "distance_metric": "cosine",  # 距离度量：cosine, l2, ip
    "anonymized_telemetry": False,  # 禁用匿名遥测
}

# ============================================
# RAG 服务配置
# ============================================
RAG_CONFIG = {
    "top_k": int(os.getenv("RAG_TOP_K", 5)),  # 检索Top-K文档
    "score_threshold": float(os.getenv("RAG_SCORE_THRESHOLD", 0.5)),  # 相似度阈值
    "max_content_length": int(os.getenv("RAG_MAX_CONTENT_LENGTH", 2000)),  # 返回内容最大长度
    "enable_rerank": False,  # 是否启用重排序
}

# ============================================
# RAG 模型预加载配置
# ============================================
# 是否在应用启动时预加载 RAG 模型
# - True（推荐）: 启动时加载模型，首次请求响应快，但启动时间长（1-3分钟）
# - False: 首次请求时加载，启动快但首次请求慢
PRELOAD_RAG_MODEL = os.getenv("PRELOAD_RAG_MODEL", "true").lower() == "true"

# ============================================
# 定时任务配置
# ============================================
SCHEDULER_CONFIG = {
    "crawler_interval_hours": int(os.getenv("CRAWLER_INTERVAL_HOURS", 24)),  # 爬虫执行间隔（小时）
    "rag_index_interval_hours": int(os.getenv("RAG_INDEX_INTERVAL_HOURS", 6)),  # RAG索引更新间隔（小时）
    "rag_index_cron": os.getenv("RAG_INDEX_CRON", None),  # Cron表达式（优先级高于interval）
    "timezone": os.getenv("SCHEDULER_TIMEZONE", "Asia/Shanghai"),  # 时区
}
