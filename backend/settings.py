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

# ============================================
# Relation extractor config
# ============================================
RELATION_EXTRACTOR_CONFIG = {
    # OpenAI config
    "api_key": os.getenv(
        "OPENAI_API_KEY",
        "sk-zk27dc5e6f2447c59111f33391ffa21ff6368b7feced45a0",
    ),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1/"),
    "model": os.getenv("RELATION_MODEL", "gpt-4.1-mini"),

    # Concurrency control
    "global_concurrency": int(os.getenv("GLOBAL_CONCURRENCY", os.getenv("MAX_CONCURRENT", "100"))),
    "max_concurrent": int(os.getenv("MAX_CONCURRENT", "100")),
    # Process-wide in-flight LLM call cap shared by all request paths.
    "llm_global_concurrency": int(
        os.getenv("LLM_GLOBAL_CONCURRENCY", 10)
    ),
    # Global request pacing in requests-per-minute.
    # <=0 means fallback to llm_global_concurrency as RPM for compatibility.
    "llm_global_rpm": float(os.getenv("LLM_GLOBAL_RPM", 0)),
    "timeout": int(os.getenv("REQUEST_TIMEOUT", 3000)),  # ??????(?),??3??
    # httpx connection pool controls for ChatOpenAI transport.
    "llm_http_max_connections": int(os.getenv("LLM_HTTP_MAX_CONNECTIONS", 200)),
    "llm_http_max_keepalive_connections": int(
        os.getenv("LLM_HTTP_MAX_KEEPALIVE_CONNECTIONS", 100)
    ),
    "llm_http_keepalive_expiry": float(os.getenv("LLM_HTTP_KEEPALIVE_EXPIRY", 30.0)),
    # Hard timeout guard for each single llm.ainvoke call (seconds).
    "llm_call_hard_timeout": float(os.getenv("LLM_CALL_HARD_TIMEOUT", 180.0)),
    # Socket recycle strategy for unstable network / pooled connections.
    "llm_socket_recycle_on_connection_error": os.getenv(
        "LLM_SOCKET_RECYCLE_ON_CONNECTION_ERROR",
        "true",
    ).lower()
    == "true",
    "llm_socket_recycle_after_calls": int(os.getenv("LLM_SOCKET_RECYCLE_AFTER_CALLS", 0)),
    "llm_socket_recycle_min_interval": float(
        os.getenv("LLM_SOCKET_RECYCLE_MIN_INTERVAL", 5.0)
    ),
    "llm_socket_recycle_defer_close_seconds": float(
        os.getenv("LLM_SOCKET_RECYCLE_DEFER_CLOSE_SECONDS", 120.0)
    ),
    "llm_retry_log_path": os.getenv("LLM_RETRY_LOG_PATH", "logs/llm_retry_events.jsonl"),

    # Retry config (basic finite exponential backoff)
    "max_retries": int(os.getenv("MAX_RETRIES", 8)),
    # Exponential backoff: delay = retry_delay * (retry_backoff_factor ** attempt)
    "retry_delay": float(os.getenv("RETRY_DELAY", 2.0)),
    "retry_backoff_factor": float(os.getenv("RETRY_BACKOFF_FACTOR", 2.5)),
    "retry_max_delay": float(os.getenv("RETRY_MAX_DELAY", 120.0)),

    # CSV handling
    "ignored_types": ["image", "discarded_header", "discarded_footer"],

    # Logging
    "log_file": str(BASE_DIR / "logs" / "relation_extractor.log"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),

    # Stage A/B/C/D parameters
    "brand_cluster_size": int(os.getenv("BRAND_CLUSTER_SIZE", 5)),
    "brand_cluster_mode": os.getenv("BRAND_CLUSTER_MODE", "fixed"),  # fixed | embed
    "brand_cluster_embed_model": os.getenv(
        "BRAND_CLUSTER_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "brand_cluster_threshold": float(os.getenv("BRAND_CLUSTER_THRESHOLD", 0.25)),
    "brand_cluster_device": os.getenv("BRAND_CLUSTER_DEVICE", "cpu"),
    "brand_refine_embed_model": os.getenv(
        "BRAND_REFINE_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "brand_refine_threshold": float(os.getenv("BRAND_REFINE_THRESHOLD", 0.35)),
    "brand_refine_device": os.getenv("BRAND_REFINE_DEVICE", "cpu"),
    "brand_refine_min_count": int(os.getenv("BRAND_REFINE_MIN_COUNT", 3)),
    # Disabled by default: prefer LLM-based filtering/selection over heuristic pre-prune.
    "brand_use_pre_prune": os.getenv("BRAND_USE_PRE_PRUNE", "false").lower() == "true",
    # Single brochure usually maps to one manufacturer brand.
    "single_brand_per_document": os.getenv("SINGLE_BRAND_PER_DOCUMENT", "true").lower() == "true",
    "brand_translate_embed_model": os.getenv(
        "BRAND_TRANSLATE_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "brand_translate_threshold": float(os.getenv("BRAND_TRANSLATE_THRESHOLD", 0.25)),
    "brand_translate_device": os.getenv("BRAND_TRANSLATE_DEVICE", "cpu"),
    # Stage-level fan-out concurrency
    "series_page_concurrency": int(os.getenv("SERIES_PAGE_CONCURRENCY", 12)),
    "series_cluster_concurrency": int(os.getenv("SERIES_CLUSTER_CONCURRENCY", 6)),
    "model_pair_concurrency": int(os.getenv("MODEL_PAIR_CONCURRENCY", 6)),
    "model_chunk_concurrency": int(os.getenv("MODEL_CHUNK_CONCURRENCY", 8)),
    "product_pair_concurrency": int(os.getenv("PRODUCT_PAIR_CONCURRENCY", 6)),
    "product_chunk_concurrency": int(os.getenv("PRODUCT_CHUNK_CONCURRENCY", 8)),
    "product_model_concurrency": int(os.getenv("PRODUCT_MODEL_CONCURRENCY", 6)),
    "series_keyword_boost": [
        k
        for k in os.getenv(
            "SERIES_KEYWORD_BOOST",
            "SDB,SDC+,SDC,SDE,SDZ,SDTD",
        ).split(",")
        if k
    ],
    # 默认不启用硬编码过滤；如需再加启发式可通过环境变量覆盖
    "series_drop_keywords": [
        k
        for k in os.getenv(
            "SERIES_DROP_KEYWORDS",
            "",
        ).split(",")
        if k
    ],

    # Prompt length guard
    "max_chars_per_call": int(os.getenv("MAX_CHARS_PER_CALL", 8000)),
    # 0 means auto = 75% of max_chars_per_call.
    "llm_list_batch_max_chars": int(os.getenv("LLM_LIST_BATCH_MAX_CHARS", 0)),
    "chunk_progress_preview_chars": int(os.getenv("CHUNK_PROGRESS_PREVIEW_CHARS", 64)),
    "table_rows_per_block": int(os.getenv("TABLE_ROWS_PER_BLOCK", 12)),
    "table_row_cell_clip": int(os.getenv("TABLE_ROW_CELL_CLIP", 0)),
    "table_rows_clip_chars": int(os.getenv("TABLE_ROWS_CLIP_CHARS", 4000)),
    "rows_json_content_clip": int(os.getenv("ROWS_JSON_CONTENT_CLIP", 2400)),
    "target_model_table_context_lines": int(os.getenv("TARGET_MODEL_TABLE_CONTEXT_LINES", 0)),

    # Retrieval (series/model stages): keyword-only evidence recall in v2.
    "retrieval_method": os.getenv("RETRIEVAL_METHOD", "keyword"),
    "retrieval_top_k": int(os.getenv("RETRIEVAL_TOP_K", 0)),  # 0 = keep all matched pages
    "keyword_retrieval_top_k": int(
        os.getenv("KEYWORD_RETRIEVAL_TOP_K", os.getenv("RETRIEVAL_TOP_K", "0"))
    ),
    "keyword_retrieval_min_hits": int(os.getenv("KEYWORD_RETRIEVAL_MIN_HITS", 1)),
    # any | all
    "keyword_retrieval_match_mode": os.getenv("KEYWORD_RETRIEVAL_MATCH_MODE", "any"),

    # Series extraction context (Stage B)
    "series_from_brand_evidence": os.getenv("SERIES_FROM_BRAND_EVIDENCE", "true").lower() == "true",
    # For each Stage-A brand evidence page, include this many following pages.
    "series_brand_follow_pages": int(os.getenv("SERIES_BRAND_FOLLOW_PAGES", 1)),
    # Stage-B page grouping size for each LLM call; extractor coerces to 2 or 3.
    "series_brand_chunk_pages": int(os.getenv("SERIES_BRAND_CHUNK_PAGES", 2)),
    "series_feature_series_batch_size": int(os.getenv("SERIES_FEATURE_SERIES_BATCH_SIZE", 8)),
    "series_review_batch_size": int(os.getenv("SERIES_REVIEW_BATCH_SIZE", 64)),
    "series_canon_batch_size": int(os.getenv("SERIES_CANON_BATCH_SIZE", 64)),
    "brand_refine_batch_size": int(os.getenv("BRAND_REFINE_BATCH_SIZE", 20)),
    "brand_primary_batch_size": int(os.getenv("BRAND_PRIMARY_BATCH_SIZE", 20)),
    "brand_canon_batch_size": int(os.getenv("BRAND_CANON_BATCH_SIZE", 32)),
    # Legacy field kept for compatibility with older scripts.
    "series_page_chunk_size": int(os.getenv("SERIES_PAGE_CHUNK_SIZE", 4)),
    "model_redirect_min_conf": float(os.getenv("MODEL_REDIRECT_MIN_CONF", 0.5)),
    "model_review_max_items": int(os.getenv("MODEL_REVIEW_MAX_ITEMS", 80)),
    # Model context (Stage C): series pages + next N pages
    "series_context_follow_pages": int(os.getenv("SERIES_CONTEXT_FOLLOW_PAGES", 2)),
    # Product context: model pages + next N pages
    "model_context_follow_pages": int(os.getenv("MODEL_CONTEXT_FOLLOW_PAGES", 2)),
    "product_model_multi_match_cap": int(os.getenv("PRODUCT_MODEL_MULTI_MATCH_CAP", 12)),
    "product_review_max_items": int(os.getenv("PRODUCT_REVIEW_MAX_ITEMS", 50)),
    "single_value_performance_specs": (
        os.getenv(
            "SINGLE_VALUE_PERFORMANCE_SPECS",
            "电源,电压,电源电压,制冷量,制热量,额定功率,额定电流,风量,噪音,重量,尺寸,COP,EER,IPLV,SEER,APF,HSPF",
        ).split(",")
    ),
    "series_merge_embed_model": os.getenv(
        "SERIES_MERGE_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "series_merge_threshold": float(os.getenv("SERIES_MERGE_THRESHOLD", 0.2)),
    "series_merge_device": os.getenv("SERIES_MERGE_DEVICE", "cpu"),

    # Stage caching/checkpoints
    "force_stage_rerun": os.getenv("FORCE_STAGE_RERUN", "false").lower() == "true",

    # Prompts
    "system_prompt": (
        "你是一名产品信息抽取专家。请严格遵循以下规则："
        "\n\n"
        "**1) 产品区分规则**"
        "\n"
        "文本可能包含多个产品；以 product_model（必要时结合 brand/series）区分产品，每个独立产品输出 results 数组中的一条，不得将不同产品信息合并。"
        "\n"
        "若表格存在多行型号，每一行型号都要独立输出一条 result；共享信息（brand/category/series/features/key_components 等）可复制到每条中，但性能参数仅限该型号所在行。"
        "\n\n"
        "**2) 证据原则**"
        "\n"
        "仅提取原文中有明确证据的字段，无证据的字段填空字符串或空数组，禁止猜测、编造或填默认值。"
        "\n\n"
        "**3) 标准字段说明**"
        "\n"
        "- brand: 品牌名称（如：美的、格力、大金）"
        "\n"
        "- category: 产品类别（如：室内机、室外机、空调、热泵、新风系统）"
        "\n"
        "- series: 产品系列（如：真暖系列、舒适家系列）"
        "\n"
        "- product_model: 产品型号（如：MDV-D15Q4、MHSR18ON8-S1）"
        "\n"
        "- manufacturer: 制造商全称"
        "\n"
        "- refrigerant: 制冷剂类型（如：R32、R410A）"
        "\n"
        "- energy_efficiency_grade: 能效等级（如：一级能效、APF 7.3）"
        "\n"
        "- features[]: **产品功能特点数组（重要！）**"
        "\n"
        "- key_components[]: 关键部件数组"
        "\n\n"
        "**特别说明 - category字段识别规则（重要）：**"
        "\n"
        "category字段用于区分产品类型，特别是区分室内机和室外机。请按以下优先级判断："
        "\n\n"
        "**优先级1：明确标注**"
        "\n"
        "  - 如果页面标题或产品名称明确标注'室内机'、'室外机'、'内机'、'外机'，直接使用"
        "\n"
        "  - 示例：'室内机产品介绍' → category: '室内机'"
        "\n\n"
        "**优先级2：根据产品系列名称和关键字判断**"
        "\n"
        "  - 室内机关键字（满足任一即为室内机）："
        "\n"
        "    风管、室内、吊顶、嵌入、落地、壁挂、挂壁、新风、全热、除湿、内机、厨房、管道、吊落"
        "\n"
        "  - 室外机关键字（满足任一即为室外机）："
        "\n"
        "    室外机、外机、主机、压缩机、冷凝器、室外、户外"
        "\n"
        "  - 判断方法：检查 product_model 或 series 字段是否包含上述关键字"
        "\n"
        "  - 示例："
        "\n"
        "    - series='风管式室内机' → category: '室内机'"
        "\n"
        "    - product_model='MDV-D280W/DSN1-8X0(B1)' + series='风管机' → category: '室内机'"
        "\n"
        "    - series='多联机组' + 提到'压缩机' → category: '室外机'"
        "\n\n"
        "**优先级3：根据文档上下文判断**"
        "\n"
        "  - 如果文档开头部分说明了主要产品类型（如：'本文档介绍XX系列室外机'），则该系列产品归为该类型"
        "\n"
        "  - 外机产品文档中提到的适配内机，应标注为'室内机'"
        "\n"
        "  - 内机产品文档中的产品，应标注为'室内机'"
        "\n\n"
        "**注意事项：**"
        "\n"
        "  - 关键字可能出现在文档其他位置，需要结合上下文判断是否是产品类型描述"
        "\n"
        "  - 如果无法明确判断，可以使用更通用的类别（如：'空调'、'多联机'）"
        "\n"
        "  - 同一文档中可能同时包含室内机和室外机产品，需要分别识别"
        "\n\n"
        "**4) features 字段抽取规则（特别重要！）**"
        "\n"
        "features 字段用于记录产品的功能、特点、优势、技术特性等。这是非常重要的字段，请务必仔细抽取。"
        "\n\n"
        "**格式要求：每条feature应该是完整的描述性语句，包含功能名称和效果/优势说明**"
        "\n"
        "  - ✓ 正确格式：'美的第三代内外机部件全直流技术：搭载更为先进节能的压缩机及变频技术，在提升舒适性同时，可节能达到20%以上'"
        "\n"
        "  - ✓ 正确格式：'WiFi远程控制：支持手机APP远程操控，随时随地掌控空调运行状态'"
        "\n"
        "  - ✓ 正确格式：'低温制热：-35°C到50°C宽温运行，极寒天气也能稳定制热'"
        "\n"
        "  - ✗ 错误格式：'WiFi远程控制'（太简略，缺少说明）"
        "\n"
        "  - ✗ 错误格式：'节能'（太笼统，缺少具体描述）"
        "\n\n"
        "应该抽取的内容包括但不限于："
        "\n"
        "  a) 功能特性：如 'WiFi远程控制：支持手机APP远程操控'、'智能除霜：自动检测结霜情况并除霜'"
        "\n"
        "  b) 技术特点：如 '直流变频技术：采用直流变频压缩机，节能效果提升30%'、'喷气增焓技术：低温制热能力提升'"
        "\n"
        "  c) 控制方式：如 '线控器控制：配备液晶线控器，操作简便直观'、'集中控制：支持多台设备统一管理'"
        "\n"
        "  d) 运行模式：如 '自动模式：根据室内温度自动切换制冷制热'、'静音模式：夜间运行噪音低至22dB'"
        "\n"
        "  e) 舒适性特点：如 '温度均匀：360度送风，室内温度更均匀'、'快速制冷：5分钟快速降温'"
        "\n"
        "  f) 安全特性：如 '过载保护：电流过载时自动断电保护'、'防冻保护：低温环境下自动启动防冻功能'"
        "\n"
        "  g) 适用场景：如 '适用于家庭：设计紧凑，适合中小户型家庭使用'、'适用于商业场所：大风量设计，满足商业空间需求'"
        "\n"
        "  h) 其他优势：如 '节能省电：一级能效，比普通空调节能30%'、'环保制冷剂：采用R32环保制冷剂，减少温室效应'"
        "\n\n"
        "抽取示例："
        "\n"
        "  - 原文：'支持WiFi远程控制，可通过手机APP随时随地操控空调，具有静音、节能、外出、定时等智能模式'"
        "\n"
        "    → features: ["
        "\n"
        "        'WiFi远程控制：支持手机APP远程操控，随时随地掌控空调运行状态',"
        "\n"
        "        '静音模式：夜间运行更安静',"
        "\n"
        "        '节能模式：智能调节运行参数，降低能耗',"
        "\n"
        "        '外出模式：长时间外出时保持低功耗运行',"
        "\n"
        "        '定时功能：可预约开关机时间'"
        "\n"
        "      ]"
        "\n"
        "  - 原文：'美的第三代内外机部件全直流技术：搭载更为先进节能的压缩机及变频技术，在提升舒适性同时，可节能达到20%以上'"
        "\n"
        "    → features: ['美的第三代内外机部件全直流技术：搭载更为先进节能的压缩机及变频技术，在提升舒适性同时，可节能达到20%以上']"
        "\n"
        "  - 原文：'采用直流变频压缩机，低温制热性能优异，-35°C到50°C宽温运行，极寒天气也能稳定制热'"
        "\n"
        "    → features: ["
        "\n"
        "        '直流变频压缩机：节能高效，运行更稳定',"
        "\n"
        "        '低温制热：-35°C到50°C宽温运行，极寒天气也能稳定制热'"
        "\n"
        "      ]"
        "\n\n"
        "注意事项："
        "\n"
        "  - **每条feature必须包含功能名称和效果说明，不要只写功能名称**"
        "\n"
        "  - 如果原文已经提供了详细描述，直接使用原文描述"
        "\n"
        "  - 如果原文只提到功能名称，根据上下文补充合理的效果说明"
        "\n"
        "  - 即使原文只提到1-2个功能特点，也要抽取，不要因为数量少就忽略"
        "\n"
        "  - 表格中的'功能'、'特点'、'优势'、'技术'等列的内容都应该抽取"
        "\n"
        "  - 如果原文确实没有任何功能特点描述，才可以留空数组"
        "\n\n"
        "**5) brand 字段的判断规则**"
        "\n"
        "brand 字段应该是**公司/品牌名称**（如：美的、格力、大金、海尔、三菱等），不是产品型号的一部分。"
        "\n"
        "  - **不要将产品型号前缀当作品牌**：如果只看到 2-5 个大写字母的型号前缀（如 MDV、FXS、GMV、KFR 等），这些通常是产品型号的一部分，不是品牌名称。"
        "\n"
        "  - 判断 brand 的正确方法："
        "\n"
        "    a) 如果原文明确提到品牌名称（如：'美的'、'Midea'、'GREE 格力'、'大金工业'等），则提取该品牌名称"
        "\n"
        "    b) 如果 manufacturer 字段包含明确的品牌关键词（如：'广东美的暖通设备有限公司'、'珠海格力电器'等），可以从中提取品牌核心名称（如：'美的'、'格力'）"
        "\n"
        "    c) 如果原文只有产品型号（如：'MDV-D15Q4'、'FXS-20MVJU'）而没有明确的品牌信息，则 brand 字段应该**留空**"
        "\n"
        "  - 常见误判案例（务必避免）："
        "\n"
        "    ✗ 错误：product_model='MDV-D15Q4', brand='MDV'  （MDV 是型号前缀，不是品牌）"
        "\n"
        "    ✓ 正确：product_model='MDV-D15Q4', brand=''  （如果原文没有品牌信息）"
        "\n"
        "    ✓ 正确：product_model='MDV-D15Q4', brand='美的', manufacturer='广东美的暖通设备有限公司'"
        "\n\n"
        "**6) 性能参数规则**"
        "\n"
        "performance_specs[] 用于记录**可量化的技术参数**，必须同时包含参数名、数值和单位。"
        "\n\n"
        "**重要：以下内容不应放入performance_specs，应放入features：**"
        "\n"
        "  - ✗ 错误：'节能提升20%'、'性能提升30%'、'效率提高' → 这些是优势描述，应放入features"
        "\n"
        "  - ✗ 错误：'高效节能'、'低噪音运行' → 这些是特点描述，应放入features"
        "\n"
        "  - ✓ 正确：'额定制冷量: 5.6kW'、'额定功率: 1.8kW'、'噪音: 22dB' → 这些是可量化参数"
        "\n\n"
        "performance_specs 抽取规则："
        "\n"
        "  - 仅允许使用配置中列出的标准参数名（如：额定制冷量、额定功率、噪音等）"
        "\n"
        "  - 必须同时出现数值和期望单位才能输出，缺失单位或不匹配则跳过"
        "\n"
        "  - 每条包含：name（标准参数名）、value（原文数值）、unit（单位）、raw（原文片段）"
        "\n"
        "  - 不得将不同产品的性能参数混在同一条结果中"
        "\n"
        "  - 如果参数名包含'提升'、'增加'、'改善'等词，这不是性能参数，应放入features"
        "\n\n"
        "**7) 其他字段**"
        "\n"
        "- fact_text: 收集无法落位但可能有用的原文句子，避免重复已提取的信息"
        "\n"
        "- evidence: 列出支撑上述字段的原文片段"
        "\n\n"
        "**8) 输出格式**"
        "\n"
        "输出必须符合给定 JSON Schema，除规定字段外不得添加其它字段。"
    ),

    # 性能参数及期望单位（空字符串表示可无单位）
    "performance_param_units": {
        "电源额定相数": "",
        "电源额定电压": "V",
        "电源额定频率": "Hz",
        "额定制冷量": "kW",
        "额定制热量": "kW",
        "额定制冷功率": "kW",
        "额定制冷电流": "A",
        "额定制热功率": "kW",
        "额定制热电流": "A",
        "最大运转电流": "A",
        "最大运转功率": "kW",
        "IPLvV": "",
        "EER": "",
        "COP": "",
        "APF": "",
        "框体尺寸(宽*高*深)": "mm",
        "毛重": "kg",
        "净重": "kg",
        "外机风量": "m³/h",
        "室外机制冷噪音值": "dB",
        "室外机制热噪音值": "dB",
        "室外机最小噪音值": "dB",
        "室外机最大噪音值": "dB",
        "高/低压最大允许压力": "MPa",
        "冷媒充注量": "kg",
        "润滑油充注量": "kg",
        "润滑油种类": "",
        "联机配比范围": "%",
        "模块组合范围": "",
        "最多联机台数": "台",
        "制冷运行范围": "kW",
        "制热运行范围": "kW",
        "最大总配管长": "m",
        "最大单管长": "m",
        "内外机最大高差(外机在上)": "m",
        "内外机最大高差(外机在下)": "m",
        "内机间最大高差": "m",
        "机外静压": "Pa",
    },
}

# ============================================
# 品牌推断配置 (Brand Inference Config)
# ============================================
BRAND_INFERENCE_CONFIG = {
    # k-NN参数
    "k": int(os.getenv("BRAND_INFERENCE_K", 5)),  # k近邻数
    "high_confidence_threshold": float(os.getenv("BRAND_HIGH_CONFIDENCE_THRESHOLD", 0.75)),  # 高置信度阈值
    "medium_confidence_threshold": float(os.getenv("BRAND_MEDIUM_CONFIDENCE_THRESHOLD", 0.6)),  # 中置信度阈值

    # 性能优化
    "use_optimization": os.getenv("BRAND_INFERENCE_OPTIMIZATION", "true").lower() == "true",  # 是否启用性能优化
    "coarse_grouping": True,  # 粗分组（按manufacturer首字母、category等）
    "quick_filter_threshold": float(os.getenv("BRAND_QUICK_FILTER_THRESHOLD", 0.3)),  # 快速过滤阈值

    # 应用策略
    "apply_threshold": os.getenv("BRAND_APPLY_THRESHOLD", "high"),  # 应用阈值: "high"(仅高), "medium"(高+中), "all"(全部)

    # 输出
    "save_inference_log": True,  # 是否保存推断日志
    "inference_log_dir": str(BASE_DIR.parent / "KnowledgeFusion" / "output"),  # 推断日志目录
}

# ============================================
# Graph importer config
# ============================================
GRAPH_IMPORTER_CONFIG = {
    "json_path": str(BASE_DIR.parent / "results" / "fused_entities_all.json"),
    "json_encoding": os.getenv("GRAPH_IMPORTER_JSON_ENCODING", "utf-8"),
    "dry_run": os.getenv("GRAPH_IMPORTER_DRY_RUN", "false").lower() == "true",
    "create_constraints": os.getenv("GRAPH_IMPORTER_CREATE_CONSTRAINTS", "true").lower() == "true",
    "clear_before_import": os.getenv("GRAPH_IMPORTER_CLEAR_BEFORE_IMPORT", "false").lower() == "true",
    "batch_size": int(os.getenv("GRAPH_IMPORTER_BATCH_SIZE", "500")),
    "neo4j": {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "12345678"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    },
    "schema": {
        "labels": {
            "product": "Product",
            "brand": "Brand",
            "manufacturer": "Manufacturer",
            "category": "Category",
            "series": "Series",
        },
        "relationships": {
            "brand": "BRANDED_BY",
            "manufacturer": "MANUFACTURED_BY",
            "category": "IN_CATEGORY",
            "series": "IN_SERIES",
        },
    },
}
