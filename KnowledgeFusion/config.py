"""
知识融合配置文件
从 backend_v2/settings.py 读取LLM配置
"""

import sys
import os
from pathlib import Path

# 添加backend_v2到路径
backend_v2_path = Path(__file__).parent.parent / "backend_v2"
sys.path.insert(0, str(backend_v2_path))

try:
    from settings import RELATION_EXTRACTOR_CONFIG, BRAND_INFERENCE_CONFIG as BRAND_INF_CFG

    # LLM配置
    LLM_CONFIG = {
        "api_key": RELATION_EXTRACTOR_CONFIG["api_key"],
        "base_url": RELATION_EXTRACTOR_CONFIG["base_url"],
        "model": RELATION_EXTRACTOR_CONFIG.get("model", "gpt-4o-mini"),
        "timeout": RELATION_EXTRACTOR_CONFIG.get("timeout", 300),
        "max_retries": RELATION_EXTRACTOR_CONFIG.get("max_retries", 3),
    }

    # 品牌推断配置
    BRAND_INFERENCE_CONFIG = {
        "k": BRAND_INF_CFG.get("k", 5),
        "high_confidence_threshold": BRAND_INF_CFG.get("high_confidence_threshold", 0.75),
        "medium_confidence_threshold": BRAND_INF_CFG.get("medium_confidence_threshold", 0.6),
        "use_optimization": BRAND_INF_CFG.get("use_optimization", True),
        "use_llm_backup": True,  # LLM二次判断（对所有k-NN结果）
        "save_inference_log": True,
        "inference_log_dir": "output",
    }

except ImportError as e:
    print(f"⚠️  无法导入backend_v2/settings.py: {e}")
    print("   使用默认配置（请设置环境变量）")

    LLM_CONFIG = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "timeout": 300,
        "max_retries": 3,
    }

    BRAND_INFERENCE_CONFIG = {
        "k": 5,
        "high_confidence_threshold": 0.75,
        "medium_confidence_threshold": 0.6,
        "use_optimization": True,
        "use_llm_backup": True,  # LLM二次判断（对所有k-NN结果）
        "save_inference_log": True,
        "inference_log_dir": "output",
    }

# 指代融合配置 ⭐ NEW
ALIAS_FUSION_CONFIG = {
    "use_highest_frequency_as_canonical": True,  # 使用最高频品牌作为规范名
    "min_confidence": 0.6,  # 最低置信度阈值
}

# 知识融合配置
FUSION_CONFIG = {
    "similarity_threshold": 0.97,  # 相似度阈值（提高到0.97，减少LLM调用）
    "use_llm": True,  # 是否使用LLM
    "max_workers": None,  # 并行计算的最大工作进程数（None=CPU核心数）
    "direct_merge_threshold": 0.99,  # 直接合并阈值（相似度高于此值直接合并）
}

# 数据路径配置
DATA_CONFIG = {
    "input_path": str(Path(__file__).parent.parent / "LLMRelationExtracter_v2" / "relation_results.json"),
    "output_dir": str(Path(__file__).parent / "output"),
    "filter_rare_brands": True,  # 是否过滤稀有品牌
    "min_brand_frequency_ratio": 0.01,  # 最小品牌频率比例（1%）
}


def load_all_configs():
    """加载并验证所有配置"""
    # 验证LLM配置
    if not LLM_CONFIG.get("api_key"):
        print("⚠️  警告: 未设置API Key")
        print("   请设置环境变量 OPENAI_API_KEY 或在 backend_v2/settings.py 中配置")

    return {
        "llm": LLM_CONFIG,
        "brand_inference": BRAND_INFERENCE_CONFIG,
        "alias_fusion": ALIAS_FUSION_CONFIG,
        "fusion": FUSION_CONFIG,
        "data": DATA_CONFIG,
    }


if __name__ == "__main__":
    configs = load_all_configs()

    print("=" * 60)
    print("配置信息")
    print("=" * 60)

    print(f"\nLLM配置:")
    print(f"  Base URL: {LLM_CONFIG['base_url']}")
    print(f"  Model: {LLM_CONFIG['model']}")
    print(f"  Timeout: {LLM_CONFIG['timeout']}秒")

    print(f"\n品牌推断配置:")
    for key, value in BRAND_INFERENCE_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\n指代融合配置:")
    for key, value in ALIAS_FUSION_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\n知识融合配置:")
    for key, value in FUSION_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\n数据配置:")
    for key, value in DATA_CONFIG.items():
        print(f"  {key}: {value}")
