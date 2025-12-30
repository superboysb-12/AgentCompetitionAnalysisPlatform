"""
RAG 单例管理器
确保 Embedding 模型和向量数据库只初始化一次，全局共享
"""

from typing import Optional, Dict, Any
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class RAGSingletonManager:
    """
    RAG 单例管理器

    管理全局唯一的 Embedding 模型和向量存储实例，避免重复加载
    使用双重检查锁定模式确保线程安全
    """

    _instance = None
    _lock = Lock()

    # 全局共享的实例
    _embeddings = None
    _vector_store = None
    _embeddings_lock = Lock()
    _vector_store_lock = Lock()

    def __new__(cls):
        """单例模式：确保只有一个管理器实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_embeddings(
        cls,
        embedding_config: Dict[str, Any],
        force_reload: bool = False
    ):
        """
        获取全局共享的 Embedding 实例

        Args:
            embedding_config: Embedding 配置
            force_reload: 是否强制重新加载

        Returns:
            BGEM3Embeddings 实例
        """
        if cls._embeddings is None or force_reload:
            with cls._embeddings_lock:
                # 双重检查
                if cls._embeddings is None or force_reload:
                    logger.info("=" * 60)
                    logger.info("🚀 初始化全局 Embedding 模型...")

                    from .embeddings import BGEM3Embeddings

                    cls._embeddings = BGEM3Embeddings(
                        model_name=embedding_config['model_name'],
                        cache_dir=embedding_config['model_cache_dir'],
                        device=embedding_config['device'],
                        batch_size=embedding_config['batch_size'],
                        max_length=embedding_config['max_length'],
                        normalize_embeddings=embedding_config['normalize_embeddings'],
                    )

                    # 预加载模型
                    logger.info("⏳ 预加载模型（首次加载可能需要几分钟下载）...")
                    cls._embeddings._load_model()
                    logger.info("✓ Embedding 模型加载完成！")
                    logger.info("=" * 60)

        return cls._embeddings

    @classmethod
    def get_vector_store(
        cls,
        chroma_config: Dict[str, Any],
        force_reload: bool = False
    ):
        """
        获取全局共享的向量存储实例

        Args:
            chroma_config: Chroma 配置
            force_reload: 是否强制重新加载

        Returns:
            ChromaVectorStore 实例
        """
        if cls._vector_store is None or force_reload:
            with cls._vector_store_lock:
                # 双重检查
                if cls._vector_store is None or force_reload:
                    logger.info("🗄️  初始化全局向量数据库...")

                    from .vector_store import ChromaVectorStore

                    cls._vector_store = ChromaVectorStore(
                        persist_directory=chroma_config['persist_directory'],
                        collection_name=chroma_config['collection_name'],
                        distance_metric=chroma_config['distance_metric'],
                        anonymized_telemetry=chroma_config['anonymized_telemetry'],
                    )

                    logger.info("✓ 向量数据库初始化完成！")

        return cls._vector_store

    @classmethod
    def is_embeddings_loaded(cls) -> bool:
        """检查 Embedding 模型是否已加载"""
        return cls._embeddings is not None and cls._embeddings._model is not None

    @classmethod
    def is_vector_store_loaded(cls) -> bool:
        """检查向量数据库是否已加载"""
        return cls._vector_store is not None

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """获取加载状态"""
        return {
            "embeddings_loaded": cls.is_embeddings_loaded(),
            "vector_store_loaded": cls.is_vector_store_loaded(),
            "embeddings_config": {
                "device": cls._embeddings.device if cls._embeddings else None,
                "batch_size": cls._embeddings.batch_size if cls._embeddings else None,
                "model_name": cls._embeddings.model_name if cls._embeddings else None,
            } if cls._embeddings else None
        }

    @classmethod
    def clear(cls):
        """清除所有缓存实例（仅用于测试或重新加载）"""
        logger.warning("⚠️  清除所有 RAG 单例实例")
        cls._embeddings = None
        cls._vector_store = None
