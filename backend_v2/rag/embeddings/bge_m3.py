"""
BGE M3 Embedding 模型封装
基于 FlagEmbedding 实现的 BGE M3 模型
"""

from typing import List, Optional
import logging
from pathlib import Path
from .base import BaseEmbeddings

logger = logging.getLogger(__name__)


class BGEM3Embeddings(BaseEmbeddings):
    """
    BGE M3 Embedding 模型

    BGE M3 是智源研究院开源的多语言、多任务、多粒度的 Embedding 模型
    支持中英文，向量维度为 1024
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        """
        初始化 BGE M3 模型

        Args:
            model_name: 模型名称或路径
            cache_dir: 模型缓存目录
            device: 运行设备 (cpu 或 cuda)
            batch_size: 批处理大小
            max_length: 最大文本长度
            normalize_embeddings: 是否归一化向量
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # 延迟加载模型
        self._model = None
        self._dimension = 1024  # BGE M3 固定维度

        logger.info(f"BGE M3 Embeddings 配置完成: device={device}, batch_size={batch_size}")

    def _load_model(self):
        """延迟加载模型（首次使用时加载）"""
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info(f"正在加载 BGE M3 模型: {self.model_name}")

                # 确保缓存目录存在
                if self.cache_dir:
                    Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

                # 加载模型
                self._model = BGEM3FlagModel(
                    model_name_or_path=self.model_name,
                    use_fp16=(self.device == "cuda"),  # GPU 时使用 FP16
                    device=self.device,
                )

                logger.info(f"✓ BGE M3 模型加载成功")

            except ImportError:
                error_msg = "FlagEmbedding 库未安装，请运行: pip install FlagEmbedding"
                logger.error(error_msg)
                raise ImportError(error_msg)
            except Exception as e:
                logger.error(f"✗ BGE M3 模型加载失败: {e}")
                raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文档文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        if not texts:
            return []

        # 确保模型已加载
        self._load_model()

        try:
            # 使用 BGE M3 的 encode 方法
            # return_dense=True 返回稠密向量（1024维）
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,  # 返回稠密向量
                return_sparse=False,  # 不返回稀疏向量
                return_colbert_vecs=False,  # 不返回 ColBERT 向量
            )

            # embeddings['dense_vecs'] 是 numpy array，转为 list
            vectors = embeddings['dense_vecs'].tolist()

            logger.debug(f"成功编码 {len(texts)} 个文档，向量维度: {len(vectors[0])}")
            return vectors

        except Exception as e:
            logger.error(f"✗ 文档编码失败: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        将查询文本转换为向量

        Args:
            text: 查询文本

        Returns:
            List[float]: 向量
        """
        # 查询和文档使用相同的编码方式
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []

    @property
    def dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度 (BGE M3 固定为 1024)
        """
        return self._dimension
