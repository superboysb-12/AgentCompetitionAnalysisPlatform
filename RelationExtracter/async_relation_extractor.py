import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from paddlenlp import Taskflow
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from config_loader import get_config, get_schema

logger = logging.getLogger(__name__)


def normalize_text(s: str) -> str:
    """标准化文本格式"""
    s = re.sub(r'\r', '\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def dedup_paragraphs(paras: List[str], min_length: int = 20, fingerprint_length: int = 80) -> List[str]:
    """去重段落"""
    seen = set()
    kept = []
    for p in paras:
        key = re.sub(r'\s+', '', p)
        if len(key) < min_length:
            kept.append(p)
            continue
        # 粗糙去重：截断做指纹
        fp = key[:fingerprint_length]
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(p)
    return kept


def split_sentences(para: str) -> List[str]:
    """分割句子"""
    parts = re.split(r'(?<=[。！？；.!?])\s*', para)
    return [p for p in parts if p and p.strip()]


def make_chunks(sentences: List[str], chunk_min: int = 300, chunk_max: int = 800, stride_ratio: float = 0.25) -> List[Tuple[str, Tuple[int, int]]]:
    """
    创建文本块
    返回 [(chunk_text, (sent_start_idx, sent_end_idx)), ...]
    记录句子级起止索引，便于回映射/去重。
    """
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        j = i + 1
        while j < len(sentences) and len(chunk) < chunk_max:
            next_s = sentences[j]
            if len(chunk) < chunk_min or len(chunk) + len(next_s) <= chunk_max:
                chunk += next_s
                j += 1
            else:
                break
        chunks.append((chunk, (i, j-1)))
        # 滑窗前进（保留重叠）
        step = max(1, int((j - i) * (1 - stride_ratio)))
        i += step
    return chunks


class AsyncRelationExtractor:
    """异步关系提取器，适合后端服务使用"""

    def __init__(
        self,
        config_name: str = "standard",
        schema: Optional[Dict[str, List[str]]] = None,
        schema_key: Optional[str] = None,
        **overrides
    ):
        """
        初始化异步关系提取器

        Args:
            config_name: 配置名称，从config.yaml中加载
            schema: 直接指定schema（优先级最高）
            schema_key: 指定schema键名（优先级中等）
            **overrides: 其他配置覆盖参数
        """
        # 加载配置
        self.config = get_config(config_name)

        # 应用覆盖参数
        if overrides:
            # 更新模型配置
            model_keys = ['model', 'batch_size', 'precision', 'schema_lang', 'max_workers']
            for key in model_keys:
                if key in overrides:
                    self.config['model'][key] = overrides[key]

            # 更新文本处理配置
            text_keys = ['chunk_min', 'chunk_max', 'stride_ratio', 'dedup_min_length', 'fingerprint_length']
            for key in text_keys:
                if key in overrides:
                    self.config['text_processing'][key] = overrides[key]

        # 确定最终使用的schema
        if schema:
            self.schema = schema
        elif schema_key:
            self.schema = get_schema(schema_key)
        else:
            self.schema = self.config['schema']

        # 提取配置参数
        model_config = self.config['model']
        self.model = model_config['name']
        self.batch_size = model_config['batch_size']
        self.precision = model_config['precision']
        self.schema_lang = model_config['schema_lang']
        self.max_workers = model_config['max_workers']

        # 文本处理配置
        self.text_config = self.config['text_processing']

        self._ie = None
        self._executor = None
        self._initialized = False

    async def initialize(self):
        """异步初始化模型（避免阻塞事件循环）"""
        if self._initialized:
            return

        def _init_model():
            return Taskflow(
                "information_extraction",
                schema=self.schema,
                model=self.model,
                schema_lang=self.schema_lang,
                batch_size=self.batch_size,
                precision=self.precision
            )

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        loop = asyncio.get_event_loop()

        try:
            self._ie = await loop.run_in_executor(self._executor, _init_model)
            self._initialized = True
            logger.info("RelationExtractor model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RelationExtractor model: {e}")
            raise

    async def extract_async(self, text: str) -> List[Dict[str, Any]]:
        """
        异步提取关系

        Args:
            text: 待提取的文本

        Returns:
            提取结果列表，每个元素包含 range, chunk, result 字段
        """
        if not self._initialized:
            await self.initialize()

        # 预处理文本
        text = normalize_text(text)
        paras = [p for p in re.split(r'\n{2,}', text) if p.strip()]
        paras = dedup_paragraphs(
            paras,
            min_length=self.text_config['dedup_min_length'],
            fingerprint_length=self.text_config['fingerprint_length']
        )

        chunk_infos = []
        for p in paras:
            sents = split_sentences(p)
            if not sents:
                continue
            chunk_infos += make_chunks(
                sents,
                chunk_min=self.text_config['chunk_min'],
                chunk_max=self.text_config['chunk_max'],
                stride_ratio=self.text_config['stride_ratio']
            )

        if not chunk_infos:
            return []

        chunks = [c for c, _ in chunk_infos]

        # 异步调用模型推理
        loop = asyncio.get_event_loop()

        def _extract(chunks_batch):
            return self._ie(chunks_batch)

        try:
            raw_results = await loop.run_in_executor(
                self._executor, _extract, chunks
            )
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise

        # 去重合并结果
        seen = set()
        merged = []
        for (chunk, idx_range), res in zip(chunk_infos, raw_results):
            key = str(res)
            if key in seen:
                continue
            seen.add(key)
            merged.append({
                "range": idx_range,
                "chunk": chunk,
                "result": res
            })

        return merged

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        同步提取关系（兼容原接口）

        Args:
            text: 待提取的文本

        Returns:
            提取结果列表
        """
        return asyncio.run(self.extract_async(text))

    async def batch_extract_async(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        异步批量提取关系

        Args:
            texts: 待提取的文本列表

        Returns:
            每个文本对应的提取结果列表
        """
        if not self._initialized:
            await self.initialize()

        tasks = [self.extract_async(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常情况
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract from text {i}: {result}")
                final_results.append([])
            else:
                final_results.append(result)

        return final_results

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._initialized:
                await self.initialize()

            # 简单测试
            test_text = "这是一个测试文本。"
            await self.extract_async(test_text)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def update_schema(self, new_schema: Dict[str, List[str]]):
        """
        更新抽取schema（需要重新初始化）

        Args:
            new_schema: 新的schema配置
        """
        self.schema = new_schema
        self._initialized = False
        self._ie = None

    async def close(self):
        """清理资源"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._initialized = False
        self._ie = None
        logger.info("RelationExtractor resources cleaned up")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# 便捷函数
async def extract_relations_async(
    text: str,
    config_name: str = "standard",
    schema: Optional[Dict[str, List[str]]] = None,
    schema_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    便捷的异步关系提取函数

    Args:
        text: 待提取的文本
        config_name: 配置名称
        schema: 直接指定schema
        schema_key: schema键名

    Returns:
        提取结果列表
    """
    async with AsyncRelationExtractor(
        config_name=config_name,
        schema=schema,
        schema_key=schema_key
    ) as extractor:
        return await extractor.extract_async(text)


# 快捷创建函数
def create_extractor(
    config_name: str = "standard",
    **overrides
) -> AsyncRelationExtractor:
    """
    创建关系提取器实例

    Args:
        config_name: 配置名称
        **overrides: 配置覆盖参数

    Returns:
        AsyncRelationExtractor实例
    """
    return AsyncRelationExtractor(config_name=config_name, **overrides)