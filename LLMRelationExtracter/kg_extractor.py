import json
import yaml
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from openai import OpenAI, AzureOpenAI, APITimeoutError
import tiktoken

# 尝试导入zhipuai，如果不存在则设置为None
try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZhipuAI = None
    ZHIPUAI_AVAILABLE = False

# 导入自定义OpenAI客户端
try:
    from custom_openai_client import CustomOpenAIClient
    CUSTOM_CLIENT_AVAILABLE = True
except ImportError:
    CustomOpenAIClient = None
    CUSTOM_CLIENT_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvidenceSpan:
    """证据片段位置信息

    Attributes:
        start: 证据在原文中的起始位置
        end: 证据在原文中的结束位置
        text: 证据文本内容
    """
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """将证据片段对象转换为字典格式

        Returns:
            包含start、end、text字段的字典
        """
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text
        }

@dataclass
class Attribute:
    """实体属性数据类

    Attributes:
        key: 属性名称
        value: 属性值
        value_type: 值的数据类型
        unit: 属性单位（可选）
        confidence: 置信度分数
        evidence: 支持该属性的证据文本
        evidence_spans: 证据在原文中的位置信息列表
        is_key_in_config: 该属性键是否在配置文件中定义
    """
    key: str
    value: str
    value_type: str
    unit: str = ""
    confidence: float = 1.0
    evidence: str = ""
    evidence_spans: List[EvidenceSpan] = field(default_factory=list)
    is_key_in_config: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """将属性对象转换为字典格式

        Returns:
            包含所有属性字段的字典
        """
        return {
            'key': self.key,
            'value': self.value,
            'value_type': self.value_type,
            'unit': self.unit,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'evidence_spans': [span.to_dict() for span in self.evidence_spans],
            'is_key_in_config': self.is_key_in_config
        }

@dataclass
class Entity:
    """实体数据类

    Attributes:
        name: 实体名称
        entity_type: 实体类型
        attributes: 实体属性列表
        source_url: 实体来源URL
        doc_id: 所属文档ID
        is_type_in_config: 实体类型是否在配置文件中定义
    """
    name: str
    entity_type: str
    attributes: List[Attribute] = field(default_factory=list)
    source_url: str = ""
    doc_id: str = ""
    is_type_in_config: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """将实体对象转换为字典格式

        Returns:
            包含所有实体字段的字典
        """
        return {
            'name': self.name,
            'entity_type': self.entity_type,
            'attributes': [attr.to_dict() for attr in self.attributes],
            'source_url': self.source_url,
            'doc_id': self.doc_id,
            'is_type_in_config': self.is_type_in_config
        }

@dataclass
class Relation:
    """关系三元组数据类

    Attributes:
        subject: 主体实体名称
        subject_type: 主体实体类型
        relation: 关系类型
        object: 客体实体名称
        object_type: 客体实体类型
        confidence: 置信度分数
        evidence: 支持该关系的证据文本
        evidence_spans: 证据在原文中的位置信息列表
        source_url: 关系来源URL
        doc_id: 所属文档ID
        is_relation_in_config: 关系类型是否在配置中定义
        is_subject_type_in_config: 主体类型是否在配置中定义
        is_object_type_in_config: 客体类型是否在配置中定义
    """
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    confidence: float
    evidence: str
    evidence_spans: List[EvidenceSpan]
    source_url: str = ""
    doc_id: str = ""
    is_relation_in_config: bool = True
    is_subject_type_in_config: bool = True
    is_object_type_in_config: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """将关系对象转换为字典格式

        Returns:
            包含所有关系字段的字典
        """
        return {
            'subject': self.subject,
            'subject_type': self.subject_type,
            'relation': self.relation,
            'object': self.object,
            'object_type': self.object_type,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'evidence_spans': [span.to_dict() for span in self.evidence_spans],
            'source_url': self.source_url,
            'doc_id': self.doc_id,
            'is_relation_in_config': self.is_relation_in_config,
            'is_subject_type_in_config': self.is_subject_type_in_config,
            'is_object_type_in_config': self.is_object_type_in_config
        }

# 保留Triplet作为Relation的别名，用于向后兼容
Triplet = Relation

@dataclass
class ExtractionResult:
    """知识抽取结果数据类

    Attributes:
        text: 原始输入文本
        entities: 抽取出的实体列表
        relations: 抽取出的关系列表
        triplets: relations的别名，用于向后兼容
        processing_time: 处理耗时（秒）
        token_usage: LLM token使用统计
    """
    text: str
    entities: List[Entity]
    relations: List[Relation]
    triplets: List[Relation] = None
    processing_time: float = 0.0
    token_usage: Optional[Dict[str, int]] = None

    def __post_init__(self):
        """初始化后处理，设置triplets别名以保持向后兼容"""
        if self.triplets is None:
            self.triplets = self.relations

class KnowledgeGraphExtractor:
    """知识图谱抽取器，负责从文本中抽取实体、属性和关系"""

    def __init__(self, config_path: str):
        """初始化知识图谱抽取器

        Args:
            config_path: YAML配置文件的路径
        """
        self.config = self._load_config(config_path)

        # 检测模型提供商
        self.model_provider = self._detect_model_provider()

        # 初始化客户端
        self.client = self._init_llm_client()
        self.tokenizer = self._init_tokenizer()

        # 解析新的配置格式
        self.entity_types = self._parse_entity_types()
        self.relation_types = self._parse_relation_types()
        self.entity_types_list = list(self.entity_types.keys())
        self.relation_types_list = list(self.relation_types.keys())

        # 解析实体属性定义
        self.entity_attributes = self._parse_entity_attributes()

        self.max_text_length = self.config['processing']['max_text_length']
        self.confidence_threshold = self.config['output']['confidence_threshold']

        # 分块配置
        self.enable_chunking = self.config['processing'].get('enable_chunking', True)
        self.chunk_size = self.config['processing'].get('chunk_size', 6000)
        self.chunk_overlap = self.config['processing'].get('chunk_overlap', 500)

        # JSON Schema配置
        json_schema_config = self.config.get('json_schema', {})
        self.json_schema_enabled = json_schema_config.get('enabled', False)
        self.json_schema = json_schema_config.get('schema', None) if self.json_schema_enabled else None

        logger.info(f"初始化知识图谱抽取器完成 (模型提供商: {self.model_provider})")
        logger.info(f"JSON Schema模式: {'启用' if self.json_schema_enabled else '禁用'}")
        logger.info(f"自动分块功能: {'启用' if self.enable_chunking else '禁用'}")
        if self.enable_chunking:
            logger.info(f"  块大小: {self.chunk_size} tokens, 重叠: {self.chunk_overlap} tokens")
        logger.info(f"支持的实体类型: {self.entity_types_list}")
        logger.info(f"支持的关系类型: {self.relation_types_list}")

        # 统计属性定义数量
        total_attrs = sum(len(attrs) for attrs in self.entity_attributes.values())
        logger.info(f"加载了 {total_attrs} 个属性定义")

        # 初始化few-shot管理器
        self.few_shot_manager = None
        if self.config['advanced_techniques'].get('enable_few_shot', False):
            try:
                from few_shot_manager import FewShotManager
                self.few_shot_manager = FewShotManager(self.config)
                logger.info("Few-shot支持已启用")
            except ImportError:
                logger.warning("无法导入FewShotManager，few-shot功能将被禁用")
                self.config['advanced_techniques']['enable_few_shot'] = False

    def _parse_entity_types(self) -> Dict[str, Dict[str, Any]]:
        """解析配置文件中的实体类型定义

        Returns:
            实体类型名称到配置信息的映射字典
        """
        entity_config = self.config.get('entity_types', {})

        # 兼容旧格式（列表）和新格式（字典）
        if isinstance(entity_config, list):
            return {entity: {"description": entity} for entity in entity_config}
        else:
            return entity_config

    def _parse_relation_types(self) -> Dict[str, Dict[str, Any]]:
        """解析配置文件中的关系类型定义

        Returns:
            关系类型名称到配置信息的映射字典
        """
        relation_config = self.config.get('relation_types', {})

        # 兼容旧格式（列表）和新格式（字典）
        if isinstance(relation_config, list):
            return {relation: {"description": relation} for relation in relation_config}
        else:
            return relation_config

    def _parse_entity_attributes(self) -> Dict[str, Dict[str, Any]]:
        """解析实体类型的属性定义

        Returns:
            实体类型到属性配置的映射字典
        """
        entity_attributes = {}
        entity_config = self.config.get('entity_types', {})

        for entity_type, config in entity_config.items():
            if 'attributes' in config and isinstance(config['attributes'], dict):
                entity_attributes[entity_type] = config['attributes']
            else:
                entity_attributes[entity_type] = {}

        logger.info(f"加载了 {len(entity_attributes)} 个实体类型的属性定义")
        return entity_attributes

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def _detect_model_provider(self) -> str:
        """检测LLM模型提供商类型

        Returns:
            提供商名称，可能是'zhipuai'、'azure'、'openai'、'custom_api'等
        """
        model_config = self.config['model']
        provider = model_config.get('provider', 'openai').lower()
        model_name = model_config.get('model_name', '').lower()
        api_base = model_config.get('api_base', '')
        azure_endpoint = model_config.get('azure_endpoint', '')

        # 优先使用provider字段
        if provider == 'azure':
            return 'azure'

        # 检测是否是自定义API（需要URL参数）
        if provider == 'custom_api':
            if not CUSTOM_CLIENT_AVAILABLE:
                logger.warning("检测到custom_api模式，但custom_openai_client模块未找到")
                logger.warning("请确保custom_openai_client.py在同一目录下")
                raise ImportError("CustomOpenAIClient未找到，无法使用custom_api模式")
            return 'custom_api'

        # 检测是否是Azure OpenAI（通过endpoint或api_base）
        if azure_endpoint or 'azure' in api_base or 'openai.azure.com' in api_base:
            return 'azure'

        # 检测是否是智谱AI
        if provider == 'zhipuai' or 'glm' in model_name or 'bigmodel.cn' in api_base:
            if not ZHIPUAI_AVAILABLE:
                logger.warning("检测到GLM模型，但zhipuai包未安装。请运行: pip install zhipuai")
                logger.warning("将使用OpenAI兼容模式...")
                return 'openai'
            return 'zhipuai'

        # 默认为OpenAI或兼容接口
        return 'openai'

    def _init_llm_client(self) -> Union[OpenAI, AzureOpenAI, 'ZhipuAI', 'CustomOpenAIClient']:
        """初始化LLM客户端

        支持OpenAI、Azure OpenAI、智谱AI和自定义API四种提供商

        Returns:
            对应提供商的客户端实例
        """
        model_config = self.config['model']

        if self.model_provider == 'azure':
            logger.info(f"使用Azure OpenAI客户端")
            # Azure OpenAI需要特定的参数
            azure_endpoint = model_config.get('azure_endpoint') or model_config.get('api_base')
            api_version = model_config.get('api_version', '2024-02-15-preview')

            if not azure_endpoint:
                raise ValueError("Azure OpenAI需要配置azure_endpoint或api_base参数")

            client = AzureOpenAI(
                api_key=model_config['api_key'],
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                timeout=model_config.get('timeout', 60)
            )
            logger.info(f"  - Endpoint: {azure_endpoint}")
            logger.info(f"  - API Version: {api_version}")
            logger.info(f"  - Deployment: {model_config.get('deployment_name') or model_config.get('model_name')}")
            return client

        elif self.model_provider == 'zhipuai':
            logger.info(f"使用智谱AI客户端: {model_config['model_name']}")
            client = ZhipuAI(
                api_key=model_config['api_key'],
                timeout=model_config['timeout']
            )
            return client

        elif self.model_provider == 'custom_api':
            logger.info(f"使用自定义API客户端: {model_config['model_name']}")

            # 获取URL参数配置
            url_params = model_config.get('url_params', {})

            client = CustomOpenAIClient(
                api_key=model_config['api_key'],
                base_url=model_config['api_base'],
                url_params=url_params,
                timeout=model_config.get('timeout', 60),
                max_retries=self.config['processing'].get('max_retries', 3),
                retry_delay=self.config['processing'].get('retry_delay', 1)
            )
            logger.info(f"  - Base URL: {model_config['api_base']}")
            logger.info(f"  - URL参数: {url_params}")
            return client

        else:
            logger.info(f"使用OpenAI兼容客户端: {model_config['model_name']}")
            client = OpenAI(
                api_key=model_config['api_key'],
                base_url=model_config['api_base'],
                timeout=model_config['timeout']
            )
            return client

    def _init_tokenizer(self):
        """初始化tokenizer用于计算文本token数量

        Returns:
            tiktoken编码器对象
        """
        try:
            model_name = self.config['model']['model_name']
            if 'gpt-4' in model_name.lower():
                return tiktoken.encoding_for_model("gpt-4")
            elif 'gpt-3.5' in model_name.lower():
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                return tiktoken.get_encoding("cl100k_base")
        except:
            return tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量

        Args:
            text: 输入文本

        Returns:
            token数量
        """
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text) // 4

    def _truncate_text(self, text: str) -> str:
        """截断超过最大长度限制的文本

        Args:
            text: 输入文本

        Returns:
            截断后的文本
        """
        token_count = self._count_tokens(text)
        if token_count <= self.max_text_length:
            return text

        sentences = re.split(r'[。！？\n]', text)
        truncated_text = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if current_tokens + sentence_tokens > self.max_text_length:
                break
            truncated_text += sentence + "。"
            current_tokens += sentence_tokens

        logger.warning(f"文本过长，已截断: {token_count} -> {current_tokens} tokens")
        return truncated_text

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """将长文本分割成多个带重叠的文本块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        token_count = self._count_tokens(text)

        # 如果文本足够短，不需要分块
        if token_count <= self.chunk_size:
            return [text]

        # 按句子分割文本
        sentences = re.split(r'([。！？\n])', text)
        # 重新组合句子和分隔符
        sentences_with_sep = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentences_with_sep.append(sentences[i] + sentences[i + 1])
            else:
                sentences_with_sep.append(sentences[i])
        if len(sentences) % 2 == 1:
            sentences_with_sep.append(sentences[-1])

        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_sentences = []  # 用于保存重叠部分的句子

        for sentence in sentences_with_sep:
            sentence_tokens = self._count_tokens(sentence)

            # 如果单个句子就超过chunk_size，强制添加
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(sentence)
                overlap_sentences = []
                continue

            # 如果添加这个句子会超过chunk_size，开始新的块
            if current_tokens + sentence_tokens > self.chunk_size:
                chunks.append(''.join(current_chunk))

                # 计算重叠部分：从当前块末尾选取overlap大小的句子
                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    sent_tokens = self._count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                # 开始新块，包含重叠部分
                current_chunk = overlap_sentences.copy()
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # 添加最后一个块
        if current_chunk:
            chunks.append(''.join(current_chunk))

        logger.info(f"文本分块完成: {token_count} tokens -> {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_tokens(chunk)
            logger.info(f"  块 {i+1}: {chunk_tokens} tokens")

        return chunks

    def _merge_chunk_results(self, chunk_results: List[List[Relation]]) -> List[Relation]:
        """合并多个文本块的抽取结果并去重

        Args:
            chunk_results: 各文本块的关系抽取结果列表

        Returns:
            去重后的关系列表
        """
        if not chunk_results:
            return []

        # 使用字典去重：(subject, relation, object) -> Relation
        relation_dict = {}

        for chunk_idx, relations in enumerate(chunk_results):
            for relation in relations:
                # 创建唯一键（忽略大小写）
                key = (
                    relation.subject.lower().strip(),
                    relation.relation.strip(),
                    relation.object.lower().strip()
                )

                # 如果是新的关系，直接添加
                if key not in relation_dict:
                    relation_dict[key] = relation
                else:
                    # 如果已存在，取置信度更高的或合并证据
                    existing = relation_dict[key]
                    if relation.confidence > existing.confidence:
                        relation_dict[key] = relation
                    elif relation.confidence == existing.confidence:
                        # 置信度相同，合并证据
                        if relation.evidence not in existing.evidence:
                            existing.evidence += "; " + relation.evidence
                            existing.evidence_spans.extend(relation.evidence_spans)

        merged_relations = list(relation_dict.values())
        total_before = sum(len(chunk) for chunk in chunk_results)
        logger.info(f"结果合并完成: {total_before} 个关系 -> {len(merged_relations)} 个去重后的关系")

        return merged_relations

    def _build_prompt(self, text: str) -> List[Dict[str, str]]:
        """构建LLM提示词消息列表

        Args:
            text: 待抽取的输入文本

        Returns:
            包含系统提示和用户提示的消息列表
        """
        prompts = self.config['prompts']
        advanced = self.config['advanced_techniques']

        messages = []

        # 系统提示词
        messages.append({
            "role": "system",
            "content": prompts['system_prompt']
        })

        # 构建用户提示词
        user_content = ""

        # 添加few-shot示例（如果启用）
        if advanced.get('enable_few_shot', False) and self.few_shot_manager:
            few_shot_text = self.few_shot_manager.get_few_shot_examples()
            if few_shot_text:
                user_content += few_shot_text + "\n\n"

        # 添加主要任务提示
        task_prompt = prompts['task_prompt']

        # 构建实体类型描述
        entity_types_desc = self._format_entity_types_desc()
        relation_types_desc = self._format_relation_types_desc()

        # 替换占位符
        task_prompt = task_prompt.replace('{{entity_types_desc}}', entity_types_desc)
        task_prompt = task_prompt.replace('{{relation_types_desc}}', relation_types_desc)

        user_content += task_prompt + "\n\n"

        # 添加待处理文本
        user_content += f"**待分析文本**：\n{text}\n\n请开始抽取实体、属性和关系："

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def _format_entity_types_desc(self) -> str:
        """格式化实体类型描述，包含属性定义

        Returns:
            格式化后的实体类型描述文本
        """
        descriptions = []
        for entity_type, config in self.entity_types.items():
            desc = f"- **{entity_type}**: {config.get('description', '')}"
            if 'examples' in config:
                examples = ', '.join(config['examples'][:3])
                desc += f" (例如: {examples})"

            # 添加属性定义
            if entity_type in self.entity_attributes:
                attrs = self.entity_attributes[entity_type]
                if attrs:
                    attr_list = []
                    for attr_name, attr_config in list(attrs.items())[:5]:  # 最多显示5个
                        attr_desc = f"{attr_name}"
                        if 'unit' in attr_config and attr_config['unit']:
                            attr_desc += f"({attr_config['unit']})"
                        attr_list.append(attr_desc)
                    desc += f"\n  可提取属性: {', '.join(attr_list)}"

            descriptions.append(desc)
        return '\n'.join(descriptions)

    def _format_relation_types_desc(self) -> str:
        """格式化关系类型描述

        Returns:
            格式化后的关系类型描述文本
        """
        descriptions = []
        for relation_type, config in self.relation_types.items():
            desc = f"- **{relation_type}**: {config.get('description', '')}"

            # 添加类型约束
            if 'subject_types' in config and 'object_types' in config:
                subject_types = ', '.join(config['subject_types'])
                object_types = ', '.join(config['object_types'])
                desc += f" (主体类型: {subject_types}; 客体类型: {object_types})"

            if 'examples' in config:
                examples = ', '.join(config['examples'][:2])
                desc += f" 例如: {examples}"

            descriptions.append(desc)
        return '\n'.join(descriptions)

    def _get_json_schema_for_api(self) -> Optional[Dict[str, Any]]:
        """获取用于API调用的JSON Schema配置

        Returns:
            适用于OpenAI或GLM API的response_format参数，未启用则返回None
        """
        if not self.json_schema_enabled or not self.json_schema:
            return None

        # 构建符合OpenAI Structured Outputs格式的schema
        # 参考: https://platform.openai.com/docs/guides/structured-outputs
        # 注意：由于 subject_attributes 和 object_attributes 需要支持动态键值对，
        # 我们不能使用 strict mode（strict mode 要求所有 object 都设置 additionalProperties: false）
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "knowledge_graph_extraction",
                "description": "从文本中抽取知识图谱三元组",
                "schema": self.json_schema,
                "strict": False  # 禁用严格模式以支持动态属性
            }
        }

    def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, int]]]:
        """调用大语言模型API

        Args:
            messages: LLM消息列表

        Returns:
            (响应文本, token使用统计)元组
        """
        model_config = self.config['model']
        max_retries = self.config['processing']['max_retries']
        retry_delay = self.config['processing']['retry_delay']

        # 准备API调用参数
        # Azure使用deployment_name，其他使用model_name
        if self.model_provider == 'azure':
            model_param = model_config.get('deployment_name') or model_config.get('model_name')
        else:
            model_param = model_config['model_name']

        api_params = {
            'model': model_param,
            'messages': messages,
            'temperature': model_config['temperature'],
            'max_tokens': model_config['max_tokens']
        }

        # 如果启用了JSON Schema，添加response_format参数
        if self.json_schema_enabled:
            response_format = self._get_json_schema_for_api()
            if response_format:
                api_params['response_format'] = response_format
                logger.info("已启用JSON Schema结构化输出模式")

        for attempt in range(max_retries):
            try:
                logger.info(f"正在调用LLM... (尝试 {attempt + 1}/{max_retries})")
                response = self.client.chat.completions.create(**api_params)

                # 优先检查是否有error字段（某些API返回error而不是抛出异常）
                if hasattr(response, 'error') and response.error:
                    error_msg = response.error.get('message', '未知错误')
                    error_code = response.error.get('code', '')
                    error_type = response.error.get('type', '')

                    logger.error("=" * 80)
                    logger.error(f"API返回错误:")
                    logger.error(f"错误消息: {error_msg}")
                    logger.error(f"错误代码: {error_code}")
                    logger.error(f"错误类型: {error_type}")
                    logger.error("=" * 80)

                    # 检查是否是余额不足
                    is_quota_error = any([
                        '余额不足' in error_msg,
                        'insufficient_quota' in error_type.lower(),
                        'quota_not_enough' in error_type.lower(),
                        error_code == '405',
                        'balance' in error_msg.lower() and 'insufficient' in error_msg.lower(),
                    ])

                    if is_quota_error:
                        logger.error("检测到余额不足，停止处理")
                        logger.error("请充值后重新运行，系统会自动从断点继续")
                        raise RuntimeError(f"API余额不足: {error_msg}")
                    else:
                        raise ValueError(f"API错误: {error_msg} (code: {error_code}, type: {error_type})")

                # 检查响应是否有效
                if not response or not hasattr(response, 'choices'):
                    logger.error("响应对象无效或没有choices属性")
                    logger.error(f"Response: {response}")
                    raise ValueError("Invalid response structure")

                if not response.choices or len(response.choices) == 0:
                    logger.error("choices列表为空")
                    raise ValueError("Empty choices in response")

                if not hasattr(response.choices[0], 'message'):
                    logger.error("choice没有message属性")
                    raise ValueError("No message in choice")

                content = response.choices[0].message.content

                # 检查content
                if content is None:
                    logger.error("message.content为None")
                    logger.error(f"Message object: {response.choices[0].message}")
                    raise ValueError("Content is None")

                token_usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                } if response.usage else None

                # 输出LLM原始响应内容
                logger.info("=" * 80)
                logger.info("LLM原始响应内容:")
                logger.info("=" * 80)
                logger.info(content)
                logger.info("=" * 80)
                if token_usage:
                    logger.info(f"Token使用: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}")
                logger.info("=" * 80)

                return content, token_usage

            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__

                logger.error("=" * 80)
                logger.error(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries})")
                logger.error(f"错误类型: {error_type}")
                logger.error(f"错误信息: {error_msg}")

                # 检测API超时错误
                if isinstance(e, APITimeoutError):
                    current_timeout = model_config['timeout']
                    logger.error("API调用超时！")
                    logger.error(f"当前超时设置: {current_timeout}秒")
                    logger.error("建议:")
                    logger.error(f"  1. 增加config.yaml中的timeout值 (当前{current_timeout}秒 → 建议300-600秒)")
                    logger.error(f"  2. 减少max_tokens参数 (当前{model_config.get('max_tokens', 'N/A')})")
                    logger.error(f"  3. 检查网络连接是否稳定")
                    logger.error("=" * 80)

                    if attempt < max_retries - 1:
                        logger.info(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error("已达到最大重试次数，放弃调用")
                        raise

                # 检测API欠费/余额不足错误
                # 常见的欠费错误标识：
                # 1. 错误消息中包含 "余额不足"、"insufficient_quota"、"quota_not_enough"
                # 2. 错误消息中包含 "balance"、"credit"
                is_quota_error = any([
                    '余额不足' in error_msg,
                    'insufficient_quota' in error_msg.lower(),
                    'quota_not_enough' in error_msg.lower(),
                    'insufficient funds' in error_msg.lower(),
                    'quota exceeded' in error_msg.lower(),
                    'balance' in error_msg.lower() and 'insufficient' in error_msg.lower(),
                    'out of credit' in error_msg.lower(),
                ])

                if is_quota_error:
                    logger.error("检测到API余额不足错误！")
                    logger.error("请充值后重新运行，系统会自动从断点继续")
                    logger.error("=" * 80)
                    raise RuntimeError(f"API余额不足: {error_msg}")

                import traceback
                logger.error(f"错误堆栈:\n{traceback.format_exc()}")
                logger.error("=" * 80)

                if attempt < max_retries - 1:
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error("已达到最大重试次数，放弃调用")
                    raise

    def _parse_llm_response(self, response: str) -> List[Triplet]:
        """解析LLM返回的JSON结果

        Args:
            response: LLM的响应文本

        Returns:
            解析出的关系三元组列表
        """
        triplets = []

        try:
            # 尝试多种方法提取JSON
            json_str = None

            # 方法1: 提取```json```代码块（使用更健壮的正则）
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("使用方法1提取JSON: ```json```代码块")
            else:
                # 方法2: 提取任何```代码块
                json_match = re.search(r'```\s*([\s\S]*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.info("使用方法2提取JSON: ```代码块")
                else:
                    # 方法3: 查找花括号包围的完整JSON对象（从第一个{到最后一个}）
                    # 先找到第一个包含"triplets"的JSON对象
                    start_idx = response.find('{')
                    if start_idx != -1:
                        # 使用栈来匹配括号，找到完整的JSON对象
                        bracket_count = 0
                        in_string = False
                        escape_next = False
                        end_idx = -1

                        for i in range(start_idx, len(response)):
                            char = response[i]

                            if escape_next:
                                escape_next = False
                                continue

                            if char == '\\':
                                escape_next = True
                                continue

                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue

                            if not in_string:
                                if char == '{':
                                    bracket_count += 1
                                elif char == '}':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end_idx = i + 1
                                        break

                        if end_idx != -1:
                            json_str = response[start_idx:end_idx]
                            logger.info("使用方法3提取JSON: 括号匹配")

                    if not json_str:
                        # 方法4: 尝试直接解析整个响应
                        json_str = response.strip()
                        logger.info("使用方法4提取JSON: 直接解析整个响应")

            # 清理JSON字符串
            if json_str:
                # 移除可能的换行和多余空格
                json_str = json_str.strip()
                # 尝试修复常见的JSON格式问题
                json_str = re.sub(r',\s*}', '}', json_str)  # 移除尾随逗号
                json_str = re.sub(r',\s*]', ']', json_str)  # 移除数组尾随逗号

            logger.info(f"提取的JSON字符串(前500字符): {json_str[:500] if json_str else 'None'}...")

            if not json_str:
                logger.error("无法从响应中提取JSON")
                logger.error(f"完整响应内容:\n{response}")
                return triplets

            data = json.loads(json_str)
            triplets_count = len(data.get('triplets', []))
            logger.info(f"JSON解析成功，发现 {triplets_count} 个三元组")

            # 如果没有三元组，记录信息（这是正常情况）
            if triplets_count == 0:
                logger.info("此文档未提取到任何三元组（可能文档不包含相关信息）")
                return triplets

            for item in data.get('triplets', []):
                # 检查关系和实体类型是否在配置中
                relation = item.get('relation', '').strip()
                subject_type = item.get('subject_type', '').strip()
                object_type = item.get('object_type', '').strip()

                is_relation_in_config = relation in self.relation_types_list
                is_subject_type_in_config = subject_type in self.entity_types_list
                is_object_type_in_config = object_type in self.entity_types_list

                # 解析evidence_spans
                evidence_spans = []
                for span_data in item.get('evidence_spans', []):
                    evidence_span = EvidenceSpan(
                        start=span_data.get('start', 0),
                        end=span_data.get('end', 0),
                        text=span_data.get('text', '')
                    )
                    evidence_spans.append(evidence_span)

                triplet = Triplet(
                    subject=item.get('subject', '').strip(),
                    subject_type=subject_type,
                    relation=relation,
                    object=item.get('object', '').strip(),
                    object_type=object_type,
                    confidence=float(item.get('confidence', 0.0)),
                    evidence=item.get('evidence', '').strip(),
                    evidence_spans=evidence_spans,
                    source_url="",  # 将在kg_builder中设置
                    is_relation_in_config=is_relation_in_config,
                    is_subject_type_in_config=is_subject_type_in_config,
                    is_object_type_in_config=is_object_type_in_config
                )

                # 验证三元组
                if self._validate_triplet(triplet):
                    triplets.append(triplet)
                else:
                    logger.warning(f"无效三元组被过滤: ({triplet.subject}, {triplet.relation}, {triplet.object})")

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"尝试解析的JSON字符串:\n{json_str}")
            logger.error(f"完整响应内容:\n{response}")

            # 尝试从响应中手动提取信息
            triplets = self._fallback_parse(response)

        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
            logger.error(f"完整响应内容:\n{response}")

        return triplets

    def _fallback_parse(self, response: str) -> List[Triplet]:
        """JSON解析失败时的备用文本解析方法

        Args:
            response: LLM的响应文本

        Returns:
            尝试解析出的关系三元组列表
        """
        triplets = []

        try:
            # 尝试从文本中提取三元组信息
            lines = response.split('\n')
            current_triplet = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 查找键值对
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().strip('"').strip("'")
                        value = parts[1].strip().strip(',').strip('"').strip("'")

                        if key in ['subject', 'subject_type', 'relation', 'object', 'object_type', 'evidence']:
                            current_triplet[key] = value
                        elif key == 'confidence':
                            try:
                                current_triplet[key] = float(value)
                            except:
                                current_triplet[key] = 0.5

                # 如果收集到完整的三元组信息
                if len(current_triplet) >= 6:
                    # 检查关系和实体类型是否在配置中
                    relation = current_triplet.get('relation', '')
                    subject_type = current_triplet.get('subject_type', '')
                    object_type = current_triplet.get('object_type', '')

                    is_relation_in_config = relation in self.relation_types_list
                    is_subject_type_in_config = subject_type in self.entity_types_list
                    is_object_type_in_config = object_type in self.entity_types_list

                    triplet = Triplet(
                        subject=current_triplet.get('subject', ''),
                        subject_type=subject_type,
                        relation=relation,
                        object=current_triplet.get('object', ''),
                        object_type=object_type,
                        confidence=current_triplet.get('confidence', 0.5),
                        evidence=current_triplet.get('evidence', ''),
                        evidence_spans=[],  # 备用解析无法获取位置信息
                        source_url="",
                        is_relation_in_config=is_relation_in_config,
                        is_subject_type_in_config=is_subject_type_in_config,
                        is_object_type_in_config=is_object_type_in_config
                    )

                    if self._validate_triplet(triplet):
                        triplets.append(triplet)

                    current_triplet = {}

            logger.info(f"备用解析方法提取到 {len(triplets)} 个三元组")

        except Exception as e:
            logger.error(f"备用解析也失败: {e}")

        return triplets

    def _validate_triplet(self, triplet: Triplet) -> bool:
        """验证关系三元组的有效性

        Args:
            triplet: 待验证的关系三元组

        Returns:
            是否有效
        """
        # 检查必填字段
        if not all([triplet.subject, triplet.relation, triplet.object]):
            return False

        # 检查关系类型（记录警告但不过滤）
        if triplet.relation not in self.relation_types_list:
            logger.info(f"发现非配置关系类型: {triplet.relation}")

        # 检查实体类型（记录警告但不过滤）
        if triplet.subject_type not in self.entity_types_list:
            logger.info(f"发现非配置实体类型: {triplet.subject_type}")

        if triplet.object_type not in self.entity_types_list:
            logger.info(f"发现非配置实体类型: {triplet.object_type}")

        # 检查置信度
        if triplet.confidence < self.confidence_threshold:
            return False

        return True

    def _self_consistency_check(self, text: str, triplets_list: List[List[Triplet]]) -> List[Triplet]:
        """通过多次抽取结果进行自我一致性检查

        Args:
            text: 原始文本
            triplets_list: 多次抽取的三元组结果列表

        Returns:
            一致性高的三元组列表
        """
        # 统计三元组出现频率
        triplet_counts = {}

        for triplets in triplets_list:
            for triplet in triplets:
                key = (triplet.subject, triplet.relation, triplet.object)
                if key not in triplet_counts:
                    triplet_counts[key] = {'count': 0, 'triplet': triplet, 'confidences': []}

                triplet_counts[key]['count'] += 1
                triplet_counts[key]['confidences'].append(triplet.confidence)

        # 选择一致性高的三元组
        consistent_triplets = []
        consistency_threshold = len(triplets_list) // 2 + 1  # 超过一半

        for key, data in triplet_counts.items():
            if data['count'] >= consistency_threshold:
                # 使用平均置信度
                avg_confidence = sum(data['confidences']) / len(data['confidences'])
                triplet = data['triplet']
                triplet.confidence = avg_confidence
                consistent_triplets.append(triplet)

        logger.info(f"自我一致性检查: {len(consistent_triplets)}/{len(triplet_counts)} 个三元组通过")
        return consistent_triplets

    def _derive_entities_from_relations(self, relations: List[Relation]) -> List[Entity]:
        """从关系列表中推导出实体列表

        Args:
            relations: 关系列表

        Returns:
            去重后的实体列表
        """
        entity_dict = {}  # 使用字典去重: (name, type) -> Entity

        for relation in relations:
            # 处理主实体
            subject_key = (relation.subject, relation.subject_type)
            if subject_key not in entity_dict:
                entity_dict[subject_key] = Entity(
                    name=relation.subject,
                    entity_type=relation.subject_type,
                    attributes=[],
                    source_url=relation.source_url,
                    doc_id=relation.doc_id,
                    is_type_in_config=relation.is_subject_type_in_config
                )

            # 处理客实体
            object_key = (relation.object, relation.object_type)
            if object_key not in entity_dict:
                entity_dict[object_key] = Entity(
                    name=relation.object,
                    entity_type=relation.object_type,
                    attributes=[],
                    source_url=relation.source_url,
                    doc_id=relation.doc_id,
                    is_type_in_config=relation.is_object_type_in_config
                )

        entities = list(entity_dict.values())
        logger.info(f"从 {len(relations)} 个关系中推导出 {len(entities)} 个去重实体")

        return entities

    def extract_from_text(self, text: str) -> ExtractionResult:
        """从文本中抽取实体和关系

        Args:
            text: 输入文本

        Returns:
            包含实体、关系、处理时间等信息的抽取结果
        """
        start_time = time.time()
        total_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        # 检查文本长度
        token_count = self._count_tokens(text)

        # 决定处理策略：分块 vs 截断 vs 直接处理
        if self.enable_chunking and token_count > self.chunk_size:
            # 策略1: 启用分块且文本过长 -> 分块处理
            logger.info(f"文本需要分块处理 ({token_count} tokens > {self.chunk_size})")
            chunks = self._split_text_into_chunks(text)
            all_chunk_relations = []

            # 对每个块进行处理
            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"处理块 {chunk_idx + 1}/{len(chunks)}")
                try:
                    chunk_relations, chunk_token_usage = self._extract_from_single_chunk(chunk)
                    all_chunk_relations.append(chunk_relations)

                    # 累计token使用量
                    if chunk_token_usage:
                        total_token_usage['prompt_tokens'] += chunk_token_usage.get('prompt_tokens', 0)
                        total_token_usage['completion_tokens'] += chunk_token_usage.get('completion_tokens', 0)
                        total_token_usage['total_tokens'] += chunk_token_usage.get('total_tokens', 0)

                except Exception as e:
                    logger.error(f"处理块 {chunk_idx + 1} 失败: {e}")
                    all_chunk_relations.append([])

            # 合并所有块的结果
            final_relations = self._merge_chunk_results(all_chunk_relations)

        elif not self.enable_chunking and token_count > self.max_text_length:
            # 策略2: 未启用分块但文本过长 -> 截断处理
            logger.warning(f"文本超过最大长度限制，将进行截断 ({token_count} tokens > {self.max_text_length})")
            processed_text = self._truncate_text(text)
            try:
                final_relations, token_usage = self._extract_from_single_chunk(processed_text)
                if token_usage:
                    total_token_usage = token_usage
            except Exception as e:
                logger.error(f"抽取失败: {e}")
                final_relations = []

        else:
            # 策略3: 文本长度合适 -> 直接处理
            try:
                final_relations, token_usage = self._extract_from_single_chunk(text)
                if token_usage:
                    total_token_usage = token_usage
            except Exception as e:
                logger.error(f"抽取失败: {e}")
                final_relations = []

        # 从关系中推导出实体
        final_entities = self._derive_entities_from_relations(final_relations)

        processing_time = time.time() - start_time

        logger.info(f"抽取完成: {len(final_entities)} 个实体, {len(final_relations)} 个关系, 耗时: {processing_time:.2f}s")

        return ExtractionResult(
            text=text,
            entities=final_entities,
            relations=final_relations,
            triplets=None,  # 将由__post_init__自动设置为relations
            processing_time=processing_time,
            token_usage=total_token_usage if total_token_usage['total_tokens'] > 0 else None
        )

    def _extract_from_single_chunk(self, text: str) -> Tuple[List[Relation], Optional[Dict[str, int]]]:
        """从单个文本块中抽取关系

        Args:
            text: 文本块

        Returns:
            (关系列表, token使用统计)元组
        """
        # 构建提示词
        messages = self._build_prompt(text)

        # 自我一致性检查
        if self.config['advanced_techniques'].get('enable_self_consistency', False):
            consistency_count = self.config['advanced_techniques']['consistency_count']
            all_triplets = []

            for i in range(consistency_count):
                logger.info(f"自我一致性抽取 {i+1}/{consistency_count}")
                try:
                    response, token_usage = self._call_llm(messages)
                    triplets = self._parse_llm_response(response)
                    all_triplets.append(triplets)
                except Exception as e:
                    logger.error(f"自我一致性抽取失败: {e}")

            if all_triplets:
                final_relations = self._self_consistency_check(text, all_triplets)
            else:
                final_relations = []
                token_usage = None
        else:
            # 单次抽取
            try:
                response, token_usage = self._call_llm(messages)
                final_relations = self._parse_llm_response(response)
            except Exception as e:
                logger.error(f"抽取失败: {e}")
                final_relations = []
                token_usage = None

        return final_relations, token_usage
