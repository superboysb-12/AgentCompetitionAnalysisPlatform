import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FewShotManager:
    """Few-shot示例管理器，负责从配置文件中提取和格式化Few-shot学习示例

    此类已简化，不再包含硬编码示例。所有示例应从配置文件的 prompts.few_shot_examples 字段读取。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化few-shot管理器，参数config为配置字典

        Args:
            config: 完整的配置字典，应包含 prompts.few_shot_examples 字段
        """
        self.config = config
        self.few_shot_examples_text = config.get('prompts', {}).get('few_shot_examples', '')
        logger.info("Few-shot管理器初始化完成")

    def get_few_shot_examples(self) -> str:
        """获取few-shot示例文本，直接从配置文件返回

        Returns:
            str: 格式化的few-shot示例文本
        """
        return self.few_shot_examples_text if self.few_shot_examples_text else ""