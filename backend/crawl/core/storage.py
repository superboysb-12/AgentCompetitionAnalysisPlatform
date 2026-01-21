"""
存储模块
提供多种存储方式保存爬取结果
"""

from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from crawl.core.types import CrawlResult
import json
import logging

logger = logging.getLogger(__name__)


class JSONStorage:
    """
    JSON存储类
    将爬取结果保存为JSON格式文件
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化JSON存储器

        Args:
            config: 全局配置字典，包含输出目录设置
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'results'))
        # 创建输出目录（如果不存在）
        self.output_dir.mkdir(exist_ok=True)

    async def save(self, task_name: str, results: List[CrawlResult]) -> None:
        """
        保存爬取结果到JSON文件

        Args:
            task_name: 任务名称，用于生成文件名
            results: 爬取结果列表

        Returns:
            None
        """
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{task_name}_clean_{timestamp}.json"
        filepath = self.output_dir / filename

        # 将结果对象序列化为可JSON化的字典
        serializable_results = [self._serialize_result(result) for result in results]

        # 构建完整的数据结构
        data = {
            'task_name': task_name,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'successful_results': len([r for r in results if r.content]),
            'results': serializable_results
        }

        # 写入JSON文件（UTF-8编码，带缩进）
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _serialize_result(self, result: CrawlResult) -> Dict[str, Any]:
        """
        序列化单个爬取结果为字典

        Args:
            result: 爬取结果对象

        Returns:
            Dict[str, Any]: 序列化后的字典
        """
        return {
            'url': result.url,
            'original_url': result.original_url,
            'content': result.content,
            'timestamp': result.timestamp,
            'new_tab': result.new_tab,
            'strategy_used': result.strategy_used.value if result.strategy_used else None,
            'button_info': {
                'text': result.button_info.get('text', ''),
                'selector': result.button_info.get('selector', ''),
                'href': result.button_info.get('href')
            }
        }


class StorageFactory:
    """
    存储工厂类
    根据配置创建相应的存储实例
    """

    @staticmethod
    def create_storage(storage_type: str, config: Dict[str, Any]):
        """
        创建存储实例

        Args:
            storage_type: 存储类型（支持 'json' 和 'mysql'）
            config: 配置字典

        Returns:
            Storage: 存储实例

        Raises:
            ValueError: 当存储类型不支持时抛出异常
        """
        if storage_type == 'json':
            return JSONStorage(config)
        elif storage_type == 'mysql':
            # 动态导入 MySQLStorage（避免循环依赖）
            from mysql.storage import MySQLStorage
            return MySQLStorage(config)
        else:
            raise ValueError(f"不支持的存储类型: {storage_type}，支持的类型: json, mysql")
