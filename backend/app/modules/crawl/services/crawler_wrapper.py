"""
爬虫包装器
封装爬虫逻辑，适配后端服务
"""
from typing import List, Dict, Any
import logging

from ..crawler import SmartCrawler

logger = logging.getLogger(__name__)


class CrawlerWrapper:
    """爬虫包装器,用于拦截结果并转换为字典格式"""

    def __init__(self, config_path: str):
        """
        初始化爬虫

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._crawler = None

    async def initialize(self):
        """异步初始化爬虫"""
        try:
            self._crawler = SmartCrawler(self.config_path)
            logger.info(f"爬虫初始化成功: {self.config_path}")
        except Exception as e:
            logger.error(f"爬虫初始化失败: {e}")
            raise

    async def run(self) -> List[Dict[str, Any]]:
        """
        执行爬取任务

        Returns:
            List[Dict]: 爬取结果列表
        """
        if not self._crawler:
            await self.initialize()

        try:
            # 临时替换存储逻辑以拦截结果
            original_storage = self._crawler.storage

            class ResultCollector:
                """结果收集器"""
                def __init__(self):
                    self.collected_results = []

                async def save(self, task_name: str, task_results: List):
                    """收集结果而不是保存到文件"""
                    for result in task_results:
                        # 将CrawlResult dataclass转换为字典
                        if hasattr(result, '__dict__'):
                            result_dict = {
                                'url': result.url,
                                'original_url': result.original_url,
                                'content': result.content,
                                'timestamp': result.timestamp,
                                'new_tab': result.new_tab,
                                'strategy_used': result.strategy_used.value if hasattr(result, 'strategy_used') and result.strategy_used else None,
                                'button_info': result.button_info
                            }
                        else:
                            result_dict = result

                        self.collected_results.append(result_dict)

            collector = ResultCollector()
            self._crawler.storage = collector

            # 执行爬取
            await self._crawler.start()

            results = collector.collected_results

            # 恢复原始存储
            self._crawler.storage = original_storage

            logger.info(f"爬取完成，共{len(results)}条结果")
            return results

        except Exception as e:
            logger.error(f"爬取执行失败: {e}")
            raise

    async def close(self):
        """关闭爬虫"""
        if self._crawler and hasattr(self._crawler, 'browser_manager'):
            await self._crawler.browser_manager.close()
