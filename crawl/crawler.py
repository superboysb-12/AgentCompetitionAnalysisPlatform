"""
Main Crawler Engine

重构后的主爬虫引擎，采用清洁架构
"""

from typing import List, Dict, Any
from core.browser import BrowserManager
from core.discovery import ButtonDiscovery
from core.strategies import ClickStrategyManager
from core.extractor import ContentExtractor
from core.detector import PageLoadDetector
from core.storage import StorageFactory
from core.types import CrawlResult, ButtonInfo, ClickStrategy
from config.manager import ConfigManager
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class SmartCrawler:
    """智能爬虫主引擎"""

    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # 初始化组件
        self.browser_manager = BrowserManager(self.config['settings'])
        self.button_discovery = ButtonDiscovery()
        self.click_manager = ClickStrategyManager()
        self.content_extractor = ContentExtractor()
        self.page_detector = PageLoadDetector()

        # 初始化存储
        storage_type = self.config['settings'].get('storage_type', 'json')
        self.storage = StorageFactory.create_storage(storage_type, self.config['settings'])

    async def start(self) -> None:
        """启动爬虫"""
        try:
            await self.browser_manager.initialize()

            for task_config in self.config['tasks']:
                await self._execute_task(task_config)

        finally:
            await self.browser_manager.close()

    async def _execute_task(self, task_config: Dict[str, Any]) -> None:
        """执行单个爬取任务"""
        task_name = task_config['name']

        page = await self.browser_manager.create_page()
        main_page = page

        try:
            # 访问起始页面 - 使用更快的加载策略
            await page.goto(
                task_config['start_url'],
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"
            )
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # 发现按钮
            buttons = await self.button_discovery.discover_buttons(page, task_config['button_discovery'])

            # 处理每个按钮
            results = []
            visited_urls = set()  # 用于URL去重

            for i, button in enumerate(buttons):
                logger.info(f"处理按钮 {i+1}/{len(buttons)}: {button.text[:50]}...")

                result = await self._process_button(page, button, task_config)
                if result and result.url not in visited_urls:
                    results.append(result)
                    visited_urls.add(result.url)
                    logger.info(f"成功爬取: {result.url}")
                elif result and result.url in visited_urls:
                    logger.info(f"跳过重复URL: {result.url}")

                # 清理标签页并返回主页面
                await self._cleanup_and_return(main_page, task_config)

                # 操作间隔
                if i < len(buttons) - 1:
                    await asyncio.sleep(10)

            # 保存结果
            await self.storage.save(task_name, results)

        except Exception as e:
            logger.error(f"任务 {task_name} 执行失败: {e}")
        finally:
            await page.close()

    async def _process_button(self, page, button: ButtonInfo, task_config: Dict[str, Any]) -> CrawlResult:
        """处理单个按钮点击"""
        try:
            original_url = page.url

            # 执行点击策略
            result_page = await self.click_manager.execute_click(page, button, task_config['browser'])

            if result_page != page:
                page = result_page

            # 等待新页面加载
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # 提取内容
            content = await self.content_extractor.extract(page, task_config['content_extraction'])

            return CrawlResult(
                url=page.url,
                original_url=original_url,
                content=content,
                timestamp=datetime.now().isoformat(),
                new_tab=page.url != original_url,
                button_info={
                    'text': button.text,
                    'selector': button.selector,
                    'href': button.href
                }
            )

        except Exception as e:
            logger.error(f"按钮处理失败: {e}")
            return None

    async def _cleanup_and_return(self, main_page, task_config: Dict[str, Any]) -> None:
        """清理标签页并返回主页面 - 优化导航和超时"""
        try:
            await self.browser_manager.cleanup_extra_tabs(main_page)
            await main_page.bring_to_front()

            # 使用更快的导航方式和更长的超时
            await main_page.goto(
                task_config['start_url'],
                timeout=120000,  # 增加到2分钟
                wait_until="domcontentloaded"  # 更快的加载策略
            )
            await self.page_detector.wait_for_load(main_page, task_config['browser'])
        except Exception as e:
            logger.warning(f"清理和返回主页面失败: {e}")
            # 如果导航失败，尝试重新加载页面
            try:
                await main_page.reload(timeout=60000, wait_until="domcontentloaded")
            except Exception as reload_error:
                logger.error(f"页面重新加载也失败: {reload_error}")