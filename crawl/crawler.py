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
from core.types import CrawlResult, ButtonInfo, ClickStrategy, CrawlMode
from config.manager import ConfigManager
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class SmartCrawler:
    """智能爬虫主引擎"""

    def __init__(self, config_path: str):
        # 加载配置文件并解析YAML
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # 初始化所有核心组件，遵循依赖注入原则
        self.browser_manager = BrowserManager(self.config['settings'])  # 浏览器生命周期管理
        self.button_discovery = ButtonDiscovery()  # 页面按钮发现器
        self.click_manager = ClickStrategyManager()  # 点击策略管理器
        self.content_extractor = ContentExtractor()  # 内容提取器
        self.page_detector = PageLoadDetector()  # 页面加载状态检测器

        # 根据配置创建存储实例，默认使用JSON存储
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
        """执行单个爬取任务

        根据配置的模式选择不同的爬取策略：
        - direct: 直接爬取页面内容
        - indirect: 点击按钮后爬取新页面内容
        """
        task_name = task_config['name']

        # 获取爬取模式，默认为间接模式（保持向后兼容）
        crawl_mode = task_config.get('mode', 'indirect')

        if crawl_mode == 'direct':
            await self._execute_direct_crawl(task_config)
        else:
            await self._execute_indirect_crawl(task_config)

    async def _execute_direct_crawl(self, task_config: Dict[str, Any]) -> None:
        """执行直接爬取模式

        工作流程：
        1. 访问目标页面
        2. 直接提取页面内容
        3. 保存结果
        """
        task_name = task_config['name']

        page = await self.browser_manager.create_page()

        try:
            # 访问目标页面
            await page.goto(
                task_config['start_url'],
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"
            )
            # 等待页面完全加载
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # 直接提取页面内容
            content = await self.content_extractor.extract(page, task_config['content_extraction'])

            # 创建结果
            result = CrawlResult(
                url=page.url,
                original_url=task_config['start_url'],
                content=content,
                timestamp=datetime.now().isoformat(),
                new_tab=False,
                button_info={'mode': 'direct'}
            )

            # 保存结果
            await self.storage.save(task_name, [result])
            logger.info(f"直接爬取完成: {page.url}")

        except Exception as e:
            logger.error(f"直接爬取任务 {task_name} 失败: {e}")
        finally:
            await page.close()

    async def _execute_indirect_crawl(self, task_config: Dict[str, Any]) -> None:
        """执行间接爬取模式（原有逻辑）

        工作流程：
        1. 访问起始页面
        2. 发现指定的可点击按钮
        3. 对每个按钮执行点击策略
        4. 提取内容并保存结果
        5. 清理标签页回到主页面
        """
        task_name = task_config['name']

        # 创建主页面实例
        page = await self.browser_manager.create_page()
        main_page = page  # 保持主页面引用用于后续返回

        try:
            # 访问起始页面 - 使用domcontentloaded策略加快加载速度
            await page.goto(
                task_config['start_url'],
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"  # 不等待所有资源，只等DOM加载完成
            )
            # 智能等待页面完全加载（包括JavaScript渲染）
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # 根据配置的CSS选择器发现指定的可点击按钮
            buttons = await self.button_discovery.discover_buttons(page, task_config['button_discovery'])

            # 遍历处理每个发现的按钮
            results = []
            visited_urls = set()  # 用于URL去重，避免重复爬取相同页面

            for i, button in enumerate(buttons):
                logger.info(f"处理按钮 {i+1}/{len(buttons)}: {button.text[:50]}...")

                # 执行按钮点击和内容提取
                result = await self._process_button(page, button, task_config)

                # 检查结果有效性和URL去重
                if result and result.url not in visited_urls:
                    results.append(result)
                    visited_urls.add(result.url)
                    logger.info(f"成功爬取: {result.url}")
                elif result and result.url in visited_urls:
                    logger.info(f"跳过重复URL: {result.url}")

                # 清理新打开的标签页并返回主页面，准备处理下一个按钮
                await self._cleanup_and_return(main_page, task_config)

                # 在处理下一个按钮前等待，避免过于频繁的请求
                if i < len(buttons) - 1:
                    await asyncio.sleep(10)  # 10秒间隔

            # 保存结果
            await self.storage.save(task_name, results)

        except Exception as e:
            logger.error(f"任务 {task_name} 执行失败: {e}")
        finally:
            await page.close()

    async def _process_button(self, page, button: ButtonInfo, task_config: Dict[str, Any]) -> CrawlResult:
        """处理单个按钮的点击和内容提取

        流程：
        1. 根据按钮特征选择合适的点击策略
        2. 执行点击并处理可能的标签页跳转
        3. 等待新页面完全加载
        4. 提取配置中定义的内容字段
        """
        try:
            original_url = page.url  # 保存原始URL用于比较

            # 使用策略管理器执行最适合的点击策略
            result_page = await self.click_manager.execute_click(page, button, task_config['browser'])

            # 如果返回了新页面（如新标签页），切换到新页面
            if result_page != page:
                page = result_page

            # 智能等待新页面完全加载（包括动态内容）
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # 根据配置的CSS选择器提取页面内容
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
        """清理多余标签页并返回主页面

        这个方法确保：
        1. 关闭所有新打开的标签页，避免内存泄漏
        2. 将焦点返回到主页面
        3. 重新导航到起始URL，准备处理下一个按钮
        """
        try:
            # 关闭除主页面外的所有标签页
            await self.browser_manager.cleanup_extra_tabs(main_page)
            # 将焦点切换到主页面
            await main_page.bring_to_front()

            # 重新导航回起始页面 - 使用较长超时时间应对网络延迟
            await main_page.goto(
                task_config['start_url'],
                timeout=120000,  # 2分钟超时
                wait_until="domcontentloaded"  # 快速加载策略
            )
            # 等待页面完全加载和JavaScript执行
            await self.page_detector.wait_for_load(main_page, task_config['browser'])
        except Exception as e:
            logger.warning(f"清理和返回主页面失败: {e}")
            # 如果导航失败，尝试重新加载页面作为备选方案
            try:
                await main_page.reload(timeout=60000, wait_until="domcontentloaded")
            except Exception as reload_error:
                logger.error(f"页面重新加载也失败: {reload_error}")