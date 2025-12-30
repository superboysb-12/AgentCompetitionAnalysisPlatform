"""
智能爬虫主模块
核心爬虫类，支持两种爬取模式（direct、indirect）
整合按钮发现、点击策略、内容提取等组件，实现灵活的网页爬取
"""

from typing import List, Dict, Any
from playwright.async_api import Page
from crawl.core.browser import BrowserManager
from crawl.core.discovery import ButtonDiscovery
from crawl.core.strategies import ClickStrategyManager
from crawl.core.extractor import ContentExtractor
from crawl.core.detector import PageLoadDetector
from crawl.core.storage import StorageFactory
from crawl.core.types import CrawlResult, ButtonInfo, ClickStrategy, CrawlMode
from crawl.config.manager import ConfigManager
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class SmartCrawler:
    """
    智能爬虫类
    支持两种爬取模式：
    1. direct: 直接爬取目标页面
    2. indirect: 通过点击按钮间接进入内容页
    """

    def __init__(self, config_path: str):
        """
        初始化智能爬虫

        Args:
            config_path: YAML配置文件路径
        """
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # 初始化各个组件
        self.browser_manager = BrowserManager(self.config['settings'])
        self.button_discovery = ButtonDiscovery()
        self.click_manager = ClickStrategyManager()
        self.content_extractor = ContentExtractor()
        self.page_detector = PageLoadDetector()

        # 创建存储实例
        storage_type = self.config['settings'].get('storage_type', 'json')
        self.storage = StorageFactory.create_storage(storage_type, self.config['settings'])

        # 统计总保存数量
        self.total_saved_count = 0

    async def start(self) -> int:
        """
        启动爬虫

        初始化浏览器并执行所有配置的任务

        Returns:
            int: 成功保存到数据库的记录总数
        """
        try:
            # 初始化浏览器
            await self.browser_manager.initialize()

            # 执行每个任务
            for task_config in self.config['tasks']:
                await self._execute_task(task_config)

        finally:
            # 确保关闭浏览器
            await self.browser_manager.close()

            # 关闭存储连接池
            if hasattr(self.storage, 'close'):
                self.storage.close()
                logger.info("存储资源已清理")

        return self.total_saved_count

    async def _execute_task(self, task_config: Dict[str, Any]) -> None:
        """
        根据模式执行任务

        Args:
            task_config: 任务配置字典

        Returns:
            None
        """
        task_name = task_config['name']

        # 获取爬取模式，默认为indirect
        crawl_mode = task_config.get('mode', 'indirect')

        # 根据模式选择执行方法
        if crawl_mode == 'direct':
            await self._execute_direct_crawl(task_config)
        else:  # indirect
            await self._execute_indirect_crawl(task_config)

    async def _execute_direct_crawl(self, task_config: Dict[str, Any]) -> None:
        """
        执行直接爬取模式（支持 URL 去重）

        直接访问目标URL并提取内容，不涉及按钮点击
        爬取前检查URL是否已存在于数据库中

        Args:
            task_config: 任务配置字典

        Returns:
            None
        """
        task_name = task_config['name']
        target_url = task_config['start_url']

        # 创建新页面
        page = await self.browser_manager.create_page()

        try:
            # ===== URL 去重检查 =====
            if hasattr(self.storage, 'url_exists'):
                if await self.storage.url_exists(target_url):
                    logger.info(f"⊗ URL 已存在于数据库，跳过爬取: {target_url}")
                    return
                else:
                    logger.info(f"→ URL 未存在，开始爬取: {target_url}")
            # ========================

            # 导航到目标URL
            await page.goto(
                target_url,
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"
            )
            # 等待页面加载完成
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # 提取内容
            content = await self.content_extractor.extract(page, task_config['content_extraction'])

            # 创建爬取结果
            result = CrawlResult(
                url=page.url,
                original_url=task_config['start_url'],
                content=content,
                timestamp=datetime.now().isoformat(),
                new_tab=False,
                button_info={'mode': 'direct'}
            )

            # 保存结果
            saved_count = await self.storage.save(task_name, [result])
            self.total_saved_count += saved_count
            logger.info(f"✓ 直接爬取完成: {page.url}")

        except Exception as e:
            logger.error(f"✗ 直接爬取任务 {task_name} 失败: {e}")
        finally:
            await page.close()

    async def _execute_indirect_crawl(self, task_config: Dict[str, Any]) -> None:
        """
        执行间接爬取模式（支持 URL 去重和实时保存）

        先发现页面上的按钮，然后依次点击并提取内容
        每次爬取成功后立即保存到数据库（实时保存）
        支持批量检查 URL 优化性能

        Args:
            task_config: 任务配置字典

        Returns:
            None
        """
        task_name = task_config['name']

        # 创建页面
        page = await self.browser_manager.create_page()
        main_page = page

        try:
            # 导航到起始URL
            await page.goto(
                task_config['start_url'],
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"
            )
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # 发现按钮
            buttons = await self.button_discovery.discover_buttons(page, task_config['button_discovery'])
            logger.info(f"发现 {len(buttons)} 个按钮")

            # ===== 批量检查优化：提取所有 href，批量查询已存在的 URL =====
            existing_urls = set()  # 初始化在 if 之外，避免作用域问题
            if hasattr(self.storage, 'batch_check_urls'):
                # 提取所有有 href 的按钮 URL
                button_urls = [btn.href for btn in buttons if btn.href]
                if button_urls:
                    existing_urls = await self.storage.batch_check_urls(button_urls)
                    logger.info(f"批量检查完成，{len(existing_urls)} 个 URL 已存在")
            # ================================================================

            success_count = 0
            skip_count = 0
            visited_urls = set()  # 本次运行的内存去重

            # 处理每个按钮
            for i, button in enumerate(buttons):
                logger.info(f"处理按钮 {i+1}/{len(buttons)}: {button.text[:50]}...")

                # 如果 href 在批量检查中已存在，直接跳过
                if button.href and button.href in existing_urls:
                    logger.info(f"⊗ URL 已存在（批量检查），跳过按钮")
                    skip_count += 1
                    continue

                # 点击按钮并提取内容
                result = await self._process_button(page, button, task_config)

                # 检查结果
                if result:
                    # 内存去重检查
                    if result.url not in visited_urls:
                        visited_urls.add(result.url)

                        # ===== 实时保存：立即保存到数据库 =====
                        saved_count = await self.storage.save(task_name, [result])
                        self.total_saved_count += saved_count
                        success_count += 1
                        logger.info(f"✓ 成功爬取并保存 ({success_count}): {result.url}")
                        # =====================================
                    else:
                        logger.info(f"⊗ 跳过重复 URL (内存去重): {result.url}")
                        skip_count += 1
                else:
                    # result 为 None，可能是 URL 已存在或处理失败
                    skip_count += 1

                # 清理并返回主页面
                await self._cleanup_and_return(main_page, task_config)

                # 按钮间等待（避免过快请求）
                if i < len(buttons) - 1:
                    button_interval = task_config['browser'].get('button_interval', 10)
                    await asyncio.sleep(button_interval)

            logger.info(f"✓ 任务 {task_name} 完成！成功: {success_count}，跳过: {skip_count}，总计: {len(buttons)}")

        except Exception as e:
            logger.error(f"✗ 任务 {task_name} 执行失败: {e}")
        finally:
            await page.close()

    async def _process_button(self, page, button: ButtonInfo, task_config: Dict[str, Any]) -> CrawlResult:
        """
        处理单个按钮的点击和内容提取（支持 URL 去重）

        两次去重检查：
        1. 如果按钮有 href，先检查 href 是否已存在
        2. 点击后，检查实际 URL 是否已存在

        Args:
            page: 当前页面对象
            button: 按钮信息
            task_config: 任务配置字典

        Returns:
            CrawlResult: 爬取结果对象，失败或 URL 已存在时返回 None
        """
        try:
            original_url = page.url

            # ===== 第一次检查：如果有 href，先检查是否已存在 =====
            if button.href and hasattr(self.storage, 'url_exists'):
                if await self.storage.url_exists(button.href):
                    logger.info(f"⊗ URL 已存在（按钮 href），跳过: {button.href}")
                    return None
                else:
                    logger.info(f"→ href 未存在，准备点击: {button.href}")
            # ====================================================

            # 使用策略管理器执行点击
            result_page = await self.click_manager.execute_click(page, button, task_config['browser'])

            # 更新页面引用
            if result_page != page:
                page = result_page

            # 等待页面加载
            await self.page_detector.wait_for_load(page, task_config['browser'])

            # ===== 第二次检查：点击后检查实际 URL =====
            if hasattr(self.storage, 'url_exists'):
                if await self.storage.url_exists(page.url):
                    logger.info(f"⊗ URL 已存在（实际 URL），跳过: {page.url}")
                    return None
                else:
                    logger.info(f"→ 实际 URL 未存在，开始提取内容: {page.url}")
            # ==========================================

            # 提取内容
            content = await self.content_extractor.extract(page, task_config['content_extraction'])

            # 创建爬取结果
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
            logger.error(f"✗ 按钮处理失败: {e}")
            return None

    async def _cleanup_and_return(self, main_page, task_config: Dict[str, Any]) -> None:
        """
        清理额外标签页并返回主页面

        关闭所有额外打开的标签页，重新导航到起始URL

        Args:
            main_page: 主页面对象
            task_config: 任务配置字典

        Returns:
            None
        """
        try:
            # 清理额外的标签页
            await self.browser_manager.cleanup_extra_tabs(main_page)
            # 将主页面置于前台
            await main_page.bring_to_front()

            # 重新导航到起始URL
            await main_page.goto(
                task_config['start_url'],
                timeout=120000,
                wait_until="domcontentloaded"
            )
            await self.page_detector.wait_for_load(main_page, task_config['browser'])
        except Exception as e:
            logger.warning(f"清理和返回主页面失败: {e}")
            try:
                # 如果导航失败，尝试重新加载
                await main_page.reload(timeout=60000, wait_until="domcontentloaded")
            except Exception as reload_error:
                logger.error(f"页面重新加载也失败: {reload_error}")
