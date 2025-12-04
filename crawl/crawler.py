from typing import List, Dict, Any
from playwright.async_api import Page
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

    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        self.browser_manager = BrowserManager(self.config['settings'])
        self.button_discovery = ButtonDiscovery()
        self.click_manager = ClickStrategyManager()
        self.content_extractor = ContentExtractor()
        self.page_detector = PageLoadDetector()

        storage_type = self.config['settings'].get('storage_type', 'json')
        self.storage = StorageFactory.create_storage(storage_type, self.config['settings'])

    async def start(self) -> None:
        try:
            await self.browser_manager.initialize()

            for task_config in self.config['tasks']:
                await self._execute_task(task_config)

        finally:
            await self.browser_manager.close()

    async def _execute_task(self, task_config: Dict[str, Any]) -> None:
        task_name = task_config['name']

        crawl_mode = task_config.get('mode', 'indirect')

        if crawl_mode == 'direct':
            await self._execute_direct_crawl(task_config)
        elif crawl_mode == 'multistep':
            await self._execute_multistep_crawl(task_config)
        else:
            await self._execute_indirect_crawl(task_config)

    async def _execute_direct_crawl(self, task_config: Dict[str, Any]) -> None:
        task_name = task_config['name']

        page = await self.browser_manager.create_page()

        try:
            await page.goto(
                task_config['start_url'],
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"
            )
            await self.page_detector.wait_for_load(page, task_config['browser'])

            content = await self.content_extractor.extract(page, task_config['content_extraction'])

            result = CrawlResult(
                url=page.url,
                original_url=task_config['start_url'],
                content=content,
                timestamp=datetime.now().isoformat(),
                new_tab=False,
                button_info={'mode': 'direct'}
            )

            await self.storage.save(task_name, [result])
            logger.info(f"直接爬取完成: {page.url}")

        except Exception as e:
            logger.error(f"直接爬取任务 {task_name} 失败: {e}")
        finally:
            await page.close()

    async def _execute_multistep_crawl(self, task_config: Dict[str, Any]) -> None:
        task_name = task_config['name']
        page = await self.browser_manager.create_page()
        all_results = []

        try:
            await page.goto(
                task_config['start_url'],
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"
            )
            await self.page_detector.wait_for_load(page, task_config['browser'])

            operation_sequence = task_config.get('operation_sequence', [])
            logger.info(f"开始执行 {len(operation_sequence)} 步操作序列")

            for i, step_config in enumerate(operation_sequence):
                step_num = step_config.get('step', i + 1)
                action = step_config.get('action')
                description = step_config.get('description', f'第{step_num}步')

                logger.info(f"执行步骤 {step_num}: {description}")

                if action == 'click':
                    await self._execute_click_step(page, step_config)
                elif action == 'extract':
                    result = await self._execute_extract_step(page, task_config, task_name, step_config)
                    if result:
                        all_results.append(result)
                elif action == 'wait':
                    wait_time = step_config.get('wait_time', 1)
                    logger.info(f"等待 {wait_time} 秒")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"未知操作类型: {action}")

                wait_after = step_config.get('wait_after', 1)
                if wait_after > 0:
                    logger.info(f"步骤后等待 {wait_after} 秒")
                    await asyncio.sleep(wait_after)

            if all_results:
                await self.storage.save(task_name, all_results)
                logger.info(f"多步操作完成，共提取 {len(all_results)} 个结果: {task_name}")

        except Exception as e:
            logger.error(f"多步操作任务 {task_name} 失败: {e}")
        finally:
            await page.close()

    async def _execute_click_step(self, page: Page, step_config: Dict[str, Any]) -> None:
        selector = step_config.get('selector')
        if not selector:
            logger.error("点击步骤缺少selector配置")
            return

        try:
            locator = page.locator(selector)
            await locator.wait_for(state='visible', timeout=10000)

            if await locator.is_enabled():
                await locator.click()
                logger.info(f"成功点击元素: {selector}")
            else:
                logger.warning(f"元素不可点击: {selector}")

        except Exception as e:
            logger.error(f"点击操作失败 {selector}: {e}")

    async def _execute_extract_step(self, page: Page, task_config: Dict[str, Any], task_name: str, step_config: Dict[str, Any]) -> CrawlResult:
        try:
            await asyncio.sleep(2)

            content = await self.content_extractor.extract(page, task_config['content_extraction'])

            save_id = step_config.get('save_id', f"{task_name}_step_{step_config.get('step', 'unknown')}")
            description = step_config.get('description', '内容提取')

            result = CrawlResult(
                url=page.url,
                original_url=task_config['start_url'],
                content=content,
                timestamp=datetime.now().isoformat(),
                new_tab=False,
                button_info={
                    'mode': 'multistep',
                    'description': description,
                    'save_id': save_id,
                    'step': step_config.get('step', 'unknown')
                }
            )

            logger.info(f"内容提取完成 [{save_id}]: {page.url}")
            return result

        except Exception as e:
            logger.error(f"内容提取失败: {e}")
            return None


    async def _execute_indirect_crawl(self, task_config: Dict[str, Any]) -> None:
        task_name = task_config['name']

        page = await self.browser_manager.create_page()
        main_page = page

        try:
            await page.goto(
                task_config['start_url'],
                timeout=task_config['browser']['timeout'] * 1000,
                wait_until="domcontentloaded"
            )
            await self.page_detector.wait_for_load(page, task_config['browser'])

            buttons = await self.button_discovery.discover_buttons(page, task_config['button_discovery'])

            results = []
            visited_urls = set()

            for i, button in enumerate(buttons):
                logger.info(f"处理按钮 {i+1}/{len(buttons)}: {button.text[:50]}...")

                result = await self._process_button(page, button, task_config)

                if result and result.url not in visited_urls:
                    results.append(result)
                    visited_urls.add(result.url)
                    logger.info(f"成功爬取: {result.url}")
                elif result and result.url in visited_urls:
                    logger.info(f"跳过重复URL: {result.url}")

                await self._cleanup_and_return(main_page, task_config)

                if i < len(buttons) - 1:
                    await asyncio.sleep(10)

            await self.storage.save(task_name, results)

        except Exception as e:
            logger.error(f"任务 {task_name} 执行失败: {e}")
        finally:
            await page.close()

    async def _process_button(self, page, button: ButtonInfo, task_config: Dict[str, Any]) -> CrawlResult:
        try:
            original_url = page.url

            result_page = await self.click_manager.execute_click(page, button, task_config['browser'])

            if result_page != page:
                page = result_page

            await self.page_detector.wait_for_load(page, task_config['browser'])

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
        try:
            await self.browser_manager.cleanup_extra_tabs(main_page)
            await main_page.bring_to_front()

            await main_page.goto(
                task_config['start_url'],
                timeout=120000,
                wait_until="domcontentloaded"
            )
            await self.page_detector.wait_for_load(main_page, task_config['browser'])
        except Exception as e:
            logger.warning(f"清理和返回主页面失败: {e}")
            try:
                await main_page.reload(timeout=60000, wait_until="domcontentloaded")
            except Exception as reload_error:
                logger.error(f"页面重新加载也失败: {reload_error}")
