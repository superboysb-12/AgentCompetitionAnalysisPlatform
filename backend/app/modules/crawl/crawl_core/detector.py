from typing import Dict, Any
from playwright.async_api import Page
import asyncio
import logging

logger = logging.getLogger(__name__)


class PageLoadDetector:

    async def wait_for_load(self, page: Page, config: Dict[str, Any]) -> None:
        fast_mode = config.get('fast_mode', False)

        if fast_mode:
            await self._fast_wait_for_load(page, config)
        else:
            await self._wait_for_dom(page)
            await self._wait_for_network(page, config)
            await self._wait_for_javascript(page, config)
            await self._wait_for_dynamic_content(page, config)

    async def _fast_wait_for_load(self, page: Page, config: Dict[str, Any]) -> None:
        try:
            await page.wait_for_selector('body', timeout=10000)

            await page.wait_for_load_state('domcontentloaded', timeout=15000)

            js_wait_time = config.get('fast_js_wait_time', 2)
            await asyncio.sleep(js_wait_time)

            logger.info("快速等待模式完成")

        except Exception as e:
            logger.warning(f"快速等待失败，回退到标准模式: {e}")
            await self._wait_for_network(page, config)
            await self._wait_for_javascript(page, config)

    async def _wait_for_dom(self, page: Page) -> None:
        try:
            await page.wait_for_selector('body', timeout=60000)
        except Exception as e:
            logger.warning(f"DOM等待超时: {e}")

    async def _wait_for_network(self, page: Page, config: Dict[str, Any]) -> None:
        max_attempts = config.get('network_wait_attempts', 2)
        network_timeout = config.get('network_timeout', 15000)

        for attempt in range(max_attempts):
            try:
                await page.wait_for_load_state('domcontentloaded', timeout=network_timeout)

                if config.get('wait_for_networkidle', False):
                    await page.wait_for_load_state('networkidle', timeout=10000)
                break
            except Exception as e:
                logger.debug(f"网络等待第{attempt+1}次超时: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)

    async def _wait_for_javascript(self, page: Page, config: Dict[str, Any]) -> None:
        js_wait_time = config.get('js_wait_time', 3)
        await asyncio.sleep(js_wait_time)

    async def _wait_for_dynamic_content(self, page: Page, config: Dict[str, Any]) -> None:
        if not config.get('check_content_stability', True):
            return

        max_attempts = config.get('content_stability_attempts', 3)
        stable_count = 0
        prev_content_hash = None
        check_interval = config.get('stability_check_interval', 1.5)

        for attempt in range(max_attempts):
            try:
                content_elements = await page.query_selector_all(
                    'div.article-card-container, .article-card, .content, h1, h2'
                )

                content_text = ""
                for elem in content_elements[:3]:
                    try:
                        text = await elem.text_content()
                        if text:
                            content_text += text[:30]
                    except:
                        continue

                current_hash = hash(content_text) if content_text else 0

                if current_hash == prev_content_hash:
                    stable_count += 1
                    if stable_count >= 1:
                        break
                else:
                    stable_count = 0
                    prev_content_hash = current_hash

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.debug(f"动态内容检测异常: {e}")
                break
