"""
Page Load Detection Module

智能检测页面加载状态
"""

from typing import Dict, Any
from playwright.async_api import Page
import asyncio
import logging

logger = logging.getLogger(__name__)


class PageLoadDetector:
    """页面加载检测器"""

    async def wait_for_load(self, page: Page, config: Dict[str, Any]) -> None:
        """智能等待页面加载完成"""
        await self._wait_for_dom(page)
        await self._wait_for_network(page, config)
        await self._wait_for_javascript(page, config)
        await self._wait_for_dynamic_content(page, config)

    async def _wait_for_dom(self, page: Page) -> None:
        """等待DOM加载完成"""
        try:
            await page.wait_for_selector('body', timeout=60000)
        except Exception as e:
            logger.warning(f"DOM等待超时: {e}")

    async def _wait_for_network(self, page: Page, config: Dict[str, Any]) -> None:
        """等待网络请求完成"""
        max_attempts = config.get('network_wait_attempts', 3)

        for attempt in range(max_attempts):
            try:
                await page.wait_for_load_state('networkidle', timeout=30000)
                break
            except Exception as e:
                logger.debug(f"网络等待第{attempt+1}次超时: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(3)

    async def _wait_for_javascript(self, page: Page, config: Dict[str, Any]) -> None:
        """等待JavaScript执行完成"""
        js_wait_time = config.get('js_wait_time', 5)
        await asyncio.sleep(js_wait_time)

    async def _wait_for_dynamic_content(self, page: Page, config: Dict[str, Any]) -> None:
        """等待动态内容稳定"""
        max_attempts = config.get('content_stability_attempts', 5)
        stable_count = 0
        prev_content_hash = None

        for attempt in range(max_attempts):
            try:
                content_elements = await page.query_selector_all(
                    'div.article-card-container, .article-card, .content, h1, h2'
                )

                content_text = ""
                for elem in content_elements[:5]:
                    try:
                        text = await elem.text_content()
                        if text:
                            content_text += text[:50]
                    except:
                        continue

                current_hash = hash(content_text) if content_text else 0

                if current_hash == prev_content_hash:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0
                    prev_content_hash = current_hash

                await asyncio.sleep(2)

            except Exception as e:
                logger.debug(f"动态内容检测异常: {e}")
                break