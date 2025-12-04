from typing import Optional, Dict, Any
from playwright.async_api import Page
from core.types import ButtonInfo, ClickStrategy, CrawlResult
from abc import ABC, abstractmethod
from urllib.parse import urljoin
import asyncio
import logging

logger = logging.getLogger(__name__)


class ClickStrategyBase(ABC):

    def _resolve_url(self, base_url: str, href: str) -> str:
        if not href:
            return href

        if href.startswith(('http://', 'https://', 'javascript:', '#')):
            return href

        try:
            return urljoin(base_url, href)
        except Exception as e:
            logger.warning(f"URL解析失败: {base_url} + {href}, 错误: {e}")
            return href

    @abstractmethod
    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        pass

    @abstractmethod
    def can_handle(self, button: ButtonInfo) -> bool:
        pass


class DirectNavigationStrategy(ClickStrategyBase):

    def can_handle(self, button: ButtonInfo) -> bool:
        return (button.href and
                not button.href.startswith('javascript:') and
                not button.href.startswith('#'))

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            if button.target == '_blank':
                async with page.context.expect_page() as new_page_info:
                    await locator.click()
                return await new_page_info.value
            else:
                await page.goto(button.href, timeout=60000, wait_until="domcontentloaded")
                return page
        except Exception as e:
            logger.warning(f"直接导航失败: {e}")
            return None


class NewTabMonitoringStrategy(ClickStrategyBase):

    def can_handle(self, button: ButtonInfo) -> bool:
        return True

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            async with page.context.expect_page(timeout=30000) as new_page_info:
                await locator.click()
            return await new_page_info.value
        except Exception as e:
            logger.debug(f"新标签页监听失败: {e}")
            return None


class SamePageNavigationStrategy(ClickStrategyBase):

    def can_handle(self, button: ButtonInfo) -> bool:
        return True

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            async with page.expect_navigation(timeout=60000, wait_until="domcontentloaded"):
                await locator.click()
            return page
        except Exception as e:
            logger.debug(f"同页面导航失败: {e}")
            return None


class DirectClickStrategy(ClickStrategyBase):

    def can_handle(self, button: ButtonInfo) -> bool:
        return True

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            original_url = page.url
            await locator.click()

            for wait_time in [2, 3, 5]:
                await asyncio.sleep(wait_time)

                all_pages = page.context.pages
                if len(all_pages) > 1:
                    return all_pages[-1]

                if page.url != original_url:
                    return page

            return page
        except Exception as e:
            logger.warning(f"直接点击失败: {e}")
            return None


class InternalLinkStrategy(ClickStrategyBase):

    def can_handle(self, button: ButtonInfo) -> bool:
        return ('article-card' in button.selector or
                'container' in button.selector)

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            current_url = page.url

            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            links = await locator.locator('a[href]').all()
            for link in links:
                raw_href = await link.get_attribute('href')
                target = await link.get_attribute('target')

                if raw_href and self._is_valid_article_link(raw_href):
                    href = self._resolve_url(current_url, raw_href)

                    if target == '_blank':
                        async with page.context.expect_page() as new_page_info:
                            await link.click()
                        return await new_page_info.value
                    else:
                        await page.goto(href, timeout=60000, wait_until="domcontentloaded")
                        return page
            return None
        except Exception as e:
            logger.warning(f"内部链接处理失败: {e}")
            return None

    def _is_valid_article_link(self, href: str) -> bool:
        if not href:
            return False

        invalid_patterns = ['#', 'javascript:', 'mailto:', 'tel:']
        if any(pattern in href for pattern in invalid_patterns):
            return False

        article_patterns = ['article', 'news', 'post', 'detail', 'content', '/id/', '/item/']
        return any(pattern in href.lower() for pattern in article_patterns)


class ClickStrategyManager:

    def __init__(self):
        self.strategies = [
            DirectNavigationStrategy(),
            NewTabMonitoringStrategy(),
            SamePageNavigationStrategy(),
            DirectClickStrategy(),
            InternalLinkStrategy()
        ]

    async def execute_click(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        max_retries = config.get('max_retries', 3)

        for attempt in range(max_retries):
            for strategy in self.strategies:
                if strategy.can_handle(button):
                    try:
                        result_page = await strategy.execute(page, button, config)
                        if result_page:
                            strategy_name = strategy.__class__.__name__
                            logger.info(f"使用策略: {strategy_name}")
                            return result_page
                    except Exception as e:
                        logger.debug(f"策略 {strategy.__class__.__name__} 失败: {e}")
                        continue

            if attempt < max_retries - 1:
                logger.info(f"第{attempt + 1}次尝试失败，5秒后重试...")
                await asyncio.sleep(5)

        logger.warning("所有点击策略都失败，返回原页面")
        return page
