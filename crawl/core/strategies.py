"""
Click Strategies Module

不同的点击策略实现，处理各种页面跳转场景
"""

from typing import Optional, Dict, Any
from playwright.async_api import Page
from core.types import ButtonInfo, ClickStrategy, CrawlResult
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)


class ClickStrategyBase(ABC):
    """点击策略基类"""

    @abstractmethod
    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """执行点击策略"""
        pass

    @abstractmethod
    def can_handle(self, button: ButtonInfo) -> bool:
        """判断是否可以处理该按钮"""
        pass


class DirectNavigationStrategy(ClickStrategyBase):
    """直接导航策略 - 适用于有明确href的链接"""

    def can_handle(self, button: ButtonInfo) -> bool:
        return (button.href and
                not button.href.startswith('javascript:') and
                not button.href.startswith('#'))

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            # 重新定位元素
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            # 等待元素可见
            await locator.wait_for(state='visible', timeout=5000)

            if button.target == '_blank':
                # 新标签页导航
                async with page.context.expect_page() as new_page_info:
                    await locator.click()
                return await new_page_info.value
            else:
                # 同页面导航 - 使用更快的wait_until
                await page.goto(button.href, timeout=60000, wait_until="domcontentloaded")
                return page
        except Exception as e:
            logger.warning(f"直接导航失败: {e}")
            return None


class NewTabMonitoringStrategy(ClickStrategyBase):
    """新标签页监听策略 - 监听新页面创建"""

    def can_handle(self, button: ButtonInfo) -> bool:
        return True  # 通用策略

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            # 重新定位元素
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
    """同页面导航策略 - 监听页面导航"""

    def can_handle(self, button: ButtonInfo) -> bool:
        return True  # 通用策略

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            # 重新定位元素
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
    """直接点击策略 - 点击后检测变化"""

    def can_handle(self, button: ButtonInfo) -> bool:
        return True  # 通用策略

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            # 重新定位元素
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            original_url = page.url
            await locator.click()

            # 渐进式等待并检测变化
            for wait_time in [2, 3, 5]:
                await asyncio.sleep(wait_time)

                # 检测新标签页
                all_pages = page.context.pages
                if len(all_pages) > 1:
                    return all_pages[-1]

                # 检测URL变化
                if page.url != original_url:
                    return page

            return page
        except Exception as e:
            logger.warning(f"直接点击失败: {e}")
            return None


class InternalLinkStrategy(ClickStrategyBase):
    """内部链接策略 - 查找并点击内部有效链接"""

    def can_handle(self, button: ButtonInfo) -> bool:
        return ('article-card' in button.selector or
                'container' in button.selector)

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        try:
            # 重新定位元素
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            # 查找内部链接
            links = await locator.locator('a[href]').all()
            for link in links:
                href = await link.get_attribute('href')
                target = await link.get_attribute('target')

                if href and self._is_valid_article_link(href):
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
        """判断是否是有效的文章链接"""
        if not href:
            return False

        invalid_patterns = ['#', 'javascript:', 'mailto:', 'tel:']
        if any(pattern in href for pattern in invalid_patterns):
            return False

        article_patterns = ['article', 'news', 'post', 'detail', 'content', '/id/', '/item/']
        return any(pattern in href.lower() for pattern in article_patterns)


class ClickStrategyManager:
    """点击策略管理器"""

    def __init__(self):
        self.strategies = [
            DirectNavigationStrategy(),
            NewTabMonitoringStrategy(),
            SamePageNavigationStrategy(),
            DirectClickStrategy(),
            InternalLinkStrategy()
        ]

    async def execute_click(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """执行点击策略"""
        max_retries = config.get('max_retries', 3)

        for attempt in range(max_retries):
            for strategy in self.strategies:
                if strategy.can_handle(button):
                    try:
                        result_page = await strategy.execute(page, button, config)
                        if result_page:
                            strategy_name = strategy.__class__.__name__
                            return result_page
                    except Exception as e:
                        logger.debug(f"策略 {strategy.__class__.__name__} 失败: {e}")
                        continue

            if attempt < max_retries - 1:
                await asyncio.sleep(5)

        return page  # 返回原页面