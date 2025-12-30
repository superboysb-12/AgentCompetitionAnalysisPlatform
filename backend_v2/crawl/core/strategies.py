"""
点击策略模块
实现多种页面元素点击策略，自动选择最合适的方式处理不同类型的链接和按钮
"""

from typing import Optional, Dict, Any
from playwright.async_api import Page
from crawl.core.types import ButtonInfo, ClickStrategy, CrawlResult
from crawl.core.utils import resolve_url
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)


class ClickStrategyBase(ABC):
    """
    点击策略基类
    定义所有点击策略必须实现的接口
    """

    @abstractmethod
    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """
        执行点击操作

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典

        Returns:
            Optional[Page]: 点击后的页面对象，失败时返回None
        """
        pass

    @abstractmethod
    def can_handle(self, button: ButtonInfo) -> bool:
        """
        判断当前策略是否能处理该按钮

        Args:
            button: 按钮信息

        Returns:
            bool: 如果能处理返回True
        """
        pass


class DirectNavigationStrategy(ClickStrategyBase):
    """
    直接导航策略
    适用于有明确href属性的链接，直接通过goto导航或处理新标签页
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        """
        判断是否能处理该按钮

        只处理有有效href且不是javascript伪协议或锚点的链接

        Args:
            button: 按钮信息

        Returns:
            bool: 如果能处理返回True
        """
        return (button.href and
                not button.href.startswith('javascript:') and
                not button.href.startswith('#'))

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """
        执行直接导航

        如果target为_blank则监听新标签页，否则在当前页面导航

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典

        Returns:
            Optional[Page]: 导航后的页面对象，失败时返回None
        """
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            # 如果target为_blank，监听新标签页
            if button.target == '_blank':
                async with page.context.expect_page() as new_page_info:
                    await locator.click()
                return await new_page_info.value
            else:
                # 否则直接导航到目标URL
                await page.goto(button.href, timeout=60000, wait_until="domcontentloaded")
                return page
        except Exception as e:
            logger.warning(f"直接导航失败: {e}")
            return None


class NewTabMonitoringStrategy(ClickStrategyBase):
    """
    新标签页监听策略
    点击元素并监听是否打开新标签页
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        """
        判断是否能处理该按钮

        这是一个通用策略，可以处理任何按钮

        Args:
            button: 按钮信息

        Returns:
            bool: 总是返回True
        """
        return True

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """
        执行点击并监听新标签页

        点击元素并等待新标签页打开

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典

        Returns:
            Optional[Page]: 新打开的页面对象，失败时返回None
        """
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            # 监听新标签页并点击
            async with page.context.expect_page(timeout=30000) as new_page_info:
                await locator.click()
            return await new_page_info.value
        except Exception as e:
            logger.debug(f"新标签页监听失败: {e}")
            return None


class SamePageNavigationStrategy(ClickStrategyBase):
    """
    同页面导航策略
    点击元素并等待当前页面导航完成
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        """
        判断是否能处理该按钮

        这是一个通用策略，可以处理任何按钮

        Args:
            button: 按钮信息

        Returns:
            bool: 总是返回True
        """
        return True

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """
        执行同页面导航

        点击元素并等待页面导航事件

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典

        Returns:
            Optional[Page]: 导航后的页面对象，失败时返回None
        """
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            # 等待页面导航并点击
            async with page.expect_navigation(timeout=60000, wait_until="domcontentloaded"):
                await locator.click()
            return page
        except Exception as e:
            logger.debug(f"同页面导航失败: {e}")
            return None


class DirectClickStrategy(ClickStrategyBase):
    """
    直接点击策略
    简单点击元素并等待可能的变化
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        """
        判断是否能处理该按钮

        这是一个通用策略，可以处理任何按钮

        Args:
            button: 按钮信息

        Returns:
            bool: 总是返回True
        """
        return True

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """
        执行直接点击

        点击元素后等待并检测页面变化（新标签页或URL变化）

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典

        Returns:
            Optional[Page]: 点击后的页面对象（可能是新标签页或原页面），失败时返回None
        """
        try:
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            original_url = page.url
            await locator.click()

            # 多次检查是否有新标签页或URL变化
            for wait_time in [2, 3, 5]:
                await asyncio.sleep(wait_time)

                # 检查是否打开了新标签页
                all_pages = page.context.pages
                if len(all_pages) > 1:
                    return all_pages[-1]

                # 检查URL是否变化
                if page.url != original_url:
                    return page

            # 没有变化，返回原页面
            return page
        except Exception as e:
            logger.warning(f"直接点击失败: {e}")
            return None


class InternalLinkStrategy(ClickStrategyBase):
    """
    内部链接策略
    处理包含嵌套链接的容器元素（如文章卡片）
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        """
        判断是否能处理该按钮

        只处理包含特定关键词的选择器（article-card或container）

        Args:
            button: 按钮信息

        Returns:
            bool: 如果选择器包含关键词返回True
        """
        return ('article-card' in button.selector or
                'container' in button.selector)

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """
        执行内部链接点击

        在容器内查找有效的文章链接并点击

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典

        Returns:
            Optional[Page]: 点击后的页面对象，失败时返回None
        """
        try:
            current_url = page.url

            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            # 在容器内查找所有链接
            links = await locator.locator('a[href]').all()
            for link in links:
                raw_href = await link.get_attribute('href')
                target = await link.get_attribute('target')

                # 检查是否是有效的文章链接
                if raw_href and self._is_valid_article_link(raw_href):
                    href = resolve_url(current_url, raw_href)

                    # 根据target属性决定处理方式
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
        """
        判断链接是否是有效的文章链接

        Args:
            href: 链接地址

        Returns:
            bool: 如果是有效文章链接返回True
        """
        if not href:
            return False

        # 排除无效模式
        invalid_patterns = ['#', 'javascript:', 'mailto:', 'tel:']
        if any(pattern in href for pattern in invalid_patterns):
            return False

        # 检查是否包含文章相关的关键词
        article_patterns = ['article', 'news', 'post', 'detail', 'content', '/id/', '/item/']
        return any(pattern in href.lower() for pattern in article_patterns)


class ClickStrategyManager:
    """
    点击策略管理器
    管理所有点击策略并自动选择合适的策略执行点击操作
    """

    def __init__(self):
        """
        初始化策略管理器

        按优先级顺序初始化所有可用策略
        """
        self.strategies = [
            DirectNavigationStrategy(),      # 优先级1: 直接导航
            NewTabMonitoringStrategy(),       # 优先级2: 新标签页监听
            SamePageNavigationStrategy(),     # 优先级3: 同页面导航
            DirectClickStrategy(),            # 优先级4: 直接点击
            InternalLinkStrategy()            # 优先级5: 内部链接
        ]

    async def execute_click(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """
        执行点击操作

        遍历所有策略，选择第一个能处理该按钮的策略执行，支持重试

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典，包含max_retries等参数

        Returns:
            Optional[Page]: 点击后的页面对象，所有策略都失败时返回原页面
        """
        max_retries = config.get('max_retries', 3)

        # 重试机制
        for attempt in range(max_retries):
            # 遍历所有策略
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

            # 如果还有重试机会，等待后重试
            if attempt < max_retries - 1:
                logger.info(f"第{attempt + 1}次尝试失败，5秒后重试...")
                await asyncio.sleep(5)

        # 所有策略都失败，返回原页面
        logger.warning("所有点击策略都失败，返回原页面")
        return page
