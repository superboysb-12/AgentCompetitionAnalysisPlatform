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
    """直接导航策略 - 适用于有明确href的链接

    适用场景：
    - 标准的 <a href="..."> 链接
    - 有明确URL目标的按钮
    - 非 JavaScript 和锦点链接

    优势：速度最快，最稳定
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        """检查是否有有效的href属性且非JavaScript链接"""
        return (button.href and
                not button.href.startswith('javascript:') and
                not button.href.startswith('#'))

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """执行直接导航策略

        工作流程：
        1. 重新定位元素（避免 stale element 错误）
        2. 检查 target 属性决定是否新标签页
        3. 执行相应的导航操作
        """
        try:
            # 重新定位元素，避免元素过期问题
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            # 等待元素可见才能点击
            await locator.wait_for(state='visible', timeout=5000)

            if button.target == '_blank':
                # 新标签页导航 - 监听新页面创建并返回新页面实例
                async with page.context.expect_page() as new_page_info:
                    await locator.click()
                return await new_page_info.value
            else:
                # 同页面导航 - 直接使用 goto 方法，速度更快
                await page.goto(button.href, timeout=60000, wait_until="domcontentloaded")
                return page
        except Exception as e:
            logger.warning(f"直接导航失败: {e}")
            return None


class NewTabMonitoringStrategy(ClickStrategyBase):
    """新标签页监听策略 - 监听新页面创建

    适用场景：
    - JavaScript 控制的新标签页打开
    - window.open() 调用
    - 动态生成的链接

    优势：能捕获所有新标签页情况
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        return True  # 通用策略，可处理任意按钮

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """执行新标签页监听策略

        工作流程：
        1. 设置新页面监听器
        2. 执行点击操作
        3. 等待并返回新创建的页面
        """
        try:
            # 重新定位元素
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            # 设置30秒超时的新页面监听器
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
    """直接点击策略 - 点击后检测变化

    适用场景：
    - AJAX 动态加载内容
    - SPA 单页面应用路由
    - 延迟加载的元素

    特点：使用渐进式等待算法检测变化
    """

    def can_handle(self, button: ButtonInfo) -> bool:
        return True  # 通用策略，作为最后的备选方案

    async def execute(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """执行直接点击策略

        渐进式等待算法：
        1. 点击元素
        2. 分别等待2、3、5秒检测变化
        3. 检测新标签页或URL变化
        4. 返回相应的页面实例
        """
        try:
            # 重新定位元素
            locator = page.locator(button.selector)
            if button.index is not None:
                locator = locator.nth(button.index)

            await locator.wait_for(state='visible', timeout=5000)

            original_url = page.url  # 保存原始URL用于比较
            await locator.click()

            # 渐进式等待算法：从短到长的等待时间，平衡响应速度和完整性
            for wait_time in [2, 3, 5]:
                await asyncio.sleep(wait_time)

                # 检测是否有新标签页被创建
                all_pages = page.context.pages
                if len(all_pages) > 1:
                    return all_pages[-1]  # 返回最新的页面

                # 检测URL是否发生变化（SPA路由或页面重定向）
                if page.url != original_url:
                    return page

            return page  # 如果没有检测到变化，返回当前页面
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
    """点击策略管理器 - 统一管理和调度所有点击策略

    设计原则：
    1. 策略模式：根据按钮特征选择最适合的策略
    2. 优先级排序：从最精准到最通用的策略
    3. 容错机制：支持重试和策略降级
    """

    def __init__(self):
        # 按优先级排序的策略列表：从最精准到最通用
        self.strategies = [
            DirectNavigationStrategy(),    # 1. 最高优先级：有明确href的链接
            NewTabMonitoringStrategy(),    # 2. 新标签页监听
            SamePageNavigationStrategy(),  # 3. 同页面导航
            DirectClickStrategy(),         # 4. 直接点击检测
            InternalLinkStrategy()         # 5. 内部链接特殊处理
        ]

    async def execute_click(self, page: Page, button: ButtonInfo, config: Dict[str, Any]) -> Optional[Page]:
        """执行点击策略的核心方法

        工作流程：
        1. 遍历所有可用策略，检查能否处理当前按钮
        2. 尝试执行策略，如果成功则返回结果
        3. 如果所有策略都失败，尝试重试
        4. 最终返回结果或原页面
        """
        max_retries = config.get('max_retries', 3)

        for attempt in range(max_retries):
            for strategy in self.strategies:
                # 检查策略是否能处理当前按钮
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

            # 如果所有策略都失败，等待5秒后重试
            if attempt < max_retries - 1:
                logger.info(f"第{attempt + 1}次尝试失败，5秒后重试...")
                await asyncio.sleep(5)

        logger.warning("所有点击策略都失败，返回原页面")
        return page  # 所有策略都失败时返回原页面