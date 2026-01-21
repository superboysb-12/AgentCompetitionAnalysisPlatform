"""
浏览器管理模块
负责Playwright浏览器的初始化、页面创建和标签页管理
"""

from typing import Dict, Any, List
from playwright.async_api import async_playwright, Page, Browser
import logging

logger = logging.getLogger(__name__)


class BrowserManager:
    """
    浏览器管理器类
    封装Playwright浏览器的生命周期管理和页面操作
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化浏览器管理器

        Args:
            config: 全局配置字典，包含浏览器类型、超时等设置
        """
        self.config = config
        self.browser: Browser = None

    async def initialize(self):
        """
        初始化并启动Playwright浏览器实例

        根据配置启动指定类型的浏览器（chromium/firefox/webkit）

        Returns:
            None
        """
        self.playwright_manager = async_playwright()
        self.playwright = await self.playwright_manager.__aenter__()
        browser_type = self.config.get('browser_type', 'chromium')
        browser_args = self._build_browser_options()

        self.browser = await getattr(self.playwright, browser_type).launch(**browser_args)

    async def create_page(self) -> Page:
        """
        创建新的浏览器页面并配置默认参数

        设置页面的默认超时时间和视口大小

        Returns:
            Page: 新创建的Playwright页面对象
        """
        page = await self.browser.new_page()

        # 设置默认超时时间（转换为毫秒）
        timeout = self.config.get('global_timeout', 300) * 1000
        page.set_default_timeout(timeout)

        # 设置视口大小
        viewport = self.config.get('viewport', {'width': 1920, 'height': 1080})
        await page.set_viewport_size(viewport)

        return page

    async def get_all_pages(self) -> List[Page]:
        """
        获取当前浏览器上下文中的所有页面

        Returns:
            List[Page]: 所有打开的页面列表
        """
        return self.browser.contexts[0].pages

    async def cleanup_extra_tabs(self, main_page: Page) -> None:
        """
        清理除主页面外的所有额外标签页

        用于在爬取过程中清理打开的新标签页，保持浏览器整洁

        Args:
            main_page: 主页面对象，不会被关闭

        Returns:
            None
        """
        try:
            all_pages = await self.get_all_pages()
            if len(all_pages) > 1:
                for page in all_pages:
                    if page != main_page:
                        try:
                            await page.close()
                        except Exception as e:
                            logger.warning(f"关闭标签页失败: {e}")
        except Exception as e:
            logger.warning(f"标签页清理异常: {e}")

    async def close(self):
        """
        关闭浏览器并清理Playwright资源

        安全关闭浏览器实例和Playwright管理器

        Returns:
            None
        """
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright_manager'):
            await self.playwright_manager.__aexit__(None, None, None)

    def _build_browser_options(self) -> Dict[str, Any]:
        """
        构建浏览器启动选项

        Returns:
            Dict[str, Any]: 浏览器启动参数字典
        """
        return {
            'headless': self.config.get('headless', False),
        }
