"""
Browser Management Module

处理浏览器实例创建、页面管理和标签页控制
"""

from typing import Dict, Any, List
from playwright.async_api import async_playwright, Page, Browser
import logging

logger = logging.getLogger(__name__)


class BrowserManager:
    """浏览器管理器 - 处理浏览器生命周期和标签页管理"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.browser: Browser = None

    async def initialize(self):
        """初始化浏览器"""
        self.playwright_manager = async_playwright()
        self.playwright = await self.playwright_manager.__aenter__()
        browser_type = self.config.get('browser_type', 'chromium')
        browser_args = self._build_browser_options()

        self.browser = await getattr(self.playwright, browser_type).launch(**browser_args)

    async def create_page(self) -> Page:
        """创建新页面"""
        page = await self.browser.new_page()

        # 设置页面级别的默认超时
        timeout = self.config.get('global_timeout', 300) * 1000
        page.set_default_timeout(timeout)

        # 设置视口大小 - 使用正确的字典格式
        viewport = self.config.get('viewport', {'width': 1920, 'height': 1080})
        await page.set_viewport_size(viewport)

        return page

    async def get_all_pages(self) -> List[Page]:
        """获取所有页面"""
        return self.browser.contexts[0].pages

    async def cleanup_extra_tabs(self, main_page: Page) -> None:
        """清理多余的标签页，只保留主页面"""
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
        """关闭浏览器"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright_manager'):
            await self.playwright_manager.__aexit__(None, None, None)

    def _build_browser_options(self) -> Dict[str, Any]:
        """构建浏览器启动选项"""
        return {
            'headless': self.config.get('headless', False),
        }