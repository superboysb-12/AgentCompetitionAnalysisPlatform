from typing import Dict, Any, List
from playwright.async_api import async_playwright, Page, Browser
import logging

logger = logging.getLogger(__name__)


class BrowserManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.browser: Browser = None

    async def initialize(self):
        self.playwright_manager = async_playwright()
        self.playwright = await self.playwright_manager.__aenter__()
        browser_type = self.config.get('browser_type', 'chromium')
        browser_args = self._build_browser_options()

        self.browser = await getattr(self.playwright, browser_type).launch(**browser_args)

    async def create_page(self) -> Page:
        page = await self.browser.new_page()

        timeout = self.config.get('global_timeout', 300) * 1000
        page.set_default_timeout(timeout)

        viewport = self.config.get('viewport', {'width': 1920, 'height': 1080})
        await page.set_viewport_size(viewport)

        return page

    async def get_all_pages(self) -> List[Page]:
        return self.browser.contexts[0].pages

    async def cleanup_extra_tabs(self, main_page: Page) -> None:
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
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright_manager'):
            await self.playwright_manager.__aexit__(None, None, None)

    def _build_browser_options(self) -> Dict[str, Any]:
        return {
            'headless': self.config.get('headless', False),
        }
