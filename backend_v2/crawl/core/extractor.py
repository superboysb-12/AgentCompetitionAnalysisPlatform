"""
内容提取模块
根据CSS选择器规则从页面中提取文本、链接和图片内容
"""

from typing import Dict, Any
from playwright.async_api import Page
from crawl.core.utils import resolve_url
import logging

logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    内容提取器类
    使用CSS选择器从页面中提取结构化数据
    """

    async def extract(self, page: Page, rules: Dict[str, str]) -> Dict[str, Any]:
        """
        根据规则从页面中提取内容

        Args:
            page: Playwright页面对象
            rules: 提取规则字典，键为字段名，值为CSS选择器（支持逗号分隔多个选择器）

        Returns:
            Dict[str, Any]: 提取的内容字典
        """
        current_url = page.url

        content = {}

        # 遍历每个字段的提取规则
        for field, selector in rules.items():
            try:
                # 支持逗号分隔的多个选择器
                selectors = [s.strip() for s in selector.split(',')]
                extracted_data = await self._extract_field(page, field, selectors, current_url)
                content[field] = extracted_data
            except Exception as e:
                logger.warning(f"字段 {field} 提取失败: {e}")
                content[field] = None

        return content

    async def _extract_field(self, page: Page, field: str, selectors: list, base_url: str) -> Any:
        """
        从页面中提取单个字段的数据

        根据字段名称使用不同的提取策略（链接、图片或文本）

        Args:
            page: Playwright页面对象
            field: 字段名称
            selectors: CSS选择器列表
            base_url: 基础URL

        Returns:
            Any: 提取的数据（列表或None）
        """
        # 尝试每个选择器直到成功
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                if not elements:
                    continue

                # 根据字段类型选择提取方法
                if field == 'links':
                    return await self._extract_links(elements, base_url)
                elif field == 'images':
                    return await self._extract_images(elements, base_url)
                else:
                    return await self._extract_text(elements)

            except Exception as e:
                logger.debug(f"选择器 {selector} 处理失败: {e}")
                continue

        return None

    async def _extract_links(self, elements, base_url: str) -> list:
        """
        提取链接数据

        Args:
            elements: 元素列表
            base_url: 基础URL

        Returns:
            list: 链接字典列表，每个包含href和text字段
        """
        links = []
        for elem in elements[:20]:  # 限制最多提取20个
            try:
                raw_href = await elem.get_attribute('href')
                text = await elem.text_content()
                if raw_href:
                    href = resolve_url(base_url, raw_href)
                    links.append({'href': href, 'text': (text or "").strip()})
            except:
                continue
        return links

    async def _extract_images(self, elements, base_url: str) -> list:
        """
        提取图片数据

        Args:
            elements: 元素列表
            base_url: 基础URL

        Returns:
            list: 图片字典列表，每个包含src和alt字段
        """
        images = []
        for elem in elements[:20]:  # 限制最多提取20个
            try:
                raw_src = await elem.get_attribute('src')
                alt = await elem.get_attribute('alt')
                if raw_src:
                    src = resolve_url(base_url, raw_src)
                    images.append({'src': src, 'alt': alt})
            except:
                continue
        return images

    async def _extract_text(self, elements) -> list:
        """
        提取文本数据

        Args:
            elements: 元素列表

        Returns:
            list: 文本字符串列表
        """
        texts = []
        for elem in elements:
            try:
                text = await elem.text_content()
                if text and text.strip():
                    texts.append(text.strip())
            except:
                continue
        return texts
