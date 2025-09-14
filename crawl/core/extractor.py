"""
Content Extraction Module

处理页面内容的提取和清理
"""

from typing import Dict, Any
from playwright.async_api import Page
import logging

logger = logging.getLogger(__name__)


class ContentExtractor:
    """内容提取器"""

    async def extract(self, page: Page, rules: Dict[str, str]) -> Dict[str, Any]:
        """根据规则提取页面内容"""
        content = {}

        for field, selector in rules.items():
            try:
                selectors = [s.strip() for s in selector.split(',')]
                extracted_data = await self._extract_field(page, field, selectors)
                content[field] = extracted_data
            except Exception as e:
                logger.warning(f"字段 {field} 提取失败: {e}")
                content[field] = None

        return content

    async def _extract_field(self, page: Page, field: str, selectors: list) -> Any:
        """提取单个字段的内容"""
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                if not elements:
                    continue

                if field == 'links':
                    return await self._extract_links(elements)
                elif field == 'images':
                    return await self._extract_images(elements)
                else:
                    return await self._extract_text(elements)

            except Exception as e:
                logger.debug(f"选择器 {selector} 处理失败: {e}")
                continue

        return None

    async def _extract_links(self, elements) -> list:
        """提取链接信息"""
        links = []
        for elem in elements[:20]:  # 限制数量
            try:
                href = await elem.get_attribute('href')
                text = await elem.text_content()
                if href:
                    links.append({'href': href, 'text': (text or "").strip()})
            except:
                continue
        return links

    async def _extract_images(self, elements) -> list:
        """提取图片信息"""
        images = []
        for elem in elements[:20]:  # 限制数量
            try:
                src = await elem.get_attribute('src')
                alt = await elem.get_attribute('alt')
                if src:
                    images.append({'src': src, 'alt': alt})
            except:
                continue
        return images

    async def _extract_text(self, elements) -> list:
        """提取文本内容"""
        texts = []
        for elem in elements:
            try:
                text = await elem.text_content()
                if text and text.strip():
                    texts.append(text.strip())
            except:
                continue
        return texts