from typing import Dict, Any
from playwright.async_api import Page
from urllib.parse import urljoin
import logging

logger = logging.getLogger(__name__)


class ContentExtractor:

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

    async def extract(self, page: Page, rules: Dict[str, str]) -> Dict[str, Any]:
        current_url = page.url

        content = {}

        for field, selector in rules.items():
            try:
                selectors = [s.strip() for s in selector.split(',')]
                extracted_data = await self._extract_field(page, field, selectors, current_url)
                content[field] = extracted_data
            except Exception as e:
                logger.warning(f"字段 {field} 提取失败: {e}")
                content[field] = None

        return content

    async def _extract_field(self, page: Page, field: str, selectors: list, base_url: str) -> Any:
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                if not elements:
                    continue

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
        links = []
        for elem in elements[:20]:
            try:
                raw_href = await elem.get_attribute('href')
                text = await elem.text_content()
                if raw_href:
                    href = self._resolve_url(base_url, raw_href)
                    links.append({'href': href, 'text': (text or "").strip()})
            except:
                continue
        return links

    async def _extract_images(self, elements, base_url: str) -> list:
        images = []
        for elem in elements[:20]:
            try:
                raw_src = await elem.get_attribute('src')
                alt = await elem.get_attribute('alt')
                if raw_src:
                    src = self._resolve_url(base_url, raw_src)
                    images.append({'src': src, 'alt': alt})
            except:
                continue
        return images

    async def _extract_text(self, elements) -> list:
        texts = []
        for elem in elements:
            try:
                text = await elem.text_content()
                if text and text.strip():
                    texts.append(text.strip())
            except:
                continue
        return texts
