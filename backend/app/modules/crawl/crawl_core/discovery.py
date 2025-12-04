from typing import List, Dict, Any
from playwright.async_api import Page
from .types import ButtonInfo
from urllib.parse import urljoin
import logging

logger = logging.getLogger(__name__)


class ButtonDiscovery:

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

    async def discover_buttons(self, page: Page, config: Dict[str, Any]) -> List[ButtonInfo]:
        current_url = page.url

        buttons = []
        seen_elements = set()
        selectors = config.get('selectors', [])
        max_buttons = config.get('max_buttons', 10)
        deduplicate = config.get('deduplicate', False)
        smart_discovery = config.get('smart_discovery', False)

        if smart_discovery and len(selectors) > 1:
            selectors = await self._optimize_selectors(page, selectors)

        for selector in selectors:
            try:
                locator = page.locator(selector)
                count = await locator.count()
                logger.info(f"选择器 {selector} 找到 {count} 个元素")

                for i in range(min(count, max_buttons - len(buttons))):
                    try:
                        element_locator = locator.nth(i)

                        await element_locator.wait_for(state='visible', timeout=5000)

                        if await self._is_locator_clickable(element_locator):
                            if deduplicate:
                                element_signature = await self._get_element_signature(element_locator)
                                if element_signature in seen_elements:
                                    logger.debug(f"跳过重复元素: {element_signature[:100]}...")
                                    continue
                                seen_elements.add(element_signature)

                            button_info = await self._extract_button_info_from_locator(element_locator, selector, i, current_url)
                            buttons.append(button_info)
                            logger.info(f"添加按钮 #{len(buttons)}: {button_info.text[:50]}...")

                        if len(buttons) >= max_buttons:
                            break

                    except Exception as e:
                        logger.debug(f"元素 {selector}[{i}] 处理失败: {e}")
                        continue

            except Exception as e:
                logger.warning(f"选择器 {selector} 处理失败: {e}")
                continue

        logger.info(f"总共发现 {len(buttons)} 个唯一可点击按钮")
        return buttons

    async def _optimize_selectors(self, page: Page, selectors: List[str]) -> List[str]:
        optimized = []
        element_counts = {}

        for selector in selectors:
            try:
                count = await page.locator(selector).count()
                element_counts[selector] = count
                logger.debug(f"选择器分析: {selector} -> {count} 个元素")
            except Exception as e:
                logger.debug(f"选择器分析失败 {selector}: {e}")
                element_counts[selector] = 0

        for selector in selectors:
            is_subset = False
            for other_selector in selectors:
                if selector != other_selector and self._is_selector_subset(selector, other_selector):
                    if element_counts[selector] <= element_counts[other_selector]:
                        logger.info(f"跳过重叠选择器: {selector} (被 {other_selector} 覆盖)")
                        is_subset = True
                        break

            if not is_subset:
                optimized.append(selector)

        logger.info(f"选择器优化: {len(selectors)} -> {len(optimized)}")
        return optimized

    def _is_selector_subset(self, selector1: str, selector2: str) -> bool:
        core1 = selector1.replace('div.', '.').replace(' ', '').lower()
        core2 = selector2.replace('div.', '.').replace(' ', '').lower()

        return core1 in core2 or core2 in core1

    async def _get_element_signature(self, locator) -> str:
        try:
            bounding_box = await locator.bounding_box()
            text_content = (await locator.text_content() or "").strip()
            href = await locator.get_attribute('href')
            element_id = await locator.get_attribute('id')
            class_name = await locator.get_attribute('class')

            signature_parts = []

            if element_id:
                signature_parts.append(f"id:{element_id}")
            if bounding_box:
                pos = f"pos:{int(bounding_box['x'])}_{int(bounding_box['y'])}"
                signature_parts.append(pos)
            if text_content:
                signature_parts.append(f"text:{text_content[:50]}")
            if href:
                signature_parts.append(f"href:{href}")
            if class_name:
                signature_parts.append(f"class:{class_name}")

            return "|".join(signature_parts) if signature_parts else "unknown"

        except Exception as e:
            logger.debug(f"获取元素签名失败: {e}")
            return "unknown"

    async def _is_locator_clickable(self, locator) -> bool:
        try:
            return await locator.is_visible() and await locator.is_enabled()
        except:
            return False

    async def _is_clickable(self, element) -> bool:
        try:
            return await element.is_visible() and await element.is_enabled()
        except:
            return False

    async def _extract_button_info_from_locator(self, locator, selector: str, index: int, base_url: str) -> ButtonInfo:
        try:
            text = await locator.text_content() or ""
            raw_href = await locator.get_attribute('href')
            onclick = await locator.get_attribute('onclick')
            target = await locator.get_attribute('target')

            href = self._resolve_url(base_url, raw_href) if raw_href else None

            return ButtonInfo(
                element=None,
                selector=selector,
                text=text[:100],
                href=href,
                onclick=onclick,
                target=target,
                index=index
            )
        except Exception as e:
            logger.warning(f"按钮信息提取失败: {e}")
            return ButtonInfo(
                element=None,
                selector=selector,
                text="",
                href=None,
                onclick=None,
                target=None,
                index=index
            )

    async def _extract_button_info(self, element, selector: str, base_url: str = None) -> ButtonInfo:
        try:
            text = await element.text_content() or ""
            raw_href = await element.get_attribute('href')
            onclick = await element.get_attribute('onclick')
            target = await element.get_attribute('target')

            href = self._resolve_url(base_url, raw_href) if base_url and raw_href else raw_href

            return ButtonInfo(
                element=element,
                selector=selector,
                text=text[:100],
                href=href,
                onclick=onclick,
                target=target
            )
        except Exception as e:
            logger.warning(f"按钮信息提取失败: {e}")
            return ButtonInfo(
                element=element,
                selector=selector,
                text="",
                href=None,
                onclick=None,
                target=None
            )
