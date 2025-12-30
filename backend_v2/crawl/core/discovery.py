"""
按钮发现模块
智能发现页面中的可点击元素，支持去重和选择器优化
"""

from typing import List, Dict, Any
from playwright.async_api import Page
from crawl.core.types import ButtonInfo
from crawl.core.utils import resolve_url
import logging

logger = logging.getLogger(__name__)


class ButtonDiscovery:
    """
    按钮发现器类
    在页面中查找并提取可点击元素的信息
    """

    async def discover_buttons(self, page: Page, config: Dict[str, Any]) -> List[ButtonInfo]:
        """
        发现页面中的可点击按钮

        根据配置的选择器查找页面中的可点击元素，支持去重和智能优化

        Args:
            page: Playwright页面对象
            config: 按钮发现配置，包含selectors、max_buttons、deduplicate等

        Returns:
            List[ButtonInfo]: 发现的按钮信息列表
        """
        current_url = page.url

        buttons = []
        seen_elements = set()  # 用于去重的元素签名集合
        selectors = config.get('selectors', [])
        max_buttons = config.get('max_buttons', 10)
        deduplicate = config.get('deduplicate', False)
        smart_discovery = config.get('smart_discovery', False)

        # 如果启用智能发现，优化选择器列表
        if smart_discovery and len(selectors) > 1:
            selectors = await self._optimize_selectors(page, selectors)

        # 遍历每个选择器查找元素
        for selector in selectors:
            try:
                locator = page.locator(selector)
                count = await locator.count()
                logger.info(f"选择器 {selector} 找到 {count} 个元素")

                for i in range(min(count, max_buttons - len(buttons))):
                    try:
                        element_locator = locator.nth(i)

                        # 等待元素可见
                        await element_locator.wait_for(state='visible', timeout=5000)

                        # 检查元素是否可点击
                        if await self._is_locator_clickable(element_locator):
                            # 如果启用去重，检查元素签名
                            if deduplicate:
                                element_signature = await self._get_element_signature(element_locator)
                                if element_signature in seen_elements:
                                    logger.debug(f"跳过重复元素: {element_signature[:100]}...")
                                    continue
                                seen_elements.add(element_signature)

                            # 提取按钮信息并添加到列表
                            button_info = await self._extract_button_info_from_locator(element_locator, selector, i, current_url)
                            buttons.append(button_info)
                            logger.info(f"添加按钮 #{len(buttons)}: {button_info.text[:50]}...")

                        # 达到最大按钮数时停止
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
        """
        优化选择器列表，移除重复和冗余的选择器

        分析选择器之间的覆盖关系，移除被其他选择器覆盖的冗余选择器

        Args:
            page: Playwright页面对象
            selectors: 原始选择器列表

        Returns:
            List[str]: 优化后的选择器列表
        """
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
        """
        判断两个选择器是否有包含关系

        简化选择器字符串后进行比较，判断是否存在子集关系

        Args:
            selector1: 第一个选择器
            selector2: 第二个选择器

        Returns:
            bool: 如果存在包含关系返回True
        """
        core1 = selector1.replace('div.', '.').replace(' ', '').lower()
        core2 = selector2.replace('div.', '.').replace(' ', '').lower()

        return core1 in core2 or core2 in core1

    async def _get_element_signature(self, locator) -> str:
        """
        生成元素的唯一签名用于去重

        综合元素的ID、位置、文本、href和class等特征生成签名

        Args:
            locator: Playwright locator对象

        Returns:
            str: 元素签名字符串
        """
        try:
            bounding_box = await locator.bounding_box()
            text_content = (await locator.text_content() or "").strip()
            href = await locator.get_attribute('href')
            element_id = await locator.get_attribute('id')
            class_name = await locator.get_attribute('class')

            # 构建签名各部分
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

            # 拼接各部分形成完整签名
            return "|".join(signature_parts) if signature_parts else "unknown"

        except Exception as e:
            logger.debug(f"获取元素签名失败: {e}")
            return "unknown"

    async def _is_locator_clickable(self, locator) -> bool:
        """
        检查locator是否可点击

        Args:
            locator: Playwright locator对象

        Returns:
            bool: 如果元素可见且可用返回True
        """
        try:
            return await locator.is_visible() and await locator.is_enabled()
        except:
            return False

    async def _extract_button_info_from_locator(self, locator, selector: str, index: int, base_url: str) -> ButtonInfo:
        """
        从locator提取按钮信息

        Args:
            locator: Playwright locator对象
            selector: CSS选择器
            index: 元素在选择器结果中的索引
            base_url: 基础URL用于解析相对链接

        Returns:
            ButtonInfo: 按钮信息对象
        """
        try:
            text = await locator.text_content() or ""
            raw_href = await locator.get_attribute('href')
            onclick = await locator.get_attribute('onclick')
            target = await locator.get_attribute('target')

            # 解析相对URL为绝对URL
            href = resolve_url(base_url, raw_href) if raw_href else None

            return ButtonInfo(
                element=None,  # locator模式下不保存元素引用
                selector=selector,
                text=text[:100],  # 限制文本长度
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
