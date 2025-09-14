"""
Button Discovery Module

处理页面中可点击元素的发现和识别
"""

from typing import List, Dict, Any
from playwright.async_api import Page
from core.types import ButtonInfo
import logging

logger = logging.getLogger(__name__)


class ButtonDiscovery:
    """按钮发现器 - 识别页面中的可点击元素"""

    async def discover_buttons(self, page: Page, config: Dict[str, Any]) -> List[ButtonInfo]:
        """发现页面中的可点击按钮 - 智能去重保持通用性

        算法流程：
        1. 获取配置参数（CSS选择器、最大数量等）
        2. 如果启用智能发现，先优化选择器列表去重
        3. 遍历每个选择器，找到匹配的元素
        4. 检查元素的可见性和可点击性
        5. 如果开启去重，根据元素特征进行智能去重
        6. 提取按钮信息并返回结果
        """
        buttons = []
        seen_elements = set()  # 用于存储已发现元素的特征签名，避免重复
        selectors = config.get('selectors', [])
        max_buttons = config.get('max_buttons', 10)
        deduplicate = config.get('deduplicate', False)  # 是否开启智能去重
        smart_discovery = config.get('smart_discovery', False)  # 是否启用智能发现模式

        # 如果开启智能发现模式且有多个选择器，先分析选择器重叠情况
        # 目标：减少重复工作，提高发现效率
        if smart_discovery and len(selectors) > 1:
            selectors = await self._optimize_selectors(page, selectors)

        for selector in selectors:
            try:
                # 使用Playwright的Locator API获取元素，更加稳定和高效
                locator = page.locator(selector)
                count = await locator.count()  # 获取匹配元素数量
                logger.info(f"选择器 {selector} 找到 {count} 个元素")

                # 遍历所有匹配的元素，但不超过限制数量
                for i in range(min(count, max_buttons - len(buttons))):
                    try:
                        element_locator = locator.nth(i)  # 获取第 i 个匹配元素

                        # 等待元素可见并检查是否可点击
                        await element_locator.wait_for(state='visible', timeout=5000)

                        if await self._is_locator_clickable(element_locator):
                            # 智能去重：根据元素的多个特征检查是否为重复元素
                            if deduplicate:
                                element_signature = await self._get_element_signature(element_locator)
                                if element_signature in seen_elements:
                                    logger.debug(f"跳过重复元素: {element_signature[:100]}...")
                                    continue
                                seen_elements.add(element_signature)

                            # 提取按钮元素的详细信息
                            button_info = await self._extract_button_info_from_locator(element_locator, selector, i)
                            buttons.append(button_info)
                            logger.info(f"添加按钮 #{len(buttons)}: {button_info.text[:50]}...")

                        # 检查是否达到最大数量限制
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
        """优化选择器列表，移除完全重叠的选择器"""
        optimized = []
        element_counts = {}

        # 统计每个选择器匹配的元素
        for selector in selectors:
            try:
                count = await page.locator(selector).count()
                element_counts[selector] = count
                logger.debug(f"选择器分析: {selector} -> {count} 个元素")
            except Exception as e:
                logger.debug(f"选择器分析失败 {selector}: {e}")
                element_counts[selector] = 0

        # 移除重叠选择器的逻辑
        for selector in selectors:
            # 检查是否是其他选择器的子集
            is_subset = False
            for other_selector in selectors:
                if selector != other_selector and self._is_selector_subset(selector, other_selector):
                    # 如果当前选择器是其他选择器的子集，且元素数量相同，则跳过
                    if element_counts[selector] <= element_counts[other_selector]:
                        logger.info(f"跳过重叠选择器: {selector} (被 {other_selector} 覆盖)")
                        is_subset = True
                        break

            if not is_subset:
                optimized.append(selector)

        logger.info(f"选择器优化: {len(selectors)} -> {len(optimized)}")
        return optimized

    def _is_selector_subset(self, selector1: str, selector2: str) -> bool:
        """简单判断选择器是否可能重叠"""
        # 移除空格和标点，比较核心部分
        core1 = selector1.replace('div.', '.').replace(' ', '').lower()
        core2 = selector2.replace('div.', '.').replace(' ', '').lower()

        # 如果一个选择器包含另一个的核心部分，认为可能重叠
        return core1 in core2 or core2 in core1

    async def _get_element_signature(self, locator) -> str:
        """获取元素的综合签名用于智能去重

        算法：组合多个元素特征构建唯一标识：
        1. ID属性（最优先）
        2. 位置信息（x, y 坐标）
        3. 文本内容
        4. href 属性
        5. class 属性
        """
        try:
            # 获取元素在DOM中的位置信息
            bounding_box = await locator.bounding_box()
            text_content = (await locator.text_content() or "").strip()
            href = await locator.get_attribute('href')
            element_id = await locator.get_attribute('id')
            class_name = await locator.get_attribute('class')

            # 组合多个属性构建综合签名。位置信息特别重要，因为相同元素不会在同一位置
            signature_parts = []

            if element_id:
                signature_parts.append(f"id:{element_id}")
            if bounding_box:
                # 使用像素位置信息作为签名的重要部分
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

    async def _get_element_id(self, locator) -> str:
        """获取元素的唯一标识用于去重"""
        try:
            # 尝试多种方式获取唯一标识
            element_id = await locator.get_attribute('id')
            if element_id:
                return f"id:{element_id}"

            class_name = await locator.get_attribute('class')
            text_content = await locator.text_content()
            href = await locator.get_attribute('href')

            # 组合多个属性创建唯一标识
            unique_parts = []
            if class_name:
                unique_parts.append(f"class:{class_name}")
            if text_content:
                unique_parts.append(f"text:{text_content[:100]}")
            if href:
                unique_parts.append(f"href:{href}")

            return "|".join(unique_parts) if unique_parts else "unknown"

        except Exception:
            return "unknown"

    async def _is_locator_clickable(self, locator) -> bool:
        """检查定位器元素是否可点击"""
        try:
            return await locator.is_visible() and await locator.is_enabled()
        except:
            return False

    async def _is_clickable(self, element) -> bool:
        """检查元素是否可点击"""
        try:
            return await element.is_visible() and await element.is_enabled()
        except:
            return False

    async def _extract_button_info_from_locator(self, locator, selector: str, index: int) -> ButtonInfo:
        """从Platwright定位器提取按钮信息

        提取信息包括：
        - 文本内容（限制长度避免过长）
        - href 属性（链接目标）
        - onclick 属性（JavaScript事件）
        - target 属性（打开方式）
        - 索引信息（用于后续重新定位）
        """
        try:
            # 提取元素的各项属性
            text = await locator.text_content() or ""
            href = await locator.get_attribute('href')
            onclick = await locator.get_attribute('onclick')
            target = await locator.get_attribute('target')

            return ButtonInfo(
                element=None,  # 不存储元素句柄避免内存泄漏
                selector=selector,
                text=text[:100],  # 限制文本长度避免过长日志
                href=href,
                onclick=onclick,
                target=target,
                index=index  # 保存索引信息用于后续重新定位元素
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

    async def _extract_button_info(self, element, selector: str) -> ButtonInfo:
        """提取按钮信息"""
        try:
            text = await element.text_content() or ""
            href = await element.get_attribute('href')
            onclick = await element.get_attribute('onclick')
            target = await element.get_attribute('target')

            return ButtonInfo(
                element=element,
                selector=selector,
                text=text[:100],  # 限制长度
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