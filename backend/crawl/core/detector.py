"""
页面加载检测模块
提供多种策略检测页面是否完全加载，包括DOM、网络、JavaScript和动态内容
"""

from typing import Dict, Any
from playwright.async_api import Page
import asyncio
import logging

logger = logging.getLogger(__name__)


class PageLoadDetector:
    """
    页面加载检测器类
    智能等待页面完全加载，支持快速模式和标准模式
    """

    async def wait_for_load(self, page: Page, config: Dict[str, Any]) -> None:
        """
        等待页面完全加载的主入口方法

        根据配置选择快速模式或标准模式进行页面加载检测

        Args:
            page: Playwright页面对象
            config: 任务配置字典，包含等待策略设置

        Returns:
            None
        """
        fast_mode = config.get('fast_mode', False)

        if fast_mode:
            await self._fast_wait_for_load(page, config)
        else:
            # 标准模式：执行完整的加载检测流程
            await self._wait_for_dom(page)
            await self._wait_for_network(page, config)
            await self._wait_for_javascript(page, config)
            await self._wait_for_dynamic_content(page, config)

    async def _fast_wait_for_load(self, page: Page, config: Dict[str, Any]) -> None:
        """
        快速等待模式

        使用简化的等待策略，适用于加载速度快的页面

        Args:
            page: Playwright页面对象
            config: 任务配置字典

        Returns:
            None
        """
        try:
            # 等待body元素出现
            await page.wait_for_selector('body', timeout=10000)

            # 等待DOM内容加载完成
            await page.wait_for_load_state('domcontentloaded', timeout=15000)

            # 等待JavaScript执行
            js_wait_time = config.get('fast_js_wait_time', 2)
            await asyncio.sleep(js_wait_time)

            logger.info("快速等待模式完成")

        except Exception as e:
            logger.warning(f"快速等待失败，回退到标准模式: {e}")
            await self._wait_for_network(page, config)
            await self._wait_for_javascript(page, config)

    async def _wait_for_dom(self, page: Page) -> None:
        """
        等待DOM基本结构加载完成

        Args:
            page: Playwright页面对象

        Returns:
            None
        """
        try:
            await page.wait_for_selector('body', timeout=60000)
        except Exception as e:
            logger.warning(f"DOM等待超时: {e}")

    async def _wait_for_network(self, page: Page, config: Dict[str, Any]) -> None:
        """
        等待网络请求完成

        支持多次重试和networkidle状态检测

        Args:
            page: Playwright页面对象
            config: 任务配置字典

        Returns:
            None
        """
        max_attempts = config.get('network_wait_attempts', 2)
        network_timeout = config.get('network_timeout', 15000)

        for attempt in range(max_attempts):
            try:
                # 等待DOM内容加载
                await page.wait_for_load_state('domcontentloaded', timeout=network_timeout)

                # 如果配置要求，等待网络空闲
                if config.get('wait_for_networkidle', False):
                    await page.wait_for_load_state('networkidle', timeout=10000)
                break
            except Exception as e:
                logger.debug(f"网络等待第{attempt+1}次超时: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)

    async def _wait_for_javascript(self, page: Page, config: Dict[str, Any]) -> None:
        """
        等待JavaScript代码执行完成

        通过简单的延迟等待JavaScript渲染

        Args:
            page: Playwright页面对象
            config: 任务配置字典

        Returns:
            None
        """
        js_wait_time = config.get('js_wait_time', 3)
        await asyncio.sleep(js_wait_time)

    async def _wait_for_dynamic_content(self, page: Page, config: Dict[str, Any]) -> None:
        """
        等待动态内容稳定

        通过检测内容变化来判断动态渲染是否完成

        Args:
            page: Playwright页面对象
            config: 任务配置字典

        Returns:
            None
        """
        if not config.get('check_content_stability', True):
            return

        max_attempts = config.get('content_stability_attempts', 3)
        stable_count = 0
        prev_content_hash = None
        check_interval = config.get('stability_check_interval', 1.5)

        for attempt in range(max_attempts):
            try:
                # 查询主要内容元素
                content_elements = await page.query_selector_all(
                    'div.article-card-container, .article-card, .content, h1, h2'
                )

                # 提取内容文本用于比较
                content_text = ""
                for elem in content_elements[:3]:
                    try:
                        text = await elem.text_content()
                        if text:
                            content_text += text[:30]
                    except:
                        continue

                current_hash = hash(content_text) if content_text else 0

                # 检查内容是否稳定（连续两次相同）
                if current_hash == prev_content_hash:
                    stable_count += 1
                    if stable_count >= 1:
                        break
                else:
                    stable_count = 0
                    prev_content_hash = current_hash

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.debug(f"动态内容检测异常: {e}")
                break
