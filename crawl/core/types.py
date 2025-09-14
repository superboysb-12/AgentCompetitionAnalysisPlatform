"""
Smart Crawler - Clean Architecture

现代化的Web爬虫架构，基于2024年最佳实践
- 异步架构处理复杂JavaScript网站
- 模块化设计，清晰的职责分离
- 类型提示和错误处理
- 支持新标签页和复杂导航
"""

import asyncio
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ClickStrategy(Enum):
    DIRECT_NAVIGATION = "direct_navigation"
    NEW_TAB_MONITORING = "new_tab_monitoring"
    SAME_PAGE_NAVIGATION = "same_page_navigation"
    DIRECT_CLICK = "direct_click"
    INTERNAL_LINK = "internal_link"


class CrawlMode(Enum):
    """爬取模式枚举"""
    DIRECT = "direct"      # 直接爬取模式：直接爬取页面内容，不点击按钮
    INDIRECT = "indirect"  # 间接爬取模式：点击按钮后跳转到新页面爬取内容


@dataclass
class CrawlResult:
    url: str
    original_url: str
    content: Dict[str, Any]
    timestamp: str
    new_tab: bool = False
    strategy_used: Optional[ClickStrategy] = None
    button_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class ButtonInfo:
    element: Any
    selector: str
    text: str
    href: Optional[str] = None
    onclick: Optional[str] = None
    target: Optional[str] = None
    index: Optional[int] = None  # 添加索引字段用于重新定位


class ContentExtractor(Protocol):
    async def extract(self, page: Any, rules: Dict[str, str]) -> Dict[str, Any]:
        ...


class ClickHandler(Protocol):
    async def click_and_extract(self, page: Any, button: ButtonInfo, config: Dict[str, Any]) -> Optional[CrawlResult]:
        ...


class PageLoadDetector(Protocol):
    async def wait_for_load(self, page: Any, config: Dict[str, Any]) -> None:
        ...


class ConfigValidator(Protocol):
    def validate(self, config: Dict[str, Any]) -> bool:
        ...


class Storage(Protocol):
    async def save(self, task_name: str, results: List[CrawlResult]) -> None:
        ...