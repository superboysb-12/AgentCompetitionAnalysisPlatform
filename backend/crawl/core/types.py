"""
核心类型定义模块
定义爬虫系统中使用的所有数据类、枚举和协议接口
"""

import asyncio
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ClickStrategy(Enum):
    """
    点击策略枚举
    定义处理页面元素点击的不同策略类型
    """
    DIRECT_NAVIGATION = "direct_navigation"  # 直接导航到目标URL
    NEW_TAB_MONITORING = "new_tab_monitoring"  # 监听并处理新标签页
    SAME_PAGE_NAVIGATION = "same_page_navigation"  # 同页面内导航
    DIRECT_CLICK = "direct_click"  # 直接点击元素
    INTERNAL_LINK = "internal_link"  # 内部链接处理


class CrawlMode(Enum):
    """
    爬取模式枚举
    定义爬虫的工作模式
    """
    DIRECT = "direct"  # 直接爬取模式：直接访问目标页面
    INDIRECT = "indirect"  # 间接爬取模式：通过点击按钮进入内容页


@dataclass
class CrawlResult:
    """
    爬取结果数据类
    存储单次爬取任务的所有相关信息
    """
    url: str  # 实际爬取的URL
    original_url: str  # 原始起始URL
    content: Dict[str, Any]  # 提取的内容字典
    timestamp: str  # 爬取时间戳
    new_tab: bool = False  # 是否在新标签页中打开
    strategy_used: Optional[ClickStrategy] = None  # 使用的点击策略
    button_info: Dict[str, str] = field(default_factory=dict)  # 按钮相关信息


@dataclass
class ButtonInfo:
    """
    按钮信息数据类
    存储页面中可点击元素的详细信息
    """
    element: Any  # Playwright元素对象
    selector: str  # CSS选择器
    text: str  # 按钮文本内容
    href: Optional[str] = None  # 链接地址
    onclick: Optional[str] = None  # onclick属性
    target: Optional[str] = None  # target属性（如_blank）
    index: Optional[int] = None  # 在选择器结果中的索引


class ContentExtractor(Protocol):
    """
    内容提取器协议接口
    定义内容提取器必须实现的方法
    """
    async def extract(self, page: Any, rules: Dict[str, str]) -> Dict[str, Any]:
        """
        从页面中提取内容

        Args:
            page: Playwright页面对象
            rules: 提取规则字典，键为字段名，值为CSS选择器

        Returns:
            Dict[str, Any]: 提取的内容字典
        """
        ...


class ClickHandler(Protocol):
    """
    点击处理器协议接口
    定义点击处理器必须实现的方法
    """
    async def click_and_extract(self, page: Any, button: ButtonInfo, config: Dict[str, Any]) -> Optional[CrawlResult]:
        """
        点击元素并提取内容

        Args:
            page: Playwright页面对象
            button: 按钮信息
            config: 配置字典

        Returns:
            Optional[CrawlResult]: 爬取结果，失败时返回None
        """
        ...


class PageLoadDetector(Protocol):
    """
    页面加载检测器协议接口
    定义页面加载检测器必须实现的方法
    """
    async def wait_for_load(self, page: Any, config: Dict[str, Any]) -> None:
        """
        等待页面完全加载

        Args:
            page: Playwright页面对象
            config: 配置字典

        Returns:
            None
        """
        ...


class ConfigValidator(Protocol):
    """
    配置验证器协议接口
    定义配置验证器必须实现的方法
    """
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        验证配置的有效性

        Args:
            config: 配置字典

        Returns:
            bool: 配置是否有效
        """
        ...


class Storage(Protocol):
    """
    存储器协议接口
    定义存储器必须实现的方法
    """
    async def save(self, task_name: str, results: List[CrawlResult]) -> None:
        """
        保存爬取结果

        Args:
            task_name: 任务名称
            results: 爬取结果列表

        Returns:
            None
        """
        ...
