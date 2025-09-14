"""
Package initialization for core modules
"""

from .types import CrawlResult, ButtonInfo, ClickStrategy
from .browser import BrowserManager
from .discovery import ButtonDiscovery
from .strategies import ClickStrategyManager
from .extractor import ContentExtractor
from .detector import PageLoadDetector
from .storage import StorageFactory

__all__ = [
    'CrawlResult',
    'ButtonInfo',
    'ClickStrategy',
    'BrowserManager',
    'ButtonDiscovery',
    'ClickStrategyManager',
    'ContentExtractor',
    'PageLoadDetector',
    'StorageFactory'
]