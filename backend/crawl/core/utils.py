"""
工具函数模块
提供爬虫系统中通用的辅助函数
"""

from urllib.parse import urljoin
import logging

logger = logging.getLogger(__name__)


def resolve_url(base_url: str, href: str) -> str:
    """
    解析相对URL为绝对URL

    Args:
        base_url: 基础URL（当前页面URL）
        href: 待解析的URL（可能是相对路径）

    Returns:
        str: 解析后的绝对URL
    """
    if not href:
        return href

    # 如果已经是完整URL或特殊协议，直接返回
    if href.startswith(('http://', 'https://', 'javascript:', '#')):
        return href

    try:
        # 使用urljoin拼接相对路径
        return urljoin(base_url, href)
    except Exception as e:
        logger.warning(f"URL解析失败: {base_url} + {href}, 错误: {e}")
        return href
