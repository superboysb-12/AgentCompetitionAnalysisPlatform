"""
模块系统

自动发现、加载和注册所有业务模块
"""
from .loader import module_loader

__all__ = ['module_loader']
