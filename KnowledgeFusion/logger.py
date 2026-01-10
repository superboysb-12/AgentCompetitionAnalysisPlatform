"""
统一日志系统
提供结构化的日志输出，支持控制台和文件
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """彩色控制台日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        # 添加颜色
        level_name = record.levelname
        if level_name in self.COLORS:
            record.levelname = f"{self.COLORS[level_name]}{level_name}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "knowledge_fusion",
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    设置日志系统

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件目录
        console: 是否输出到控制台
        file: 是否输出到文件

    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 日志格式
    detailed_format = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_format = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )

    # 控制台处理器（彩色）
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(
            '[%(levelname)s] %(message)s'
        ))
        logger.addHandler(console_handler)

    # 文件处理器
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 主日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_format)
        logger.addHandler(file_handler)

        logger.info(f"日志文件: {log_file}")

    return logger


# 预配置的logger实例
def get_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    获取预配置的logger实例

    Args:
        log_dir: 日志文件目录（如果为None，则不保存文件）

    Returns:
        logger实例
    """
    return setup_logger(
        name="knowledge_fusion",
        level=logging.INFO,
        log_dir=log_dir,
        console=True,
        file=log_dir is not None
    )


# 便捷函数
def log_section(logger: logging.Logger, title: str, width: int = 80):
    """输出分隔线"""
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_subsection(logger: logging.Logger, title: str, width: int = 60):
    """输出子分隔线"""
    logger.info("-" * width)
    logger.info(title)


def log_stats(logger: logging.Logger, stats: dict, title: str = "统计信息"):
    """输出统计信息"""
    log_subsection(logger, title)
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    # 测试日志系统
    logger = get_logger(log_dir=Path("output"))

    log_section(logger, "日志系统测试")

    logger.debug("这是一条DEBUG消息")
    logger.info("这是一条INFO消息")
    logger.warning("这是一条WARNING消息")
    logger.error("这是一条ERROR消息")

    log_stats(logger, {
        "总实体数": 1000,
        "融合后": 850,
        "融合率": 0.15
    })

    logger.info("测试完成")
