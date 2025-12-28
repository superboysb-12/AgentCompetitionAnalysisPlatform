#!/usr/bin/env python3
"""
智能爬虫程序入口文件
提供命令行接口来启动爬虫任务
"""

import asyncio
import sys
import logging
from crawler import SmartCrawler

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def main():
    """
    主函数：解析命令行参数，初始化并启动爬虫

    Args:
        无直接参数，从 sys.argv 读取配置文件路径

    Returns:
        None

    Raises:
        Exception: 当爬虫执行失败时打印错误信息
    """
    # 从命令行参数获取配置文件路径，默认为 config.yaml
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'

    try:
        # 创建爬虫实例并启动
        crawler = SmartCrawler(config_file)
        await crawler.start()
    except Exception as e:
        print(f"爬取失败: {e}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
