#!/usr/bin/env python3

import asyncio
import sys
import logging
from crawler import SmartCrawler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'

    try:
        crawler = SmartCrawler(config_file)
        await crawler.start()
    except Exception as e:
        print(f"爬取失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
