#!/usr/bin/env python3
"""
SAEC Product Scraper
- Scrapes product info.
- Downloads brochure PDFs into named subfolders.
- Captures screenshots of each <section> on product pages using a stable rendering strategy.
- Saves the product detail page URL to a JSON file in each product's folder.
"""

import os
import re
import time
import logging
import json  # <<< 新增：导入json模块
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- 全局设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 辅助函数 ---
def sanitize_foldername(name):
    """移除用作文件夹名称时的非法字符。"""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.strip()


def capture_section_screenshots(url, folder_path):
    """
    使用Selenium访问URL，并对页面中所有的<section>标签进行截图。
    """
    logger.info(f"开始为 {url} 进行分段截图...")
    os.makedirs(folder_path, exist_ok=True)

    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.page_load_strategy = 'none'

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        driver.get(url)

        wait = WebDriverWait(driver, 60)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        logger.info("页面Body已加载，等待页面稳定...")

        # --- 核心改进 1: 给予页面充足的“稳定期”来加载CSS和字体 ---
        time.sleep(7)

        # --- 核心改进 2: 预滚动页面，触发所有懒加载 ---
        logger.info("预滚动页面以触发所有懒加载内容...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)

        sections = driver.find_elements(By.TAG_NAME, 'section')
        if not sections:
            logger.warning(f"在页面 {url} 上没有找到 <section> 元素。")
            return

        logger.info(f"找到 {len(sections)} 个 <section> 元素，准备截图...")

        for i, section in enumerate(sections):
            try:
                # --- 核心改进 3: 使用最可靠的“等待元素可见” ---
                # 滚动到元素附近
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", section)

                # 智能等待，直到该元素在屏幕上可见
                WebDriverWait(driver, 10).until(EC.visibility_of(section))

                # 短暂等待动画效果结束
                time.sleep(1)

                if section.is_displayed() and section.size['height'] > 20:
                    screenshot_filename = os.path.join(folder_path, f"section_{i + 1}.png")
                    section.screenshot(screenshot_filename)
                    logger.info(f"成功截图: {screenshot_filename}")
                else:
                    logger.warning(f"Section {i + 1} 未显示或高度过小，跳过截图。")
            except Exception as e:
                logger.error(f"为 section {i + 1} 处理时失败: {e}")

    except TimeoutException:
        logger.error(f"等待页面Body超过60秒，页面可能无法访问或被阻止。跳过 {url} 的截图。")
    except WebDriverException as e:
        logger.error(f"Selenium处理URL {url} 时发生错误: {e}")
    finally:
        if driver:
            driver.quit()
        logger.info(f"截图流程结束: {url}")


# --- 主爬虫类 (无变动) ---
class SAECScraper:
    def __init__(self, base_url="https://www.saec.com.cn"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def download_file(self, url, folder_path):
        try:
            filename = unquote(url.split('/')[-1])
            save_path = os.path.join(folder_path, filename)
            if os.path.exists(save_path):
                logger.info(f"文件已存在，跳过下载: {save_path}")
                return save_path

            logger.info(f"正在下载 {url} 到 {save_path}")
            with self.session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"成功下载 {filename}")
            return save_path
        except requests.RequestException as e:
            logger.error(f"下载文件失败 {url}: {e}")
            return None

    def get_brochure_link(self, product_url):
        try:
            response = self.session.get(product_url, timeout=30)
            response.raise_for_status()
            match = re.search(r'(/download/[^\'"]+\.pdf)', response.text)
            if match:
                return urljoin(self.base_url, match.group(1))
            return None
        except requests.RequestException as e:
            logger.error(f"请求详情页失败 {product_url}: {e}")
            return None

    def scrape_products(self, url):
        try:
            logger.info(f"正在抓取产品列表: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            product_items = soup.select('#pprodjybg-bedroom .sec1 ul li')
            if not product_items:
                logger.error("在主页上未找到产品列表容器。")
                return []

            products = []
            for i, item in enumerate(product_items, 1):
                link_element = item.select_one('a')
                if not link_element: continue

                href = link_element.get('href', '').strip()
                if not href: continue

                product_link = urljoin(self.base_url, href)
                name_element = item.select_one('.prod-name')
                product_name = name_element.get_text(strip=True) if name_element else f"未知产品_{i}"

                products.append({
                    'position': i,
                    'name': product_name,
                    'link': product_link,
                })
            logger.info(f"从列表页成功抓取 {len(products)} 个产品。")
            return products
        except requests.RequestException as e:
            logger.error(f"请求产品列表页失败: {e}")
            return []


# --- 主执行函数 ---
def main():
    # 家用
    # 挂式
    target_url = "https://www.saec.com.cn/site/prodjybg-bedroom"
    # "https://www.saec.com.cn/site/prodjybg-room"
    # 柜式
    # "https://www.saec.com.cn/site/prodjygs"
    # 内藏式风管机
    # "https://www.saec.com.cn/site/prodsync-home"
    # 方向嵌机
    # “https://www.saec.com.cn/site/prodsyqr-home”

    # 商用
    # 内藏式风管机
    # "https://www.saec.com.cn/site/prodsync"
    # 嵌入式空调
    # "https://www.saec.com.cn/site/prodsyqr"
    # 柜机
    # "https://www.saec.com.cn/site/prodsygs"
    base_download_folder = "saec_brochures"
    os.makedirs(base_download_folder, exist_ok=True)

    scraper = SAECScraper()
    products = scraper.scrape_products(target_url)

    if not products:
        logger.info("未找到任何产品，程序退出。")
        return

    for product in products:
        product_name = product['name']
        product_link = product['link']

        logger.info(f"--- 正在处理产品: {product_name} ---")

        product_folder_name = sanitize_foldername(product_name)
        product_folder_path = os.path.join(base_download_folder, product_folder_name)
        os.makedirs(product_folder_path, exist_ok=True)

        # --- 新增代码块开始 ---
        # 保存产品URL到JSON文件
        product_info = {
            'product_page_url': product_link
        }
        json_filename = os.path.join(product_folder_path, 'product_info.json')
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(product_info, f, ensure_ascii=False, indent=4)
            logger.info(f"产品详情已保存到: {json_filename}")
        except IOError as e:
            logger.error(f"无法写入JSON文件 {json_filename}: {e}")
        # --- 新增代码块结束 ---

        brochure_link = scraper.get_brochure_link(product_link)
        if brochure_link:
            scraper.download_file(brochure_link, product_folder_path)
        else:
            logger.warning(f"未找到 {product_name} 的宣传手册链接。")

        screenshots_folder_path = os.path.join(product_folder_path, "screenshots")
        capture_section_screenshots(product_link, screenshots_folder_path)

        time.sleep(1)

    logger.info("所有产品处理完毕！")


if __name__ == "__main__":
    main()