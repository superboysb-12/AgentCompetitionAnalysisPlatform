import requests
import time
import json
import os
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging


class GreeSpecificCrawler:
    def __init__(self, base_url="https://www.gree.com/", output_dir="gree_specific_products"):
        """
        初始化爬虫
        Args:
            base_url: 格力官网基础URL
            output_dir: 数据保存目录
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = requests.Session()

        # 指定的产品页面URL列表
        self.target_urls = [
            "https://www.gree.com/cmsProduct/view/2255",
            "https://www.gree.com/cmsProduct/view/1233",
            "https://www.gree.com/cmsProduct/view/1232",
            "https://www.gree.com/cmsProduct/view/1231"

        ]

        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        os.makedirs(output_dir, exist_ok=True)
        self.products_dir = output_dir
        os.makedirs(self.products_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{output_dir}/crawler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_driver(self):
        """设置Selenium WebDriver"""
        chrome_options = Options()
        # chrome_options.add_argument('--headless')  # 暂时关闭无头模式以便调试
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(60)
            # 添加反自动化检测的脚本
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            self.logger.error(f"Chrome driver setup failed: {e}")
            return None

    def create_product_image_folder(self, product_name, product_index):
        """
        为每个产品创建独立的图片文件夹，使用产品名称命名
        Args:
            product_name: 产品名称
            product_index: 产品序号
        Returns:
            str: 图片文件夹路径
        """
        # 清理产品名称，创建安全的文件夹名称
        safe_name = re.sub(r'[<>:"/\\|?*]', '', product_name.strip())
        safe_name = re.sub(r'[^\w\s\u4e00-\u9fff\-]', '', safe_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name[:100]

        # 如果处理后的名称为空，使用备用名称
        if not safe_name:
            safe_name = f"产品_{product_index}"

        # 避免重复文件夹名称
        counter = 1
        original_safe_name = safe_name
        product_folder = os.path.join(self.products_dir, safe_name)
        while os.path.exists(product_folder):
            safe_name = f"{original_safe_name}_{counter}"
            product_folder = os.path.join(self.products_dir, safe_name)
            counter += 1

        # 创建产品主文件夹
        os.makedirs(product_folder, exist_ok=True)

        # 创建product和introduction子文件夹
        product_img_folder = os.path.join(product_folder, "product")
        introduction_img_folder = os.path.join(product_folder, "introduction")
        os.makedirs(product_img_folder, exist_ok=True)
        os.makedirs(introduction_img_folder, exist_ok=True)

        self.logger.info(f"创建产品文件夹: {safe_name}")
        return product_folder

    def extract_specific_images(self, driver, product_name, product_index):
        """
        提取产品图片和海报图片
        Args:
            driver: WebDriver实例
            product_name: 产品名称
            product_index: 产品序号
        Returns:
            dict: 包含提取的图片信息
        """
        images_info = {
            'product_images': [],
            'poster_images': [],
            'product_folder': ''
        }
        product_folder = self.create_product_image_folder(product_name, product_index)
        images_info['product_folder'] = product_folder

        try:
            # 等待图片区域加载
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "product-details-intro")))
            self.logger.info("产品详情区域已加载。")

            # 1. 提取产品图片
            product_img_elements = []
            product_img_selectors = [
                "//div[@id='pika-list']//ul/li//img",
                "//div[contains(@class, 'pika-thumb')]//li//img",
                "//div[contains(@class, 'jcarousel-list-vertical')]//img",
                "//div[@class='pikachoose']//img"
            ]

            for selector in product_img_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        self.logger.info(f"成功通过XPath找到 {len(elements)} 个产品图片元素: {selector}")
                        product_img_elements = elements
                        break
                except Exception:
                    continue

            if not product_img_elements:
                self.logger.warning("所有备选XPath均未能找到任何产品图片。")
            else:
                for element in product_img_elements:
                    src = element.get_attribute("src")
                    if not src:
                        src = element.get_attribute("data-src") or element.get_attribute("data-original")

                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        full_url = urljoin(self.base_url, src)
                        if full_url not in images_info['product_images']:
                            images_info['product_images'].append(full_url)
                            self.logger.info(f"找到产品图片: {full_url}")

            # 2. 提取海报图片
            poster_img_xpath_base = "/html/body/div[2]/div[2]/div[1]/div[3]/div/div/p[{}]/img"
            for i in range(1, 30):
                try:
                    poster_img_element = driver.find_element(By.XPATH, poster_img_xpath_base.format(i))
                    src = poster_img_element.get_attribute("src") or poster_img_element.get_attribute(
                        "data-src") or poster_img_element.get_attribute("data-original")
                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        full_url = urljoin(self.base_url, src)
                        if full_url not in images_info['poster_images']:
                            images_info['poster_images'].append(full_url)
                            self.logger.info(f"找到海报图片: {full_url}")
                except NoSuchElementException:
                    continue

            # 下载图片
            self.download_categorized_images(images_info, product_name)

        except TimeoutException:
            self.logger.error("等待产品详情区域加载超时。")
        except Exception as e:
            self.logger.error(f"提取指定图片时发生错误: {e}")

        return images_info

    def download_categorized_images(self, images_info, product_name):
        """
        下载分类的图片到对应文件夹
        Args:
            images_info: 图片信息字典
            product_name: 产品名称
        """
        product_folder = images_info['product_folder']
        product_img_folder = os.path.join(product_folder, "product")
        introduction_img_folder = os.path.join(product_folder, "introduction")

        try:
            # 下载产品图片到product文件夹
            for i, img_url in enumerate(images_info['product_images']):
                try:
                    response = self.session.get(img_url, timeout=15)
                    if response.status_code == 200:
                        ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
                        filename = f"product_{i + 1}{ext}"
                        filepath = os.path.join(product_img_folder, filename)

                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                        self.logger.info(f"下载产品图片: {filename}")

                except Exception as e:
                    self.logger.error(f"下载产品图片 {img_url} 失败: {e}")

            # 下载海报图片到introduction文件夹
            for i, img_url in enumerate(images_info['poster_images']):
                try:
                    response = self.session.get(img_url, timeout=15)
                    if response.status_code == 200:
                        ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
                        filename = f"introduction_{i + 1}{ext}"
                        filepath = os.path.join(introduction_img_folder, filename)

                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                        self.logger.info(f"下载介绍图片: {filename}")

                except Exception as e:
                    self.logger.error(f"下载海报图片 {img_url} 失败: {e}")

        except Exception as e:
            self.logger.error(f"下载分类图片时出错: {e}")

    def extract_product_description(self, driver):
        """
        提取产品信息区域的所有文字内容
        """
        description = ""
        info_container_xpath = "/html/body/div[2]/div[2]/div[1]/div[1]/div/div/div[2]"

        try:
            self.logger.info(f"正在提取产品信息: {info_container_xpath}")
            wait = WebDriverWait(driver, 10)
            info_container = wait.until(
                EC.presence_of_element_located((By.XPATH, info_container_xpath))
            )

            description = info_container.text
            if description:
                self.logger.info("成功提取到产品信息文本。")
            else:
                self.logger.warning("指定XPath的元素内未找到文本内容。")

        except TimeoutException:
            self.logger.error(f"等待产品信息容器加载超时，XPath: {info_container_xpath}")
        except Exception as e:
            self.logger.error(f"提取产品描述时出错: {e}")

        return description.strip()

    def extract_current_page_params(self, driver):
        """从当前可见的参数列表中提取参数"""
        parameters = {}
        try:
            wait = WebDriverWait(driver, 5)
            params_container = wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.prd-conf-list"))
            )
            param_elements = params_container.find_elements(By.CSS_SELECTOR, "div.col-lg-6.conf-list")

            for param_element in param_elements:
                text = param_element.text.strip()
                if '：' in text:
                    key, value = text.split('：', 1)
                    parameters[key.strip()] = value.strip()
        except Exception as e:
            self.logger.error(f"提取当前页面参数时出错: {e}")
        return parameters

    def extract_product_specifications(self, driver):
        """
        提取产品规格和参数
        """
        specifications = {}
        all_models_data = {}

        try:
            # 1. 点击"功能参数"选项卡
            self.logger.info("正在点击 '功能参数' 选项卡...")
            wait = WebDriverWait(driver, 10)
            param_tab = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), '功能参数')]"))
            )
            driver.execute_script("arguments[0].click();", param_tab)
            wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.prd-conf-tabs-info"))
            )
            self.logger.info("'功能参数' 选项卡内容已加载。")

            # 2. 找到左侧所有可点击的产品型号
            model_switcher_elements = driver.find_elements(By.CSS_SELECTOR, "ul.prd-conf-tab > li > a")
            if not model_switcher_elements:
                self.logger.warning("未找到产品型号切换按钮。将尝试爬取当前页面显示的参数。")
                page_params = self.extract_current_page_params(driver)
                if page_params:
                    model_name = page_params.get("产品型号", "default_model")
                    all_models_data[model_name] = page_params
                return specifications, all_models_data

            num_models = len(model_switcher_elements)
            self.logger.info(f"找到 {num_models} 个不同的产品型号。")

            # 3. 循环点击每个型号并抓取其参数
            for i in range(num_models):
                try:
                    current_model_button = driver.find_elements(By.CSS_SELECTOR, "ul.prd-conf-tab > li > a")[i]
                    model_name_from_button = current_model_button.text.strip()
                    if not model_name_from_button:
                        self.logger.warning(f"型号按钮 {i + 1} 没有文本，跳过。")
                        continue

                    self.logger.info(f"正在处理型号 ({i + 1}/{num_models}): {model_name_from_button}")

                    # 点击按钮加载此型号的参数
                    driver.execute_script("arguments[0].click();", current_model_button)
                    time.sleep(2)

                    # 抓取当前显示型号的参数表
                    current_model_params = self.extract_current_page_params(driver)

                    actual_model_name = current_model_params.get("产品型号", model_name_from_button)
                    all_models_data[actual_model_name] = current_model_params
                    self.logger.info(f"成功爬取型号 {actual_model_name} 的 {len(current_model_params)} 个参数。")

                except Exception as e:
                    self.logger.error(f"处理型号索引 {i} 时出错: {e}")
                    continue

        except TimeoutException:
            self.logger.error("点击或加载'功能参数'选项卡时超时。")
        except Exception as e:
            self.logger.error(f"提取产品规格时发生错误: {e}")

        return specifications, all_models_data

    def get_folder_name_from_description(self, description, fallback_name, product_index):
        """
        从描述的第一行获取用作文件夹名称的产品名
        """
        if description and description.strip():
            first_line = description.strip().split('\n')[0].strip()
            if first_line:
                first_line = ' '.join(first_line.split())
                self.logger.info(f"使用描述第一行作为文件夹名: {first_line}")
                return first_line

        self.logger.warning(f"描述为空或第一行为空，使用备用名称: {fallback_name}")
        return fallback_name if fallback_name else f"产品_{product_index}"

    def crawl_product_detail(self, product_url, product_index):
        """
        爬取单个产品的详细信息
        Args:
            product_url: 产品详情页URL
            product_index: 产品序号
        Returns:
            dict: 产品信息字典
        """
        product_data = {
            'url': product_url,
            'product_name': '',
            'images_info': {},
            'description': '',
            'specifications': {},
            'parameters': {}
        }

        driver = self.setup_driver()
        if not driver:
            return product_data

        try:
            self.logger.info(f"正在爬取产品: {product_url}")
            driver.get(product_url)
            time.sleep(3)

            # 1. 获取产品名称
            try:
                product_name_element = driver.find_element(By.CSS_SELECTOR, "h1, .product-title, .hotProduct-header")
                product_data['product_name'] = product_name_element.text.strip()
            except NoSuchElementException:
                product_data['product_name'] = driver.title.split('-')[0].strip()

            if not product_data['product_name']:
                product_data['product_name'] = f"产品_{product_index}"

            # 2. 获取产品描述
            product_data['description'] = self.extract_product_description(driver)

            # 3. 从描述的第一行获取用作文件夹名称的产品名
            folder_name = self.get_folder_name_from_description(
                product_data['description'], product_data['product_name'], product_index
            )

            # 4. 提取图片
            product_data['images_info'] = self.extract_specific_images(
                driver, folder_name, product_index
            )

            # 5. 获取产品规格和参数
            product_data['specifications'], product_data['parameters'] = self.extract_product_specifications(driver)

            self.logger.info(f"成功爬取产品: {product_data['product_name']}")

        except Exception as e:
            self.logger.error(f"爬取产品 {product_url} 时出错: {e}")
        finally:
            driver.quit()

        return product_data

    def crawl_specific_products(self):
        """爬取指定的产品页面"""
        all_products_data = []

        self.logger.info(f"开始爬取 {len(self.target_urls)} 个指定的产品页面")

        for i, product_url in enumerate(self.target_urls, 1):
            try:
                # 爬取单个产品
                product_data = self.crawl_product_detail(product_url, i)
                product_data['crawl_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                product_data['index'] = i

                all_products_data.append(product_data)

                # 保存产品数据到对应的产品文件夹中
                product_folder = product_data.get('images_info', {}).get('product_folder', '')
                if product_folder and os.path.exists(product_folder):
                    json_filename = "product_data.json"
                    json_filepath = os.path.join(product_folder, json_filename)

                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(product_data, f, ensure_ascii=False, indent=2)

                    self.logger.info(f"产品数据已保存到: {json_filepath}")

                self.logger.info(f"完成产品 {i}/{len(self.target_urls)}: {product_data['product_name']}")

                # 添加延时避免过于频繁的请求
                time.sleep(3)

            except Exception as e:
                self.logger.error(f"爬取产品 {product_url} 失败: {e}")
                continue

        return all_products_data

    def run(self):
        """运行爬虫"""
        self.logger.info("开始爬取格力官网指定产品数据")
        start_time = time.time()

        try:
            # 爬取指定产品
            products_data = self.crawl_specific_products()

            if products_data:
                # 保存汇总数据
                summary_file = os.path.join(self.output_dir, "products_summary.json")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(products_data, f, ensure_ascii=False, indent=2)

                end_time = time.time()
                duration = end_time - start_time

                self.logger.info(f"爬虫执行完成！总耗时: {duration:.2f}秒")
                self.logger.info(f"成功爬取 {len(products_data)} 个产品")
                self.logger.info(f"汇总数据保存在: {summary_file}")
                return products_data
            else:
                self.logger.warning("未找到任何产品数据")
                return []

        except Exception as e:
            self.logger.error(f"爬虫运行时出错: {e}")
            return []


def main():
    """主函数"""
    try:
        # 创建爬虫实例
        crawler = GreeSpecificCrawler(output_dir="gree_specific_products")

        # 运行爬虫
        results = crawler.run()

        if results:
            print(f"\n爬虫执行完成！共爬取 {len(results)} 个产品页面")
            print(f"数据保存在: {os.path.abspath(crawler.output_dir)}")
            print("\n爬取的产品URL列表:")
            for i, url in enumerate(crawler.target_urls, 1):
                print(f"{i}. {url}")
        else:
            print("爬取失败，请检查日志文件")

    except KeyboardInterrupt:
        print("\n爬虫被用户中断")
    except Exception as e:
        print(f"爬虫执行出错: {e}")


if __name__ == "__main__":
    main()