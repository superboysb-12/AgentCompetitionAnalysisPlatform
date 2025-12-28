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


class GreeCrawler:
    def __init__(self, base_url="https://www.gree.com/", output_dir="gree_products", target_category=None):
        """
        初始化爬虫
        Args:
            base_url: 格力官网基础URL
            output_dir: 数据保存目录
            target_category: (可选) 指定要爬取的单一产品类别名称
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.target_category = target_category
        self.session = requests.Session()

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
        chrome_options.add_argument('--headless')  # 暂时关闭无头模式以便调试
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
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            self.logger.error(f"Chrome driver setup failed: {e}")
            return None

    def get_all_product_categories(self, driver):
        """
        获取中央空调产品类别的产品链接
        """
        all_categories = {}
        target_url = "https://www.gree.com/cmsProduct/list/51"

        try:
            self.logger.info(f"正在访问产品页面: {target_url}")
            driver.get(target_url)

            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.row.category-result-list")))

            all_categories_config = []

            try:
                page_count = 1
                while True:
                    self.logger.info(f"正在处理产品系列第 {page_count} 页")

                    category_elements = driver.find_elements(By.CSS_SELECTOR, ".allproductCategory-item")
                    current_page_categories = []

                    for i, element in enumerate(category_elements):
                        try:
                            category_name = element.get_attribute("data-categoryname")
                            if category_name:
                                if not any(cat['name'] == category_name for cat in all_categories_config):
                                    is_default = "active" in element.get_attribute("class")
                                    category_config = {
                                        'name': category_name,
                                        'tab_text': category_name,
                                        'is_default': is_default,
                                        'element_index': i,
                                        'discovery_page': page_count,
                                        'selector': f'.allproductCategory-item[data-categoryname="{category_name}"]'
                                    }
                                    current_page_categories.append(category_config)
                                    all_categories_config.append(category_config)
                        except Exception as e:
                            self.logger.warning(f"处理第 {i} 个选项卡时出错: {e}")
                            continue

                    if not current_page_categories:
                        self.logger.warning(f"第 {page_count} 页未发现新的产品系列")

                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, ".swiper-gree-next")
                        if next_button.is_displayed() and "swiper-button-disabled" not in next_button.get_attribute(
                                "class"):

                            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                                                  next_button)
                            time.sleep(1)
                            driver.execute_script("arguments[0].click();", next_button)

                            time.sleep(2)
                            page_count += 1

                        else:
                            self.logger.info("下一页按钮不可用或已到达最后一页，停止分页")
                            break
                    except Exception as e:
                        self.logger.info(f"未找到下一页按钮或已到达最后一页: {e}")
                        break

                try:
                    reset_attempts = 0
                    max_reset_attempts = 10
                    while reset_attempts < max_reset_attempts:
                        try:
                            prev_button = driver.find_element(By.CSS_SELECTOR, ".swiper-gree-prev")
                            if prev_button.is_displayed() and "swiper-button-disabled" not in prev_button.get_attribute(
                                    "class"):
                                driver.execute_script("arguments[0].click();", prev_button)
                                time.sleep(1)
                                reset_attempts += 1
                            else:
                                break
                        except:
                            break
                except Exception as e:
                    self.logger.warning(f"重置到第一页时出错: {e}")

                for category in all_categories_config:
                    category['page'] = self.find_category_actual_page(driver, category['name'])

                categories_config = all_categories_config

                if not categories_config:
                    self.logger.error("未发现任何产品系列选项卡")
                    return {}

                self.logger.info(f"总共发现 {len(categories_config)} 个产品系列，分布在 {page_count} 页中")

            except Exception as e:
                self.logger.error(f"动态发现产品系列时出错: {e}")
                return {}

            categories_to_process = categories_config

            categories_by_page = {}
            for category in categories_to_process:
                page = category.get('page', 1)
                if page not in categories_by_page:
                    categories_by_page[page] = []
                categories_by_page[page].append(category)

            for page_num in sorted(categories_by_page.keys()):
                page_categories = categories_by_page[page_num]
                self.logger.info(f"开始处理第 {page_num} 页的 {len(page_categories)} 个产品系列")

                if page_num > 1:
                    try:
                        reset_attempts = 0
                        max_reset_attempts = 10
                        while reset_attempts < max_reset_attempts:
                            try:
                                prev_button = driver.find_element(By.CSS_SELECTOR, ".swiper-gree-prev")
                                if prev_button.is_displayed() and "swiper-button-disabled" not in prev_button.get_attribute(
                                        "class"):
                                    driver.execute_script("arguments[0].click();", prev_button)
                                    time.sleep(1)
                                    reset_attempts += 1
                                else:
                                    break
                            except:
                                break
                    except:
                        pass

                    current_page = 1
                    while current_page < page_num:
                        try:
                            next_button = driver.find_element(By.CSS_SELECTOR, ".swiper-gree-next")
                            if next_button.is_displayed() and "swiper-button-disabled" not in next_button.get_attribute(
                                    "class"):
                                driver.execute_script("arguments[0].click();", next_button)
                                time.sleep(2)
                                current_page += 1
                            else:
                                self.logger.error(f"无法导航到第 {page_num} 页")
                                break
                        except Exception as e:
                            self.logger.error(f"导航到第 {page_num} 页时出错: {e}")
                            break

                for category in page_categories:
                    try:
                        category_name = category['name']
                        self.logger.info(f"开始爬取 {category_name} 类别 (第 {page_num} 页)")

                        if not category['is_default']:
                            custom_selector = category.get('selector', '')
                            if custom_selector:
                                time.sleep(2)

                                try:
                                    # 滚动到产品系列选项卡容器
                                    category_container = driver.find_element(By.CSS_SELECTOR,
                                                                             "#allproductCategory-list")
                                    driver.execute_script(
                                        "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                                        category_container)
                                    time.sleep(1)
                                except Exception as e:
                                    self.logger.debug(f"滚动到选项卡容器失败: {e}")

                                click_attempts = 0
                                max_click_attempts = 5
                                clicked_successfully = False

                                while click_attempts < max_click_attempts and not clicked_successfully:
                                    try:
                                        tab_element = driver.find_element(By.CSS_SELECTOR, custom_selector)
                                        if tab_element.is_displayed() and tab_element.is_enabled():
                                            driver.execute_script(
                                                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                                                tab_element)
                                            time.sleep(1)

                                            driver.execute_script("arguments[0].click();", tab_element)

                                            time.sleep(3)
                                            wait.until(EC.presence_of_element_located(
                                                (By.CSS_SELECTOR, "ul.row.category-result-list")))
                                            clicked_successfully = True
                                        else:
                                            self.logger.warning(
                                                f"元素不可见或不可点击 (尝试 {click_attempts + 1}): {custom_selector}")

                                            if click_attempts == 1:
                                                try:
                                                    prev_btn = driver.find_element(By.CSS_SELECTOR, ".swiper-gree-prev")
                                                    if prev_btn.is_displayed():
                                                        driver.execute_script("arguments[0].click();", prev_btn)
                                                        time.sleep(1)
                                                except:
                                                    pass
                                            elif click_attempts == 2:
                                                try:
                                                    next_btn = driver.find_element(By.CSS_SELECTOR, ".swiper-gree-next")
                                                    if next_btn.is_displayed():
                                                        driver.execute_script("arguments[0].click();", next_btn)
                                                        time.sleep(1)
                                                except:
                                                    pass

                                            click_attempts += 1
                                            time.sleep(1)
                                    except Exception as e:
                                        self.logger.warning(
                                            f"尝试 {click_attempts + 1} 点击 {category_name} 时出错: {e}")
                                        click_attempts += 1
                                        time.sleep(1)

                                if not clicked_successfully:
                                    self.logger.error(f"多次尝试后仍无法点击 {category_name} 选项卡")
                                    continue
                            else:
                                self.logger.warning(f"未为 {category_name} 配置选择器")
                                continue
                        else:
                            self.logger.info(f"{category_name} 是默认类别，无需点击")

                        product_links = self.get_visible_product_links(driver)

                        if product_links:
                            all_categories[category_name] = product_links
                            self.logger.info(f"{category_name} 获取到 {len(product_links)} 个产品链接")
                        else:
                            self.logger.warning(f"{category_name} 未找到产品链接")

                    except Exception as e:
                        self.logger.error(f"处理 {category['name']} 类别时出错: {e}")
                        continue

        except TimeoutException:
            self.logger.error("访问产品页面超时")
        except Exception as e:
            self.logger.error(f"获取产品分类时发生严重错误: {e}")

        return all_categories

    def find_category_actual_page(self, driver, category_name):
        """
        从第一页开始查找产品系列的实际位置
        Args:
            driver: WebDriver实例
            category_name: 产品系列名称
        Returns:
            int: 产品系列所在的页面号
        """
        try:
            current_page = 1
            max_pages = 5

            while current_page <= max_pages:
                try:
                    element = driver.find_element(By.CSS_SELECTOR,
                                                  f'.allproductCategory-item[data-categoryname="{category_name}"]')
                    if element.is_displayed():
                        return current_page
                except:
                    pass

                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, ".swiper-gree-next")
                    if next_button.is_displayed() and "swiper-button-disabled" not in next_button.get_attribute(
                            "class"):
                        driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(1)
                        current_page += 1
                    else:
                        break
                except:
                    break

            self.logger.warning(f"未能找到 {category_name} 的实际位置，默认设为第1页")
            return 1

        except Exception as e:
            self.logger.error(f"查找 {category_name} 实际位置时出错: {e}")
            return 1

    def get_visible_product_links(self, driver):
        """
        获取当前页面可见的所有产品链接
        Returns:
            list: 当前可见的产品链接列表
        """
        visible_links = []

        try:
            xpath_selector = "//a[contains(@href, '/cmsProduct/view/')]"
            elements = driver.find_elements(By.XPATH, xpath_selector)

            for element in elements:
                try:
                    href = element.get_attribute("href")
                    if href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in visible_links:
                            visible_links.append(full_url)
                except Exception as e:
                    self.logger.debug(f"获取链接时出错: {e}")

        except Exception as e:
            self.logger.error(f"获取可见产品链接时出错: {e}")

        return visible_links

    def create_product_image_folder(self, product_name, product_index, category_name):
        """
        为每个产品创建独立的图片文件夹，在产品系列子文件夹下使用产品名称命名
        Args:
            product_name: 产品名称（来自描述的第一行）
            product_index: 产品序号
            category_name: 产品系列名称
        Returns:
            str: 图片文件夹路径
        """
        safe_category_name = re.sub(r'[<>:"/\\|?*]', '', category_name.strip())
        safe_category_name = re.sub(r'[^\w\s\u4e00-\u9fff\-]', '', safe_category_name)
        safe_category_name = re.sub(r'\s+', '_', safe_category_name)

        category_folder = os.path.join(self.products_dir, safe_category_name)
        os.makedirs(category_folder, exist_ok=True)

        safe_name = re.sub(r'[<>:"/\\|?*]', '', product_name.strip())
        safe_name = re.sub(r'[^\w\s\u4e00-\u9fff\-]', '', safe_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name[:100]

        if not safe_name:
            safe_name = f"{category_name}_产品_{product_index}"

        counter = 1
        original_safe_name = safe_name
        product_folder = os.path.join(category_folder, safe_name)
        while os.path.exists(product_folder):
            safe_name = f"{original_safe_name}_{counter}"
            product_folder = os.path.join(category_folder, safe_name)
            counter += 1

        os.makedirs(product_folder, exist_ok=True)

        product_img_folder = os.path.join(product_folder, "product")
        introduction_img_folder = os.path.join(product_folder, "introduction")
        os.makedirs(product_img_folder, exist_ok=True)
        os.makedirs(introduction_img_folder, exist_ok=True)

        return product_folder

    def extract_specific_images(self, driver, product_name, product_index, category_name):
        """
        提取产品图片和产品介绍内容
        Args:
            driver: WebDriver实例
            product_name: 产品名称
            product_index: 产品序号
            category_name: 产品类别
        Returns:
            dict: 包含提取的图片信息和介绍文本
        """
        images_info = {
            'product_images': [],
            'introduction_images': [],
            'introduction_text': '',
            'product_folder': ''
        }
        product_folder = self.create_product_image_folder(product_name, product_index, category_name)
        images_info['product_folder'] = product_folder

        try:
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "product-details-intro")))

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

            # 2. 提取产品介绍区域的内容（#product-details-list > div）
            try:
                intro_container = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#product-details-list > div"))
                )

                intro_text = intro_container.text.strip()
                if intro_text:
                    images_info['introduction_text'] = intro_text

                intro_images = intro_container.find_elements(By.TAG_NAME, "img")

                for idx, img_element in enumerate(intro_images, 1):
                    try:
                        src = img_element.get_attribute("src")
                        if not src:
                            src = img_element.get_attribute("data-src") or \
                                  img_element.get_attribute("data-original") or \
                                  img_element.get_attribute("data-lazy")

                        if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                            full_url = urljoin(self.base_url, src)
                            if full_url not in images_info['introduction_images']:
                                images_info['introduction_images'].append(full_url)
                    except Exception as e:
                        self.logger.debug(f"提取介绍图片 {idx} 时出错: {e}")

            except TimeoutException:
                self.logger.warning("未找到产品介绍区域 #product-details-list > div")
            except Exception as e:
                self.logger.error(f"提取产品介绍内容时出错: {e}")

            self.download_categorized_images(images_info, product_name, category_name)

        except TimeoutException:
            self.logger.error("等待产品详情区域加载超时，页面可能未完全显示。")
        except Exception as e:
            self.logger.error(f"提取指定图片时发生未知错误: {e}")

        return images_info

    def download_categorized_images(self, images_info, product_name, category_name):
        """
        下载分类的图片到对应文件夹
        Args:
            images_info: 图片信息字典
            product_name: 产品名称
            category_name: 产品类别
        """
        product_folder = images_info['product_folder']
        product_img_folder = os.path.join(product_folder, "product")
        introduction_img_folder = os.path.join(product_folder, "introduction")

        try:
            for i, img_url in enumerate(images_info['product_images']):
                try:
                    response = self.session.get(img_url, timeout=15)
                    if response.status_code == 200:
                        ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
                        filename = f"product_{i + 1}{ext}"
                        filepath = os.path.join(product_img_folder, filename)

                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                except Exception as e:
                    self.logger.error(f"下载产品图片 {img_url} 失败: {e}")

            for i, img_url in enumerate(images_info.get('introduction_images', [])):
                try:
                    response = self.session.get(img_url, timeout=15)
                    if response.status_code == 200:
                        ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
                        filename = f"introduction_{i + 1}{ext}"
                        filepath = os.path.join(introduction_img_folder, filename)

                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                except Exception as e:
                    self.logger.error(f"下载介绍图片 {img_url} 失败: {e}")

        except Exception as e:
            self.logger.error(f"下载分类图片时出错: {e}")

    def crawl_product_detail(self, product_url, product_index, category_name):
        """
        爬取单个产品的详细信息
        Args:
            product_url: 产品详情页URL
            product_index: 产品序号
            category_name: 产品类别
        Returns:
            dict: 产品信息字典
        """
        product_data = {
            'url': product_url,
            'category': category_name,
            'product_name': '',
            'description': '',
            'introduction_text': '',
            'images_info': {},
            'specifications': {},
            'parameters': {}
        }

        driver = self.setup_driver()
        if not driver:
            self.logger.error(f"无法为产品 {product_url} 启动浏览器驱动")
            return product_data

        try:
            self.logger.info(f"正在爬取{category_name}产品: {product_url}")
            driver.get(product_url)
            time.sleep(3)

            # 1. 获取产品名称
            try:
                product_name_element = driver.find_element(By.CSS_SELECTOR, "h1, .product-title, .hotProduct-header")
                product_data['product_name'] = product_name_element.text.strip()
            except NoSuchElementException:
                try:
                    product_data['product_name'] = driver.title.split('-')[0].strip()
                except:
                    product_data['product_name'] = f"{category_name}_Product_{product_index}"

            if not product_data['product_name']:
                product_data['product_name'] = f"{category_name}_Product_{product_index}"

            # 2. 先获取产品描述
            try:
                product_data['description'] = self.extract_product_description(driver)
            except Exception as e:
                self.logger.error(f"提取产品描述失败: {e}")
                product_data['description'] = ""

            # 3. 从描述的第一行获取用作文件夹名称的产品名
            folder_name = self.get_folder_name_from_description(product_data['description'],
                                                                product_data['product_name'], product_index,
                                                                category_name)

            # 4. 使用描述第一行作为文件夹名提取指定图片
            try:
                images_info = self.extract_specific_images(
                    driver, folder_name, product_index, category_name
                )
                product_data['images_info'] = images_info
                if images_info.get('introduction_text'):
                    product_data['introduction_text'] = images_info['introduction_text']

            except Exception as e:
                self.logger.error(f"提取产品图片失败: {e}")
                product_data['images_info'] = {'product_images': [], 'introduction_images': [], 'introduction_text': '',
                                               'product_folder': ''}

            # 5. 中央空调产品没有功能参数选项，跳过规格和参数提取
            product_data['specifications'] = {}
            product_data['parameters'] = {}

            self.logger.info(f"成功爬取{category_name}产品: {product_data['product_name']}")

        except KeyboardInterrupt:
            self.logger.info("产品爬取被用户中断")
            raise
        except Exception as e:
            self.logger.error(f"爬取{category_name}产品 {product_url} 时出错: {e}")
        finally:
            try:
                driver.quit()
            except Exception as e:
                self.logger.error(f"关闭产品详情页面浏览器时出错: {e}")

        return product_data

    def get_folder_name_from_description(self, description, fallback_name, product_index, category_name):
        """
        从描述的第一行获取用作文件夹名称的产品名
        Args:
            description: 产品描述文本
            fallback_name: 备用名称
            product_index: 产品序号
            category_name: 产品类别
        Returns:
            str: 用作文件夹名称的产品名
        """
        if description and description.strip():
            first_line = description.strip().split('\n')[0].strip()
            if first_line:
                first_line = ' '.join(first_line.split())
                return first_line

        return fallback_name if fallback_name else f"{category_name}_产品_{product_index}"

    def extract_product_description(self, driver):
        """
        使用精确XPath提取产品信息区域的所有文字内容。
        """
        description = ""
        info_container_xpath = "/html/body/div[2]/div[2]/div[1]/div[1]/div/div/div[2]"

        try:
            wait = WebDriverWait(driver, 10)
            info_container = wait.until(
                EC.presence_of_element_located((By.XPATH, info_container_xpath))
            )
            description = info_container.text

        except TimeoutException:
            self.logger.error(f"等待产品信息容器加载超时，XPath: {info_container_xpath}")
        except Exception as e:
            self.logger.error(f"提取产品描述时出错: {e}")

        return description.strip()

    def crawl_all_products(self):
        """爬取所有产品信息"""
        all_products_data = []

        driver = self.setup_driver()
        if not driver:
            self.logger.error("无法启动浏览器驱动")
            return []

        try:
            all_categories = self.get_all_product_categories(driver)

            total_products = sum(len(links) for links in all_categories.values())
            self.logger.info(f"总共需要爬取 {total_products} 个产品")

            product_counter = 1

            for category_name, product_links in all_categories.items():

                for i, product_url in enumerate(product_links, 1):
                    try:
                        self.logger.info(f"开始爬取产品 {product_counter}/{total_products}: {product_url}")

                        product_data = self.crawl_product_detail(product_url, i, category_name)

                        if product_data.get('product_name'):
                            product_data['global_index'] = product_counter
                            product_data['category_index'] = i
                            product_data['crawl_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

                            all_products_data.append(product_data)

                            product_folder = product_data.get('images_info', {}).get('product_folder', '')
                            if product_folder and os.path.exists(product_folder):
                                json_filename = "product_data.json"
                                json_filepath = os.path.join(product_folder, json_filename)

                                try:
                                    with open(json_filepath, 'w', encoding='utf-8') as f:
                                        json.dump(product_data, f, ensure_ascii=False, indent=2)
                                except Exception as save_error:
                                    self.logger.error(f"保存产品数据失败: {save_error}")

                        else:
                            self.logger.warning(f"产品数据无效，跳过: {product_url}")

                        product_counter += 1

                        time.sleep(3)

                    except KeyboardInterrupt:
                        self.logger.info("用户中断了爬取过程")
                        raise
                    except Exception as e:
                        self.logger.error(f"爬取{category_name}产品 {product_url} 失败: {e}")
                        product_counter += 1
                        continue

                self.logger.info(f"{category_name} 分类爬取完成")

        except KeyboardInterrupt:
            self.logger.info("爬取过程被用户中断")
        except Exception as e:
            self.logger.error(f"爬取过程中发生严重错误: {e}")
        finally:
            try:
                driver.quit()
                self.logger.info("浏览器驱动已关闭")
            except Exception as e:
                self.logger.error(f"关闭浏览器驱动时出错: {e}")

        return all_products_data

    def run(self):
        """运行爬虫"""
        self.logger.info("开始爬取格力官网产品数据")
        start_time = time.time()

        try:
            products_data = self.crawl_all_products()

            if products_data:
                end_time = time.time()
                duration = end_time - start_time

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
        crawler = GreeCrawler(output_dir="zykongtiao/shangyong")

        results = crawler.run()

        if results:
            print(f"\n爬虫执行完成！共爬取 {len(results)} 个产品页面")
        else:
            print("爬取失败，请检查日志文件")

    except KeyboardInterrupt:
        print("\n爬虫被用户中断")
    except Exception as e:
        print(f"爬虫执行出错: {e}")


if __name__ == "__main__":
    main()