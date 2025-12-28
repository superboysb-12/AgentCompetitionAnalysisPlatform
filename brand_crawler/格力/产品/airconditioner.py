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
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            self.logger.error(f"Chrome driver setup failed: {e}")
            return None

    def get_all_product_categories(self, driver):
        """
        获取所有产品类别的产品链接
        包括挂式空调、柜式空调和特种空调
        """
        all_categories = {}
        target_url = "https://www.gree.com/cmsProduct/list/41"

        try:
            self.logger.info(f"正在访问产品页面: {target_url}")
            driver.get(target_url)

            # 等待页面加载完成
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.row.category-result-list")))
            self.logger.info("产品页面已加载完成")

            # 定义三个产品类别及其对应的选择器
            categories_config = [
                {
                    'name': '挂式空调',
                    'tab_text': '挂式空调',
                    'is_default': False
                },
                {
                    'name': '柜式空调',
                    'tab_text': '柜式空调',
                    'is_default': False
                },
                {
                    'name': '特种空调',
                    'tab_text': '特种空调',
                    'is_default': False
                }
            ]


            if self.target_category:
                self.logger.info(f"已指定目标类别: {self.target_category}")
                categories_to_process = [cat for cat in categories_config if cat['name'] == self.target_category]
                if not categories_to_process:
                    self.logger.error(f"指定的目标类别 '{self.target_category}' 未在配置中找到。程序将退出。")
                    return {}
            else:
                self.logger.info("未指定目标类别，将爬取所有类别。")
                categories_to_process = categories_config

            for category in categories_to_process:
                try:
                    category_name = category['name']
                    self.logger.info(f"开始爬取 {category_name} 类别")

                    if not category['is_default']:
                        tab_selectors = [
                            f"//a[@class='allproductCategory-item' and @data-categoryname='{category['tab_text']}']",
                            f"//a[.//p[@class='allcategory-name' and normalize-space(text())='{category['tab_text']}']]",
                            f"//*[normalize-space(text())='{category['tab_text']}']"
                        ]

                        tab_element = None
                        for selector in tab_selectors:
                            try:
                                elements = driver.find_elements(By.XPATH, selector)
                                for element in elements:
                                    if element.is_displayed() and element.is_enabled():
                                        tab_element = element
                                        break
                                if tab_element:
                                    break
                            except Exception as e:
                                self.logger.debug(f"选择器 {selector} 失败: {e}")
                                continue

                        if not tab_element:
                            self.logger.error(f"无法找到 {category_name} 选项卡元素")
                            continue

                        try:
                            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                                                  tab_element)
                            time.sleep(1)

                            click_success = False
                            try:
                                driver.execute_script("arguments[0].click();", tab_element)
                                click_success = True
                            except Exception as e:
                                self.logger.error(f"点击 {category_name} 选项卡时出错: {e}")
                                continue

                            time.sleep(2)
                            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.row.category-result-list")))

                        except Exception as e:
                            self.logger.error(f"点击或等待 {category_name} 选项卡时出错: {e}")
                            continue

                    product_links = self.extract_product_links_with_load_more(driver)

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

    def extract_product_links_with_load_more(self, driver):
        """
        从产品列表页面提取所有产品链接，包括点击"查看更多"按钮后的产品
        Returns:
            list: 产品链接列表
        """
        product_links = []
        wait = WebDriverWait(driver, 15)

        try:
            initial_links = self.get_visible_product_links(driver)
            product_links.extend(initial_links)
            self.logger.info(f"获取到初始产品链接数量: {len(initial_links)}")

            load_more_attempts = 0
            max_attempts = 10

            while load_more_attempts < max_attempts:
                try:
                    load_more_selectors = [
                        "//button[contains(text(), '查看更多')]",
                        "//a[contains(text(), '查看更多')]",
                        "//div[contains(text(), '查看更多')]",
                        "//span[contains(text(), '查看更多')]",
                        "//*[contains(@class, 'load-more')]",
                        "//*[contains(@class, 'show-more')]"
                    ]

                    load_more_button = None
                    for selector in load_more_selectors:
                        try:
                            elements = driver.find_elements(By.XPATH, selector)
                            for element in elements:
                                if element.is_displayed() and element.is_enabled():
                                    load_more_button = element
                                    break
                            if load_more_button:
                                break
                        except:
                            continue

                    if not load_more_button:
                        self.logger.info("未找到可点击的'查看更多'按钮，可能已显示所有产品")
                        break

                    before_click_count = len(self.get_visible_product_links(driver))

                    driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", load_more_button)


                    time.sleep(3)

                    after_click_count = len(self.get_visible_product_links(driver))
                    if after_click_count > before_click_count:
                        new_links = self.get_visible_product_links(driver)
                        for link in new_links:
                            if link not in product_links:
                                product_links.append(link)
                    else:
                        self.logger.info("点击后没有加载新产品，可能已到达列表末尾")
                        break

                    load_more_attempts += 1

                except Exception as e:
                    self.logger.error(f"点击'查看更多'按钮时出错: {e}")
                    break

            self.logger.info(f"最终获取到总计 {len(product_links)} 个产品链接")

        except Exception as e:
            self.logger.error(f"提取产品链接时出错: {e}")

        return list(set(product_links))  # 返回去重后的列表

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
        为每个产品创建独立的图片文件夹，使用产品名称命名
        Args:
            product_name: 产品名称（来自描述的第一行）
            product_index: 产品序号
            category_name: 产品类别
        Returns:
            str: 图片文件夹路径
        """
        safe_name = re.sub(r'[<>:"/\\|?*]', '', product_name.strip())
        safe_name = re.sub(r'[^\w\s\u4e00-\u9fff\-]', '', safe_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name[:100]

        if not safe_name:
            safe_name = f"{category_name}_产品_{product_index}"

        counter = 1
        original_safe_name = safe_name
        product_folder = os.path.join(self.products_dir, safe_name)
        while os.path.exists(product_folder):
            safe_name = f"{original_safe_name}_{counter}"
            product_folder = os.path.join(self.products_dir, safe_name)
            counter += 1

        os.makedirs(product_folder, exist_ok=True)

        product_img_folder = os.path.join(product_folder, "product")
        introduction_img_folder = os.path.join(product_folder, "introduction")
        os.makedirs(product_img_folder, exist_ok=True)
        os.makedirs(introduction_img_folder, exist_ok=True)

        return product_folder

    def extract_specific_images(self, driver, product_name, product_index, category_name):
        """
        使用多种备选选择器和更强的等待机制来提取产品图片和海报图片。
        Args:
            driver: WebDriver实例
            product_name: 产品名称
            product_index: 产品序号
            category_name: 产品类别
        Returns:
            dict: 包含提取的图片信息
        """
        images_info = {
            'product_images': [],
            'poster_images': [],
            'product_folder': ''
        }
        product_folder = self.create_product_image_folder(product_name, product_index, category_name)
        images_info['product_folder'] = product_folder

        try:
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "product-details-intro")))
            self.logger.info("产品详情区域已加载。")

            # 1. 提取产品图片 - 使用多种备选选择器
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

            self.download_categorized_images(images_info, product_name, category_name)

        except TimeoutException:
            self.logger.error("等待产品详情区域加载超时，页面可能未完全显示。")
        except Exception as e:
            self.logger.error(f"提取指定图片时发生未知错误: {e}")

        return images_info

    def download_categorized_images(self, images_info, product_name, category_name):
        """
        下载分类的图片到对应文件夹，并在产品文件夹中保存产品数据JSON
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

            for i, img_url in enumerate(images_info['poster_images']):
                try:
                    response = self.session.get(img_url, timeout=15)
                    if response.status_code == 200:
                        ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
                        filename = f"introduction_{i + 1}{ext}"
                        filepath = os.path.join(introduction_img_folder, filename)

                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                except Exception as e:
                    self.logger.error(f"下载海报图片 {img_url} 失败: {e}")

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
            'images_info': {},
            'description': '',
            'specifications': {},
            'parameters': {}
        }

        driver = self.setup_driver()
        if not driver:
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
                product_data['product_name'] = driver.title.split('-')[0].strip()

            if not product_data['product_name']:
                product_data['product_name'] = f"{category_name}_Product_{product_index}"

            # 2. 先获取产品描述和介绍
            product_data['description'] = self.extract_product_description(driver)

            # 3. 从描述的第一行获取用作文件夹名称的产品名
            folder_name = self.get_folder_name_from_description(product_data['description'],
                                                                product_data['product_name'], product_index,
                                                                category_name)

            # 4. 使用描述第一行作为文件夹名提取指定图片
            product_data['images_info'] = self.extract_specific_images(
                driver, folder_name, product_index, category_name
            )

            # 5. 获取产品规格和参数
            product_data['specifications'], product_data['parameters'] = self.extract_product_specifications(driver)

            self.logger.info(f"成功爬取{category_name}产品: {product_data['product_name']}")

        except Exception as e:
            self.logger.error(f"爬取{category_name}产品 {product_url} 时出错: {e}")
        finally:
            driver.quit()

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
        使用用户指定的精确XPath提取产品信息区域的所有文字内容。
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

    def extract_current_page_params(self, driver):
        """从当前可见的参数列表中提取参数。"""
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
        提取产品规格和参数。
        通过模拟点击每个产品型号，来获取其对应的详细参数。
        """
        specifications = {}
        all_models_data = {}

        try:
            # 1. 点击"功能参数"选项卡以显示参数区域
            wait = WebDriverWait(driver, 10)
            param_tab = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), '功能参数')]"))
            )
            driver.execute_script("arguments[0].click();", param_tab)
            # 等待参数内容加载
            wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.prd-conf-tabs-info"))
            )

            # 2. 找到左侧所有可点击的产品型号
            model_switcher_elements = driver.find_elements(By.CSS_SELECTOR, "ul.prd-conf-tab > li > a")
            if not model_switcher_elements:
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

                    driver.execute_script("arguments[0].click();", current_model_button)

                    time.sleep(2)

                    # 4. 抓取当前显示型号的参数表
                    current_model_params = self.extract_current_page_params(driver)

                    actual_model_name = current_model_params.get("产品型号", model_name_from_button)
                    all_models_data[actual_model_name] = current_model_params

                except Exception as e:
                    self.logger.error(f"处理型号索引 {i} 时出错: {e}")
                    continue

        except TimeoutException:
            self.logger.error("点击或加载'功能参数'选项卡时超时。")
        except Exception as e:
            self.logger.error(f"提取产品规格时发生严重错误: {e}")

        return specifications, all_models_data


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
                self.logger.info(f"开始爬取 {category_name} 分类的 {len(product_links)} 个产品")

                for i, product_url in enumerate(product_links, 1):
                    try:
                        product_data = self.crawl_product_detail(product_url, i, category_name)
                        product_data['global_index'] = product_counter
                        product_data['category_index'] = i
                        product_data['crawl_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

                        all_products_data.append(product_data)

                        product_folder = product_data.get('images_info', {}).get('product_folder', '')
                        if product_folder and os.path.exists(product_folder):
                            json_filename = "product_data.json"
                            json_filepath = os.path.join(product_folder, json_filename)

                            with open(json_filepath, 'w', encoding='utf-8') as f:
                                json.dump(product_data, f, ensure_ascii=False, indent=2)


                        self.logger.info(
                            f"完成产品 {product_counter}/{total_products} - {category_name}: {product_data['product_name']}")
                        product_counter += 1

                        time.sleep(3)

                    except Exception as e:
                        self.logger.error(f"爬取{category_name}产品 {product_url} 失败: {e}")
                        product_counter += 1
                        continue

                self.logger.info(f"{category_name} 分类爬取完成")

        finally:
            driver.quit()

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
        crawler = GreeCrawler(output_dir="gree_tezhong", target_category="特种空调")
        results = crawler.run()

        if results:
            print(f"\n爬虫执行完成！共爬取 {len(results)} 个产品页面")
            print(f"数据保存在: {os.path.abspath(crawler.output_dir)}")
        else:
            print("爬取失败，请检查日志文件")

    except KeyboardInterrupt:
        print("\n爬虫被用户中断")
    except Exception as e:
        print(f"爬虫执行出错: {e}")


if __name__ == "__main__":
    main()