from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from urllib.parse import urljoin, urlparse
import time
import json
import os
import re
import logging


class GreeProductCrawler:
    def __init__(self, base_url="https://www.gree.com/", output_dir="zykongtiao/jiayong"):
        """
        初始化爬虫
        Args:
            base_url: 格力官网基础URL
            output_dir: 数据保存目录
        """
        self.base_url = base_url
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_driver(self):
        """设置Selenium WebDriver"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # 取消注释以启用无头模式
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(60)
            return driver
        except Exception as e:
            self.logger.error(f"Chrome driver启动失败: {e}")
            return None

    def get_all_product_links(self, url):
        """
        获取产品列表页的所有产品链接（支持动态加载）
        Args:
            url: 产品列表页URL
        Returns:
            list: 产品链接列表
        """
        driver = self.setup_driver()
        if not driver:
            return []

        product_links = []

        try:
            self.logger.info(f"正在访问: {url}")
            driver.get(url)

            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#category-result-list')))
            self.logger.info("页面加载完成")

            more_button_selector = '#gree-allproduct-category-result > div > div > div.allproduct-category-result-hander > a'
            click_count = 0

            while True:
                try:
                    # 滚动到页面底部
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)

                    more_button = driver.find_element(By.CSS_SELECTOR, more_button_selector)

                    if more_button.is_displayed():
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_button)
                        time.sleep(0.5)

                        try:
                            more_button.click()
                        except ElementClickInterceptedException:
                            driver.execute_script("arguments[0].click();", more_button)

                        click_count += 1
                        self.logger.info(f"第 {click_count} 次点击'查看更多'按钮")
                        time.sleep(2)
                    else:
                        self.logger.info("'查看更多'按钮不可见，已加载所有内容")
                        break

                except NoSuchElementException:
                    self.logger.info("未找到'查看更多'按钮，已加载所有内容")
                    break
                except Exception as e:
                    self.logger.info(f"点击过程结束: {e}")
                    break

            time.sleep(2)

            # 提取所有产品链接
            self.logger.info("\n开始提取产品链接...")

            # 初始加载的产品链接
            initial_links = driver.find_elements(By.CSS_SELECTOR, '#category-result-list > li > a')
            self.logger.info(f"初始加载的产品: {len(initial_links)} 个")

            # 动态加载的产品链接
            more_links = driver.find_elements(By.CSS_SELECTOR, '#allproduct-category-result-more > li > a')
            self.logger.info(f"动态加载的产品: {len(more_links)} 个")

            all_links = initial_links + more_links

            for idx, link in enumerate(all_links, 1):
                try:
                    href = link.get_attribute('href')
                    if href:
                        full_url = urljoin(self.base_url, href)
                        product_links.append(full_url)
                        self.logger.info(f"[{idx}] {full_url}")
                except Exception as e:
                    self.logger.error(f"提取第 {idx} 个产品链接时出错: {e}")

            self.logger.info(f"\n共找到 {len(product_links)} 个产品链接")
        except Exception as e:
            self.logger.error(f"获取产品链接时出错: {e}")
        finally:
            driver.quit()

        return product_links

    def create_product_folder(self, product_name, product_index):
        """
        为每个产品创建独立的文件夹
        Args:
            product_name: 产品名称
            product_index: 产品序号
        Returns:
            str: 产品文件夹路径
        """
        # 清理产品名称，移除非法字符
        safe_name = re.sub(r'[<>:"/\\|?*]', '', product_name.strip())
        safe_name = re.sub(r'[^\w\s\u4e00-\u9fff\-]', '', safe_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name[:100]

        if not safe_name:
            safe_name = f"Product_{product_index}"

        counter = 1
        original_safe_name = safe_name
        product_folder = os.path.join(self.output_dir, safe_name)
        while os.path.exists(product_folder):
            safe_name = f"{original_safe_name}_{counter}"
            product_folder = os.path.join(self.output_dir, safe_name)
            counter += 1

        os.makedirs(product_folder, exist_ok=True)

        product_img_folder = os.path.join(product_folder, "product")
        introduction_img_folder = os.path.join(product_folder, "introduction")
        os.makedirs(product_img_folder, exist_ok=True)
        os.makedirs(introduction_img_folder, exist_ok=True)

        return product_folder

    def extract_product_description(self, driver):
        """
        提取产品信息区域的所有文字内容
        Returns:
            str: 产品描述文本
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
            self.logger.error(f"等待产品信息容器加载超时")
        except Exception as e:
            self.logger.error(f"提取产品描述时出错: {e}")

        return description.strip()

    def extract_product_images(self, driver):
        """
        提取产品图片和产品介绍内容
        Returns:
            dict: 包含产品图片、介绍图片和介绍文本的字典
        """
        images_info = {
            'product_images': [],
            'introduction_images': [],
            'introduction_text': ''
        }

        try:
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "product-details-intro")))

            # 1. 提取产品图片
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
                        for element in elements:
                            src = element.get_attribute("src")
                            if not src:
                                src = element.get_attribute("data-src") or element.get_attribute("data-original")

                            if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                                full_url = urljoin(self.base_url, src)
                                if full_url not in images_info['product_images']:
                                    images_info['product_images'].append(full_url)
                        break
                except Exception:
                    continue

            # 2. 提取产品介绍区域的内容（#product-details-list > div）
            try:
                intro_container = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#product-details-list > div"))
                )

                intro_text = intro_container.text.strip()
                if intro_text:
                    images_info['introduction_text'] = intro_text
                    self.logger.info(f"提取到介绍文本，共 {len(intro_text)} 个字符")

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

        except Exception as e:
            self.logger.error(f"提取图片时出错: {e}")

        return images_info

    def download_images(self, images_info, product_folder):
        """
        下载图片到对应文件夹，并保存介绍文本
        Args:
            images_info: 图片信息字典
            product_folder: 产品文件夹路径
        """
        import requests

        product_img_folder = os.path.join(product_folder, "product")
        introduction_img_folder = os.path.join(product_folder, "introduction")

        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # 下载产品图片
        for i, img_url in enumerate(images_info['product_images']):
            try:
                response = session.get(img_url, timeout=15)
                if response.status_code == 200:
                    ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
                    filename = f"product_{i + 1}{ext}"
                    filepath = os.path.join(product_img_folder, filename)

                    with open(filepath, 'wb') as f:
                        f.write(response.content)
            except Exception as e:
                self.logger.error(f"下载产品图片 {img_url} 失败: {e}")

        # 下载介绍图片
        for i, img_url in enumerate(images_info.get('introduction_images', [])):
            try:
                response = session.get(img_url, timeout=15)
                if response.status_code == 200:
                    ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
                    filename = f"introduction_{i + 1}{ext}"
                    filepath = os.path.join(introduction_img_folder, filename)

                    with open(filepath, 'wb') as f:
                        f.write(response.content)
            except Exception as e:
                self.logger.error(f"下载介绍图片 {img_url} 失败: {e}")

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
            'product_index': product_index,
            'product_name': '',
            'description': '',
            'images_info': {},
            'product_folder': '',
            'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        driver = self.setup_driver()
        if not driver:
            self.logger.error(f"无法为产品 {product_url} 启动浏览器驱动")
            return product_data

        try:
            self.logger.info(f"正在爬取产品 {product_index}: {product_url}")
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
                    product_data['product_name'] = f"Product_{product_index}"

            if not product_data['product_name']:
                product_data['product_name'] = f"Product_{product_index}"

            # 2. 提取产品描述
            try:
                product_data['description'] = self.extract_product_description(driver)
            except Exception as e:
                self.logger.error(f"提取产品描述失败: {e}")

            # 3. 从描述的第一行获取用作文件夹名称的产品名
            if product_data['description']:
                first_line = product_data['description'].strip().split('\n')[0].strip()
                if first_line:
                    folder_name = ' '.join(first_line.split())
                else:
                    folder_name = product_data['product_name']
            else:
                folder_name = product_data['product_name']

            # 4. 创建产品文件夹
            product_folder = self.create_product_folder(folder_name, product_index)
            product_data['product_folder'] = product_folder

            # 5. 提取图片
            try:
                images_info = self.extract_product_images(driver)
                product_data['images_info'] = images_info

                # 下载图片
                self.download_images(images_info, product_folder)
            except Exception as e:
                self.logger.error(f"提取产品图片失败: {e}")

            # 6. 保存产品数据到JSON文件
            json_filepath = os.path.join(product_folder, "product_data.json")
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(product_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"产品数据已保存: {json_filepath}")
            except Exception as e:
                self.logger.error(f"保存产品数据失败: {e}")

            self.logger.info(f"成功爬取产品: {product_data['product_name']}")

        except Exception as e:
            self.logger.error(f"爬取产品 {product_url} 时出错: {e}")
        finally:
            driver.quit()

        return product_data

    def run(self, list_url):
        """
        运行完整的爬虫流程
        Args:
            list_url: 产品列表页URL
        """

        start_time = time.time()

        # 1. 获取所有产品链接
        product_links = self.get_all_product_links(list_url)

        if not product_links:
            self.logger.warning("未找到任何产品链接")
            return []

        # 2. 爬取每个产品的详情
        all_products_data = []
        total = len(product_links)

        for idx, product_url in enumerate(product_links, 1):
            try:
                self.logger.info(f"\n处理产品 {idx}/{total}")
                product_data = self.crawl_product_detail(product_url, idx)

                if product_data.get('product_name'):
                    all_products_data.append(product_data)

                if idx < total:
                    time.sleep(3)

            except KeyboardInterrupt:
                self.logger.info("用户中断了爬取过程")
                break
            except Exception as e:
                self.logger.error(f"处理产品 {product_url} 时出错: {e}")
                continue


        self.logger.info("=" * 60)
        self.logger.info(f"成功爬取: {len(all_products_data)}/{total} 个产品")
        self.logger.info(f"数据保存在: {os.path.abspath(self.output_dir)}")
        self.logger.info("=" * 60)

        return all_products_data


def main():
    """主函数"""
    # 目标URL
    url = 'https://www.gree.com/cmsProduct/list/161'

    crawler = GreeProductCrawler(output_dir="zykongtiao/jiayong")

    # 运行爬虫
    try:
        results = crawler.run(url)

        if results:
            print(f"\n爬虫执行完成！共爬取 {len(results)} 个产品")
        else:
            print("爬取失败，请检查日志文件")

    except KeyboardInterrupt:
        print("\n爬虫被用户中断")
    except Exception as e:
        print(f"爬虫执行出错: {e}")


if __name__ == '__main__':
    main()