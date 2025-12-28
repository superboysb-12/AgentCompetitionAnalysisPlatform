import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin, urlparse
import time
import json
import re

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


class HisenseScraperSelenium:
    def __init__(self):
        self.base_url = "https://www.zykthisense.com"
        # 家用
        # 家用变频多联机
        self.target_url = "https://www.zykthisense.com/product/index.aspx?nodeid=656"
        # 全直流变频风管机


        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        self.driver = self.setup_driver()

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.create_directories()

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式，在后台运行
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--force-device-scale-factor=1")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")

        print("初始化 Selenium WebDriver...")
        try:

            service = Service()
            driver = webdriver.Chrome(service=service, options=chrome_options)

            driver.set_window_size(1920, 1080)

            driver.implicitly_wait(10)

            return driver
        except WebDriverException as e:
            print(f"WebDriver 初始化失败。请确保您的电脑已安装 Google Chrome 浏览器。")
            print(f"错误详情: {e}")
            return None

    def create_directories(self):
        """创建保存文件的目录"""
        directories = ['page_content_screenshots', 'products', 'data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def clean_filename(self, filename):
        """清理文件名，移除非法字符"""
        if not filename:
            return "未命名内容"
        filename = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', filename)
        filename = re.sub(r'_+', '_', filename)  # 合并多个下划线
        filename = filename.strip('_. ')  # 移除开头和结尾的下划线、点和空格

        if len(filename) > 100:
            filename = filename[:100]

        return filename if filename else "未命名内容"

    def scrape_w1170_content_with_screenshot(self):
        """
        只对 productHome 下的 <div class="w1170"> 容器内的每个直接子元素进行截图和文字提取。
        """
        if not self.driver:
            return

        print("\n 使用 Selenium 对 productHome 下的 w1170 容器进行动态截图和文本提取...")
        output_dir = "page_content_screenshots"

        try:
            wait = WebDriverWait(self.driver, 15)

            product_home_elements = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "productHome")))

            if not product_home_elements:
                print(" 未找到 productHome 容器")
                return


            for product_home_idx, product_home_element in enumerate(product_home_elements):

                try:
                    w1170_elements = product_home_element.find_elements(By.CLASS_NAME, "w1170")

                    if not w1170_elements:
                        print(f"   ️ productHome 容器 {product_home_idx + 1} 内未找到 w1170 容器")
                        continue

                    for w1170_idx, w1170_element in enumerate(w1170_elements):
                        try:
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", w1170_element)
                            time.sleep(2)

                            self.driver.execute_script("window.scrollTo(0, arguments[0].offsetTop - 100);",
                                                       w1170_element)
                            time.sleep(1)

                            container_folder = os.path.join(output_dir,
                                                            f"productHome_{product_home_idx + 1}_w1170_{w1170_idx + 1}")
                            os.makedirs(container_folder, exist_ok=True)

                            container_screenshot_path = os.path.join(container_folder, "w1170_container_full.png")

                            container_size = w1170_element.size
                            if container_size['height'] <= 0 or container_size['width'] <= 0:
                                print(
                                    f" 跳过容器截图：尺寸为0 (高度:{container_size['height']}, 宽度:{container_size['width']})")
                            else:
                                self.driver.execute_script("arguments[0].style.border = '2px solid red';",
                                                           w1170_element)
                                time.sleep(0.5)
                                w1170_element.screenshot(container_screenshot_path)
                                self.driver.execute_script("arguments[0].style.border = '';", w1170_element)

                            container_text = w1170_element.text
                            container_text_path = os.path.join(container_folder, "w1170_container_text.txt")
                            with open(container_text_path, "w", encoding="utf-8") as f:
                                f.write(container_text)

                            self.process_all_child_elements_dynamically(w1170_element, container_folder)

                        except Exception as e:
                            print(
                                f" 处理 productHome_{product_home_idx + 1} 下的 w1170 容器 {w1170_idx + 1} 时出错: {e}")

                except Exception as e:
                    print(f" 处理 productHome 容器 {product_home_idx + 1} 时出错: {e}")

        except TimeoutException:
            print(" 页面加载超时或未找到任何 'productHome' 容器。")
        except Exception as e:
            print(f" 在截图过程中发生未知异常: {e}")

    def take_element_screenshot_alternative(self, element, save_path):
        """
        备用截图方法：当直接截图失败时，使用全页截图+裁剪的方式
        """
        try:
            location = element.location
            size = element.size

            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(1)

            scroll_y = self.driver.execute_script("return window.pageYOffset;")
            scroll_x = self.driver.execute_script("return window.pageXOffset;")

            full_screenshot = self.driver.get_screenshot_as_png()

            from PIL import Image
            import io

            img = Image.open(io.BytesIO(full_screenshot))

            left = max(0, location['x'] - scroll_x)
            top = max(0, location['y'] - scroll_y)
            right = min(img.width, left + size['width'])
            bottom = min(img.height, top + size['height'])

            if right > left and bottom > top:
                cropped_img = img.crop((left, top, right, bottom))
                cropped_img.save(save_path)
                return True
            else:
                print(f"       备用截图方法失败：裁剪区域无效")
                return False

        except ImportError:
            print(f"       备用截图方法需要安装 Pillow 库: pip install Pillow")
            return False
        except Exception as e:
            print(f"       备用截图方法异常: {e}")
            return False

    def process_all_child_elements_dynamically(self, container_element, container_folder):
        """
        动态处理w1170容器内的所有子元素，不区分元素类型
        """
        try:
            child_elements = container_element.find_elements(By.XPATH, "./*")

            if not child_elements:
                print("   ℹ 未找到任何直接子元素")
                return

            print(f"    找到 {len(child_elements)} 个直接子元素，开始逐个处理...")

            children_folder = os.path.join(container_folder, "child_elements")
            os.makedirs(children_folder, exist_ok=True)

            for child_idx, child_element in enumerate(child_elements):
                try:
                    element_info = self.get_element_info(child_element, child_idx)

                    print(f"    处理子元素 {child_idx + 1}: {element_info['description']}")

                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", child_element)
                    time.sleep(1)

                    self.driver.execute_script("""
                        var rect = arguments[0].getBoundingClientRect();
                        var viewHeight = Math.max(document.documentElement.clientHeight, window.innerHeight);
                        if (rect.bottom > viewHeight) {
                            window.scrollBy(0, rect.bottom - viewHeight + 50);
                        }
                        if (rect.top < 0) {
                            window.scrollBy(0, rect.top - 50);
                        }
                    """, child_element)
                    time.sleep(1)

                    element_folder = os.path.join(children_folder, element_info['folder_name'])
                    os.makedirs(element_folder, exist_ok=True)

                    screenshot_path = os.path.join(element_folder, f"{element_info['file_prefix']}_screenshot.png")
                    try:
                        element_size = child_element.size
                        if element_size['height'] <= 0 or element_size['width'] <= 0:
                            print(
                                f"       跳过截图：元素尺寸为0 (高度:{element_size['height']}, 宽度:{element_size['width']})")
                        else:
                            self.driver.execute_script("arguments[0].style.border = '1px solid blue';", child_element)
                            time.sleep(0.2)

                            child_element.screenshot(screenshot_path)

                            self.driver.execute_script("arguments[0].style.border = '';", child_element)
                    except Exception as screenshot_error:
                        print(f"       截图失败: {screenshot_error}")
                        # 尝试使用整页截图然后裁剪的方式
                        try:
                            print(f"       尝试备用截图方法...")
                            self.take_element_screenshot_alternative(child_element, screenshot_path)
                        except Exception as alt_error:
                            print(f"       备用截图方法也失败: {alt_error}")
                            continue

                    element_text = child_element.text.strip()
                    if element_text:
                        try:
                            text_path = os.path.join(element_folder, f"{element_info['file_prefix']}_text.txt")
                            with open(text_path, "w", encoding="utf-8") as f:
                                f.write(element_text)
                        except Exception as text_error:
                            print(f"       文本保存失败: {text_error}")

                    try:
                        info_path = os.path.join(element_folder, f"{element_info['file_prefix']}_info.json")
                        with open(info_path, "w", encoding="utf-8") as f:
                            json.dump(element_info, f, ensure_ascii=False, indent=2)
                    except Exception as info_error:
                        print(f"       信息保存失败: {info_error}")

                    self.process_nested_elements(child_element, element_folder, element_info['file_prefix'])

                except Exception as e:
                    print(f"       处理子元素 {child_idx + 1} 时出错: {e}")

        except Exception as e:
            print(f"    动态处理子元素时出错: {e}")

    def get_element_info(self, element, index):
        """
        获取元素的详细信息，用于命名和描述
        """
        try:
            tag_name = element.tag_name.lower()
            element_classes = element.get_attribute("class") or ""
            element_id = element.get_attribute("id") or ""

            title = ""
            try:
                for title_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    title_elements = element.find_elements(By.TAG_NAME, title_tag)
                    if title_elements:
                        title = title_elements[0].text.strip()[:50]
                        break

                if not title:
                    alt_text = element.get_attribute("alt")
                    if alt_text:
                        title = alt_text[:50]
                    elif element.get_attribute("title"):
                        title = element.get_attribute("title")[:50]
                    elif element.text.strip():
                        title = element.text.strip()[:30]

            except:
                pass

            description_parts = [f"<{tag_name}>"]
            if element_classes:
                description_parts.append(f"class='{element_classes}'")
            if element_id:
                description_parts.append(f"id='{element_id}'")
            if title:
                description_parts.append(f"title='{title}'")

            description = " ".join(description_parts)

            folder_name_parts = [f"element_{index + 1:02d}", tag_name]
            if element_classes:
                main_class = element_classes.split()[0] if element_classes else ""
                if main_class:
                    folder_name_parts.append(main_class)

            folder_name = "_".join(folder_name_parts)
            folder_name = self.clean_filename(folder_name)

            file_prefix = f"element_{index + 1:02d}_{tag_name}"
            if title:
                clean_title = self.clean_filename(title)[:20]
                file_prefix += f"_{clean_title}"

            return {
                'index': index + 1,
                'tag_name': tag_name,
                'classes': element_classes,
                'id': element_id,
                'title': title,
                'description': description,
                'folder_name': folder_name,
                'file_prefix': file_prefix
            }

        except Exception as e:
            return {
                'index': index + 1,
                'tag_name': 'unknown',
                'classes': '',
                'id': '',
                'title': '',
                'description': f'Element {index + 1}',
                'folder_name': f'element_{index + 1:02d}',
                'file_prefix': f'element_{index + 1:02d}'
            }

    def process_nested_elements(self, parent_element, parent_folder, prefix):
        """
        处理嵌套元素（如果子元素内还有重要的子元素）
        """
        try:
            nested_elements = parent_element.find_elements(By.XPATH, ".//*[@class or @id]")

            important_nested = []
            for elem in nested_elements:
                try:
                    size = elem.size
                    if size['height'] > 50 and size['width'] > 50:
                        important_nested.append(elem)
                except:
                    continue

            if len(important_nested) > 0 and len(important_nested) <= 10:
                nested_folder = os.path.join(parent_folder, "nested_elements")
                os.makedirs(nested_folder, exist_ok=True)


                for nested_idx, nested_elem in enumerate(important_nested):
                    try:
                        nested_info = self.get_element_info(nested_elem, nested_idx)
                        nested_screenshot = os.path.join(nested_folder, f"{prefix}_nested_{nested_idx + 1}.png")

                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", nested_elem)
                        time.sleep(0.3)
                        nested_elem.screenshot(nested_screenshot)

                        if nested_elem.text.strip():
                            nested_text_path = os.path.join(nested_folder, f"{prefix}_nested_{nested_idx + 1}_text.txt")
                            with open(nested_text_path, "w", encoding="utf-8") as f:
                                f.write(nested_elem.text.strip())


                    except Exception as e:
                        print(f"         处理嵌套元素 {nested_idx + 1} 时出错: {e}")

        except Exception as e:
            print(f"         处理嵌套元素时出错: {e}")

    def download_image(self, img_url, save_path, filename):
        """下载图片到指定路径"""
        try:
            if img_url.startswith('..'):
                img_url = urljoin(self.base_url, '/'.join(self.target_url.split('/')[:-1]) + '/' + img_url)
            elif img_url.startswith('/'):
                img_url = self.base_url + img_url
            elif not img_url.startswith('http'):
                img_url = urljoin(self.target_url, img_url)

            response = self.session.get(img_url, timeout=10)
            if response.status_code == 200:
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                return False
        except Exception as e:
            return False

    def parse_product_table(self, table):
        """解析产品参数表格 - 改进版，正确处理表头"""
        parameters = {}
        table_data = {
            'headers': [],
            'rows': []
        }

        try:
            rows = table.find_all('tr')
            if not rows:
                return parameters, table_data

            all_rows_data = []
            for i, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text().strip() for cell in cells]
                    if any(cell.strip() for cell in row_data):
                        all_rows_data.append(row_data)

            if not all_rows_data:
                return parameters, table_data

            headers = all_rows_data[0] if all_rows_data else []
            data_rows = all_rows_data[1:] if len(all_rows_data) > 1 else []

            table_data['headers'] = headers
            table_data['rows'] = data_rows

            if headers and data_rows:
                for row_data in data_rows:
                    if len(row_data) > 1:
                        param_name = row_data[0]
                        param_value = row_data[1]
                        if param_name:
                            parameters[param_name] = param_value

            return parameters, table_data

        except Exception as e:
            return parameters, table_data

    def scrape_product_info(self, soup):
        """爬取产品信息 - 改进版"""
        products = []

        try:

            product_ul = soup.find('ul', class_='picList')
            if not product_ul:
                print(" 无法找到产品列表容器 (ul.picList)")
                return products

            product_items = product_ul.find_all('li')
            print(f" 找到 {len(product_items)} 个产品项")

            for i, product_li in enumerate(product_items):
                print(f"\n--- 处理产品 {i + 1} ---")

                product_data = {
                    'index': i + 1,
                    'name': '',
                    'image_url': '',
                    'folder_path': '',
                    'parameters': {}
                }

                img = product_li.find('img')
                if img and img.get('src'):
                    product_data['image_url'] = img.get('src')

                name_div = product_li.find('div', class_='parnum')
                product_data['name'] = name_div.get_text(strip=True) if name_div else f'产品_{i + 1}'

                table = product_li.find('table')
                if table:
                    table_params, _ = self.parse_product_table(table)
                    product_data['parameters'].update(table_params)

                clean_name = self.clean_filename(product_data['name'])
                product_folder = os.path.join('products', clean_name)
                product_data['folder_path'] = product_folder
                os.makedirs(product_folder, exist_ok=True)

                if product_data['image_url']:
                    filename = f"product_image.jpg"
                    self.download_image(product_data['image_url'], product_folder, filename)

                if product_data['parameters']:
                    df = pd.DataFrame(list(product_data['parameters'].items()), columns=['参数名称', '参数值'])
                    csv_filename = os.path.join(product_folder, 'parameters.csv')
                    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

                products.append(product_data)

            print(f"\n 产品信息爬取完成: {len(products)}个产品")
            return products

        except Exception as e:
            print(f" 产品信息爬取异常: {str(e)}")
            return products

    def close_driver(self):
        """关闭 Selenium WebDriver"""
        if self.driver:
            print("\n 关闭 WebDriver...")
            self.driver.quit()

    def run(self):
        """主运行函数"""
        if not self.driver:
            print("WebDriver 未成功启动，程序退出。")
            return

        print("-" * 50)
        print(f" 开始爬取: {self.target_url}")

        try:
            self.driver.get(self.target_url)
            print(" 页面加载完成。")

            self.scrape_w1170_content_with_screenshot()

            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            products = self.scrape_product_info(soup)

            if products:
                summary_data = {
                    'url': self.target_url,
                    'scrape_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_products': len(products),
                    'products': [p['name'] for p in products]
                }
                with open('data/scraped_data.json', 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, ensure_ascii=False, indent=2)

            print("\n 爬取完成!")

        except Exception as e:
            print(f" 在主运行过程中发生严重错误: {e}")
        finally:
            self.close_driver()


if __name__ == "__main__":
    scraper = HisenseScraperSelenium()
    scraper.run()