import requests
from bs4 import BeautifulSoup
import time
import re
import json
import os
import signal
import sys
from urllib.parse import urljoin, urlparse
from PIL import Image
import shutil

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, \
    StaleElementReferenceException


BASE_URL = "https://consumer.panasonic.cn"
OUTPUT_ROOT_DIR = "panasonic_products"

# 全局变量，用于优雅退出
interrupted = False


def signal_handler(signum, frame):
    """信号处理器"""
    global interrupted
    interrupted = True
    print("\n收到中断信号，正在退出...")
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

# 确保根输出目录存在
if not os.path.exists(OUTPUT_ROOT_DIR):
    os.makedirs(OUTPUT_ROOT_DIR)


def clean_filename(text):
    """
    清理字符串，使其适合作为文件名或文件夹名。
    """
    text = str(text)
    text = text.strip()
    text = re.sub(r'[，。！？、；：""''【】（）《》—]', '_', text)
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '', text)
    text = re.sub(r'^-+|-+$', '', text)
    text = text[:100]
    return text


def initialize_selenium_driver():
    """
    初始化Selenium WebDriver。
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except WebDriverException as e:
        print(f"初始化Selenium WebDriver失败: {e}")
        print("请确保您已安装Chrome浏览器和匹配版本的ChromeDriver，并将其添加到系统PATH中。")
        sys.exit(1)


def fetch_page_content_requests(url, retries=3, delay=5):
    """
    使用requests获取页面内容，并处理重试。
    适用于静态加载的页面内容。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"请求页面 {url} 失败 (尝试 {i + 1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(delay)
    return None


def scrape_product_links_requests(base_list_url, selector, max_pages=5):
    """
    使用requests从给定的基础URL和分页范围抓取所有产品链接。
    适用于产品列表页是静态渲染的情况。
    """
    all_product_links = set()

    base_url_without_page = base_list_url.rsplit('_', 1)[0] if '_1' in base_list_url else base_list_url

    print(f"开始从 {base_url_without_page} 抓取产品列表链接 (最多 {max_pages} 页)...")

    for page_num in range(1, max_pages + 1):
        if interrupted:
            print("停止抓取产品列表页 (收到中断信号)。")
            break

        current_page_url = f"{base_url_without_page}_{page_num}/" if page_num > 1 else base_list_url
        print(f"  正在抓取第 {page_num} 页: {current_page_url}")

        page_content = fetch_page_content_requests(current_page_url)
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            elements = soup.select(selector)

            if not elements and page_num > 1:
                print(f"  第 {page_num} 页未找到产品链接，停止分页抓取。")
                break

            for element in elements:
                href = element.get('href')
                if href:
                    if not href.startswith(('http', 'https')):
                        href = urljoin(BASE_URL, href)
                    all_product_links.add(href)
            print(f"  从第 {page_num} 页抓取到 {len(elements)} 个链接。")
        else:
            print(f"  未能获取第 {page_num} 页内容，跳过。")
            if page_num > 1:
                break

        time.sleep(1)

    print(f"产品列表链接抓取完成。总计找到 {len(all_product_links)} 个唯一产品链接。")
    return list(all_product_links)


def extract_images_from_swiper_selenium(driver, product_url):
    """
    使用Selenium访问产品详情页，等待图片加载，并从Swiper组件中提取图片URL。
    用于处理动态加载的图片。
    """
    image_urls = []
    seen_urls = set()

    try:
        driver.get(product_url)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.detail-product-left div.swiper-wrapper img'))
        )

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        swiper_imgs = soup.select('div.detail-product-left div.swiper-wrapper img')

        exclude_keywords = ['icon', 'logo', 'button', 'loading', 'sprite', 'banner', 'nav', 'menu']

        for img_tag in swiper_imgs:
            src = img_tag.get('data-src') or img_tag.get('src')
            if not src:
                src = img_tag.get('src')

            if src:
                src = src.strip()
                if not src.startswith(('http', 'https')):
                    src = urljoin(BASE_URL, src)

                if '/static/upload/image/' in src and src not in seen_urls:
                    if any(keyword in src.lower() for keyword in exclude_keywords):
                        continue
                    if not any(src.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        continue
                    # 避免抓取过小的图片
                    if len(src.split('/')[-1]) < 10:
                        continue

                    image_urls.append(src)
                    seen_urls.add(src)
                    if len(image_urls) >= 6:  # 限制主图数量
                        break

        if not image_urls:
            print(f"  警告: 未能从产品页 {product_url} 提取到任何有效主图。")

    except TimeoutException:
        print(f"  加载产品详情页 {product_url} 超时，可能Swiper图片未完全加载。")
    except Exception as e:
        print(f"  使用Selenium提取图片时发生错误: {e}")

    return image_urls


def extract_all_models(container, product_name):
    """
    提取所有产品型号，处理多种可能的格式
    """
    models = []

    # 方法1: 查找明确标注"型号"的区域，例如 h3, h2 标签，或包含"型号"文字的div
    model_elements_ev_style = container.select('div.content-style-params > h3')
    for elem in model_elements_ev_style:
        model_text = elem.get_text(strip=True)
        match = re.search(r'\((.*?)\)', model_text)
        if match:
            models.append(match.group(1))
        else:
            model_codes = re.findall(r'\b([A-Z]{2,}[0-9]{2,}[A-Z0-9]*)\b', model_text)
            models.extend(model_codes)

    # 方法2: H2系列：#c2 > div.detail-product-content-article > div > div:nth-child > ul > li > p:nth-child(2)
    model_elements_h2_style = container.select('div.content-style-params > ul > li > p:nth-child(2)')
    for elem in model_elements_h2_style:
        model_text = elem.get_text(strip=True)
        match = re.search(r'型号[：:]\s*(\S+)', model_text)
        if match:
            models.append(match.group(1))
        else:
            model_codes = re.findall(r'\b([A-Z]{2,}[0-9]{2,}[A-Z0-9]*)\b', model_text)
            models.extend(model_codes)

    # 方法3: 在参数列表中查找型号 (原有的逻辑，作为通用回退)
    if not models:
        param_rows = container.select("div.camera-params-list > div")
        for row in param_rows:
            text = row.get_text()
            if '型号' in text:
                model_codes = re.findall(r'\b([A-Z]{2,}[0-9]{2,}[A-Z0-9]*)\b', text)
                models.extend(model_codes)
                break

    # 方法4: 从产品名称中提取 (原有的逻辑，作为通用回退)
    if not models and product_name != 'N/A':
        model_codes = re.findall(r'\b([A-Z]{2,}[0-9]{2,}[A-Z0-9]*)\b', product_name)
        models.extend(model_codes)

    # 去重并保持顺序
    seen = set()
    unique_models = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    return unique_models


def extract_param_from_row(row):
    """
    从参数行中提取参数名和值
    """
    # 方法1: 标准的两列布局（参数名 | 值列表）
    columns = row.find_all("div", recursive=False)

    if len(columns) >= 2:
        param_name_div = columns[0]
        param_name = param_name_div.get_text(strip=True)

        values = []
        value_container = columns[1]

        value_list = value_container.find("ul")
        if value_list:
            for li in value_list.find_all("li"):
                value_div = li.find("div")
                if value_div:
                    val = value_div.get_text(strip=True)
                else:
                    val = li.get_text(strip=True)
                if val:
                    values.append(val)
        else:
            val = value_container.get_text(strip=True)
            if val:
                values.append(val)

        return {'name': param_name, 'values': values}

    # 方法2: 单列布局，参数名和值用分隔符分开
    text = row.get_text(strip=True)

    if '：' in text or ':' in text:
        separator = '：' if '：' in text else ':'
        parts = text.split(separator, 1)
        if len(parts) == 2:
            return {'name': parts[0].strip(), 'values': [parts[1].strip()]}

    parts = re.split(r'\s{2,}|\t', text)
    if len(parts) >= 2:
        return {'name': parts[0].strip(), 'values': [parts[1].strip()]}

    return None


def extract_params_without_types(container):
    """
    提取没有明确类型划分的参数
    """
    params = []

    param_list = container.select_one("div.camera-params-list")
    if not param_list:
        return params

    rows = param_list.find_all("div", recursive=False)
    for row in rows:
        param_info = extract_param_from_row(row)
        if param_info and param_info['name'] != '型号':
            params.append(param_info)

    return params


def extract_parameter_structure(container):
    """
    提取参数结构，返回按类型组织的参数列表
    结构: [{'type': '参数类型', 'params': [{'name': '参数名', 'values': [值列表]}]}]
    """
    structure = []

    # 查找所有参数类型标题
    param_type_headers = container.select("div.camera-params-title")

    for type_header in param_type_headers:
        param_type = type_header.get_text(strip=True)

        if param_type in ['规格参数', '产品参数']:
            continue

        param_list = type_header.find_next_sibling("div", class_="camera-params-list")

        if not param_list:
            continue

        type_params = {'type': param_type, 'params': []}

        param_rows = param_list.find_all("div", recursive=False)

        for row in param_rows:
            param_info = extract_param_from_row(row)
            if param_info and param_info['name'] != '型号':
                type_params['params'].append(param_info)

        if type_params['params']:
            structure.append(type_params)

    if not structure:
        all_params = extract_params_without_types(container)
        if all_params:
            structure.append({'type': '基本参数', 'params': all_params})

    return structure


def organize_parameters_by_model(models, param_structure):
    """
    根据型号数量和参数值数量，智能分配参数到各个型号
    """
    num_models = len(models)
    model_parameters = []

    if num_models == 0:
        return []

    for model_name in models:
        model_parameters.append({
            'model_name': model_name,
            'parameters': {}
        })

    for type_data in param_structure:
        param_type = type_data['type']

        for param in type_data['params']:
            param_name = param['name']
            param_values = param['values']
            num_values = len(param_values)

            full_key = f"{param_type}_{param_name}"

            if num_values == num_models:
                for i in range(num_models):
                    model_parameters[i]['parameters'][full_key] = param_values[i]

            elif num_values == 1:
                for i in range(num_models):
                    model_parameters[i]['parameters'][full_key] = param_values[0]

            elif num_values > num_models and num_values % num_models == 0:
                values_per_model = num_values // num_models
                for i in range(num_models):
                    start_idx = i * values_per_model
                    end_idx = start_idx + values_per_model
                    combined_value = " / ".join(param_values[start_idx:end_idx])
                    model_parameters[i]['parameters'][full_key] = combined_value

            else:
                if num_values == 0:
                    for i in range(num_models):
                        model_parameters[i]['parameters'][full_key] = "N/A"
                else:
                    combined_value = " / ".join(param_values)
                    for i in range(num_models):
                        model_parameters[i]['parameters'][full_key] = combined_value

    return model_parameters


def extract_product_parameters_universal(soup_static, product_name):
    """
    通用的产品参数提取函数，能够处理多种HTML结构
    """
    models = []
    model_parameters = []

    param_container = soup_static.select_one("#c2 > div.detail-product-content-article > div")

    if not param_container:
        param_container = soup_static.select_one(".detail-product-content-article > div")

    if not param_container:
        param_container = soup_static.select_one(".detail-product-content-item.content-style-params")

    if not param_container:
        print("  警告: 未找到产品参数容器")
        return models, model_parameters

    models = extract_all_models(param_container, product_name)

    if not models:
        print("  警告: 未找到任何型号，使用产品名称作为默认型号")
        models = [product_name if product_name != 'N/A' else '未知型号']

    print(f"  找到 {len(models)} 个型号: {models}")

    param_structure = extract_parameter_structure(param_container)

    model_parameters = organize_parameters_by_model(models, param_structure)

    return models, model_parameters


def scrape_product_details_hybrid(driver, product_url):
    """
    混合抓取：使用requests获取大部分静态信息，使用Selenium获取动态加载的图片URL。
    新增产品型号和参数的抓取。
    """
    details = {
        'url': product_url,
        'name': 'N/A',
        'image_urls': [],
        'local_image_paths': [],
        'info': [],
        'features': [],
        'models': [],
        'model_parameters': []
    }

    page_content_static = fetch_page_content_requests(product_url)
    if page_content_static:
        soup_static = BeautifulSoup(page_content_static, 'html.parser')

        name_selectors = [
            "body > div:nth-child(7) > div > div.detail-product-right > div.detail-product-title > h1",
            ".detail-product-title h1",
            ".product-title h1",
            "h1.product-name",
            "h1"
        ]
        for selector in name_selectors:
            name_element = soup_static.select_one(selector)
            if name_element:
                details['name'] = name_element.get_text(strip=True)
                break
        if details['name'] == 'N/A':
            print(f"  警告: 未找到产品名称: {product_url}")

        info_selectors = [
            "div.detail-product-head-desc > ul > li > p",
            ".product-info p",
            ".detail-product-right .product-desc p"
        ]
        for selector in info_selectors:
            info_elements = soup_static.select(selector)
            if info_elements:
                for p_tag in info_elements:
                    info_text = p_tag.get_text(strip=True)
                    if info_text and info_text not in details['info']:
                        details['info'].append(info_text)
                break

        features_selectors = [
            "#c1 > div.detail-product-content-lists > div > ul",
            ".detail-product-content-lists ul",
            ".product-features ul"
        ]
        for selector in features_selectors:
            features_ul = soup_static.select_one(selector)
            if features_ul:
                feature_lis = features_ul.select('li')
                for li in feature_lis:
                    feature_data = {
                        'poster_url': 'N/A',
                        'local_poster_path': 'N/A',
                        'name': 'N/A',
                        'description': 'N/A'
                    }
                    poster_img = li.select_one('img')
                    if poster_img:
                        src_attrs = ['src', 'data-src', 'data-original', 'data-lazy']
                        for attr in src_attrs:
                            poster_src = poster_img.get(attr)
                            if poster_src:
                                if not poster_src.startswith(('http', 'https')):
                                    poster_src = urljoin(BASE_URL, poster_src)
                                feature_data['poster_url'] = poster_src
                                break
                    feature_name = li.select_one('h3, .feature-name, .feature-title')
                    if feature_name:
                        feature_data['name'] = feature_name.get_text(strip=True)
                    feature_description = li.select_one('p, .feature-desc, .feature-description')
                    if feature_description:
                        feature_data['description'] = feature_description.get_text(strip=True)
                    details['features'].append(feature_data)
                break

        details['models'], details['model_parameters'] = extract_product_parameters_universal(soup_static,
                                                                                              details['name'])

    else:
        print(f"  警告: 无法通过requests获取产品详情页 {product_url} 静态内容。")

    details['image_urls'] = extract_images_from_swiper_selenium(driver, product_url)

    return details


def download_image_with_requests(image_url, final_save_path, timeout=30, retries=5, retry_delay=5):
    """
    使用 requests 库下载图片，并进行 Pillow 验证。
    增加了重试机制和更长的超时时间。
    """
    global interrupted
    if interrupted:
        return False

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    temp_file_path = final_save_path + ".tmp"

    for i in range(retries):
        if interrupted:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return False

        try:
            response = requests.get(image_url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if interrupted:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        return False
                    f.write(chunk)

            if os.path.getsize(temp_file_path) == 0:
                print(f"  错误 (尝试 {i + 1}/{retries}): 图片文件 {image_url} 为空。")
                os.remove(temp_file_path)
                return False

            try:
                img = Image.open(temp_file_path)
                img.verify()
            except (IOError, SyntaxError) as pil_e:
                print(f"  错误 (尝试 {i + 1}/{retries}): 图片 {image_url} Pillow 验证失败: {pil_e}")
                os.remove(temp_file_path)
                return False
            finally:
                if 'img' in locals():
                    img.close()

            shutil.move(temp_file_path, final_save_path)
            return True

        except requests.exceptions.Timeout as e:
            print(f"  错误 (尝试 {i + 1}/{retries}): 下载图片 {image_url} 超时: {e}")
        except requests.exceptions.RequestException as e:
            print(f"  错误 (尝试 {i + 1}/{retries}): 下载图片 {image_url} 失败: {e}")
        except Exception as e:
            print(f"  错误 (尝试 {i + 1}/{retries}): 下载图片 {image_url} 时发生未知错误: {e}")

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if i < retries - 1:
            time.sleep(retry_delay)

    print(f"  警告: 未能成功下载图片: {image_url}")
    return False


if __name__ == "__main__":
    selenium_driver = None
    try:
        selenium_driver = initialize_selenium_driver()
        if not selenium_driver:
            sys.exit(1)

        product_list_base_url = "https://consumer.panasonic.cn/product/air-conditioner/split-wall-mounted"
        product_link_selector = "div.product-right > div.product-lists > ul > li > div > a"
        max_pagination_pages = 5

        product_links = scrape_product_links_requests(product_list_base_url, product_link_selector,
                                                      max_pages=max_pagination_pages)

        if product_links:
            all_product_details_with_local_paths = []

            for i, link in enumerate(product_links):
                if interrupted:
                    print("停止抓取 (收到中断信号)。")
                    break

                print(f"\n[{i + 1}/{len(product_links)}] 正在处理产品: {link}")
                details = scrape_product_details_hybrid(selenium_driver, link)

                product_name_for_folder = details['name'] if details['name'] != 'N/A' else f"product_{i + 1}"
                product_folder_name = clean_filename(product_name_for_folder)
                product_output_dir = os.path.join(OUTPUT_ROOT_DIR, product_folder_name)

                os.makedirs(os.path.join(product_output_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(product_output_dir, "posters"), exist_ok=True)

                local_image_paths = []
                for img_idx, img_url in enumerate(details['image_urls']):
                    if interrupted:
                        break

                    parsed_url = urlparse(img_url)
                    img_name_from_url = os.path.basename(parsed_url.path)
                    if not img_name_from_url or '.' not in img_name_from_url:
                        img_name_from_url = f"main_image_{img_idx + 1}.png"

                    name_part, ext_part = os.path.splitext(img_name_from_url)
                    cleaned_name = clean_filename(name_part)
                    if not ext_part or ext_part.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                        ext_part = '.png'

                    cleaned_img_name = f"{cleaned_name}{ext_part}"
                    local_path = os.path.join(product_output_dir, "images", cleaned_img_name)

                    if download_image_with_requests(img_url, local_path):
                        local_image_paths.append(os.path.relpath(local_path, OUTPUT_ROOT_DIR))
                    else:
                        print(f"  警告: 未能成功下载主图: {img_url}")

                details['local_image_paths'] = local_image_paths

                for feature_idx, feature in enumerate(details['features']):
                    if interrupted:
                        break

                    if feature['poster_url'] != 'N/A':
                        poster_url = feature['poster_url']
                        parsed_url = urlparse(poster_url)
                        poster_name_from_url = os.path.basename(parsed_url.path)

                        if not poster_name_from_url or '.' not in poster_name_from_url:
                            poster_name_from_url = f"feature_poster_{feature_idx + 1}.jpg"

                        name_part, ext_part = os.path.splitext(poster_name_from_url)
                        cleaned_name = clean_filename(name_part)
                        if not ext_part or ext_part.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                            ext_part = '.jpg'

                        cleaned_poster_name = f"{cleaned_name}{ext_part}"
                        local_path = os.path.join(product_output_dir, "posters", cleaned_poster_name)

                        if download_image_with_requests(poster_url, local_path):
                            feature['local_poster_path'] = os.path.relpath(local_path, OUTPUT_ROOT_DIR)
                        else:
                            print(f"  警告: 未能成功下载海报图: {poster_url}")

                all_product_details_with_local_paths.append(details)

                product_data_json_path = os.path.join(product_output_dir, f"{product_folder_name}.json")
                with open(product_data_json_path, 'w', encoding='utf-8') as f:
                    json.dump(details, f, ensure_ascii=False, indent=4)
                print(f"  产品 '{details['name']}' 数据已保存到 {product_output_dir}")

                if not interrupted:
                    time.sleep(1)

            print(f"\n抓取完成！共处理 {len(all_product_details_with_local_paths)} 个产品。")

        else:
            print("未能抓取到任何产品链接。")

    finally:
        if selenium_driver:
            selenium_driver.quit()
            print("Selenium WebDriver 已关闭。")