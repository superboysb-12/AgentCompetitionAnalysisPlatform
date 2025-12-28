import os
import time
import requests
import json
import random
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

SAVE_DIR = "haier_products_jy"
LIST_URL = "https://www.haier.com/air_conditioners/?spm=cn.home_pc.header_1_20241118.2"
BASE_URL = "https://www.haier.com"
MAX_RETRIES = 3
MAX_PAGES_TO_SCRAPE = 2

def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--log-level=3')
    options.add_argument("start-maximized")
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"')
    return webdriver.Chrome(options=options)


def _parse_introduction_content(soup_obj):

    parsed_data = {
        "basic_attributes": {},
        "selling_points": [],
        "features": {},
        "notes": "",
        "image_urls": []
    }

    # 1. 定位到产品介绍内容的总容器
    intro_container = soup_obj.select_one('div[data-obj="js_introduceTab"]')
    if not intro_container:
        return parsed_data

    # 2. 提取基础属性
    basic_items_container = intro_container.select_one('.basic_attribute.content_warp')
    if basic_items_container:
        items = basic_items_container.select('.item')
        for item in items:
            key_el = item.select_one('.name')
            value_el = item.select_one('.value')
            if key_el and value_el and key_el.text.strip():
                parsed_data["basic_attributes"][key_el.text.strip()] = value_el.text.strip()

    # 3. 提取卖点
    sell_points_container = intro_container.select_one('.sell_point.content_warp')
    if sell_points_container:
        points = sell_points_container.select('.item .value')
        parsed_data["selling_points"] = [point.text.strip() for point in points if point.text.strip()]

    # 4. 提取图文特性
    features = intro_container.select('.haier_product_feature')
    if features:
        for feature in features:
            title_el = feature.select_one('.firstTitle_key')
            desc_el = feature.select_one('.secondTitle_key')
            if title_el and desc_el:
                title = title_el.get_text(strip=True)
                desc = desc_el.get_text(strip=True)
                if title:
                    parsed_data["features"][title] = desc

    # 5. 提取底部的产品说明/注释
    notes_el = intro_container.select_one('.product_explanation')
    if notes_el:
        parsed_data["notes"] = notes_el.get_text(separator='\n', strip=True)

    # 6. 提取整个介绍区域内的所有图片
    all_images = intro_container.select('img[src]')
    parsed_data["image_urls"] = [urljoin(BASE_URL, img.get('src')) for img in all_images if img.get('src')]

    return parsed_data
def _extract_specifications(soup_obj):
    """从BeautifulSoup对象中提取规格参数"""
    specifications = {};
    spec_container = soup_obj.select_one('#pro-detail-con, .p-standard, div[data-obj="js_specificationTab"]')
    if spec_container:
        spec_rows = spec_container.select('.p-guige-item, .item, .item_content')
        for row in spec_rows:
            key_el, value_el = row.select_one('.p-guige-item-name, .name'), row.select_one(
                '.p-guige-item-value, .value')
            if key_el and value_el and key_el.text.strip(): specifications[key_el.text.strip()] = value_el.text.strip()
    return specifications


def _download_image_set(urls, subfolder_name, base_dir, referer):
    """模块化下载函数"""
    if not urls: return
    target_dir = os.path.join(base_dir, subfolder_name);
    os.makedirs(target_dir, exist_ok=True)
    headers = {'Referer': referer,
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'}
    for i, img_url in enumerate(urls):
        try:
            img_response = requests.get(img_url, headers=headers, timeout=15);
            img_response.raise_for_status()
            file_name = f"image_{i + 1}{os.path.splitext(img_url.split('?')[0])[1] or '.jpg'}"
            with open(os.path.join(target_dir, file_name), 'wb') as f:
                f.write(img_response.content)
        except Exception as e:
            print(f"    - 下载图片失败: {img_url}, 错误: {e}")

def get_product_urls_with_selenium(list_url, max_pages_to_scrape):
    """
     从产品列表页获取所有产品的详情页 URL 和对应的产品型号。
    支持自动点击“下一页”进行翻页，并可通过 max_pages_to_scrape 参数控制最大翻页次数。
    """
    product_info = {}
    driver = create_driver()
    try:
        driver.get(list_url)

        current_page = 1
        while current_page <= max_pages_to_scrape:
            print(f"\n--- 正在处理第 {current_page} 页 ---")

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li.proitem"))
            )

            scroll_pause_time = 3
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            product_elements = soup.select("li.proitem")
            print(f"在第 {current_page} 页找到 {len(product_elements)} 个产品项。")

            if not product_elements:
                print("当前页未找到产品，可能是最后一页，程序终止翻页。")
                break

            for element in product_elements:
                link_element = element.select_one("a.tit1")
                model_element = element.select_one("p.t2")
                if link_element and model_element:
                    url = urljoin(BASE_URL, link_element.get('href'))
                    model = model_element.text.strip()
                    if url and model:
                        product_info[url] = model

            if current_page == max_pages_to_scrape:
                print(f"已达到设置的最大爬取页数: {max_pages_to_scrape} 页。")
                break

            try:
                next_page_button_xpath = "/html/body/div[1]/div/div[10]/div[1]/div[2]/div/a[6]"
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, next_page_button_xpath))
                )

                if 'lose' in next_button.get_attribute('class'):
                    print("检测到'下一页'按钮为禁用状态，已是最后一页。")
                    break

                print("找到'下一页'按钮，准备点击进入下一页...")
                driver.execute_script("arguments[0].click();", next_button)
                current_page += 1
                time.sleep(random.uniform(3, 5))
            except (NoSuchElementException, TimeoutException):
                print("未找到可点击的'下一页'按钮，已到达最后一页。")
                break

    except Exception as e:
        print(f"提取列表信息时出错: {e}")
    finally:
        driver.quit()
        print("用于获取列表的浏览器已关闭。")
    return product_info

def scrape_and_save_product_details(driver, product_url, product_model, base_save_dir):
    """使用智能判断、分类归档策略，爬取所有信息。"""
    print(f"\n正在处理详情页: {product_url}")
    for attempt in range(MAX_RETRIES):
        try:
            driver.get(product_url)
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            product_name = soup.select_one('h1').get_text(strip=True)
            if not product_name: raise ValueError("关键元素'产品名称'未找到。")
            safe_folder_name = "".join(c for c in f"{product_name}-{product_model}" if c not in r'\/:*?"<>|')
            product_dir = os.path.join(base_save_dir, safe_folder_name)
            os.makedirs(product_dir, exist_ok=True)

            product_data = {
                'url': product_url,
                'name': product_name,
                'model': product_model,
                'introduction': '',
                'basic_attributes': {},
                'selling_points': [],
                'features': {},
                'notes': '',
                'specifications': {},
                'images': {'product': [], 'introduction': [], 'parameter': []}
            }

            introduction_parts = []
            price_text_raw = soup.select_one('.detail_top_content_right_price, .price').get_text(
                strip=True) if soup.select_one('.detail_top_content_right_price, .price') else ''
            price_match = re.search(r'(\d[\d,\.]*)', price_text_raw)
            if price_match:
                price = price_match.group(1)
                introduction_parts.append(f"参考价: ￥{price}")

            label_elements = soup.select('div.detail_top_content_right_label a.label_item, div.label_box a')
            if label_elements:
                labels = list(dict.fromkeys(
                    [label.get_text(strip=True) for label in label_elements if label.get_text(strip=True)]))
                if labels:
                    introduction_parts.append(f"产品标签: {' | '.join(labels)}")
            product_data['introduction'] = '\n\n'.join(filter(None, introduction_parts)) or "N/A"

            product_data['images']['product'] = [urljoin(BASE_URL, img.get('src')) for img in soup.select(
                'div.detail_top_product_preview_small li img[src], #thumblist li img[src]') if img.get('src')]

            intro_data = _parse_introduction_content(soup)

            if not intro_data.get("features"):
                print("  - 未发现默认展示的介绍特性，尝试点击选项卡...")
                try:
                    intro_button = driver.find_element(By.ID, 'introduceTab')
                    driver.execute_script("arguments[0].click();", intro_button)
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'div[data-obj="js_introduceTab"] .feature_js_introduce_main')))
                    clicked_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    intro_data = _parse_introduction_content(clicked_soup)
                except Exception as e:
                    print(f"  - 点击'产品介绍'选项卡失败或无内容: {e}")

            product_data['basic_attributes'] = intro_data.get('basic_attributes', {})
            product_data['selling_points'] = intro_data.get('selling_points', [])
            product_data['features'] = intro_data.get('features', {})
            product_data['notes'] = intro_data.get('notes', '')
            product_data['images']['introduction'] = intro_data.get('image_urls', [])

            try:
                spec_button = driver.find_element(By.ID, 'specificationTab')
                driver.execute_script("arguments[0].click();", spec_button)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-obj="js_specificationTab"] .js_params')))
                spec_soup = BeautifulSoup(driver.page_source, 'html.parser')
                product_data['specifications'] = _extract_specifications(spec_soup)
                spec_container = spec_soup.select_one('div[data-obj="js_specificationTab"]')
                if spec_container:
                    product_data['images']['parameter'] = [urljoin(BASE_URL, img.get('src')) for img in
                                                           spec_container.select('img[src]')]
            except Exception:
                print("  - '规格参数'选项卡不存在或无需点击。")

            with open(os.path.join(product_dir, 'details.json'), 'w', encoding='utf-8') as f:
                json.dump(product_data, f, ensure_ascii=False, indent=4)

            _download_image_set(product_data['images']['product'], 'product', product_dir, product_url)
            _download_image_set(product_data['images']['introduction'], 'introduction', product_dir, product_url)
            _download_image_set(product_data['images']['parameter'], 'parameter', product_dir, product_url)

            return

        except Exception as e:
            print(f"  [尝试 {attempt + 1}/{MAX_RETRIES}] 加载或处理页面失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(random.uniform(3, 5))
            else:
                print(f"  [严重错误] 达到最大重试次数，放弃处理此页面: {product_url}")

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    product_urls_map = get_product_urls_with_selenium(LIST_URL, MAX_PAGES_TO_SCRAPE)

    if not product_urls_map:
        print("未能获取到任何产品信息，程序终止。")
    else:
        detail_driver = create_driver()
        try:
            for url, model in product_urls_map.items():
                scrape_and_save_product_details(detail_driver, url, model, SAVE_DIR)
                print(f"  ...任务间隔，休眠 {random.uniform(1.5, 3.5):.2f} 秒...")
                time.sleep(random.uniform(1.5, 3.5))
        finally:
            detail_driver.quit()
            print("\n所有产品信息已成功爬取并保存！")