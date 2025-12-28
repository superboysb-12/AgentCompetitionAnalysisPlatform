import os
import json
import re
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


BASE_DATA_FOLDER = "casarte_zy"


def sanitize_filename(name):
    """
    清理字符串，使其可以作为有效的文件或文件夹名。
    """
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def get_all_product_urls(list_url):
    """
    使用Selenium获取产品列表页面上所有产品的详情页URL。

    :param list_url: 产品列表页的URL
    :return: 包含所有产品URL的列表
    """
    print("步骤1: 开始获取所有产品详情页的URL...")
    product_links = []
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('log-level=3')
    options.add_argument(
        'user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"')

    driver = None
    try:
        service = ChromeService(executable_path=ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(list_url)
        time.sleep(5)  # 等待初始页面加载

        while True:
            try:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".load_more.js_load_more"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                time.sleep(1)  # 等待滚动动画
                load_more_button.click()
                print("  > 成功点击 '加载更多' 按钮，等待新产品加载...")
                time.sleep(3)  # 等待新产品加载完成
            except TimeoutException:
                print("  > '加载更多' 按钮未找到，已加载所有产品。")
                break
            except Exception as e:
                print(f"  > 点击 '加载更多' 时发生错误: {e}")
                break

        print("  > 开始从完全加载的页面中提取所有产品链接...")
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        product_list_ul = soup.find('ul', class_='product_list')
        if product_list_ul:
            product_items = product_list_ul.find_all('li', class_='product_item')
            for item in product_items:
                if 'data-url' in item.attrs:
                    relative_url = item['data-url']
                    if relative_url.startswith('//'):
                        full_url = 'https:' + relative_url
                        if full_url not in product_links:
                            product_links.append(full_url)
            print(f"成功找到 {len(product_links)} 个产品链接。")
        else:
            print("错误：在产品列表页未能找到产品列表容器。")

    except Exception as e:
        print(f"获取产品URL列表时出错: {e}")
    finally:
        if driver:
            driver.quit()
    return product_links


def scroll_to_bottom(driver, pause_time=2):
    """
    使用Selenium模拟浏览器滚动到底部，以触发所有懒加载内容的加载。
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)

        try:
            feature_element = driver.find_element(By.CLASS_NAME, "product_feature")
            driver.execute_script("arguments[0].scrollIntoView(true);", feature_element)
            time.sleep(1)
        except:
            pass

        try:
            spec_element = driver.find_element(By.ID, "js_specification")
            driver.execute_script("arguments[0].scrollIntoView(true);", spec_element)
            time.sleep(1)
        except:
            pass

        new_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count += 1

        if new_height == last_height or scroll_count > 5:
            break
        last_height = new_height

    time.sleep(3)


def scrape_product_features(soup, driver=None):
    """
    从BeautifulSoup对象中提取产品特性信息。
    增强版：尝试多种选择器策略
    """
    features = []

    features_container = soup.find('div', class_='product_feature')

    if not features_container:
        print("  > 警告：未找到产品特性主容器 (div.product_feature)。")

        features_container = soup.find('div', class_=re.compile('product.*feature', re.I))

    if not features_container:
        if driver:
            try:
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "product_feature")))
                # 重新获取页面源码
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                features_container = soup.find('div', class_='product_feature')
            except:
                print("  > 等待产品特性加载超时")

    if not features_container:
        return features

    feature_boxes = []

    selectors = [
        'trs_customize_features_fragment_Box',
        'trs_customize_features',
        'fragment_Box',
        'feature_item'
    ]

    for selector in selectors:
        boxes = features_container.find_all('div', class_=re.compile(selector, re.I))
        if boxes:
            feature_boxes.extend(boxes)
            break

    if not feature_boxes:
        feature_boxes = features_container.find_all('div', recursive=False)
        feature_boxes = [box for box in feature_boxes if box.get('class') and
                         any('feature' in ' '.join(box.get('class', [])) or
                             'fragment' in ' '.join(box.get('class', [])) for cls in box.get('class', []))]


    for idx, box in enumerate(feature_boxes):
        title = ""
        description = ""
        image_url = ""

        text_containers = box.find_all('div', class_=re.compile('(title|text|desc|content)', re.I))

        for container in text_containers:
            text = container.get_text(strip=True)
            if text and not title and 'title' in container.get('class', []):
                em_tag = container.find('em')
                if em_tag:
                    for element in em_tag.contents:
                        if element.name == 'br':
                            break
                        if isinstance(element, str):
                            title += element.strip()
                else:
                    title = text
            elif text and not description:
                description = text

        if not description:
            em_tags = box.find_all('em')
            for em in em_tags:
                br_tag = em.find('br')
                if br_tag:
                    texts = list(em.stripped_strings)
                    if len(texts) >= 2:
                        if not title:
                            title = texts[0]
                        description = texts[1] if len(texts) > 1 else ""

        img_selectors = [
            ('div', {'class': 'all_main_imgbox'}),
            ('div', {'class': re.compile('img.*box', re.I)}),
            ('div', {'class': re.compile('.*image.*', re.I)})
        ]

        img_tag = None
        for tag_name, attrs in img_selectors:
            img_container = box.find(tag_name, attrs)
            if img_container:
                img_tag = img_container.find('img')
                if img_tag:
                    break

        if not img_tag:
            img_tag = box.find('img')

        if img_tag:
            raw_url = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('data-original')
            if raw_url and raw_url.strip():
                image_url = 'https:' + raw_url if raw_url.startswith('//') else raw_url

        if title or description or image_url:
            features.append({
                'title': title,
                'description': description,
                'image_url': image_url
            })

    return features


def scrape_specifications(soup, driver=None):
    """
    从BeautifulSoup对象中提取规格参数信息。
    增强版：添加更多等待和查找策略
    """
    specifications = {}
    main_image_url = ""

    spec_area = soup.find('div', id='js_specification')

    if not spec_area:
        spec_area = soup.find('div', class_=re.compile('specification', re.I))

    if not spec_area and driver:
        try:
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.ID, "js_specification")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            spec_area = soup.find('div', id='js_specification')
        except:
            print("  > 等待规格参数加载超时")

    if not spec_area:
        print("  > 警告：未找到规格参数模块。")
        return specifications, main_image_url

    picture_tags = spec_area.find_all(['picture', 'img'])
    for tag in picture_tags:
        if tag.name == 'picture':
            source_tag = tag.find('source', attrs={'srcset': True})
            if source_tag and source_tag['srcset'].strip():
                raw_url = source_tag['srcset'].strip().split(' ')[0]
                main_image_url = 'https:' + raw_url if raw_url.startswith('//') else raw_url
                break
        elif tag.name == 'img' and tag.get('src'):
            raw_url = tag.get('src')
            main_image_url = 'https:' + raw_url if raw_url.startswith('//') else raw_url
            break

    param_lists = spec_area.find_all('ul')
    for ul in param_lists:
        for item in ul.find_all('li'):
            label = None
            value = None

            label_div = item.find('div', class_='label')
            value_div = item.find('div', class_='value')

            if label_div and value_div:
                label = label_div.get_text(strip=True)
                value = value_div.get_text(strip=True)
            else:
                spans = item.find_all('span')
                if len(spans) >= 2:
                    label = spans[0].get_text(strip=True)
                    value = spans[1].get_text(strip=True)
                else:
                    text = item.get_text(strip=True)
                    if '：' in text:
                        parts = text.split('：', 1)
                        label = parts[0].strip()
                        value = parts[1].strip()

            if label and value:
                specifications[label] = value

    return specifications, main_image_url


def download_image(url, folder_path, filename, headers):
    """
    下载单个图片并保存到指定路径。
    """
    if not url or not url.startswith('http'):
        return None

    relative_folder = os.path.relpath(folder_path, BASE_DATA_FOLDER)
    relative_path = os.path.join(relative_folder, filename).replace('\\', '/')

    try:
        img_response = requests.get(url, headers=headers, timeout=20)
        img_response.raise_for_status()

        os.makedirs(folder_path, exist_ok=True)
        img_path = os.path.join(folder_path, filename)

        with open(img_path, 'wb') as f:
            f.write(img_response.content)
        return relative_path
    except requests.exceptions.RequestException as img_e:
        print(f"  > 下载图片失败: {url}, 错误: {img_e}")
        return None



def scrape_and_save_product_details(product_url, save_debug=True):
    """
    从单个产品详情页抓取信息并保存。
    """
    print(f"\n正在处理产品页面: {product_url}")
    options = webdriver.ChromeOptions()
    # 启用headless模式以在后台运行
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('log-level=3')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"')

    driver = None
    try:
        service = ChromeService(executable_path=ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)

        driver.get(product_url)
        time.sleep(5)

        scroll_to_bottom(driver)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        operate_box = soup.find('div', class_='operate_box')
        if not operate_box:
            print("警告：在此页面上未找到 'operate_box' 容器，尝试其他方式获取产品信息。")
            product_name = "未知产品"
            product_model = "未知型号"
            product_price = "N/A"

            title_elem = soup.find('title')
            if title_elem:
                product_name = title_elem.get_text().split('-')[0].strip()
        else:
            product_name = operate_box.get('data-pname', '未知产品名称')
            product_model = operate_box.get('data-promodelno', '未知型号')
            product_price = operate_box.get('data-price', 'N/A')

        tags = [span.text.strip() for span in soup.select('div.sale_point span.item')]

        print(f"  > 名称: {product_name}")
        print(f"  > 型号: {product_model}")

        folder_name = sanitize_filename(f"{product_name}_{product_model}")
        product_folder_path = os.path.join(BASE_DATA_FOLDER, folder_name)

        banner_image_urls = []
        image_container = soup.find('div', class_='banner_img_swiper')
        if image_container:
            all_media = image_container.find_all(['source', 'img'])
            for media in all_media:
                url = media.get('srcset', '').strip().split(' ')[0] or media.get('src', '')
                if url and url.strip().startswith('//'):
                    full_img_url = 'https:' + url
                    if full_img_url not in banner_image_urls:
                        banner_image_urls.append(full_img_url)

        features = scrape_product_features(soup, driver)
        specifications, spec_main_image_url = scrape_specifications(soup, driver)

        banner_images_path = os.path.join(product_folder_path, "images", "banner")
        features_images_path = os.path.join(product_folder_path, "images", "features")
        specs_images_path = os.path.join(product_folder_path, "images", "specs")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': product_url,
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }

        # 下载轮播图
        downloaded_banner_images = []
        for i, img_url in enumerate(banner_image_urls):
            local_path = download_image(img_url, banner_images_path, f"{i + 1}.png", headers)
            if local_path:
                downloaded_banner_images.append({'url': img_url, 'local_path': local_path})

        # 下载特性图片
        downloaded_features = []
        for i, feature in enumerate(features):
            local_path = None
            if feature.get('image_url'):
                local_path = download_image(feature['image_url'], features_images_path, f"{i + 1}.png", headers)
            downloaded_features.append({
                'title': feature['title'],
                'description': feature['description'],
                'image': {'url': feature.get('image_url', ''), 'local_path': local_path}
            })

        # 下载规格图片
        downloaded_spec_image = None
        if spec_main_image_url:
            spec_local_path = download_image(spec_main_image_url, specs_images_path, "main.png", headers)
            if spec_local_path:
                downloaded_spec_image = {'url': spec_main_image_url, 'local_path': spec_local_path}


        # 保存数据到JSON
        product_data = {
            'product_name': product_name,
            'product_model': product_model,
            'price': product_price,
            'tags': tags,
            'product_url': product_url,
            'banner_images': downloaded_banner_images,
            'features': downloaded_features,
            'specifications': {
                'main_image': downloaded_spec_image,
                'parameters': specifications
            }
        }

        json_path = os.path.join(product_folder_path, 'details.json')
        os.makedirs(product_folder_path, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(product_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        import traceback
        print(f"处理产品 {product_url} 时发生严重错误: {e}")
        traceback.print_exc()
    finally:
        if driver:
            driver.quit()

def main():
    """
    主执行函数
    """
    # 空调
    # target_list_url = 'https://www.casarte.com/air-conditioners/?spm=cn.news-list_pc.header-nav-20240219.3'
    # 中央空调
    target_list_url = "https://www.casarte.com/central-air-conditioning/"
    product_urls = get_all_product_urls(target_list_url)

    if not product_urls:
        print("未能获取到任何产品链接，程序退出。")
        return

    os.makedirs(BASE_DATA_FOLDER, exist_ok=True)

    print("\n开始抓取每个产品的详细信息...")
    for idx, url in enumerate(product_urls):
        print(f"\n进度: {idx + 1}/{len(product_urls)}")
        scrape_and_save_product_details(url)
        time.sleep(2)  # 增加延迟避免被反爬

    print("\n所有产品数据抓取完成！")


if __name__ == '__main__':
    main()