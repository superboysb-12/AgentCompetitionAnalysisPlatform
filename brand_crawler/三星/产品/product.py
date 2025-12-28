import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import os
import json
import re
from urllib.parse import urlparse


def _fix_image_url(url):
    """确保URL有完整的scheme"""
    if url and url.startswith('//'):
        return 'https:' + url
    return url


def scrape_product_list(url):
    """使用Selenium爬取产品列表页的所有产品链接，并处理“查看更多”按钮"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument(
        'User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    product_links = []

    try:
        driver = webdriver.Chrome(options=chrome_options)
        print(f"正在加载页面: {url}")
        driver.get(url)

        wait = WebDriverWait(driver, 20)

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.pd12-product-card__content')))
        except:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

        time.sleep(5)

        load_more_button_selector = '#content > div > div > div.responsivegrid.aem-GridColumn.aem-GridColumn--default--12 > div > div.bu-pd-g-product-finder.aem-GridColumn.aem-GridColumn--default--12 > div > div > div.pd12-product-finder__inner > div.pd12-product-finder__content > div.pd12-product-finder__content-cta.js-pf-cta-area > button'
        while True:
            try:
                load_more_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, load_more_button_selector))
                )
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                                      load_more_button)
                time.sleep(1)

                load_more_button.click()
                time.sleep(5)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            except Exception as e:
                print(f"  未能找到或点击'查看更多'按钮，或按钮不再可点击。可能已加载所有产品或按钮不存在: {e}")
                break  # 退出循环

        print("所有产品加载完毕，开始提取产品链接...")

        product_cards = driver.find_elements(By.CSS_SELECTOR, 'div.pd12-product-card__content')
        print(f"找到 {len(product_cards)} 个产品卡片")

        for card in product_cards:
            try:
                link_element = card.find_element(By.CSS_SELECTOR, 'a.cta')
                href = link_element.get_attribute('href')
                if href and '/business/system-air-conditioners/' in href:
                    product_links.append(href)
                    print(f"找到产品链接: {href}")
            except:
                continue

        if not product_links:
            all_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/business/system-air-conditioners/"]')
            for link in all_links:
                href = link.get_attribute('href')
                if href and 'all-system-air-conditioners' not in href and href not in product_links:
                    if link.get_attribute('data-modelcode'):
                        product_links.append(href)
                        print(f"找到产品链接: {href}")

        driver.quit()

    except Exception as e:
        print(f"Selenium爬取失败: {e}")

    return product_links


def download_image(image_url, folder, filename):
    """下载图片到指定文件夹 """
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, filename)
    image_url = _fix_image_url(image_url)

    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = session.get(image_url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✓ 图片下载成功: {filename}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"  ✗ 图片下载失败 {image_url}: {e}")
        return None


def extract_base_url_from_detail_url(url):
    """从详情页URL中提取基础URL路径"""
    url = url.rstrip('/')
    parts = url.split('/')
    base_parts = parts[:-1]
    return '/'.join(base_parts) + '/'


def build_model_url_from_input(base_url, input_element, current_url):
    """
    根据input元素的属性构建型号URL
    """
    try:
        modelname = input_element.get_attribute('data-modelname')
        modelcode = input_element.get_attribute('data-modelcode')
        modeldisplay = input_element.get_attribute('data-modeldisplay')

        # 从当前URL中提取前缀模式
        current_url_clean = current_url.rstrip('/')
        url_parts = current_url_clean.split('/')
        last_part = url_parts[-1]

        # *** MODIFICATION START ***
        # 原来的 r'^(.+?)-ac[0-9]' 无法匹配 'am' 开头的型号
        # 新的 regex 匹配 -ac 或 -am
        prefix_match = re.match(r'^(.+?)-(ac|am)[0-9]', last_part, re.IGNORECASE)
        # *** MODIFICATION END ***

        url_prefix = prefix_match.group(1) if prefix_match else 'commercial'

        clean_modelname = modelname.replace('/', '-').lower()

        # 构建最终URL
        url_suffix = f"{url_prefix}-{clean_modelname}"
        result_url = base_url + url_suffix + '/'

        print(f"    构建URL: {result_url}")
        return result_url

    except Exception as e:
        print(f"    构建URL时出错: {e}")
        import traceback
        print(f"    详细错误: {traceback.format_exc()}")
        return None


def extract_model_options_from_page(driver, base_url, current_url):
    """
    从当前页面提取所有型号选项及其URL
    """
    models = []
    seen_modelcodes = set()

    try:
        time.sleep(3)

        # 使用Path选择器
        xpath = "/html/body/div/div[4]/div/div/div[4]/div/div[2]/section/div[2]/div[1]/div/ul/li/div/input"

        try:
            model_inputs = driver.find_elements(By.XPATH, xpath)
        except:
            xpath_alt = "//input[@data-modelcode and @data-modelname]"
            model_inputs = driver.find_elements(By.XPATH, xpath_alt)

        for idx, input_elem in enumerate(model_inputs, 1):
            try:
                modelcode = input_elem.get_attribute('data-modelcode')
                modelname = input_elem.get_attribute('data-modelname')
                modeldisplay = input_elem.get_attribute('data-modeldisplay') or modelname

                if modelcode and modelcode not in seen_modelcodes:
                    seen_modelcodes.add(modelcode)

                    print(f"  型号 {idx}: {modelname} | 代码: {modelcode} | 显示名: {modeldisplay}")

                    model_url = build_model_url_from_input(base_url, input_elem, current_url)

                    if model_url:
                        model_info = {
                            'modelcode': modelcode,
                            'modelname': modelname,
                            'modeldisplay': modeldisplay,
                            'url': model_url
                        }
                        models.append(model_info)
                    else:
                        print(f"    URL构建失败，跳过该型号")

            except Exception as e:
                import traceback
                print(f"    {traceback.format_exc()}")
                continue

    except Exception as e:
        import traceback
        print(f"  {traceback.format_exc()}")

    return models


def download_spec_for_model(driver, product_folder, model_name, max_wait=40):
    """
    为特定型号下载规格参数文件
    """
    try:
        print(f"  等待页面完全加载...")
        time.sleep(5)

        # 滚动到页面底部,确保specs部分加载
        print(f"  滚动页面以加载specs部分...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        try:
            specs_section = driver.find_element(By.ID, 'specs')
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", specs_section)
            time.sleep(3)
        except Exception as e:
            print(f"  继续尝试查找下载按钮...")

        files_before = set(os.listdir(product_folder))

        selector = '#specs > div.spec-highlight__downloads__cta-wrap > a'

        # 备用选择器列表
        backup_selectors = [
            'div.spec-highlight__downloads__cta-wrap > a',
            '#specs a[href*=".pdf"]',
            '#specs a[href*=".xlsx"]',
            '#specs a.cta',
            'a[download][href*="spec"]',
        ]

        spec_button = None
        used_selector = None

        try:
            wait = WebDriverWait(driver, 15)

            # 先尝试主选择器
            try:
                spec_button = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                used_selector = selector
            except:
                # 尝试备用选择器
                for backup_sel in backup_selectors:
                    try:
                        spec_button = driver.find_element(By.CSS_SELECTOR, backup_sel)
                        used_selector = backup_sel
                        break
                    except:
                        continue

            if not spec_button:
                raise Exception("所有选择器都未找到下载按钮")

            # 滚动到按钮位置
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", spec_button)
            time.sleep(2)

            # 等待按钮可点击
            try:
                spec_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, used_selector))
                )
            except:
                print(f"按钮可能未完全可点击,继续尝试...")

            # 尝试点击下载按钮
            click_success = False

            try:
                spec_button.click()
                click_success = True
            except Exception as e1:
                print(f"  点击失败: {e1}")

                # 方法2: JavaScript点击
                try:
                    driver.execute_script("arguments[0].click();", spec_button)
                    click_success = True
                except Exception as e2:
                    print(f"  JavaScript点击失败: {e2}")

                    # 方法3: 模拟鼠标移动后点击
                    try:
                        from selenium.webdriver.common.action_chains import ActionChains
                        actions = ActionChains(driver)
                        actions.move_to_element(spec_button).click().perform()
                        click_success = True
                    except Exception as e3:
                        print(f"  ActionChains点击失败: {e3}")

            if not click_success:
                print(f"  ✗ 所有点击方法都失败")
                return None

            # 等待文件下载完成
            start_time = time.time()
            downloaded_file_path = None

            check_count = 0
            while time.time() - start_time < max_wait:
                time.sleep(1)
                check_count += 1

                try:
                    files_after = set(os.listdir(product_folder))
                    new_files = files_after - files_before

                    if new_files:
                        # 检查是否有完整下载的文件
                        for file in new_files:
                            if not file.endswith(('.crdownload', '.tmp', '.part', '.download')):
                                old_path = os.path.join(product_folder, file)

                                # 确保文件写入完成
                                file_size = os.path.getsize(old_path)
                                time.sleep(1)
                                new_size = os.path.getsize(old_path)

                                if file_size == new_size and file_size > 0:
                                    # 文件大小稳定,说明下载完成
                                    file_ext = os.path.splitext(file)[1].lower()  # 转小写以便比较
                                    clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(' ', '_')

                                    # *** MODIFICATION START ***
                                    if file_ext == '.xlsx':
                                        # 转换为 CSV
                                        new_filename = f"spec_{clean_model_name}.csv"
                                        new_path = os.path.join(product_folder, new_filename)

                                        if os.path.exists(new_path):
                                            os.remove(new_path)

                                        try:
                                            print(f"  .xlsx 文件找到: {file}. 正在转换为 CSV...")
                                            df = pd.read_excel(old_path)
                                            df.to_csv(new_path, index=False, encoding='utf-8-sig')

                                            # 删除原始xlsx文件
                                            os.remove(old_path)

                                            downloaded_file_path = new_path
                                            print(f"  ✓ 文件下载完成并转换为: {new_filename}")

                                        except Exception as convert_e:
                                            print(f"  ✗ 转换为CSV失败: {convert_e}. 保留原始 .xlsx 文件...")
                                            # 回退到重命名 .xlsx
                                            new_filename_xlsx = f"spec_{clean_model_name}{file_ext}"
                                            new_path_xlsx = os.path.join(product_folder, new_filename_xlsx)
                                            if os.path.exists(new_path_xlsx):
                                                os.remove(new_path_xlsx)
                                            os.rename(old_path, new_path_xlsx)
                                            downloaded_file_path = new_path_xlsx
                                            print(f"  ✓ 文件下载完成并重命名为 (原始格式): {new_filename_xlsx}")

                                    else:
                                        # 非xlsx文件,按原逻辑重命名
                                        new_filename = f"spec_{clean_model_name}{file_ext}"
                                        new_path = os.path.join(product_folder, new_filename)

                                        if os.path.exists(new_path):
                                            os.remove(new_path)

                                        time.sleep(0.5)
                                        os.rename(old_path, new_path)
                                        downloaded_file_path = new_path
                                        print(f"  ✓ 文件下载完成并重命名 ({file_ext}): {new_filename}")

                                    # *** MODIFICATION END ***
                                    break  # 退出 for file in new_files 循环

                        if downloaded_file_path:
                            break  # 退出 while 循环


                except Exception as e:
                    # 忽略文件检查过程中的错误,继续等待
                    if check_count % 10 == 0:
                        print(f"  ... 检查文件时出错,继续等待: {e}")
                    continue

            if downloaded_file_path:
                file_size = os.path.getsize(downloaded_file_path)
                print(f"  规格文件下载成功: {os.path.basename(downloaded_file_path)} ({file_size} bytes)")
                return downloaded_file_path
            else:
                print(f"  文件下载超时({max_wait}秒内未完成)")
                # 显示可能的部分下载文件
                try:
                    files_after = set(os.listdir(product_folder))
                    partial_files = [f for f in (files_after - files_before) if
                                     f.endswith(('.crdownload', '.tmp', '.part'))]
                    if partial_files:
                        print(f"  发现部分下载文件: {partial_files}")
                except:
                    pass
                return None

        except Exception as e:
            print(f"  未找到或无法点击下载按钮: {e}")
            # 尝试输出页面信息用于调试
            try:
                print(f"  当前页面URL: {driver.current_url}")
                print(f"  页面标题: {driver.title}")
            except:
                pass
            return None

    except Exception as e:
        print(f"  ✗ 下载规格文件时出错: {e}")
        import traceback
        print(f"  错误详情: {traceback.format_exc()}")
        return None


def clean_filename(text):
    cleaned_text = re.sub(r'[^\w\s\-\.（）]', '', text)
    cleaned_text = re.sub(r'\s+', '-', cleaned_text)  # 将空格替换为连字符
    cleaned_text = cleaned_text.strip('-._')  # 移除开头和结尾的特殊字符
    return cleaned_text


def check_if_url_already_scraped(url, base_folder="samsung_products"):
    """检查URL是否已经被爬取过"""
    if not os.path.exists(base_folder):
        return False, None

    # 规范化URL用于比较
    normalized_url = url.rstrip('/')

    # 遍历所有产品文件夹
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 查找该文件夹中的JSON文件
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        for json_file in json_files:
            json_path = os.path.join(folder_path, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('url', '').rstrip('/') == normalized_url:
                        return True, folder_path
            except Exception as e:
                continue

    return False, None


def get_unique_folder_name(base_folder, desired_name):
    """获取唯一的文件夹名称，如果存在重名则添加后缀"""
    folder_path = os.path.join(base_folder, desired_name)

    if not os.path.exists(folder_path):
        return desired_name

    # 如果文件夹存在，添加数字后缀
    counter = 1
    while True:
        new_name = f"{desired_name}_{counter}"
        new_path = os.path.join(base_folder, new_name)
        if not os.path.exists(new_path):
            return new_name
        counter += 1


def scrape_product_detail(detail_url, product_name_for_folder="unknown_product"):
    """使用Selenium爬取产品详情页的所有内容"""

    # 检查URL是否已经爬取过
    already_scraped, existing_folder = check_if_url_already_scraped(detail_url)
    if already_scraped:
        print(f"  ✓ 该URL已爬取过，跳过: {existing_folder}")
        return None

    # 提取基础URL
    base_url = extract_base_url_from_detail_url(detail_url)
    print(f"  基础URL: {base_url}")

    # 创建临时driver获取产品名称
    temp_chrome_options = Options()
    temp_chrome_options.add_argument('--headless')
    temp_chrome_options.add_argument('--no-sandbox')
    temp_chrome_options.add_argument('--disable-dev-shm-usage')
    temp_chrome_options.add_argument('User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    product_name = None

    try:
        temp_driver = webdriver.Chrome(options=temp_chrome_options)
        temp_driver.get(detail_url)
        time.sleep(5)

        # 获取产品名称
        name_selectors = [
            'div.pdd-buying-tool__info h2',
            'section.pdd-buying-tool h2',
            'div.pdp-header__buying-tool h2',
            'h2[class*="product"]',
            'h1'
        ]
        for selector in name_selectors:
            try:
                name_element = temp_driver.find_element(By.CSS_SELECTOR, selector)
                product_name = name_element.text.strip()
                if product_name:
                    break
            except:
                continue

        temp_driver.quit()
    except Exception as e:
        print(f"  获取产品名称时出错: {e}")
        if 'temp_driver' in locals():
            temp_driver.quit()

    # 确定文件夹名称
    if product_name:
        # 使用新的清理函数
        product_name_for_folder = clean_filename(product_name)
    if not product_name_for_folder or product_name_for_folder == "unknown_product":
        url_parts = detail_url.rstrip('/').split('/')
        product_name_for_folder = clean_filename(url_parts[-1])  # 对URL部分也进行清理

    # 获取唯一的文件夹名称（处理重名情况）
    base_products_folder = "samsung_products"
    unique_folder_name = get_unique_folder_name(base_products_folder, product_name_for_folder)

    if unique_folder_name != product_name_for_folder:
        print(f"  检测到重名，使用唯一名称: {unique_folder_name}")

    product_folder = os.path.join(base_products_folder, unique_folder_name)

    # 创建文件夹结构
    product_images_folder = os.path.join(product_folder, "product_images")
    poster_images_folder = os.path.join(product_folder, "poster_images")
    os.makedirs(product_images_folder, exist_ok=True)
    os.makedirs(poster_images_folder, exist_ok=True)

    # 产品详情数据
    product_details = {
        "url": detail_url,
        "name": product_name,
        "main_images": [],
        "posters": [],
        "models": []
    }

    print(f"正在爬取详情页: {detail_url}")

    # 配置主driver
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    prefs = {"download.default_directory": os.path.abspath(product_folder)}
    chrome_options.add_experimental_option("prefs", prefs)

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(detail_url)
        time.sleep(5)

        # 先提取型号列表(包含URL)
        models = extract_model_options_from_page(driver, base_url, detail_url)
        print(f"  共找到 {len(models)} 个不同型号")

        # 滚动页面加载内容
        print("\n  滚动页面加载所有内容...")
        for i in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # 提取产品图片
        main_image_selectors = [
            'img.first-image__main.first-image__desktop',
            'img.first-image__main',
            'div.pdp-header img.image__main',
            'img[class*="first-image"]',
            'div[id*="gallery"] img.image__main',
            'div.pdp-header img[class*="image"]'
        ]
        found_images = set()
        for selector in main_image_selectors:
            try:
                image_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for img_element in image_elements:
                    img_url = img_element.get_attribute('src') or img_element.get_attribute('data-src')
                    if img_url and img_url not in found_images:
                        found_images.add(img_url)
                        filename = f"product_image_{len(product_details['main_images']) + 1}.jpg"
                        filepath = download_image(img_url, product_images_folder, filename)
                        if filepath:
                            product_details["main_images"].append({"url": img_url, "local_path": filepath})
            except:
                continue
        print(f"  共找到 {len(product_details['main_images'])} 张产品图片")

        # 提取海报 - 使用多种策略
        print("\n  开始提取海报和特性描述...")

        # 策略1: 尝试标准的benefit区域
        poster_containers = []
        try:
            benefit_section = driver.find_element(By.ID, 'benefit')
            containers = benefit_section.find_elements(By.CSS_SELECTOR, 'div.feature-benefit')
            if containers:
                print(f"  [策略1] 在#benefit中找到 {len(containers)} 个feature-benefit容器")
                poster_containers.extend(containers)
        except Exception as e:
            print(f"  [策略1] 未找到#benefit区域: {e}")

        # 策略2: 查找所有可能的特性容器
        if not poster_containers:
            container_selectors = [
                'div.feature-benefit',
                'div[class*="feature"]',
                'section[class*="feature"]',
                'div[class*="benefit"]',
                'div.product-feature',
                'div.pd-feature',
                'article[class*="feature"]'
            ]
            for selector in container_selectors:
                try:
                    containers = driver.find_elements(By.CSS_SELECTOR, selector)
                    if containers:
                        print(f"  [策略2] 使用选择器 '{selector}' 找到 {len(containers)} 个容器")
                        poster_containers.extend(containers)
                        break
                except:
                    continue

        # 策略3: 查找包含大图和文字的通用容器
        if not poster_containers:
            print(f"  [策略3] 尝试查找通用的图文容器...")
            try:
                # 查找所有section或div，筛选出同时包含图片和标题的容器
                all_sections = driver.find_elements(By.CSS_SELECTOR, 'section, div[class*="section"], div[class*="content"]')
                for section in all_sections:
                    try:
                        # 检查是否同时包含图片和标题
                        imgs = section.find_elements(By.CSS_SELECTOR, 'img')
                        headings = section.find_elements(By.CSS_SELECTOR, 'h1, h2, h3, h4')
                        if imgs and headings:
                            poster_containers.append(section)
                    except:
                        continue
                if poster_containers:
                    print(f"  [策略3] 找到 {len(poster_containers)} 个通用图文容器")
            except Exception as e:
                print(f"  [策略3] 失败: {e}")

        # 提取每个容器中的内容
        for idx, container in enumerate(poster_containers, 1):
            poster_data = {"image_url": None, "image_local_path": None, "title": None, "explanation": None}

            # 提取图片 - 尝试多种选择器
            image_selectors = [
                'img.image__main.responsive-img',
                'img.image__main',
                'img.responsive-img',
                'img[class*="feature"]',
                'img[class*="benefit"]',
                'img[class*="main"]',
                'picture img',
                'img'
            ]

            for img_selector in image_selectors:
                try:
                    img_element = container.find_element(By.CSS_SELECTOR, img_selector)
                    img_url = img_element.get_attribute('src') or img_element.get_attribute('data-src')
                    if img_url and not img_url.endswith('.svg'):  # 排除svg图标
                        poster_data["image_url"] = img_url
                        filename = f"poster_{len(product_details['posters']) + 1}.jpg"
                        poster_data["image_local_path"] = download_image(img_url, poster_images_folder, filename)
                        break
                except:
                    continue

            # 提取标题 - 尝试多种选择器
            title_selectors = [
                'div.feature-benefit__text-wrap h2',
                'div[class*="text"] h2',
                'h2',
                'h3',
                'div[class*="title"]',
                'div[class*="heading"]'
            ]

            for title_selector in title_selectors:
                try:
                    title_element = container.find_element(By.CSS_SELECTOR, title_selector)
                    title_text = title_element.text.strip()
                    if title_text and len(title_text) > 0:
                        poster_data["title"] = title_text
                        break
                except:
                    continue

            # 提取说明文字 - 尝试多种选择器
            explanation_selectors = [
                'div.feature-benefit__text-wrap p',
                'div[class*="text"] p',
                'p',
                'div[class*="description"]',
                'div[class*="desc"]'
            ]

            for exp_selector in explanation_selectors:
                try:
                    exp_elements = container.find_elements(By.CSS_SELECTOR, exp_selector)
                    # 合并所有段落
                    exp_texts = []
                    for exp_elem in exp_elements:
                        exp_text = exp_elem.text.strip()
                        if exp_text and len(exp_text) > 0:
                            exp_texts.append(exp_text)

                    if exp_texts:
                        poster_data["explanation"] = '\n'.join(exp_texts)
                        break
                except:
                    continue

            # 只保存有效的海报数据（至少有图片或标题）
            if poster_data["image_url"] or poster_data["title"]:
                product_details["posters"].append(poster_data)
                print(f"  ✓ 海报 {len(product_details['posters'])}: 图片={'✓' if poster_data['image_url'] else '✗'}, 标题={'✓' if poster_data['title'] else '✗'}, 说明={'✓' if poster_data['explanation'] else '✗'}")

        print(f"  共找到 {len(product_details['posters'])} 张海报")

        # 为每个型号下载规格参数
        if models:
            print(f"\n  开始为 {len(models)} 个型号下载规格参数...")

            for idx, model in enumerate(models, 1):
                print(f"\n  >>> 处理型号 {idx}/{len(models)}: {model['modelname']} <<<")

                model_url = model.get('url')
                if not model_url:
                    print(f"  未能生成URL,跳过该型号")
                    model_data = {
                        'modelcode': model['modelcode'],
                        'modelname': model['modelname'],
                        'modeldisplay': model['modeldisplay'],
                        'url': None,
                        'spec_file': None
                    }
                    product_details['models'].append(model_data)
                    continue

                print(f"  访问URL: {model_url}")

                try:
                    driver.get(model_url)
                    time.sleep(5)

                    # 下载规格文件
                    spec_file = download_spec_for_model(driver, product_folder, model['modelname'])

                    model_data = {
                        'modelcode': model['modelcode'],
                        'modelname': model['modelname'],
                        'modeldisplay': model['modeldisplay'],
                        'url': model_url,
                        'spec_file': spec_file
                    }
                    product_details['models'].append(model_data)

                except Exception as e:
                    print(f"  处理型号时出错: {e}")
                    model_data = {
                        'modelcode': model['modelcode'],
                        'modelname': model['modelname'],
                        'modeldisplay': model['modeldisplay'],
                        'url': model_url,
                        'spec_file': None
                    }
                    product_details['models'].append(model_data)

                time.sleep(2)
        else:
            print("  未找到型号选择器，将当前页面作为单个型号处理...")

            # 尝试从URL提取型号信息
            url_parts = detail_url.rstrip('/').split('/')
            last_part = url_parts[-1] if url_parts else "unknown"

            # 尝试从页面提取型号代码
            model_code = None
            model_name = last_part

            try:
                # 尝试多个可能的选择器来查找型号代码
                code_selectors = [
                    'div.pdd-buying-tool__info span[class*="model"]',
                    'span.model-code',
                    'div[class*="model-code"]',
                    'span[class*="sku"]'
                ]

                for selector in code_selectors:
                    try:
                        code_element = driver.find_element(By.CSS_SELECTOR, selector)
                        model_code = code_element.text.strip()
                        if model_code:
                            print(f"  从页面提取到型号代码: {model_code}")
                            break
                    except:
                        continue

                # 如果没找到，尝试从URL提取
                if not model_code:
                    # 匹配类似 am045mnldeh-sc 的模式
                    code_match = re.search(r'(am|ac)[\w-]+', last_part, re.IGNORECASE)
                    if code_match:
                        model_code = code_match.group(0).upper()
                        print(f"  从URL提取到型号代码: {model_code}")
                    else:
                        model_code = last_part.upper()
                        print(f"  使用URL最后部分作为型号代码: {model_code}")

            except Exception as e:
                print(f"  提取型号代码时出错: {e}")
                model_code = last_part.upper()

            print(f"\n  >>> 处理单个型号: {model_name} <<<")

            # 下载规格文件
            spec_file = download_spec_for_model(driver, product_folder, model_name)

            model_data = {
                'modelcode': model_code,
                'modelname': model_name,
                'modeldisplay': product_name or model_name,
                'url': detail_url,
                'spec_file': spec_file
            }
            product_details['models'].append(model_data)
            print(f"  ✓ 单个型号数据已添加")

        # 保存JSON
        json_filename = f"{product_name_for_folder}.json"
        json_filepath = os.path.join(product_folder, json_filename)
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(product_details, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ 数据已保存到: {json_filepath}")

        driver.quit()

    except Exception as e:
        print(f"  ✗ 爬取详情页时发生错误: {e}")
        if 'driver' in locals():
            driver.quit()

    return product_details


if __name__ == "__main__":
    product_list_url = "https://www.samsung.com.cn/business/system-air-conditioners/all-system-air-conditioners/"

    print(f"\n正在从 {product_list_url} 爬取产品列表...")

    links = scrape_product_list(product_list_url)
    all_product_data = []

    if links:
        print(f"\n✓ 成功爬取到 {len(links)} 个产品链接")
        print("=" * 60)
        for i, link in enumerate(links):
            print(f"\n>>> 正在处理产品 {i + 1}/{len(links)} <<<")
            product_data = scrape_product_detail(link)

            # *** MODIFICATION START ***
            # 只有当 product_data 不是 None (即没有被跳过) 时才添加
            if product_data:
                all_product_data.append(product_data)
            # *** MODIFICATION END ***

            time.sleep(3)
    else:
        print("\n✗ 未能爬取到任何产品链接")

    print("\n" + "=" * 60)

    print("\n程序执行完成!")