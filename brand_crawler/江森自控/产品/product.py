import requests
from bs4 import BeautifulSoup
import time
import os
import json
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def configure_requests_session():
    """配置带有重试策略的requests会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def initialize_selenium_driver(driver_path=None, headless=True):
    """初始化Selenium WebDriver"""
    chrome_options = webdriver.ChromeOptions()
    if headless:
        chrome_options.add_argument("--headless")  # 无头模式,不显示浏览器界面
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # 解决无法加载扩展的问题
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-application-cache')
    chrome_options.add_argument('--disable-setuid-sandbox')
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--disable-browser-side-navigation')
    chrome_options.add_argument('--enable-features=NetworkServiceInProcess')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')

    if driver_path:
        service = Service(executable_path=driver_path)
    else:
        service = Service()

    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
        })  # 反爬虫JS
        return driver
    except WebDriverException as e:
        print(f"初始化Selenium WebDriver失败: {e}")
        print(
            "请确保已安装Chrome浏览器,并下载了对应版本的ChromeDriver,并将其路径添加到系统PATH中,或在脚本中指定driver_path。")
        return None


def scrape_product_info_and_save(base_list_page_url, output_dir="hitachi_products", selenium_driver_path=None):
    """
    使用Selenium处理分页,从产品列表页爬取产品链接和名称,
    并使用requests从每个产品详情页爬取其余指定信息。
    """
    # Requests session for detail pages
    requests_session = configure_requests_session()
    requests_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Selenium driver for list page pagination
    driver = initialize_selenium_driver(driver_path=selenium_driver_path, headless=True)
    if driver is None:
        return

    try:
        print(f"正在使用Selenium打开产品列表页: {base_list_page_url}")
        driver.get(base_list_page_url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "familyfilterresult"))
        )


        page_num = 1
        max_pages_to_scrape = 5  # 设置一个最大页数,防止无限循环

        while page_num <= max_pages_to_scrape:

            current_page_html = driver.page_source
            soup = BeautifulSoup(current_page_html, 'html.parser')

            product_cards = soup.select('#familyfilterresult > div > div > div.productcard-title > a')

            if not product_cards:
                print(f"第 {page_num} 页未找到任何产品卡片,可能已到达最后一页或选择器失效。")
                break

            print(f"第 {page_num} 页找到 {len(product_cards)} 个产品。")

            for card_link_tag in product_cards:
                product_url = card_link_tag.get('href')
                if not product_url:
                    continue
                if not product_url.startswith('http'):
                    product_url = requests.compat.urljoin(base_list_page_url,
                                                          product_url)  # Using requests.compat for urljoin

                name_tag_from_list = card_link_tag.find('h2')
                product_name = name_tag_from_list.get_text(strip=True) if name_tag_from_list else '未知产品_' + str(
                    int(time.time()))

                sanitized_product_name = re.sub(r'[\\/:*?"<>|]', '_', product_name)
                sanitized_product_name = sanitized_product_name[:100].strip().rstrip('.')
                product_folder_path = os.path.join(output_dir, sanitized_product_name)

                if not os.path.exists(product_folder_path):
                    os.makedirs(product_folder_path)

                print(f"\n正在爬取产品详情页: {product_url}")

                try:
                    detail_response = requests_session.get(product_url, headers=requests_headers, timeout=15,
                                                           verify=False)
                    detail_response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"请求产品详情页失败: {product_url} - {e}")
                    print("跳过此产品。")
                    continue

                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

                model_tag = detail_soup.select_one(
                    '#content > div:nth-child(2) > section > div > div.col-7 > section > div > h2')
                product_model = model_tag.get_text(strip=True) if model_tag else 'N/A'

                image_tag = detail_soup.select_one('#content > div:nth-child(2) > section > div > div.col-5 > img')
                image_url = image_tag.get('src') if image_tag else None

                description_list_tags = detail_soup.select(
                    '#content > div:nth-child(2) > section > div > div.col-7 > section > div > ul > li')
                description_points = [li.get_text(strip=True) for li in
                                      description_list_tags] if description_list_tags else []

                overview_tag = detail_soup.select_one(
                    '#content > div:nth-child(4) > section > div > div > div:nth-child(1) > div > p')
                overview = overview_tag.get_text(strip=True) if overview_tag else 'N/A'

                # --- 处理宣传册或产品特性 (与上版相同) ---
                final_brochure_download_link = None
                product_features = []

                brochure_middle_page_link_tag = detail_soup.select_one('#slick-slide00 > div:nth-child(1) > div > a')
                if not brochure_middle_page_link_tag:
                    brochure_middle_page_link_tag = detail_soup.select_one(
                        '#slick-slide00 > div.slick-slide.slick-active > div > a')

                if brochure_middle_page_link_tag:
                    middle_page_url = brochure_middle_page_link_tag.get('href')
                    if not middle_page_url:
                        print("宣传册中间页链接href属性为空,跳过宣传册下载。")
                    elif not middle_page_url.startswith('http'):
                        middle_page_url = requests.compat.urljoin(product_url, middle_page_url)

                    print(f"正在访问宣传册中间页: {middle_page_url}")
                    time.sleep(1)
                    try:
                        middle_page_response = requests_session.get(middle_page_url, headers=requests_headers,
                                                                    timeout=10, verify=False)
                        middle_page_response.raise_for_status()
                        middle_page_soup = BeautifulSoup(middle_page_response.text, 'html.parser')

                        final_download_tag = middle_page_soup.select_one('#download')
                        if final_download_tag and final_download_tag.get('href'):
                            final_brochure_download_link = final_download_tag.get('href')
                            if not final_brochure_download_link.startswith('http'):
                                final_brochure_download_link = requests.compat.urljoin(middle_page_url,
                                                                                       final_brochure_download_link)
                            print(f"已找到最终宣传册下载链接: {final_brochure_download_link}")
                        else:
                            print("在宣传册中间页未找到最终下载链接(选择器 #download)。")

                    except requests.exceptions.RequestException as e:
                        print(f"请求宣传册中间页失败: {middle_page_url} - {e}")
                else:
                    print("未找到精确的宣传册中间页链接。尝试通过文本查找下载链接...")
                    found_general_brochure_link = False
                    all_links = detail_soup.find_all('a', href=True)
                    for link in all_links:
                        link_text = link.get_text(strip=True).lower()
                        if any(keyword in link_text or keyword in link['href'].lower() for keyword in
                               ['宣传册', '手册', '下载', 'brochure', 'manual', 'download']):
                            middle_page_url = link.get('href')
                            if not middle_page_url.startswith('http'):
                                middle_page_url = requests.compat.urljoin(product_url, middle_page_url)

                            print(f"通过通用文本/URL搜索找到疑似下载中间页: {middle_page_url}")
                            time.sleep(1)
                            try:
                                middle_page_response = requests_session.get(middle_page_url, headers=requests_headers,
                                                                            timeout=10,
                                                                            verify=False)
                                middle_page_response.raise_for_status()
                                middle_page_soup = BeautifulSoup(middle_page_response.text, 'html.parser')

                                final_download_tag = middle_page_soup.select_one('#download')
                                if final_download_tag and final_download_tag.get('href'):
                                    final_brochure_download_link = final_download_tag.get('href')
                                    if not final_brochure_download_link.startswith('http'):
                                        final_brochure_download_link = requests.compat.urljoin(middle_page_url,
                                                                                               final_brochure_download_link)
                                    found_general_brochure_link = True
                                    break
                                else:
                                    print(f"在 {middle_page_url} 页面未找到最终下载链接(#download)。")
                            except requests.exceptions.RequestException as e:
                                print(f"请求通用下载中间页失败: {middle_page_url} - {e}")

                    if not found_general_brochure_link:
                        print("未通过任何方式找到宣传册下载链接。")
                        print("尝试查找产品特性...")
                        features_section_title = detail_soup.find(lambda tag: tag.name in ['h2', 'h3', 'h4'] and (
                                '产品特性' in tag.get_text(strip=True) or 'Features' in tag.get_text(strip=True)))
                        if features_section_title:
                            next_ul = features_section_title.find_next_sibling('ul')
                            if next_ul:
                                product_features = [li.get_text(strip=True) for li in next_ul.find_all('li')]
                            else:
                                next_table = features_section_title.find_next_sibling('table')
                                if next_table:
                                    product_features = [li.get_text(strip=True) for li in next_table.find_all('li')]
                                else:
                                    features_lis = detail_soup.select(
                                        '#content > div:nth-child(6) > section > div > div > table li')
                                    if features_lis:
                                        product_features = [li.get_text(strip=True) for li in features_lis]
                                    else:
                                        print("未找到产品特性。")
                        else:
                            features_lis = detail_soup.select(
                                '#content > div:nth-child(6) > section > div > div > table li')
                            if features_lis:
                                product_features = [li.get_text(strip=True) for li in features_lis]
                            else:
                                print("未找到产品特性。")

                product_info = {
                    'url': product_url,
                    'name': product_name,
                    'model': product_model,
                    'image_url': image_url,
                    'description_points': description_points,
                    'overview': overview,
                    'brochure_download_link': final_brochure_download_link,
                    'features': product_features
                }

                json_file_path = os.path.join(product_folder_path, 'info.json')
                try:
                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(product_info, f, ensure_ascii=False, indent=4)
                except IOError as e:
                    print(f"保存产品信息到JSON文件失败: {json_file_path} - {e}")

                if image_url:
                    if not image_url.startswith('http'):
                        image_url = requests.compat.urljoin(product_url, image_url)
                    try:
                        img_response = requests_session.get(image_url, headers=requests_headers, stream=True,
                                                            timeout=10, verify=False)
                        img_response.raise_for_status()
                        image_name = os.path.basename(image_url.split('?')[0])
                        image_path = os.path.join(product_folder_path, image_name)
                        with open(image_path, 'wb') as img_f:
                            for chunk in img_response.iter_content(1024):
                                img_f.write(chunk)
                    except requests.exceptions.RequestException as e:
                        print(f"下载产品图片失败: {image_url} - {e}")
                else:
                    print("未找到产品图片链接。")

                # 下载产品宣传册
                if final_brochure_download_link:
                    try:
                        brochure_response = requests_session.get(final_brochure_download_link, headers=requests_headers,
                                                                 stream=True,
                                                                 timeout=20, verify=False)
                        brochure_response.raise_for_status()
                        brochure_name = os.path.basename(final_brochure_download_link.split('?')[0])
                        brochure_path = os.path.join(product_folder_path, brochure_name)
                        with open(brochure_path, 'wb') as brochure_f:
                            for chunk in brochure_response.iter_content(1024):
                                brochure_f.write(chunk)
                    except requests.exceptions.RequestException as e:
                        print(f"下载产品宣传册失败: {final_brochure_download_link} - {e}")
                else:
                    print("未找到最终产品宣传册下载链接。")

                time.sleep(1)

            next_page_selector = '#content > section.staticproductfamily.productfamily > div > section > div > div:nth-child(2) > div > div > a:nth-child(4)'

            try:
                try:
                    overlay = driver.find_element(By.CSS_SELECTOR, 'div.truste_overlay')
                    if overlay.is_displayed():
                        print("检测到覆盖层,尝试使用JavaScript关闭...")
                        driver.execute_script("arguments[0].style.display = 'none';", overlay)
                        time.sleep(0.5)
                except NoSuchElementException:
                    pass

                try:
                    close_buttons = driver.find_elements(By.CSS_SELECTOR,
                                                         'button.close, a.close, [class*="close"], [id*="close"]')
                    for btn in close_buttons:
                        if btn.is_displayed():
                            btn.click()
                            time.sleep(0.5)
                            break
                except:
                    pass

                next_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, next_page_selector))
                )

                if 'disableprev' in next_button.get_attribute('class'):
                    print("下一页按钮已被禁用或已到达最后一页,停止分页。")
                    break

                print(f"点击下一页按钮 (当前第 {page_num} 页)。")

                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                time.sleep(0.5)

                try:
                    driver.execute_script("arguments[0].click();", next_button)
                except:
                    next_button.click()
                time.sleep(3)

                page_num += 1

            except (TimeoutException, NoSuchElementException) as e:
                print(f"未找到下一页按钮或按钮不可点击 (可能已是最后一页): {e}")
                break
            except WebDriverException as e:
                print(f"点击下一页时发生WebDriver错误: {e}")
                break

    finally:
        if driver:
            driver.quit()
            print("Selenium WebDriver 已关闭。")

    print("\n所有产品信息爬取和保存完成。")


if __name__ == "__main__":
    # 风冷冷热水机组
    list_page_url = "https://cn.johnsoncontrols.com/hvac-equipment/air-cooled-chillers-and-heat-pump-units"
    # 水冷冷热水机组
    # "https://cn.johnsoncontrols.com/hvac-equipment/water-cooled-chillers-and-heat-pump-units"
    # 空气测产品
    # "https://cn.johnsoncontrols.com/hvac-equipment/airside"
    # 溴化锂吸收式机组
    # "https://cn.johnsoncontrols.com/hvac-equipment/absorption-products"
    output_directory = "johnson_controls_products"

    chrome_driver_path = None

    scrape_product_info_and_save(list_page_url, output_directory, selenium_driver_path=chrome_driver_path)