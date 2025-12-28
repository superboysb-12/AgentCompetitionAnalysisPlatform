import os
import json
import re
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def sanitize_filename(filename):
    """移除文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()


def download_image(url, folder_path, filename):
    """下载单个图片"""
    try:
        os.makedirs(folder_path, exist_ok=True)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except:
        return False


def scrape_product_details(driver, base_folder="scraped_data"):
    """爬取产品详情页内容"""
    try:
        product_name_element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.product_right > h1'))
        )
        product_name = product_name_element.text
        print(f"  正在处理: {product_name}")

        product_data = {
            'name': product_name,
            'source_url': driver.current_url,
            'product_image_urls': [],
            'poster_image_urls': []
        }

        try:
            price_text = driver.find_element(By.CSS_SELECTOR, 'div.floor_price_act span b').text
            product_data['price'] = price_text
        except NoSuchElementException:
            product_data['price'] = "价格未找到"

        spec_params = {}
        try:
            spec_container = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#product_spec'))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", spec_container)
            time.sleep(1)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#product_spec > div > table > tbody > tr > td'))
            )

            all_rows = spec_container.find_elements(By.CSS_SELECTOR, 'table tbody tr')
            for row in all_rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 2:
                        key = cells[0].get_attribute('textContent').strip()
                        value = cells[1].get_attribute('textContent').strip()
                        if key and value:
                            spec_params[key] = value
                except:
                    continue

            product_data['specifications'] = spec_params

        except TimeoutException:
            product_data['specifications'] = {}
            print("     规格参数加载超时")

        safe_folder_name = sanitize_filename(product_name)
        product_folder = os.path.join(base_folder, safe_folder_name)
        product_images_folder = os.path.join(product_folder, 'images', 'product_images')
        product_posters_folder = os.path.join(product_folder, 'images', 'product_posters')

        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#thumbnails')))
            product_image_elements = driver.find_elements(By.CSS_SELECTOR, '#thumbnails li a img')
            success_count = 0
            for i, img_element in enumerate(product_image_elements):
                img_url = img_element.get_attribute('src') or img_element.get_attribute('data-src')
                if img_url:
                    product_data['product_image_urls'].append(img_url)
                    filename = f"image_{i + 1}.jpg"
                    if download_image(img_url, product_images_folder, filename):
                        success_count += 1
        except TimeoutException:
            print("    ✗ 未找到产品图片")

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        poster_image_elements = driver.find_elements(By.CSS_SELECTOR, '#product_intro > div > img')
        success_count = 0
        for i, img_element in enumerate(poster_image_elements):
            img_url = img_element.get_attribute('src') or img_element.get_attribute('data-src')
            if img_url:
                product_data['poster_image_urls'].append(img_url)
                filename = f"poster_{i + 1}.jpg"
                if download_image(img_url, product_posters_folder, filename):
                    success_count += 1

        json_path = os.path.join(product_folder, 'details.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(product_data, f, ensure_ascii=False, indent=4)
        print(f"    数据已保存")
        return True

    except Exception as e:
        print(f"    处理失败: {e}")
        return False


# --- 主控制函数 ---

def main():
    # 壁挂式
    START_URL = 'https://www.midea.cn/s/search/search.html?addr_code=&category_id=10002&attr_list=127:364'
    # 立柜式
    # START_URL = 'https://www.midea.cn/s/search/search.html?addr_code=440000%2C440100%2C440106&category_id=10003&attr_list=127:365'
    # 中央空调
    # START_URL = 'https://www.midea.cn/s/search/search.html?addr_code=440000%2C440100%2C440106&category_id=10004'
    # 移动空调
    # START_URL = 'https://www.midea.cn/s/search/search.html?addr_code=440000%2C440100%2C440106&category_id=10007&attr_list=127:368'

    BASE_DATA_FOLDER = "midea_products"

    driver = webdriver.Chrome()
    driver.get(START_URL)

    current_page = 1
    while True:
        print(f"\n===== 第 {current_page} 页 =====")
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.search_list_wrap'))
            )
            time.sleep(2)

            product_elements = driver.find_elements(By.CSS_SELECTOR, 'li.hproduct > a')
            product_links = [elem.get_attribute('href') for elem in product_elements if elem.get_attribute('href')]

            if not product_links:
                print("未找到产品，爬取结束")
                break

            print(f"找到 {len(product_links)} 个产品")

            for index, link in enumerate(product_links):
                print(f"\n-> 第 {index + 1}/{len(product_links)} 个产品")
                driver.switch_to.new_window('tab')
                driver.get(link)
                scrape_product_details(driver, BASE_DATA_FOLDER)
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(1)

        except TimeoutException:
            print("页面加载超时，退出")
            break

        try:
            next_page_button = driver.find_element(By.CSS_SELECTOR, 'span.page-end')
            if 'ban' in next_page_button.get_attribute('class'):
                print("已到最后一页")
                break
            driver.execute_script("arguments[0].click();", next_page_button)
            current_page += 1
            time.sleep(3)
        except NoSuchElementException:
            print("未找到下一页按钮")
            break

    driver.quit()
    print("\n===== 爬取完成 =====")


if __name__ == '__main__':
    main()