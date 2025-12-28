import requests
import os
import json
import re
import time
from datetime import datetime

# 导入Selenium相关库
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


def sanitize_filename(filename):
    """移除文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()


def get_filtered_news_list_with_selenium():
    """
    使用Selenium驱动浏览器，智能等待并点击“加载更多”，获取所有符合条件的新闻信息。
    """
    start_url = "http://www.casarte.com/about/news/?spm=casarte2023.homepage_pc.header-nav-20240219.1"
    base_url = "http://www.casarte.com"
    stop_date = datetime(2024, 1, 1)

    print("正在启动浏览器...")
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    print(f"正在访问页面: {start_url}")
    driver.get(start_url)

    # --- 这是本次修改的核心逻辑 ---
    while True:
        try:
            # 1. 点击前，获取当前新闻条目的数量
            item_count_before = len(driver.find_elements(By.CSS_SELECTOR, "div.news_list_item"))

            # 2. 检查最后一条新闻的日期，决定是否需要继续加载
            all_dates = driver.find_elements(By.CSS_SELECTOR, "div.news_date")
            if all_dates:
                last_date_str = all_dates[-1].text.strip()
                last_article_date = datetime.strptime(last_date_str, '%Y.%m.%d')
                if last_article_date < stop_date:
                    print(f"发现最后一条新闻日期为 {last_date_str}，早于目标日期，停止加载。")
                    break

            # 3. 找到并点击“加载更多”按钮
            load_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "div.load_more_box"))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_button)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", load_more_button)

            # 4. 智能等待：等待新闻条目数量增加
            print(f"已点击“加载更多”，当前有 {item_count_before} 条，等待数量增加...")
            WebDriverWait(driver, 10).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, "div.news_list_item")) > item_count_before
            )
            item_count_after = len(driver.find_elements(By.CSS_SELECTOR, "div.news_list_item"))
            print(f"内容已加载，现在有 {item_count_after} 条。")

        except TimeoutException:
            # 如果找不到按钮，或者点击后数量在10秒内没有增加，说明已经到底了
            print("未找到“加载更多”按钮或内容未增加，已加载全部内容。")
            break
        except Exception as e:
            print(f"发生未知错误: {e}")
            break
    # --- 核心逻辑修改结束 ---

    print("数据加载完成，正在解析HTML...")
    page_html = driver.page_source
    driver.quit()
    print("浏览器已关闭。")

    soup = BeautifulSoup(page_html, 'html.parser')
    news_items = soup.find_all('div', class_='news_list_item')

    all_news_data = []
    for item in news_items:
        date_tag = item.find('div', class_='news_date')
        link_tag = item.find('a', href=True)
        title_tag = item.find('a', class_='news_title')

        if date_tag and link_tag and title_tag:
            date_str = date_tag.text.strip()
            article_date = datetime.strptime(date_str, '%Y.%m.%d')

            if article_date >= stop_date:
                title = title_tag.text.strip()
                href = link_tag['href']
                full_url = ""
                if href.startswith('//'):
                    full_url = "http:" + href
                elif href.startswith('/'):
                    full_url = base_url + href

                if full_url:
                    all_news_data.append({
                        'url': full_url, 'title': title, 'date': date_str,
                    })

    print(f"\n总共收集到 {len(all_news_data)} 条发布于2025年1月1日之后的新闻。")
    return all_news_data


def fetch_and_save_details(news_data, session):
    """访问详情页，获取内容和图片，并保存。"""
    # (此函数无需修改)
    base_url = "http://www.casarte.com"
    output_dir = "casarte_news_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有文件将被保存在 '{output_dir}' 文件夹中。")

    for i, news in enumerate(news_data):
        print(f"\n--- 正在处理第 {i + 1}/{len(news_data)} 条: {news['title']} ---")
        url = news['url']

        try:
            detail_response = session.get(url)
            detail_response.raise_for_status()
            detail_response.encoding = 'utf-8'

            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
            content_div = detail_soup.find('div', class_='newsdetail_content')

            text_content = ""
            image_urls = []

            if content_div:
                paragraphs = content_div.find_all('p')
                text_content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

                image_tags = content_div.find_all('img')
                for img in image_tags:
                    src = img.get('src')
                    if src:
                        if src.startswith('//'):
                            full_image_url = 'http:' + src
                        elif src.startswith('/'):
                            full_image_url = base_url + src
                        else:
                            full_image_url = src
                        image_urls.append(full_image_url)

            news['content'] = text_content
            news['images'] = image_urls

            folder_name = sanitize_filename(news['title'])
            article_path = os.path.join(output_dir, folder_name)
            os.makedirs(article_path, exist_ok=True)

            json_path = os.path.join(article_path, 'data.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(news, f, ensure_ascii=False, indent=4)
            print(f"  - JSON文件已保存")

            if news['images']:
                print(f"  - 发现 {len(news['images'])} 张图片，开始下载...")
                for j, img_url in enumerate(news['images']):
                    try:
                        image_headers = {'Referer': url}
                        img_response = session.get(img_url, headers=image_headers, timeout=15)
                        img_response.raise_for_status()
                        img_name = img_url.split('/')[-1].split('?')[0]
                        if '.' not in img_name: img_name += '.jpg'
                        img_path = os.path.join(article_path, f"{j + 1:02d}_{img_name}")
                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                    except requests.RequestException as e:
                        print(f"    - 下载图片失败: {img_url}, 错误: {e}")
            else:
                print("  - 该资讯未发现图片。")

        except requests.RequestException as e:
            print(f"  - 爬取详情页失败: {e}")


if __name__ == '__main__':
    filtered_news_list = get_filtered_news_list_with_selenium()

    if filtered_news_list:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        fetch_and_save_details(filtered_news_list, session)
        print("\n所有任务完成！")
    else:
        print("未能爬取到任何符合条件的数据。")