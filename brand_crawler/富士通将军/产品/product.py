import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import time
import json


def scrape_fujitsu_products():
    """
    爬取富士通将军官网的产品信息和图片，并为每个产品生成一个包含URL元数据的JSON文件。
    增加了对下载失败的自动重试机制。
    """
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_BACKOFF_FACTOR = 2  # 重试等待时间的基数（秒）

    # Base URL for the entire site
    base_url = "https://www.fujitsu-general.com.cn/"
    # List page URL
    # 中央空调
    list_page_url = "https://www.fujitsu-general.com.cn/pros/list.php?catid=26"
    # 分体空调
    # “https://www.fujitsu-general.com.cn/pros/list.php?catid=27”
    # 新风系统
    # “https://www.fujitsu-general.com.cn/pros/list.php?catid=28”
    main_folder = "fujitsu_products"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print(f"正在访问产品列表页: {list_page_url}")
        response = requests.get(list_page_url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')
        product_links = soup.select('.produt-list a')

        if not product_links:
            print("在列表页未找到任何产品链接，请检查选择器是否正确。")
            return

        print(f"共找到 {len(product_links)} 个产品。")
        print("-" * 30)

        for link in product_links:
            product_name_element = link.select_one('.text')
            if not product_name_element:
                continue

            product_name = product_name_element.text.strip()
            product_relative_url = link.get('href')
            if not product_relative_url:
                continue

            product_url = urljoin(list_page_url, product_relative_url)
            safe_folder_name = re.sub(r'[\\/*?:"<>|]', "", product_name)
            product_folder = os.path.join(main_folder, safe_folder_name)

            if not os.path.exists(product_folder):
                os.makedirs(product_folder)

            print(f"正在处理产品: {product_name}")

            time.sleep(1)

            try:
                detail_response = requests.get(product_url, headers=headers, timeout=10)
                detail_response.raise_for_status()
                detail_response.encoding = 'utf-8'
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                image_tags = detail_soup.select('.product-info img')

                if not image_tags:
                    print(f" -> 在 '{product_name}' 的详情页未找到任何图片。")
                    print("-" * 30)
                    continue

                product_data = {
                    "product_url": product_url,
                    "images": []
                }

                image_counter = 0

                for img_tag in image_tags:
                    img_src = img_tag.get('data-src') or img_tag.get('original') or img_tag.get('src')

                    if not img_src or 'lazy' in img_src.lower():
                        continue

                    img_url = urljoin(base_url, img_src)
                    parsed_url = urlparse(img_url)
                    img_name_with_ext = os.path.basename(parsed_url.path)

                    if not img_name_with_ext:
                        continue

                    img_base_name, img_ext = os.path.splitext(img_name_with_ext)
                    image_counter += 1
                    clean_base_name = os.path.basename(img_base_name)
                    new_img_name = f"{clean_base_name}_{image_counter}{img_ext}"
                    img_path = os.path.join(product_folder, new_img_name)

                    image_info = {"url": img_url, "path": new_img_name}
                    product_data["images"].append(image_info)

                    download_success = False
                    for attempt in range(MAX_RETRIES):
                        try:
                            img_response = requests.get(img_url, stream=True, headers=headers, timeout=15)
                            img_response.raise_for_status()

                            with open(img_path, 'wb') as f:
                                for chunk in img_response.iter_content(1024):
                                    f.write(chunk)
                            download_success = True
                            break  # 下载成功，跳出重试循环

                        except requests.exceptions.RequestException as e:
                            print(f" -> 下载失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {new_img_name}")
                            if attempt < MAX_RETRIES - 1:
                                wait_time = RETRY_BACKOFF_FACTOR * (attempt + 1)
                                print(f"    将在 {wait_time} 秒后重试...")
                                time.sleep(wait_time)
                            else:
                                print(f"    所有重试均失败。错误: {e}")

                    # 每次图片下载尝试之间也加入短暂延时
                    if download_success:
                        time.sleep(0.5)

                json_path = os.path.join(product_folder, 'product_info.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(product_data, f, ensure_ascii=False, indent=4)

            except requests.exceptions.RequestException as e:
                print(f"访问产品详情页失败: {product_url}, 错误: {e}")

            print("-" * 30)

    except requests.exceptions.RequestException as e:
        print(f"访问产品列表页失败: {list_page_url}, 错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    scrape_fujitsu_products()