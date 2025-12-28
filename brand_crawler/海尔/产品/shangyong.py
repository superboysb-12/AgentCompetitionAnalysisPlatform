import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json
import re


def sanitize_filename(filename):
    """
    清理字符串，使其成为一个有效的文件/文件夹名称。
    """
    filename = str(filename).strip()
    filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
    filename = re.sub(r'[\x00-\x1f]', '', filename)
    return filename[:200]


def scrape_product_list(list_url, session, headers, base_url):
    """
    爬取产品列表页，获取所有产品的名称和详情页链接。
    """
    print(f"正在爬取列表页: {list_url}")
    try:
        response = session.get(list_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"无法访问列表页: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    products = []

    product_container = soup.select_one('div.kt_list_cont.sd_module')
    if not product_container:
        print("未找到产品列表容器 'div.kt_list_cont.sd_module'")
        return []

    product_items = product_container.find_all('div', recursive=False)

    if not product_items:
        product_items = soup.select('div.kt_list_cont.sd_module > div')

    for item in product_items:
        name_tag = item.select_one('div.textbox h2')
        link_tag = item.select_one('div.imgbox a')

        if name_tag and link_tag and link_tag.has_attr('href'):
            name = name_tag.get_text(strip=True)
            relative_url = link_tag['href']
            absolute_url = urljoin(base_url, relative_url)

            products.append({
                "name": name,
                "url": absolute_url
            })

    if not products:
        print("警告：在列表页未提取到任何产品链接。选择器可能已失效。")

    return products


def scrape_product_detail(product_url, session, headers, base_url):
    """
    爬取产品详情页，获取产品图、适用场景和所有介绍内容（文字和图片）。
    """
    print(f"  正在爬取详情页: {product_url}")
    try:
        response = session.get(product_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  无法访问详情页 {product_url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    product_data = {
        "product_images": [],
        "scenarios": [],
        "introduction_content": []
    }


    # 1. 提取产品图片 - 尝试多种选择器
    selectors_to_try = [
        ('完整路径', 'div.busi_pro_topInfo_box ul li div img'),
        ('midBox路径', 'div.midBox ul li img'),
        ('简化路径1', 'div.busi_pro_topInfo ul li img'),
        ('简化路径2', '.busi_pro_topInfo_box img'),
        ('ul下所有li中的img', 'ul li img'),
        ('所有topInfo区域img', 'div.busi_pro_topInfo img'),
    ]

    image_tags = []
    for selector_name, selector in selectors_to_try:
        temp_tags = soup.select(selector)
        print(f"  - [{selector_name}] 找到 {len(temp_tags)} 个图片")
        if temp_tags and len(temp_tags) > len(image_tags):
            image_tags = temp_tags

    for img in image_tags:
        img_url_val = None
        for attr in ['df', 'data-src', 'data-original', 'data-lazy', 'src', 'srcset']:
            if img.has_attr(attr):
                img_url_val = img[attr]
                if attr == 'srcset':
                    img_url_val = img_url_val.split()[0].split(',')[0]
                break

        if img_url_val and not img_url_val.startswith('data:image') and 'haier2019_station_bitmap' not in img_url_val:
            img_url = urljoin(base_url, img_url_val)
            product_data["product_images"].append(img_url)

    # 2. 提取适用场景
    scenario_tags = soup.select('div.busi_pro_scene a span')
    for tag in scenario_tags:
        scenario_text = tag.get_text(strip=True)
        if scenario_text:
            product_data["scenarios"].append(scenario_text)

    # 3. 提取产品介绍
    intro_container_selector = 'div.busi_pro_introduce > div > div'
    intro_container = soup.select_one(intro_container_selector)

    if intro_container:
        intro_blocks = intro_container.find_all('div', recursive=False)

        for block in intro_blocks:
            block_content = {
                "text_content": [],
                "image_urls": []
            }

            all_text_nodes = block.find_all(string=True, recursive=True)
            cleaned_text_list = []
            for text in all_text_nodes:
                if text.parent.name in ['script', 'style']:
                    continue
                stripped_text = text.strip()
                if stripped_text:
                    cleaned_text_list.append(stripped_text)

            if cleaned_text_list:
                block_content["text_content"] = list(dict.fromkeys(cleaned_text_list))

            image_tags_intro = block.select('img')
            for img in image_tags_intro:
                img_url_val = None
                for attr in ['src', 'data-src', 'data-original', 'data-lazy']:
                    if img.has_attr(attr):
                        img_url_val = img[attr]
                        break

                if img_url_val and not img_url_val.startswith('data:image'):
                    img_url = urljoin(base_url, img_url_val)
                    block_content["image_urls"].append(img_url)

            if block_content["text_content"] or block_content["image_urls"]:
                product_data["introduction_content"].append(block_content)
    else:
        print(f"  - 未找到产品介绍容器: {intro_container_selector}")

    product_data["product_images"] = list(dict.fromkeys(product_data["product_images"]))

    return product_data


def download_image(img_url, save_folder, session, headers):
    """
    下载单个图片到指定文件夹。
    """
    try:
        filename = img_url.split('/')[-1].split('?')[0]
        if not filename:
            filename = f"image_{hash(img_url)}.jpg"

        filename = sanitize_filename(filename)
        save_path = os.path.join(save_folder, filename)

        if os.path.exists(save_path):
            print(f"    - [跳过] {filename} (已存在)")
            return

        response = session.get(img_url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"    - [失败] 无法下载 {img_url}: {e}")
    except IOError as e:
        print(f"    - [失败] 无法保存 {filename}: {e}")
    except Exception as e:
        print(f"    - [失败] 下载时发生未知错误 {img_url}: {e}")


def save_product_data(product_data, product_dir, session, headers):
    """
    将所有数据（JSON和图片）保存到指定的产品文件夹中。
    """
    # 1. 创建图片子文件夹
    product_img_dir = os.path.join(product_dir, "product")
    intro_img_dir = os.path.join(product_dir, "posters")
    os.makedirs(product_img_dir, exist_ok=True)
    os.makedirs(intro_img_dir, exist_ok=True)

    # 2. 保存 JSON 文件
    json_file_path = os.path.join(product_dir, "product_data.json")
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(product_data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"  [失败] 无法写入JSON文件 {json_file_path}: {e}")

    # 3. 下载 'product' 图片
    for img_url in product_data['images']:
        download_image(img_url, product_img_dir, session, headers)

    # 4. 下载 'introduction_content' 中的所有图片
    all_intro_img_urls = []
    for block in product_data['introduction_content']:
        all_intro_img_urls.extend(block['image_urls'])

    unique_intro_img_urls = list(dict.fromkeys(all_intro_img_urls))
    total_intro_images = len(unique_intro_img_urls)

    for img_url in unique_intro_img_urls:
        download_image(img_url, intro_img_dir, session, headers)


def main():
    """
    主执行函数。
    """

    base_url = "https://www.haier.com"
    # 智慧暖通
    # 水机系列
    start_url = "https://www.haier.com/business/central-air-conditioning/product/zhnt-sjxl/?spm=cn.central-air-conditioning_pc.haiercn2021_bac_home_part01_20240731.1"
    # 物联多联机系列
    # "https://www.haier.com/business/central-air-conditioning/product/zhnt-dljxl/"
    # 特种空调系列
    # "https://www.haier.com/business/central-air-conditioning/product/zhnt-tzktxl/"

    # 能源管理
    # 热泵采暖
    # "https://www.haier.com/business/central-air-conditioning/product/nygl-rbcn/"

    # 商用壁挂柜机系列
    # "https://www.haier.com/business/central-air-conditioning/product/sybggj/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    output_base_dir = "haier_products_sy/水机系列"
    os.makedirs(output_base_dir, exist_ok=True)

    session = requests.Session()

    products_to_scrape = scrape_product_list(start_url, session, headers, base_url)

    if not products_to_scrape:
        print("列表页没有爬取到任何产品，程序退出。")
        return

    print(f"\n成功获取到 {len(products_to_scrape)} 个产品链接，准备爬取详情...")

    total_saved = 0

    for product in products_to_scrape:
        product_name = product["name"]
        product_url = product["url"]

        print(f"\n--- 开始处理: {product_name} ---")

        detail_data = scrape_product_detail(product_url, session, headers, base_url)

        if detail_data:
            full_product_data = {
                "product_name": product_name,
                "product_url": product_url,
                "scenarios": detail_data["scenarios"],
                "images": detail_data["product_images"],
                "introduction_content": detail_data["introduction_content"]
            }

            safe_dir_name = sanitize_filename(product_name)
            product_dir = os.path.join(output_base_dir, safe_dir_name)
            os.makedirs(product_dir, exist_ok=True)

            save_product_data(full_product_data, product_dir, session, headers)

            total_saved += 1

    print("\n--- 爬取完成 ---")
    print(f"总共成功处理了 {total_saved} 个产品的数据。")


if __name__ == "__main__":
    main()