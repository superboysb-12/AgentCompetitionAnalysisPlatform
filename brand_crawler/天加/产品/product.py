import os
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def sanitize_filename(filename):
    """
    移除目录或文件名中不允许的特殊字符。
    """
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()


def scrape_and_save_product_data(product_name, product_url, image_src, base_dir, base_url):
    """
    根据传入的产品名称、详情页URL和图片URL，爬取、整合并保存所有数据。
    (已更新为多策略选择器，以兼容"干冷器"等特殊页面结构)
    """
    try:
        # --- 1. 创建目录 ---
        sanitized_name = sanitize_filename(product_name)
        product_dir = os.path.join(base_dir, sanitized_name)
        os.makedirs(product_dir, exist_ok=True)

        # --- 2. 爬取详情页 ---
        print("  [*] 正在爬取详情页文本...")
        detail_response = requests.get(product_url, timeout=15)
        detail_response.raise_for_status()
        soup = BeautifulSoup(detail_response.content, 'html.parser')

        main_img_full_url = urljoin(base_url, "/public/pc/" + image_src) if image_src else "未提供图片URL"
        product_data = {
            "product_name": product_name,
            "main_image_url": main_img_full_url
        }

        # 简介
        intro_element = soup.select_one('div.detail-play-descBoxs.w104 > div.play-descBoxs > p')
        product_data['introduction'] = intro_element.text.strip() if intro_element else "未找到产品简介"

        # 特点
        print("  [*] 正在爬取产品特点...")
        product_data['features'] = []

        feature_containers = soup.select('div.play-parameter-box .parame-txt')

        if not feature_containers:
            feature_containers = soup.select('div.advantage > div.play-parameter-box > div > div > div.parame-txt')

        if not feature_containers:
            feature_containers = soup.select('div.advantage > div.play-parameter-box > div > div')

        for item in feature_containers:
            title_element = item.select_one('h4')

            desc_element = item.select_one('p')
            if not desc_element:
                desc_element = item.select_one('div:not(.parame-txt)')
                if not desc_element:
                    desc_divs = item.select('div')
                    for div in desc_divs:
                        if div.text.strip() and div != item and not div.select('h4'):
                            desc_element = div
                            break

            if title_element and desc_element:
                feature = {
                    "title": title_element.text.strip(),
                    "description": ' '.join(desc_element.text.strip().split())
                }
                product_data['features'].append(feature)

        if not product_data['features']:
            print("  [!] 未找到任何产品特点信息。")

        print("  [*] 正在爬取核心技术...")
        product_data['core_technologies'] = []

        technology_items = soup.select('div.technologyBoxs div.swiper-slide')

        if not technology_items:
            technology_items = soup.select('div.advantage2 div.technologyBoxs div.swiper-slide')

        if not technology_items:
            print("  [!] 未在本页找到核心技术信息。")
        else:
            for item in technology_items:
                tech_name_element = None
                tech_desc_elements = []

                standard_container = item.select_one('div.tech-desc-list')
                if standard_container:
                    tech_name_element = standard_container.select_one('h6')
                    tech_desc_elements = standard_container.select('p')
                else:
                    tech_name_element = item.select_one('h6')
                    tech_desc_elements = item.select('p')

                    if not tech_desc_elements:
                        text_divs = item.select('div:not(.tech-img-box)')
                        for div in text_divs:
                            p_tags = div.select('p')
                            if p_tags:
                                tech_desc_elements.extend(p_tags)

                tech_img_element = item.select_one('div.tech-img-box > img')
                if not tech_img_element:
                    tech_img_element = item.select_one('img')

                if tech_name_element and tech_desc_elements:
                    full_description = "\n".join([p.text.strip() for p in tech_desc_elements])
                    tech_img_src = tech_img_element.get('src') if tech_img_element else None
                    tech_data = {
                        "name": tech_name_element.text.strip(),
                        "description": full_description,
                        "image_source_url": tech_img_src
                    }
                    product_data['core_technologies'].append(tech_data)

        if not product_data['core_technologies']:
            print("  [!] 未找到任何核心技术信息。")

        # --- 3. 保存JSON文件 ---
        json_path = os.path.join(product_dir, 'details.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(product_data, f, ensure_ascii=False, indent=4)

        # --- 4. 下载主图片 ---
        if not image_src:
            print("  [!] 未提供产品主图片链接，跳过下载。")
        else:
            print("  [*] 正在下载列表页主图片...")
            try:
                img_response = requests.get(main_img_full_url, stream=True, timeout=15)
                img_response.raise_for_status()
                img_filename = os.path.basename(main_img_full_url.split("?")[0])
                if not os.path.splitext(img_filename)[1]:
                    img_filename += ".jpg"
                img_path = os.path.join(product_dir, img_filename)
                with open(img_path, 'wb') as f:
                    for chunk in img_response.iter_content(1024):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"  [!] 下载主图片失败 {main_img_full_url}: {e}")

        # --- 5. 下载核心技术原理图 ---
        if product_data['core_technologies']:
            print("  [*] 正在下载核心技术原理图...")
            tech_img_dir = os.path.join(product_dir, "technology_images")
            os.makedirs(tech_img_dir, exist_ok=True)
            for tech in product_data['core_technologies']:
                tech_img_src = tech.get('image_source_url')
                if not tech_img_src:
                    continue
                try:
                    tech_img_url = urljoin(base_url, "/public/pc/" + tech_img_src)
                    tech_img_response = requests.get(tech_img_url, stream=True, timeout=15)
                    tech_img_response.raise_for_status()
                    tech_img_filename = os.path.basename(tech_img_url.split("?")[0])
                    tech_img_path = os.path.join(tech_img_dir, tech_img_filename)
                    with open(tech_img_path, 'wb') as f:
                        for chunk in tech_img_response.iter_content(1024):
                            f.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"    [!] 下载技术图片失败 {tech_img_url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"  [!] 爬取详情页失败 {product_url}: {e}")
    except Exception as e:
        print(f"  [!] 处理产品时发生未知错误 {product_url}: {e}")


def main():
    """
    主函数，用于执行爬虫。
    """
    base_url = "https://www.tica.com"
    list_url = "https://www.tica.com/index/goods/product.html?cid=11"
    # "https://www.tica.com/index/goods/product.html?cid=11&cid2=1"
    # list_url = "https://www.tica.com/index/goods/product.html?cid=13"
    # "https://www.tica.com/index/goods/product.html?cid=13&cid2=195"
    # "https://www.tica.com/index/goods/product.html?cid=13&cid2=10"
    # "https://www.tica.com/index/goods/product.html?cid=13&cid2=196"
    # list_url = "https://www.tica.com/index/goods/product.html?cid=15"
    # list_url = "https://www.tica.com/index/goods/product.html?cid=14"
    # "https://www.tica.com/index/goods/product.html?cid=14&cid2=224"
    # "https://www.tica.com/index/goods/product.html?cid=14&cid2=225"
    # list_url = "https://www.tica.com/index/goods/product.html?cid=12"
    # list_url = "https://www.tica.com/index/goods/product.html?cid=16"
    # list_url = "https://www.tica.com/index/goods/product.html?cid=17"
    # "https://www.tica.com/index/goods/product.html?cid=17&cid2=15"
    # list_url = "https://www.tica.com/index/goods/product.html?cid=61"
    output_dir = "TICA_products"

    os.makedirs(output_dir, exist_ok=True)
    print(f"爬取结果将保存在 '{output_dir}' 文件夹中。")

    try:
        response = requests.get(list_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        product_items = soup.select('div.hprmlist.clearfix > ul > a')

        if not product_items:
            print("在列表页上未找到任何产品项目")
            return

        print(f"发现 {len(product_items)} 个产品，开始处理...")
        for item in product_items:
            name_element = item.select_one('div.hprmlistdesc > h4')
            img_element = item.select_one('div.hprmlistimg > img')
            href = item.get('href')

            if name_element and href and img_element:
                product_name = name_element.text.strip()
                image_src = img_element.get('src')
                full_product_url = urljoin(base_url, href)

                print(f"\n{'=' * 50}")
                print(f"正在处理产品: {product_name}")

                scrape_and_save_product_data(product_name, full_product_url, image_src, output_dir, base_url)
            else:
                print(f"\n{'=' * 50}")
                print("[!] 警告：找到一个产品项目，但其名称、链接或图片信息不完整，已跳过。")

        print(f"\n{'=' * 50}")
        print("所有产品爬取完成！")

    except requests.exceptions.RequestException as e:
        print(f"爬取产品列表页失败 {list_url}: {e}")


if __name__ == '__main__':
    main()