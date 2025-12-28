import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from urllib.parse import urljoin, urlparse

BASE_URL = "https://www.lg.com"
MAIN_URL = "https://www.lg.com/cn/business/air-conditioning"

all_products_summary = []


def get_html_content(url):
    """
    获取指定URL的HTML内容
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def download_image(image_url, save_path):
    """
    下载图片到指定路径
    """
    if not image_url:
        return None

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()

        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return None


def parse_main_page(html_content):
    """
    解析主页，提取产品分类和产品链接
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    products_info = []

    main_categories_ul = soup.select_one(
        '#gnbNV00046153 > div.sublayer-inner > div.columns > div.column2 > div:nth-child(1) > ul'
    )

    if main_categories_ul:
        for main_li in main_categories_ul.find_all('li', recursive=False):
            category_name_div = main_li.find('div', class_='sub-link')
            category_name = category_name_div.get_text(strip=True) if category_name_div else "未知分类"

            product_list_ul = main_li.find('ul', class_='link-list')
            if product_list_ul:
                for product_li in product_list_ul.find_all('li'):
                    product_link_tag = product_li.find('a')
                    if product_link_tag and 'href' in product_link_tag.attrs:
                        product_name = product_link_tag.get_text(strip=True)
                        product_url = product_link_tag['href']

                        if not product_url.startswith('http'):
                            product_url = BASE_URL + product_url

                        products_info.append({
                            'category': category_name,
                            'name': product_name,
                            'url': product_url
                        })
    else:
        print("未找到主分类列表，请检查选择器。")

    return products_info


def parse_product_detail(html_content, product_url, product_name_from_main, product_dir):
    """
    解析产品详情页，提取产品信息，包括首图、产品文字、海报图和海报文字。
    现在会迭代查找所有 class="iw_component" 的 div，并从中提取信息。
    会跳过第一个和最后三个 iw_component 元素，并下载海报图片。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    detail_data = {
        'product_url': product_url,
        'product_name': product_name_from_main,
        'main_image_url': None,
        'product_main_text': None,
        'model_name': None,
        'features': [],
        'specifications': {},
        'description': None,
        'poster_sections': []
    }

    # 1. 提取产品首图URL
    main_image_tag = soup.select_one('#content > div > div > div > div.visual-area > img.mobile.lazyloaded')
    if main_image_tag and 'src' in main_image_tag.attrs:
        img_url = BASE_URL + main_image_tag['src']
        detail_data['main_image_url'] = img_url
        main_img_folder = os.path.join(product_dir, 'main_image')
        os.makedirs(main_img_folder, exist_ok=True)
        img_filename = os.path.basename(urlparse(img_url).path)
        main_img_path = os.path.join(main_img_folder, img_filename)
        downloaded_path = download_image(img_url, main_img_path)
        if downloaded_path:
            detail_data['main_image_local_path'] = os.path.relpath(downloaded_path, product_dir) # 记录相对路径
    elif main_image_tag and 'data-src' in main_image_tag.attrs:
        img_url = BASE_URL + main_image_tag['data-src']
        detail_data['main_image_url'] = img_url
        main_img_folder = os.path.join(product_dir, 'main_image')
        os.makedirs(main_img_folder, exist_ok=True)
        img_filename = os.path.basename(urlparse(img_url).path)
        main_img_path = os.path.join(main_img_folder, img_filename)
        downloaded_path = download_image(img_url, main_img_path)
        if downloaded_path:
            detail_data['main_image_local_path'] = os.path.relpath(downloaded_path, product_dir)

    # 2. 提取产品文字 (主标题)
    product_main_text_tag = soup.select_one('#waGPC0055_0 > h1')
    if product_main_text_tag:
        detail_data['product_main_text'] = product_main_text_tag.get_text(strip=True)
        detail_data['product_name'] = product_main_text_tag.get_text(strip=True)

    # 3. 提取型号名称
    model_tag = soup.select_one('p.model-info span.model-number')
    if model_tag:
        detail_data['model_name'] = model_tag.get_text(strip=True)
    else:
        model_tag_alt = soup.select_one('div.product-model-details dl dt:contains("型号") + dd')
        if model_tag_alt:
            detail_data['model_name'] = model_tag_alt.get_text(strip=True)

    # 4. 提取产品特点
    features_list = soup.select('div.features-section ul.feature-list li')
    for feature_li in features_list:
        detail_data['features'].append(feature_li.get_text(strip=True))

    # 5. 提取产品描述
    description_tag = soup.select_one('div.product-description p')
    if description_tag:
        detail_data['description'] = description_tag.get_text(strip=True)

    # 6. 提取产品规格
    spec_table = soup.select_one('table.specifications-table')
    if spec_table:
        for row in spec_table.find_all('tr'):
            cols = row.find_all(['th', 'td'])
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                detail_data['specifications'][key] = value
    else:
        spec_dl = soup.select_one('dl.product-specs')
        if spec_dl:
            dt_tags = spec_dl.find_all('dt')
            dd_tags = spec_dl.find_all('dd')
            for dt, dd in zip(dt_tags, dd_tags):
                key = dt.get_text(strip=True)
                value = dd.get_text(strip=True)
                detail_data['specifications'][key] = value


    all_iw_components = soup.find_all('div', class_='iw_component')

    # 计算需要跳过的索引
    num_components = len(all_iw_components)
    components_to_process = []

    for idx, container in enumerate(all_iw_components):
        # 跳过第一个和最后三个
        if idx == 0 or idx >= num_components - 3:
            print(f"  跳过 iw_component 索引 {idx} (ID: {container.get('id', 'N/A')})")
            continue
        components_to_process.append(container)

    # 为海报图片创建子文件夹
    poster_image_folder = os.path.join(product_dir, 'poster_images')
    os.makedirs(poster_image_folder, exist_ok=True)


    for idx, container in enumerate(components_to_process):
        poster_info = {
            'component_id': container.get('id'),
            'image_url': None,
            'image_local_path': None, # 新增：用于存储本地图片路径
            'text_content': None,
            'title': None
        }

        img_tag = container.select_one('img.mobile.lazyloaded, img.pc.lazyloaded, img[src^="/"], img[data-src^="/"]')
        if img_tag:
            img_src = img_tag.get('src') or img_tag.get('data-src')
            if img_src:
                if not img_src.startswith('http'):
                    full_img_url = BASE_URL + img_src
                else:
                    full_img_url = img_src
                poster_info['image_url'] = full_img_url

                # 下载图片
                img_filename = os.path.basename(urlparse(full_img_url).path)
                img_filename = re.sub(r'\?.*$', '', img_filename) # 去除URL中的查询参数
                if not img_filename:
                    img_filename = f"poster_image_{poster_info['component_id']}_{idx}.jpg"

                local_img_path = os.path.join(poster_image_folder, img_filename)
                downloaded_path = download_image(full_img_url, local_img_path)
                if downloaded_path:
                    poster_info['image_local_path'] = os.path.relpath(downloaded_path, product_dir)


        title_tag = container.select_one('h2.title, h3.title, h2, h3')
        if title_tag:
            poster_info['title'] = title_tag.get_text(strip=True)

        text_content_tag = container.select_one('div.copy, p.description, div.text-content, div.contents-area > div > div > div > div > div.copy')
        if text_content_tag:
            poster_info['text_content'] = text_content_tag.get_text(separator='\n', strip=True)
        else:
            all_text_elements = container.find_all(string=True)
            visible_texts = []
            for t in all_text_elements:
                parent_tag = t.parent.name
                if parent_tag not in ['script', 'style', 'head', 'title', 'meta'] \
                   and t.strip() and len(t.strip()) > 10 \
                   and (not poster_info['title'] or t.strip() != poster_info['title']):
                    visible_texts.append(t.strip())
            if visible_texts:
                combined_text = ' '.join(sorted(list(set(visible_texts)), key=visible_texts.index))
                if poster_info['title'] and combined_text.startswith(poster_info['title']):
                     combined_text = combined_text[len(poster_info['title']):].strip()
                if combined_text:
                    poster_info['text_content'] = combined_text

        if poster_info['image_url'] or poster_info['text_content'] or poster_info['title']:
            detail_data['poster_sections'].append(poster_info)

    return detail_data


def sanitize_filename(filename):
    """
    清理文件名，移除或替换非法字符
    """
    filename = re.sub(r'[\\/:*?"<>|]', '', filename)
    filename = filename.replace(' ', '_')
    filename = filename.strip()
    if not filename:
        filename = "untitled_product"
    return filename


def main():
    print(f"开始爬取主页: {MAIN_URL}")
    main_page_html = get_html_content(MAIN_URL)

    if main_page_html:
        product_links = parse_main_page(main_page_html)
        print(f"主页中共找到 {len(product_links)} 个产品链接。")

        output_base_dir = "lg_products"
        os.makedirs(output_base_dir, exist_ok=True)

        for i, product in enumerate(product_links):
            print(f"\n[{i + 1}/{len(product_links)}] 正在爬取产品详情页: {product['name']} - {product['url']}")
            detail_html = get_html_content(product['url'])
            if detail_html:
                folder_name = sanitize_filename(product['name'])
                product_dir = os.path.join(output_base_dir, folder_name)
                os.makedirs(product_dir, exist_ok=True)

                detail_data = parse_product_detail(detail_html, product['url'], product['name'], product_dir)

                if detail_data.get('product_name') and detail_data['product_name'] != product['name']:
                    old_product_dir = product_dir
                    folder_name = sanitize_filename(detail_data['product_name'])
                    product_dir = os.path.join(output_base_dir, folder_name)
                    if old_product_dir != product_dir and os.path.exists(old_product_dir):
                        try:
                            os.rename(old_product_dir, product_dir)
                            if 'main_image_local_path' in detail_data and detail_data['main_image_local_path']:
                                detail_data['main_image_local_path'] = os.path.join(os.path.relpath(old_product_dir, output_base_dir), detail_data['main_image_local_path'])
                                detail_data['main_image_local_path'] = os.path.relpath(os.path.join(output_base_dir, detail_data['main_image_local_path']), product_dir)


                            for section in detail_data['poster_sections']:
                                if 'image_local_path' in section and section['image_local_path']:
                                    section['image_local_path'] = os.path.join(os.path.relpath(old_product_dir, output_base_dir), section['image_local_path'])
                                    section['image_local_path'] = os.path.relpath(os.path.join(output_base_dir, section['image_local_path']), product_dir)

                        except OSError as e:
                            print(f"  重命名文件夹 '{old_product_dir}' 到 '{product_dir}' 失败: {e}. 将继续使用旧文件夹名。")
                            product_dir = old_product_dir # 失败则回滚

                output_filename = os.path.join(product_dir, f"{folder_name}.json")
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(detail_data, f, ensure_ascii=False, indent=4)

                all_products_summary.append({
                    'category': product['category'],
                    'product_name': detail_data.get('product_name') or product['name'],
                    'product_url': product['url'],
                    'data_file': output_filename
                })

            time.sleep(1)

    summary_filename = os.path.join(output_base_dir, 'products_summary.json')
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(all_products_summary, f, ensure_ascii=False, indent=4)

    print(f"\n爬取完成！所有产品数据已保存到 '{output_base_dir}' 目录。")
    print(f"共抓取到 {len(all_products_summary)} 个产品的详细信息。")


if __name__ == "__main__":
    main()