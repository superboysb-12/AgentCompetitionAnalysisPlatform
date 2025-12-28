import requests
from bs4 import BeautifulSoup
import os
import json
import re


def sanitize_filename(name):
    """
    清理字符串，使其适合作为文件名或文件夹名。
    移除特殊字符，并将空格替换为下划线。
    """
    s = re.sub(r'[^\w\s\u4e00-\u9fa5-]', '', name)
    s = re.sub(r'\s+', '_', s)
    s = s.strip('_')
    return s


def get_product_links(list_page_url):
    """
    从产品列表页爬取所有产品链接。
    """
    print(f"正在访问列表页: {list_page_url}")
    try:
        response = requests.get(list_page_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"  访问列表页失败: {e}")
        return []

    product_links = []
    product_items = soup.select('div.pro_list02 > div > ul > li.li_')

    if not product_items:
        print("  未在列表页找到任何产品项。请检查选择器 'div.pro_list02 > div > ul > li.li_' 是否正确。")

    for item in product_items:
        link_tag = item.select_one('div.cover > div.cover_bott.fix > a.learn_more')
        if link_tag and 'href' in link_tag.attrs:
            full_link = requests.compat.urljoin(list_page_url, link_tag['href'])
            product_links.append(full_link)

    return product_links


def get_product_details(detail_page_url):
    """
    从产品详情页爬取产品图片、名称、参数、适用范围和产品特点。
    """
    print(f"  正在访问详情页: {detail_page_url}")
    try:
        response = requests.get(detail_page_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"  访问详情页失败: {e}")
        return {'name': '未知产品', 'error': str(e)}

    product_details = {}

    # 产品名称
    name_tag = soup.select_one(
        'body > section > div.pro_detail02 > div > div.left > div.intro > div > div.tit > span.leibie')
    product_details['name'] = name_tag.get_text(strip=True) if name_tag else "未知产品"

    # 产品图片
    img_tag = soup.select_one('body > section > div.pro_detail02 > div > div.left > div.img_ > div > img')
    if img_tag and 'src' in img_tag.attrs:
        product_details['image_url'] = requests.compat.urljoin(detail_page_url, img_tag['src'])
    else:
        product_details['image_url'] = None

    # 产品参数
    params_elements = soup.select(
        'body > section > div.pro_detail02 > div > div.left > div.intro > div > div.con > ul > li')
    product_details['parameters'] = []
    for li in params_elements:
        param_items = []
        for p_tag in li.find_all('p'):
            text = p_tag.get_text(strip=True)
            if text:
                param_items.append(text)
        if param_items:
            product_details['parameters'].append(" ".join(param_items))
        else:
            text = li.get_text(strip=True)
            if text:
                product_details['parameters'].append(text)

    # 适用范围
    scope_tag = soup.select_one('body > section > div.pro_detail02 > div > div.left > div.intro > div > div.txt > p')
    product_details['application_scope'] = scope_tag.get_text(strip=True) if scope_tag else None

    # 产品特点
    product_details['features'] = []
    feature_list_items = soup.select('div.content li')  # 假设特点在 'div.content' 下的 'li' 中

    if not feature_list_items:
        print("  警告: 未能找到任何产品特点内容。请检查选择器 'div.content li' 是否正确或页面结构是否变化。")

    for item in feature_list_items:
        feature_title_tag = item.select_one('div.t')
        feature_content_tags = item.select('div.c > p')

        feature_title = feature_title_tag.get_text(strip=True) if feature_title_tag else ''
        feature_content = [p.get_text(strip=True) for p in feature_content_tags if p.get_text(strip=True)]

        if feature_title or feature_content:
            product_details['features'].append({
                'title': feature_title,
                'content': "\n".join(feature_content)
            })

    return product_details


if __name__ == "__main__":
    # 水冷冷（热）水系列
    base_list_url = "http://www.dunham-bush.cn/product/watl.jsp"
    # 风冷（冷）热水系列
    # "http://www.dunham-bush.cn/product/fengl.jsp"
    # 空气侧产品
    # "http://www.dunham-bush.cn/product/airpro.jsp"
    # 空调辅助设备
    # "http://www.dunham-bush.cn/product/airfz.jsp"
    output_base_dir = "product_data"

    os.makedirs(output_base_dir, exist_ok=True)

    all_product_links = []
    page_num = 1

    print("开始爬取产品列表页（自动分页）...")

    while True:
        if page_num == 1:
            current_url = base_list_url
        else:
            current_url = f"{base_list_url}?nowPage={page_num}"

        links_on_page = get_product_links(current_url)

        if not links_on_page:
            if page_num == 1:
                print("列表页第一页爬取失败或为空，程序终止。")
            else:
                print(f"  在第 {page_num} 页未找到产品或页面无法访问，分页结束。")
            break

        new_links_found = []
        for link in links_on_page:
            if link not in all_product_links:  # 检查这个链接是否是全新的
                new_links_found.append(link)

        if not new_links_found:
            print(f"  在第 {page_num} 页未发现新产品（检测到内容重复），分页结束。")
            break

        print(f"  在第 {page_num} 页找到 {len(new_links_found)} 个新产品。")
        all_product_links.extend(new_links_found)  # 只添加新发现的链接
        page_num += 1

    if not all_product_links:
        print("没有找到产品链接，程序退出。")
    else:
        print(f"\n共找到 {len(all_product_links)} 个唯一产品，开始爬取产品详情并保存数据...")

        for i, link in enumerate(all_product_links):
            print(f"\n--- 正在处理产品 {i + 1}/{len(all_product_links)} ---")
            details = get_product_details(link)

            product_name = details.get('name', '未知产品')
            sanitized_product_name = sanitize_filename(product_name)

            product_output_dir = os.path.join(output_base_dir, sanitized_product_name)
            os.makedirs(product_output_dir, exist_ok=True)
            print(f"  产品目录: {product_output_dir}")

            json_filename = os.path.join(product_output_dir, f"{sanitized_product_name}.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(details, f, ensure_ascii=False, indent=4)

            image_url = details.get('image_url')
            if image_url:
                try:
                    img_response = requests.get(image_url, stream=True, timeout=30)
                    img_base_name = os.path.basename(image_url).split('?')[0]

                    _, ext = os.path.splitext(img_base_name)
                    if not ext:
                        ext = '.jpg'

                    image_filename_safe = f"{sanitized_product_name}{ext}"
                    image_filename = os.path.join(product_output_dir, image_filename_safe)

                    with open(image_filename, 'wb') as img_f:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            img_f.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"  下载图片失败 {image_url}: {e}")
            else:
                print("  没有找到产品图片链接。")

        print("\n所有产品数据爬取并保存完成！")