import requests
from bs4 import BeautifulSoup
import time
import json
import os
import re
import pandas as pd
import urllib.parse
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def clean_filename(filename):
    """
    清理字符串，使其可以作为有效的文件名或文件夹名。
    """
    cleaned_filename = re.sub(r'[\\/:*?"<>|]', '', filename)
    cleaned_filename = cleaned_filename.replace(' ', '_')
    cleaned_filename = cleaned_filename[:100]  # 限制文件名长度
    return cleaned_filename.strip()


def download_image(image_url, folder_path, filename):
    """
    下载图片到指定文件夹。
    """
    if not image_url or image_url == 'N/A' or not image_url.startswith('http'):
        return 'N/A (Invalid URL or N/A)'

    try:
        response = requests.get(image_url, stream=True, timeout=15)
        response.raise_for_status()

        image_path = os.path.join(folder_path, filename)
        with open(image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return image_path
    except requests.exceptions.RequestException as e:
        print(f"下载图片 {image_url} 失败: {e}")
        return f"Download failed: {e}"
    except Exception as e:
        print(f"处理图片下载时发生未知错误: {e}")
        return f"Unknown error during download: {e}"


def scrape_product_data(base_url, list_page_url, list_item_selector,
                        detail_name_selector, detail_intro_selector, detail_poster_selector,
                        outdoor_unit_section_selector,
                        outdoor_unit_item_base_selector,
                        outdoor_unit_name_relative_selector,
                        outdoor_unit_image_relative_selector,
                        outdoor_unit_table_relative_selector,
                        indoor_unit_section_selector,
                        indoor_unit_item_base_selector,
                        indoor_unit_name_relative_selector,
                        indoor_unit_image_relative_selector,
                        indoor_unit_table_relative_selector,
                        accessories_section_selector,
                        accessories_item_base_selector,
                        accessories_name_relative_selector,
                        accessories_image_relative_selector,
                        accessories_intro_parent_relative_selector,
                        output_base_dir="hitachi_products"):
    """
    爬取产品数据，并保存到以产品名命名的文件夹中。
    同时爬取外机、内机和选配件数据。
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    print(f"正在访问列表页: {list_page_url}")
    try:
        response = requests.get(list_page_url, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding if response.apparent_encoding else 'utf-8'
    except requests.exceptions.RequestException as e:
        print(f"访问列表页时发生错误: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    parts = list_item_selector.split('>')
    if len(parts) >= 1:
        list_parent_selector_parts = [p.strip() for p in parts[:-1] if p.strip()]
        if list_parent_selector_parts:
            list_parent_selector = ' > '.join(list_parent_selector_parts)
        else:
            list_parent_selector = list_item_selector
    else:
        list_parent_selector = list_item_selector

    list_items = []
    try:
        parent_element_for_list_items = soup.select_one(list_parent_selector)
        if parent_element_for_list_items:
            list_items = parent_element_for_list_items.find_all('li', recursive=False)
            if not list_items:
                list_items = soup.select(list_item_selector)
        else:
            print(f"未能找到列表项的父元素: {list_parent_selector}，尝试直接使用: {list_item_selector}")
            list_items = soup.select(list_item_selector)

    except Exception as e:
        print(f"查找列表项时发生错误: {e}")
        return

    if not list_items:
        print(f"未能找到任何产品列表项，请检查选择器: {list_item_selector}")
        return

    print(f"找到 {len(list_items)} 个产品列表项。")

    for i, item in enumerate(list_items):
        product_data = {}
        product_detail_url = 'N/A'

        list_image_url_raw = 'N/A'
        try:
            img_box = item.find('div', class_='imgBox')
            if img_box:
                img_tag = img_box.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    list_image_url_raw = urllib.parse.urljoin(base_url, img_tag['src'])
        except Exception as e:
            print(f"提取列表项 {i + 1} 图片URL时发生错误: {e}")
        product_data['list_image_url'] = list_image_url_raw

        link_tag = item.find('a')
        if link_tag and 'href' in link_tag.attrs:
            relative_link = link_tag['href']
            product_detail_url = urllib.parse.urljoin(base_url, relative_link)
            product_data['url'] = product_detail_url
            print(f"\n[{i + 1}/{len(list_items)}] 正在访问详情页: {product_detail_url}")

            try:
                time.sleep(1)
                detail_response = requests.get(product_detail_url, timeout=10)
                detail_response.raise_for_status()
                detail_response.encoding = detail_response.apparent_encoding if detail_response.apparent_encoding else 'utf-8'

                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

                # --- 提取详情页通用内容 ---
                name_element = detail_soup.select_one(detail_name_selector)
                product_name = name_element.get_text(strip=True) if name_element else f"未知产品_{i + 1}"
                product_data['name'] = product_name

                intro_element = detail_soup.select_one(detail_intro_selector)
                product_data['intro'] = intro_element.get_text(strip=True) if intro_element else 'N/A'

                poster_img_tag = detail_soup.select_one(detail_poster_selector)
                poster_image_url_raw = 'N/A'
                if poster_img_tag and 'src' in poster_img_tag.attrs:
                    poster_image_url_raw = urllib.parse.urljoin(base_url, poster_img_tag['src'])
                product_data['poster_image_url'] = poster_image_url_raw


                # --- 创建产品文件夹 ---
                cleaned_product_name = clean_filename(product_name)
                product_folder = os.path.join(output_base_dir, cleaned_product_name)
                if not os.path.exists(product_folder):
                    os.makedirs(product_folder)

                # 下载列表页图片
                list_image_filename = f"list_image{os.path.splitext(product_data['list_image_url'])[1] if product_data['list_image_url'] != 'N/A' and os.path.splitext(product_data['list_image_url'])[1] else '.jpg'}"
                downloaded_list_image_path = download_image(product_data['list_image_url'], product_folder,
                                                            list_image_filename)
                product_data['downloaded_list_image_path'] = downloaded_list_image_path or 'N/A'

                # 下载产品海报
                poster_image_filename = f"poster_image{os.path.splitext(product_data['poster_image_url'])[1] if product_data['poster_image_url'] != 'N/A' and os.path.splitext(product_data['poster_image_url'])[1] else '.jpg'}"
                downloaded_poster_image_path = download_image(product_data['poster_image_url'], product_folder,
                                                              poster_image_filename)
                product_data['downloaded_poster_image_path'] = downloaded_poster_image_path or 'N/A'

                # --- 提取外机参数 ---
                outdoor_unit_list = []
                outdoor_params_folder = os.path.join(product_folder, 'outdoor_unit_params')
                if not os.path.exists(outdoor_params_folder):
                    os.makedirs(outdoor_params_folder)

                outdoor_section = detail_soup.select_one(outdoor_unit_section_selector)
                if outdoor_section:
                    outdoor_unit_items = outdoor_section.select(outdoor_unit_item_base_selector)

                    if not outdoor_unit_items:
                        print(f"  未找到产品 '{product_name}' 的外机参数项。")

                    for ou_idx, ou_item in enumerate(outdoor_unit_items):
                        outdoor_unit_data = {}

                        ou_name_element = ou_item.select_one(outdoor_unit_name_relative_selector)
                        ou_name = ou_name_element.get_text(strip=True) if ou_name_element else f"未知外机_{ou_idx + 1}"
                        outdoor_unit_data['name'] = ou_name
                        print(f"  正在处理外机: {ou_name}")

                        ou_image_url_raw = 'N/A'
                        ou_image_tag = ou_item.select_one(outdoor_unit_image_relative_selector)
                        if ou_image_tag and 'src' in ou_image_tag.attrs:
                            ou_image_url_raw = urllib.parse.urljoin(base_url, ou_image_tag['src'])
                        outdoor_unit_data['image_url'] = ou_image_url_raw

                        ou_image_filename = f"{clean_filename(ou_name)}_image{os.path.splitext(ou_image_url_raw)[1] if ou_image_url_raw != 'N/A' and os.path.splitext(ou_image_url_raw)[1] else '.jpg'}"
                        downloaded_ou_image_path = download_image(ou_image_url_raw, outdoor_params_folder,
                                                                  ou_image_filename)
                        outdoor_unit_data['downloaded_image_path'] = downloaded_ou_image_path or 'N/A'

                        table_data = []
                        table_element = ou_item.select_one(outdoor_unit_table_relative_selector)
                        if table_element:
                            rows = table_element.find_all('tr')
                            if rows:
                                headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
                                if headers:
                                    for row in rows[1:]:
                                        row_cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                                        if row_cells:
                                            table_data.append(row_cells)

                                    if table_data:
                                        max_cols = max(len(row) for row in table_data) if table_data else 0
                                        if len(headers) < max_cols:
                                            headers.extend([f"Column_{j + 1}" for j in range(len(headers), max_cols)])
                                        elif len(headers) > max_cols and max_cols > 0:
                                            headers = headers[:max_cols]

                                        processed_table_data = []
                                        for row_cells in table_data:
                                            if len(row_cells) < len(headers):
                                                row_cells.extend([''] * (len(headers) - len(row_cells)))
                                            elif len(row_cells) > len(headers):
                                                row_cells = row_cells[:len(headers)]
                                            processed_table_data.append(row_cells)

                                        df_data = pd.DataFrame(processed_table_data, columns=headers)
                                        ou_csv_filename = f"{clean_filename(ou_name)}.csv"
                                        ou_csv_path = os.path.join(outdoor_params_folder, ou_csv_filename)
                                        df_data.to_csv(ou_csv_path, index=False, encoding='utf-8-sig')
                                        outdoor_unit_data['parameters_csv_path'] = ou_csv_path
                                        print(f"  外机 '{ou_name}' 参数已保存到 '{ou_csv_path}'")
                                    else:
                                        outdoor_unit_data['parameters_csv_path'] = 'N/A (No valid table data rows)'
                                else:
                                    outdoor_unit_data['parameters_csv_path'] = 'N/A (No valid table headers)'
                            else:
                                outdoor_unit_data['parameters_csv_path'] = 'N/A (No table rows found)'
                        else:
                            outdoor_unit_data['parameters_csv_path'] = 'N/A (Table not found)'

                        outdoor_unit_list.append(outdoor_unit_data)
                else:
                    print(f"  未找到产品 '{product_name}' 的外机参数主区域: {outdoor_unit_section_selector}")

                product_data['outdoor_units'] = outdoor_unit_list

                # --- 提取内机参数 ---
                indoor_unit_list = []
                indoor_params_folder = os.path.join(product_folder, 'indoor_unit_params')
                if not os.path.exists(indoor_params_folder):
                    os.makedirs(indoor_params_folder)

                indoor_section = detail_soup.select_one(indoor_unit_section_selector)
                if indoor_section:
                    indoor_unit_items = indoor_section.select(indoor_unit_item_base_selector)

                    if not indoor_unit_items:
                        print(f"  未找到产品 '{product_name}' 的内机参数项。")

                    for iu_idx, iu_item in enumerate(indoor_unit_items):
                        indoor_unit_data = {}

                        iu_name_element = iu_item.select_one(indoor_unit_name_relative_selector)
                        iu_name = iu_name_element.get_text(strip=True) if iu_name_element else f"未知内机_{iu_idx + 1}"
                        indoor_unit_data['name'] = iu_name
                        print(f"  正在处理内机: {iu_name}")

                        iu_image_url_raw = 'N/A'
                        iu_image_tag = iu_item.select_one(indoor_unit_image_relative_selector)
                        if iu_image_tag and 'src' in iu_image_tag.attrs:
                            iu_image_url_raw = urllib.parse.urljoin(base_url, iu_image_tag['src'])
                        indoor_unit_data['image_url'] = iu_image_url_raw

                        iu_image_filename = f"{clean_filename(iu_name)}_image{os.path.splitext(iu_image_url_raw)[1] if iu_image_url_raw != 'N/A' and os.path.splitext(iu_image_url_raw)[1] else '.jpg'}"
                        downloaded_iu_image_path = download_image(iu_image_url_raw, indoor_params_folder,
                                                                  iu_image_filename)
                        indoor_unit_data['downloaded_image_path'] = downloaded_iu_image_path or 'N/A'

                        table_data = []
                        table_element = iu_item.select_one(indoor_unit_table_relative_selector)
                        if table_element:
                            rows = table_element.find_all('tr')
                            if rows:
                                headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
                                if headers:
                                    for row in rows[1:]:
                                        row_cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                                        if row_cells:
                                            table_data.append(row_cells)

                                    if table_data:
                                        max_cols = max(len(row) for row in table_data) if table_data else 0
                                        if len(headers) < max_cols:
                                            headers.extend([f"Column_{j + 1}" for j in range(len(headers), max_cols)])
                                        elif len(headers) > max_cols and max_cols > 0:
                                            headers = headers[:max_cols]

                                        processed_table_data = []
                                        for row_cells in table_data:
                                            if len(row_cells) < len(headers):
                                                row_cells.extend([''] * (len(headers) - len(row_cells)))
                                            elif len(row_cells) > len(headers):
                                                row_cells = row_cells[:len(headers)]
                                            processed_table_data.append(row_cells)

                                        df_data = pd.DataFrame(processed_table_data, columns=headers)
                                        iu_csv_filename = f"{clean_filename(iu_name)}.csv"
                                        iu_csv_path = os.path.join(indoor_params_folder, iu_csv_filename)
                                        df_data.to_csv(iu_csv_path, index=False, encoding='utf-8-sig')
                                        indoor_unit_data['parameters_csv_path'] = iu_csv_path
                                        print(f"  内机 '{iu_name}' 参数已保存到 '{iu_csv_path}'")
                                    else:
                                        indoor_unit_data['parameters_csv_path'] = 'N/A (No valid table data rows)'
                                else:
                                    indoor_unit_data['parameters_csv_path'] = 'N/A (No valid table headers)'
                            else:
                                indoor_unit_data['parameters_csv_path'] = 'N/A (No table rows found)'
                        else:
                            indoor_unit_data['parameters_csv_path'] = 'N/A (Table not found)'

                        indoor_unit_list.append(indoor_unit_data)
                else:
                    print(f"  未找到产品 '{product_name}' 的内机参数主区域: {indoor_unit_section_selector}")

                product_data['indoor_units'] = indoor_unit_list

                accessories_list = []
                accessories_folder = os.path.join(product_folder, 'accessories')
                if not os.path.exists(accessories_folder):
                    os.makedirs(accessories_folder)

                accessories_section = detail_soup.select_one(accessories_section_selector)
                if accessories_section:
                    accessories_items = accessories_section.select(accessories_item_base_selector)

                    if not accessories_items:
                        print(f"  未找到产品 '{product_name}' 的选配件项。")

                    for acc_idx, acc_item in enumerate(accessories_items):
                        accessory_data = {}

                        # 选配件名称 (p)
                        acc_name_element = acc_item.select_one(accessories_name_relative_selector)
                        acc_name = acc_name_element.get_text(
                            strip=True) if acc_name_element else f"未知选配件_{acc_idx + 1}"
                        accessory_data['name'] = acc_name
                        print(f"  正在处理选配件: {acc_name}")

                        # 选配件图片 (div.imgBox > img)
                        acc_image_url_raw = 'N/A'
                        acc_image_tag = acc_item.select_one(accessories_image_relative_selector)
                        if acc_image_tag and 'src' in acc_image_tag.attrs:
                            acc_image_url_raw = urllib.parse.urljoin(base_url, acc_image_tag['src'])
                        accessory_data['image_url'] = acc_image_url_raw

                        # 下载选配件图片
                        acc_image_filename = f"{clean_filename(acc_name)}_image{os.path.splitext(acc_image_url_raw)[1] if acc_image_url_raw != 'N/A' and os.path.splitext(acc_image_url_raw)[1] else '.jpg'}"
                        downloaded_acc_image_path = download_image(acc_image_url_raw, accessories_folder,
                                                                   acc_image_filename)
                        accessory_data['downloaded_image_path'] = downloaded_acc_image_path or 'N/A'

                        # 选配件介绍提取
                        acc_intro_parent_element = acc_item.select_one(accessories_intro_parent_relative_selector)
                        acc_intro_text = 'N/A'
                        if acc_intro_parent_element:

                            p_elements = acc_intro_parent_element.find_all('p')
                            if p_elements:
                                acc_intro_text = '\n'.join(
                                    [p.get_text(strip=True) for p in p_elements if p.get_text(strip=True)])
                            elif acc_intro_parent_element.get_text(strip=True):
                                acc_intro_text = acc_intro_parent_element.get_text(strip=True)

                        accessory_data['intro'] = acc_intro_text

                        accessories_list.append(accessory_data)
                else:
                    print(f"  未找到产品 '{product_name}' 的选配件主区域: {accessories_section_selector}")

                product_data['accessories'] = accessories_list

                # --- 保存产品详情到JSON ---
                json_file_path = os.path.join(product_folder, 'product_details.json')
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(product_data, f, ensure_ascii=False, indent=4)
                print(f"产品 '{product_name}' 的所有数据已保存到 '{product_folder}'")

            except requests.exceptions.RequestException as e:
                print(f"访问详情页 {product_detail_url} 时发生错误: {e}")
            except Exception as e:
                print(f"解析详情页 {product_detail_url} 或保存数据时发生错误: {e}")
        else:
            print(f"在列表项 {i + 1} 中未能找到有效的详情页链接，跳过该产品。")

    print(f"\n所有产品数据已保存到 '{output_base_dir}' 文件夹。")


# --- 配置 ---
base_website_url = "https://www.hisensehitachi.com"
# 家用
# 变频中央空调
product_list_page_url = "https://www.hisensehitachi.com/homeProductList/index_10.html"
# 地暖中央空调
# "https://www.hisensehitachi.com/homeProductList/index_11.html"
# 热水中央空调
# "https://www.hisensehitachi.com/homeProductList/index_12.html"
# 户式水机
# "https://www.hisensehitachi.com/homeProductList/index_13.html"
# 新风系统
# "https://www.hisensehitachi.com/homeProductList/index_14.html"
# 家用空调
# "https://www.hisensehitachi.com/homeProductList/index_15.html"

# 商用
# "https://www.hisensehitachi.com/common/index_19.html"
# 列表页选择器
list_item_base_selector = "body > div.commerProductsListsPage.houseHoldProductsListsPage > section.pcontBodyCont > div > div > div.rightCont > ul > li"

# 详情页通用选择器
detail_name_selector = "body > div.commerProductDetailsPage.houseHoldProductsDetailsPage > section.pcontBodyCont > div.w1400d > div > div.rightConBox > div.qProductViewBox > div.txtBox > div.qTop > div"
detail_intro_selector = "body > div.commerProductDetailsPage.houseHoldProductsDetailsPage > section.pcontBodyCont > div.w1400d > div > div.rightConBox > div.qProductViewBox > div.txtBox > div.qTop > dl > dd:nth-child(2)"
detail_poster_selector = "body > div.commerProductDetailsPage.houseHoldProductsDetailsPage > section.pcontBodyCont > div.w1400d > div > div.rightConBox > div.qConTabBox > ul > li:nth-child(1) > div > img"

# 外机参数相关选择器
outdoor_unit_section_selector = "body > div.commerProductDetailsPage.houseHoldProductsDetailsPage > section.pcontBodyCont > div.w1400d > div > div.rightConBox > div.qConTabBox > ul > li:nth-child(2)"
outdoor_unit_item_base_selector = "dl > dd"
outdoor_unit_name_relative_selector = "p"
outdoor_unit_image_relative_selector = "div.imgBox > img"
outdoor_unit_table_relative_selector = "div.custSolutHidden > div > div.table > table"

# 内机参数相关选择器
indoor_unit_section_selector = "body > div.commerProductDetailsPage.houseHoldProductsDetailsPage > section.pcontBodyCont > div.w1400d > div > div.rightConBox > div.qConTabBox > ul > li:nth-child(3)"
indoor_unit_item_base_selector = "dl > dd"
indoor_unit_name_relative_selector = "p"
indoor_unit_image_relative_selector = "div.imgBox > img"
indoor_unit_table_relative_selector = "div.custSolutHidden > div > div.table > table"

# 选配件及控制器相关选择器 (已修改)
accessories_section_selector = "body > div.commerProductDetailsPage.houseHoldProductsDetailsPage > section.pcontBodyCont > div.w1400d > div > div.rightConBox > div.qConTabBox > ul > li:nth-child(4)"
accessories_item_base_selector = "dl > dd"  # 每一个选配件项的基准选择器
accessories_name_relative_selector = "p"
accessories_image_relative_selector = "div.imgBox > img"
# **修改后的选择器：改为指向包含所有 <p> 标签的父元素**
accessories_intro_parent_relative_selector = "div.custSolutHidden > div > div.topBoxCont > div.rightCont"

# --- 执行爬取 ---
if __name__ == "__main__":
    scrape_product_data(
        base_website_url,
        product_list_page_url,
        list_item_base_selector,
        detail_name_selector,
        detail_intro_selector,
        detail_poster_selector,
        outdoor_unit_section_selector,
        outdoor_unit_item_base_selector,
        outdoor_unit_name_relative_selector,
        outdoor_unit_image_relative_selector,
        outdoor_unit_table_relative_selector,
        indoor_unit_section_selector,
        indoor_unit_item_base_selector,
        indoor_unit_name_relative_selector,
        indoor_unit_image_relative_selector,
        indoor_unit_table_relative_selector,
        accessories_section_selector,
        accessories_item_base_selector,
        accessories_name_relative_selector,
        accessories_image_relative_selector,
        accessories_intro_parent_relative_selector  # 传入修改后的选择器变量
    )