import requests
from bs4 import BeautifulSoup
import json
import os
import re
import csv


def sanitize_filename(name):
    """
    移除Windows和Linux文件名中的非法字符，并将换行符替换为空格。
    """
    if not name or name == 'N/A':
        return None
    name = re.sub(r'[\r\n]+', ' ', name)
    name = name.replace('/', '-').replace('\\', '-')
    name = name.replace(':', '：')
    return re.sub(r'[*?"<>|]', '', name).strip()


def find_product_image(soup, product_url):
    """
    查找产品图片 - 简化版本
    """
    selectors = [
        '.section img', 'img[src*="upload"]', '.homeupImg img', 'img'
    ]
    for selector in selectors:
        try:
            elements = soup.select(selector)
            for elem in elements:
                src = elem.get('src', '').strip()
                if not src: continue
                skip_patterns = [
                    'logo', 'banner', 'icon', 'bg', 'background', 'nav', 'menu',
                    'footer', 'header', 'button', 'arrow', 'bullet', 'dot', 'line'
                ]
                if any(pattern in src.lower() for pattern in skip_patterns):
                    continue
                full_url = f"https://www.yorkvrfchina.com{src if src.startswith('/') else '/' + src}" if not src.startswith(
                    'http') else src
                if any(ext in full_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    print(f"  - 找到产品图片: {full_url}")
                    return full_url
        except Exception:
            continue
    return None


def scrape_product_details(product_url):
    """
    从单个产品页面抓取详细信息，包括室内机图片。
    """
    try:
        response = requests.get(product_url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        product_name = 'N/A'
        name_container = soup.select_one('.homeupTxt > div > p')
        if name_container:
            name_part1_tag = name_container.find('span')
            name_part1 = name_part1_tag.text.strip() if name_part1_tag else ''
            br_tag = name_container.find('br')
            name_part2 = br_tag.next_sibling.strip() if br_tag and br_tag.next_sibling and isinstance(
                br_tag.next_sibling, str) else ''
            full_name = f"{name_part1} {name_part2}".strip()
            if full_name: product_name = full_name

        product_image_url = find_product_image(soup, product_url) or 'N/A'
        applicable_scope_elements = soup.select('.homeupTxt > div > div > p')
        applicable_scope = ' | '.join(
            [p.text.strip() for p in applicable_scope_elements]) if applicable_scope_elements else 'N/A'

        introductions = []
        intro_elements = soup.select('.navCont.hdInter.homeOne > ul > li')
        for item in intro_elements:
            title_element = item.select_one('div > h3')
            detail_element = item.select_one('div > div > p')
            image_element = item.select_one('img')
            intro_image_url = 'N/A'
            if image_element and image_element.get('src'):
                src = image_element['src'].strip()
                intro_image_url = f"https://www.yorkvrfchina.com{src if src.startswith('/') else '/' + src}" if not src.startswith(
                    'http') else src

            if title_element and detail_element:
                clean_title = " ".join(title_element.get_text(strip=True).split())
                introductions.append({
                    "标题": clean_title,
                    "详情": detail_element.get_text(strip=True),
                    "介绍图片链接": intro_image_url
                })

        outdoor_params_data = []
        outdoor_section = soup.select_one('.navCont.hdOutdoor.w960')
        if outdoor_section:
            elements = outdoor_section.find_all(['h3', 'table'], recursive=False)
            current_title = ""
            for el in elements:
                if el.name == 'h3':
                    current_title = el.text.strip()
                elif el.name == 'table':
                    table_data = [row for row in
                                  [[td.text.strip() for td in r.find_all('td')] for r in el.find_all('tr')] if any(row)]
                    if table_data:
                        outdoor_params_data.append({"参数表标题": current_title, "参数": table_data})
                        current_title = ""

        indoor_params_data = []
        indoor_section = soup.select_one('.navCont.hdIndoor')
        if indoor_section:
            image_urls = []
            for item in indoor_section.select('ul > li'):
                img_tag = item.select_one('p > img')
                if img_tag and img_tag.get('src'):
                    src = img_tag['src'].strip()
                    image_urls.append(
                        f"https://www.yorkvrfchina.com{src if src.startswith('/') else '/' + src}" if not src.startswith(
                            'http') else src)
                else:
                    image_urls.append('N/A')

            title_elements = indoor_section.select('.boxTit')
            for idx, title_el in enumerate(title_elements):
                current_title = title_el.get_text(strip=True)
                table_el = title_el.find_next(class_='tabpar')
                if table_el and table_el.find('table'):
                    table_data = [row for row in [[td.get_text(strip=True) for td in r.find_all('td')] for r in
                                                  table_el.find('table').find_all('tr')] if any(row)]
                    if table_data:
                        matched_image_url = image_urls[idx] if idx < len(image_urls) else 'N/A'
                        indoor_params_data.append({
                            "参数表标题": current_title, "参数": table_data,
                            "室内机图片链接": matched_image_url
                        })

        return {
            "产品名称": product_name, "产品图片链接": product_image_url, "适用范围": applicable_scope,
            "产品介绍": introductions, "室外机参数": outdoor_params_data, "室内机参数": indoor_params_data,
            "源URL": product_url
        }
    except requests.exceptions.RequestException as e:
        print(f"请求错误 {product_url}: {e}")
        return None
    except Exception as e:
        print(f"解析页面时发生未知错误 {product_url}: {e}")
        return None


def main():
    base_url = "https://www.yorkvrfchina.com"
    # 家用中央空调
    list_page_url = "https://www.yorkvrfchina.com/airconditiondetail/index.aspx?nodeid=3684"
    # 商用中央空调
    # list_page_url = "https://www.yorkvrfchina.com/airconditiondetail/index.aspx?nodeid=3303"
    main_output_folder = "york_jiayong"
    os.makedirs(main_output_folder, exist_ok=True)

    try:
        response = requests.get(list_page_url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        product_links = sorted(list(set(
            f"{base_url}{a['href'] if a['href'].startswith('/') else '/' + a['href']}"
            for a in soup.select('#Form1 > div:nth-child(3) > div.tuwen > div.nav.w960.yorkNav > div.navSide > div > a[href]')
        )))

        if not product_links:
            print("未能找到任何产品链接，请检查列表页的选择器。")
            return

        print(f"共找到 {len(product_links)} 个产品，开始爬取...")

        for link in product_links:
            print(f"正在处理: {link}")
            product_data = scrape_product_details(link)

            if not product_data:
                print(f"  - [跳过] 未能从此链接获取任何数据。")
                continue

            product_name_sanitized = sanitize_filename(product_data['产品名称'])
            if not product_name_sanitized:
                print(f"  - [跳过] 因产品名称无效 ('{product_data['产品名称']}')，无法创建文件夹。")
                continue

            product_folder_path = os.path.join(main_output_folder, product_name_sanitized)
            os.makedirs(product_folder_path, exist_ok=True)

            outdoor_csv_files = []
            if product_data.get("室外机参数"):
                for table in product_data["室外机参数"]:
                    title, data = table.get("参数表标题"), table.get("参数")
                    if not data: continue
                    sanitized_title = sanitize_filename(title) or "室外机参数表"
                    csv_filename = f"{sanitized_title}.csv"
                    try:
                        with open(os.path.join(product_folder_path, csv_filename), 'w', newline='',
                                  encoding='utf-8-sig') as f:
                            csv.writer(f).writerows(data)
                        print(f"  -  室外机参数表 '{sanitized_title}' 已保存。")
                        outdoor_csv_files.append(csv_filename)
                    except Exception as e:
                        print(f"  -  保存室外机CSV '{sanitized_title}' 失败: {e}")

            indoor_info_for_json = []
            if product_data.get("室内机参数"):
                indoor_img_folder = os.path.join(product_folder_path, "indoor_unit_images")
                os.makedirs(indoor_img_folder, exist_ok=True)
                for item in product_data["室内机参数"]:
                    title, data, img_url = item.get("参数表标题"), item.get("参数"), item.get("室内机图片链接")
                    csv_filename, sanitized_title = None, sanitize_filename(title)
                    if not sanitized_title:
                        print(f"  - [跳过] 因室内机标题无效 ('{title}')，无法保存文件。")
                        continue
                    if data:
                        csv_filename = f"室内机 - {sanitized_title}.csv"
                        try:
                            with open(os.path.join(product_folder_path, csv_filename), 'w', newline='',
                                      encoding='utf-8-sig') as f:
                                csv.writer(f).writerows(data)
                            print(f"  -  室内机参数表 '{sanitized_title}' 已保存。")
                        except Exception as e:
                            print(f"  -  保存室内机CSV '{sanitized_title}' 失败: {e}")
                            csv_filename = None
                    if img_url and img_url != 'N/A':
                        try:
                            res = requests.get(img_url, timeout=30)
                            res.raise_for_status()
                            _, ext = os.path.splitext(os.path.basename(img_url.split('?')[0]))
                            img_path = os.path.join(indoor_img_folder, f"{sanitized_title}{ext or '.jpg'}")
                            with open(img_path, 'wb') as f:
                                f.write(res.content)
                            print(f"  -  室内机图片 '{sanitized_title}' 已保存。")
                        except Exception as e:
                            print(f"  -  下载室内机图片 '{sanitized_title}' 失败: {e}")
                    indoor_info_for_json.append(
                        {"型号名称": title, "参数表CSV": csv_filename or "N/A", "图片链接": img_url})

            del product_data["室外机参数"]
            del product_data["室内机参数"]
            product_data["室外机参数CSV"] = outdoor_csv_files
            product_data["室内机信息"] = indoor_info_for_json

            with open(os.path.join(product_folder_path, 'product_info.json'), 'w', encoding='utf-8') as f:
                json.dump(product_data, f, ensure_ascii=False, indent=4)

            if product_data.get("产品介绍"):
                intro_img_folder = os.path.join(product_folder_path, "introduction_images")
                os.makedirs(intro_img_folder, exist_ok=True)
                for i, item in enumerate(product_data["产品介绍"]):
                    img_url, title = item.get("介绍图片链接"), item.get("标题")
                    if img_url and img_url != 'N/A':
                        try:
                            res = requests.get(img_url, timeout=30)  # 增加超时
                            res.raise_for_status()
                            fname = sanitize_filename(title) or f"intro_{i}"
                            _, ext = os.path.splitext(os.path.basename(img_url.split('?')[0]))
                            img_path = os.path.join(intro_img_folder, f"{fname}{ext or '.jpg'}")
                            with open(img_path, 'wb') as f:
                                f.write(res.content)
                        except Exception as e:
                            print(f"  -  下载介绍图片 '{title}' 失败: {e}")

            main_image_url = product_data.get('产品图片链接')
            if main_image_url and main_image_url != 'N/A':
                try:
                    res = requests.get(main_image_url, timeout=30)  # 增加超时
                    res.raise_for_status()
                    fname = os.path.basename(main_image_url.split('?')[0])
                    img_path = os.path.join(product_folder_path,
                                            fname if fname and '.' in fname else f"{product_name_sanitized}.jpg")
                    with open(img_path, 'wb') as f:
                        f.write(res.content)
                except Exception as e:
                    print(f"  -  下载主产品图片失败: {e}")
            else:
                print("  - ⓘ 未找到主产品图片链接。")

    except requests.exceptions.RequestException as e:
        print(f"无法访问产品列表页 {list_page_url}: {e}")
    except Exception as e:
        print(f"发生致命错误: {e}")

    print(f"\n任务完成。所有数据已保存在 '{main_output_folder}' 文件夹中。")


if __name__ == '__main__':
    main()
