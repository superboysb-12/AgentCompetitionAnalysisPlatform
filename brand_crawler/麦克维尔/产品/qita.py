import requests
from bs4 import BeautifulSoup
import time
import re
import os
import json


def generate_safe_filename(text, replacement='_'):
    if not text:
        return ""
    text = re.sub(r'[^\w\u4e00-\u9fa5-]+', replacement, text)
    text = re.sub(f'{re.escape(replacement)}+', replacement, text)
    text = text.strip(replacement)
    if not text:
        return "default_name"
    return text


def scrape_mcquay_products_to_files():
    base_url = "https://www.mcquay.com.cn"
    # 轻商用机
    # 模块式空调机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=5&typeid=37"
    # output_dir = "mcquay_轻商用机/模块式空调机组"
    # 多联式中央空调机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=5&typeid=36"
    # output_dir = "mcquay_轻商用机/多联式中央空调机组"
    # 水源热泵机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=5&typeid=35"
    # output_dir = "mcquay_轻商用机/水源热泵机组"
    # 热泵热水机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=5&typeid=39"
    # output_dir = "mcquay_轻商用机/热泵热水机组"
    # 单元式空调机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=5&typeid=38"
    # output_dir = "mcquay_轻商用机/单元式空调机组"

    # 中央空调
    # 水冷离心式冷水机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=4&typeid=40"
    # output_dir = "mcquay_中央空调/水冷离心式冷水机组"
    # 水冷螺杆式冷水机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=4&typeid=41"
    # output_dir = "mcquay_中央空调/水冷螺杆式冷水机组"
    # 风冷冷热水机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=4&typeid=43"
    # output_dir = "mcquay_中央空调/风冷冷热水机组"
    # 水地源热泵机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=4&typeid=50"
    # output_dir = "mcquay_中央空调/水地源热泵机组"
    # 末端空调机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/preview_20.aspx?lcid=4&typeid=44"
    # output_dir = "mcquay_中央空调/末端空调机组"

    # 特种空调
    # 高温热水机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=3&typeid=100"
    # output_dir = "mcquay_特种空调/高温热水机组"
    # 商业冷冻机组
    list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=3&typeid=46"
    output_dir = "mcquay_特种空调/商业冷冻机组"
    # 工业冷冻机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=3&typeid=45"
    # output_dir = "mcquay_特种空调/工业冷冻机组"
    # 粮仓专用空调机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=3&typeid=98"
    # output_dir = "mcquay_特种空调/粮仓专用空调机组"
    # 烟草专用机组
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=3&typeid=99"
    # output_dir = "mcquay_特种空调/烟草专用机组"

    # 控制系统
    # 温控器
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=2&typeid=47"
    # output_dir = "mcquay_控制系统/温控器"
    # 阀类产品
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=2&typeid=48"
    # output_dir = "mcquay_控制系统/阀类产品"
    # 怡控系统控制
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=2&typeid=49"
    # output_dir = "mcquay_控制系统/怡控系统控制"
    # 小麦云控
    # list_page_url = "https://www.mcquay.com.cn/sycp/list_20.aspx?lcid=2&typeid=140"
    # output_dir = "mcquay_控制系统/小麦云控"

    os.makedirs(output_dir, exist_ok=True)

    print(f"正在访问列表页: {list_page_url}")
    try:
        response = requests.get(list_page_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"访问列表页失败: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    product_links = soup.select('body > section > div > ul > li > div > a')

    if not product_links:
        print("未找到产品链接，请检查选择器是否正确。")
        return

    print(f"找到 {len(product_links)} 个产品链接。")

    for i, link_tag in enumerate(product_links):
        product_detail_url = base_url + link_tag['href']
        print(f"\n正在爬取产品详情页 ({i + 1}/{len(product_links)}): {product_detail_url}")

        product_info = {
            'url': product_detail_url,
            'image_url': None,
            'name': None,
            'features': [],
            'parameters': {}
        }

        try:
            detail_response = requests.get(product_detail_url, timeout=10)
            detail_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"访问产品详情页失败: {e}")
            time.sleep(1)
            continue

        detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

        name_tag = detail_soup.select_one('#content > div > div.titles, #icontent > h2 > span')
        if name_tag:
            product_info['name'] = name_tag.get_text(strip=True)
            print(f"  产品名称: {product_info['name']}")
        else:
            print("  未找到产品名称，跳过此产品或使用默认名称。")
            product_info['name'] = f"Unnamed_Product_{i + 1}"

        folder_name = generate_safe_filename(product_info['name'])
        if not folder_name:
            folder_name = f"product_{i + 1}"

        product_folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(product_folder_path, exist_ok=True)

        img_tag = detail_soup.select_one('#sidebar > div > img')
        if img_tag and 'src' in img_tag.attrs:
            img_relative_url = img_tag['src']
            product_info['image_url'] = base_url + img_relative_url

            try:
                img_response = requests.get(product_info['image_url'], stream=True, timeout=10)
                img_response.raise_for_status()
                img_filename = os.path.basename(img_relative_url.split('?')[0])
                if not img_filename:
                    img_filename = "product_image.jpg"

                img_filename = generate_safe_filename(img_filename)
                if not img_filename:
                    img_filename = f"product_image_{i + 1}.jpg"

                img_path = os.path.join(product_folder_path, img_filename)
                with open(img_path, 'wb') as f:
                    for chunk in img_response.iter_content(1024):
                        f.write(chunk)
                product_info['local_image_path'] = img_path
            except requests.exceptions.RequestException as e:
                print(f"  下载图片失败: {e}")

        feature_spans = detail_soup.select('#content > div > div.rightDiv > div > p > span:nth-child(2)')
        for span in feature_spans:
            text = span.get_text(strip=True)
            if text:
                product_info['features'].append(text)

        params_container = detail_soup.select_one('#content > div > div.js')
        if params_container:
            all_param_texts = []

            # 找到 div.js 内所有 p 标签下的 span，以及直接在 div.js 下的 span
            # 这样可以覆盖所有参数可能的包装方式
            all_spans_in_container = params_container.find_all('span')

            for span_tag in all_spans_in_container:
                text = span_tag.get_text(strip=True)
                if text and text != '...':  # 过滤掉空的或只有省略号的文本
                    all_param_texts.append(text)

            # 匹配 "键: 值" 或 "键：值" 的模式
            param_pattern_with_colon = re.compile(r'([\w\u4e00-\u9fa5\s]+)\s*[:：]\s*(.+)', re.IGNORECASE)
            # 匹配一个以冒号结尾的潜在键的模式
            potential_key_pattern = re.compile(r'([\w\u4e00-\u9fa5\s]+)\s*[:：]\s*$', re.IGNORECASE)

            # 使用一个循环来匹配键值对，处理相邻的 span
            idx = 0
            while idx < len(all_param_texts):
                current_text = all_param_texts[idx]

                # 优先级1: 尝试用冒号模式匹配当前文本 (键:值 或 键：值)
                match = param_pattern_with_colon.match(current_text)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    if key and value:
                        product_info['parameters'][key] = value
                    idx += 1
                # 优先级2: 检查当前文本是否是以冒号结尾的键，并且尝试从下一个文本中获取值
                else:  # 如果不是完整的键值对，尝试将其视为一个键
                    key_match = potential_key_pattern.match(current_text)
                    if key_match:
                        key = key_match.group(1).strip()
                        # 检查下一个文本是否存在且不是另一个潜在的键
                        if idx + 1 < len(all_param_texts):
                            next_text = all_param_texts[idx + 1]
                            # 如果下一个文本不是以冒号结尾，我们就认为是它的值
                            if not potential_key_pattern.match(next_text):
                                value = next_text
                                if key and value:
                                    product_info['parameters'][key] = value
                                idx += 2  # 键和值都已处理
                                continue  # 继续下一个循环迭代
                        # 如果没有下一个文本或者下一个文本也是个键，则当前键视为没有值
                        if key not in product_info['parameters']:  # 避免重复添加
                            # product_info['parameters'][key] = None # 或者你可以给它一个空值
                            pass  # 暂时不添加没有值的键
                        idx += 1
                    else:
                        # 当前文本既不是完整的键值对，也不是以冒号结尾的键
                        # 这种情况可能是孤立的值，或者不符合我们识别的模式
                        # 我们可以选择忽略，或者如果需要，添加更复杂的启发式规则
                        # 例如，如果前一个参数没有值，这里可能是它的补充值
                        # 但为避免错误，目前暂时忽略
                        idx += 1


        # --- 参数提取部分结束 ---

        json_file_path = os.path.join(product_folder_path, "product_info.json")
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(product_info, f, ensure_ascii=False, indent=4)
            print(f"  产品信息已保存到: {json_file_path}")
        except Exception as e:
            print(f"  保存 JSON 文件失败: {e}")

        time.sleep(1)

    print("\n--- 爬取完成 ---")


if __name__ == "__main__":
    scrape_mcquay_products_to_files()