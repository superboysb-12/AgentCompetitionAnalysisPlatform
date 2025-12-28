import os
import requests
import urllib3  # 导入urllib3库
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
from datetime import datetime

# 禁用因忽略SSL验证而出现的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 基础URL
BASE_URL = "https://www.midea.com.cn"
# 新闻列表页URL
NEWS_LIST_URL = "https://www.midea.com.cn/zh/about-midea/news"
# 创建一个主文件夹来保存所有内容
OUTPUT_DIR = "midea_news"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_soup(url):
    """发送请求并获取BeautifulSoup对象"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"请求页面时发生错误: {url} - {e}")
        return None


def scrape_news_detail(news_url, title_text=""):
    """爬取单个新闻详情页的内容（文字和图片）"""
    print(f"正在爬取详情页: {news_url}")
    soup = get_soup(news_url)
    if not soup:
        return

    # 创建安全的文件夹名称
    safe_title = "".join(c for c in title_text if c.isalnum() or c in (' ', '-', '_')).rstrip()
    if not safe_title:
        safe_title = news_url.strip('/').split('/')[-1]

    dir_name = os.path.join(OUTPUT_DIR, safe_title[:50])  # 限制文件夹名长度
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # 初始化新闻数据结构
    news_data = {
        "title": "",
        "url": news_url,
        "content": [],
        "images": [],
        "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- 爬取文字内容 ---
    # 尝试多种可能的内容区域选择器
    content_selectors = [
        'div.mgws-news-detail-content',  # 修正后的类名
        'div[class*="news-detail-content"]'  # 模糊匹配
    ]

    content_div = None
    for selector in content_selectors:
        content_div = soup.select_one(selector)
        if content_div:
            print(f"  - 使用选择器找到内容: {selector}")
            break

    if content_div:
        # 获取标题
        title_selectors = [
            'div.mgws-news-detail-title',
            'div[class*="news-detail-title"]'
        ]

        title = title_text
        for title_selector in title_selectors:
            title_tag = soup.select_one(title_selector)
            if title_tag:
                title = title_tag.get_text(strip=True)
                break

        news_data["title"] = title

        # 提取所有段落内容
        paragraphs = content_div.find_all('p')

        for p in paragraphs:
            # 跳过只有&nbsp;的段落
            text = p.get_text(strip=True)
            if text and text != '' and '&nbsp;' not in text:
                news_data["content"].append(text)

        if news_data["content"]:
            print(f"  - 提取到 {len(news_data['content'])} 个段落")
        else:
            print(f"  - 内容区域找到但没有有效文字内容")
    else:
        print(f"  - 未找到新闻内容区域，尝试全局搜索...")
        # 获取标题
        news_data["title"] = title_text

        # 调试：打印页面中所有包含"content"的div
        all_content_divs = soup.find_all('div', class_=lambda x: x and 'content' in ' '.join(x).lower())
        for div in all_content_divs:
            print(f"    找到内容相关div: {div.get('class', [])}")

        # 尝试直接提取所有p标签的内容
        all_paragraphs = soup.find_all('p')

        for p in all_paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 10 and '&nbsp;' not in text:  # 过滤掉很短的段落
                news_data["content"].append(text)

        if news_data["content"]:
            print(f"  - 通过全局搜索提取到 {len(news_data['content'])} 个段落")

    # --- 爬取图片信息和下载图片 ---
    images = soup.find_all('img')
    img_count = 0
    downloaded_imgs = 0

    # 定义需要过滤的logo关键词
    logo_keywords = ['logo', 'Logo', 'LOGO', 'brand', 'Brand']

    for img in images:
        img_url = img.get('src') or img.get('data-src')  # 有些网站使用懒加载
        if img_url and not img_url.startswith('data:'):  # 跳过base64图片
            # 转换为绝对URL
            full_img_url = urljoin(BASE_URL, img_url)

            # 检查是否为logo图片
            is_logo = False
            img_name_check = os.path.basename(img_url.split('?')[0]).lower()
            img_alt = img.get('alt', '').lower()
            img_class = ' '.join(img.get('class', [])).lower()

            # 检查文件名、alt属性、class属性是否包含logo关键词
            for keyword in logo_keywords:
                if (keyword.lower() in img_name_check or
                        keyword.lower() in img_alt or
                        keyword.lower() in img_class):
                    is_logo = True
                    break

            # 如果是logo图片，跳过不处理
            if is_logo:
                continue

            try:
                img_response = requests.get(full_img_url, stream=True, verify=False, timeout=15)
                img_response.raise_for_status()

                # 获取文件扩展名
                img_name = os.path.basename(img_url.split('?')[0])
                # URL解码文件名
                try:
                    from urllib.parse import unquote
                    img_name = unquote(img_name)
                except:
                    pass

                if not img_name or '.' not in img_name:
                    img_count += 1
                    content_type = img_response.headers.get('content-type', '')
                    if 'jpeg' in content_type:
                        ext = '.jpg'
                    elif 'png' in content_type:
                        ext = '.png'
                    elif 'gif' in content_type:
                        ext = '.gif'
                    elif 'webp' in content_type:
                        ext = '.webp'
                    else:
                        ext = '.jpg'
                    img_name = f"image_{img_count}{ext}"

                # 确保文件名不会太长
                if len(img_name) > 100:
                    img_count += 1
                    ext = os.path.splitext(img_name)[1] if '.' in img_name else '.jpg'
                    img_name = f"image_{img_count}{ext}"

                img_path = os.path.join(dir_name, img_name)

                # 分块下载，避免内存问题
                with open(img_path, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        if chunk:  # 过滤掉keep-alive的新块
                            f.write(chunk)

                # 只保留URL信息到JSON中
                news_data["images"].append(full_img_url)
                downloaded_imgs += 1
                print(f"  - 图片已下载 ({downloaded_imgs}): {img_name}")

            except requests.exceptions.RequestException as e:
                print(f"  - 下载图片失败: {full_img_url} - {e}")
                # 即使下载失败也记录URL
                news_data["images"].append(full_img_url)
            except Exception as e:
                print(f"  - 保存图片时出错: {e}")
                # 即使保存失败也记录URL
                news_data["images"].append(full_img_url)

    print(f"  - 共处理 {len(news_data['images'])} 张非logo图片，成功下载 {downloaded_imgs} 张")

    # 保存JSON文件
    json_file_path = os.path.join(dir_name, "news_data.json")
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=2)
        print(f"  - 新闻数据已保存至: {json_file_path}")
    except Exception as e:
        print(f"  - 保存JSON文件失败: {e}")

    return news_data


def main():
    """主函数，用于爬取新闻列表并触发详情页爬取"""
    print("开始爬取美的集团新闻列表...")
    soup = get_soup(NEWS_LIST_URL)
    if not soup:
        print("无法获取新闻列表页，程序退出。")
        return

    # 打印页面结构以便调试
    print("页面标题:", soup.title.get_text() if soup.title else "无标题")

    # 尝试多种可能的选择器
    selectors = [
        'a[href*="/news/"]',
        'a[href*="/zh/about-midea/news/"]'
    ]

    news_links = []
    for selector in selectors:
        news_links = soup.select(selector)
        print(f"使用选择器 '{selector}' 找到 {len(news_links)} 个链接")
        if news_links:
            break

    print(f"共找到 {len(news_links)} 条新闻。")

    # 添加选项让用户选择爬取数量
    max_news = min(len(news_links), 30)  # 限制最多爬取20条，避免过多
    print(f"限制爬取前 {max_news} 条新闻")

    # 存储所有新闻数据的汇总
    all_news_data = []

    for i, link in enumerate(news_links[:max_news], 1):
        relative_path = link.get('href')
        if not relative_path.startswith('http'):
            full_url = urljoin(BASE_URL, relative_path)
        else:
            full_url = relative_path

        # 获取链接文本作为标题
        title_text = link.get_text(strip=True)

        print(f"\n[{i}/{max_news}] 处理新闻: {title_text[:50]}{'...' if len(title_text) > 50 else ''}")

        try:
            news_data = scrape_news_detail(full_url, title_text)
            if news_data:
                all_news_data.append(news_data)
        except KeyboardInterrupt:
            print(f"\n用户中断了程序，已处理 {i - 1} 条新闻")
            break
        except Exception as e:
            print(f"  - 处理新闻时出错: {e}")

        print("-" * 50)

        # 添加延时避免被反爬虫
        time.sleep(2)


if __name__ == "__main__":
    main()