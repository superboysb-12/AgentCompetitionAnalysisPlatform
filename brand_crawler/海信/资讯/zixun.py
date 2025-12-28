import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class HisenseNewsScraperImproved:
    def __init__(self):
        self.base_url = "https://www.zykthisense.com"
        self.news_list_url = "https://www.zykthisense.com/news/index.aspx?nodeid=86"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Selenium 设置
        self.driver = None
        self.setup_selenium()

        # 创建保存目录
        self.create_directories()

    def setup_selenium(self):
        """设置Selenium WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # 无头模式
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')

            self.driver = webdriver.Chrome(options=chrome_options)
            print("Selenium WebDriver 初始化成功")
        except Exception as e:
            print(f"Selenium WebDriver 初始化失败: {e}")
            print("将使用纯requests方式爬取")
            self.driver = None

    def create_directories(self):
        """创建必要的目录"""
        directories = ['news_data', 'news_data/articles', 'news_data/images']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")

    def get_page_with_selenium(self, url, wait_time=10):
        """使用Selenium获取页面内容"""
        if not self.driver:
            return None

        try:
            print(f"使用Selenium获取页面: {url}")
            self.driver.get(url)

            # 等待页面加载完成
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # 额外等待，确保动态内容加载
            time.sleep(2)

            return self.driver.page_source
        except TimeoutException:
            print(f"页面加载超时: {url}")
            return None
        except Exception as e:
            print(f"获取页面失败: {e}")
            return None

    def get_page_content(self, url, max_retries=3):
        """获取页面内容，带重试机制"""
        # 首先尝试Selenium
        if self.driver:
            content = self.get_page_with_selenium(url)
            if content:
                return content

        # 如果Selenium失败，使用requests
        for attempt in range(max_retries):
            try:
                print(f"正在获取页面 (requests): {url}")
                response = self.session.get(url, timeout=15)
                response.raise_for_status()

                # 尝试不同的编码
                encodings = ['utf-8', 'gb2312', 'gbk']
                for encoding in encodings:
                    try:
                        response.encoding = encoding
                        content = response.text
                        # 检查是否包含中文字符，判断编码是否正确
                        if '海信' in content or '新闻' in content or '资讯' in content:
                            print(f"使用编码: {encoding}")
                            return content
                    except:
                        continue

                # 如果都不行，就用默认的
                response.encoding = response.apparent_encoding or 'utf-8'
                return response.text

            except requests.exceptions.RequestException as e:
                print(f"获取页面失败 (尝试 {attempt + 1}/{max_retries}): {url}")
                print(f"错误: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    return None

    def extract_news_links_from_html(self, html_content):
        """从HTML中提取新闻链接"""
        soup = BeautifulSoup(html_content, 'html.parser')
        news_links = []

        print("开始提取新闻链接...")

        # 根据您提供的HTML结构，新闻链接在 ul.newList 的 li 中
        news_lists = soup.find_all('ul', class_='newList')
        print(f"找到 {len(news_lists)} 个新闻列表容器")

        for news_list in news_lists:
            items = news_list.find_all('li')
            print(f"新闻列表中有 {len(items)} 个新闻项")

            for item in items:
                # 查找链接
                link = item.find('a', href=True)
                if link:
                    href = link.get('href')
                    title = link.get_text(strip=True)

                    if href and title:
                        # 处理相对URL
                        if href.startswith('/'):
                            full_url = self.base_url + href
                        else:
                            full_url = urljoin(self.base_url, href)

                        # 检查是否是新闻详情页
                        if 'contentid=' in href or 'detail.aspx' in href:
                            news_links.append({
                                'url': full_url,
                                'title': title
                            })
                            print(f"找到新闻: {title[:50]}...")

        # 如果没有找到，尝试其他方式
        if not news_links:
            print("使用备选方式查找新闻链接...")

            # 查找所有包含 contentid 的链接
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link.get('href', '')
                if 'contentid=' in href and 'page=ContentPage' in href:
                    title = link.get_text(strip=True)
                    if title and len(title) > 5:
                        full_url = urljoin(self.base_url, href)
                        news_links.append({
                            'url': full_url,
                            'title': title
                        })
                        print(f"备选方式找到: {title[:50]}...")

        return news_links

    def handle_pagination(self, max_pages=5):
        """处理分页，获取多页新闻链接"""
        all_news_links = []

        if not self.driver:
            print("没有Selenium支持，无法处理分页")
            # 只获取第一页
            html_content = self.get_page_content(self.news_list_url)
            if html_content:
                return self.extract_news_links_from_html(html_content)
            return []

        try:
            # 获取第一页
            print("获取第一页新闻...")
            self.driver.get(self.news_list_url)
            time.sleep(3)

            # 提取第一页的新闻
            page_source = self.driver.page_source
            page_links = self.extract_news_links_from_html(page_source)
            all_news_links.extend(page_links)
            print(f"第1页找到 {len(page_links)} 条新闻")

            # 尝试获取更多页面
            for page_num in range(2, max_pages + 1):
                try:
                    print(f"尝试获取第{page_num}页...")

                    # 查找下一页按钮
                    next_buttons = self.driver.find_elements(By.XPATH,
                                                             "//a[contains(@class, 'next') or contains(text(), '下一页') or contains(text(), '>')]")

                    if not next_buttons:
                        # 尝试查找页码链接
                        page_links_elements = self.driver.find_elements(By.XPATH,
                                                                        f"//a[contains(@href, 'pagenum={page_num}') or text()='{page_num}']")

                        if page_links_elements:
                            page_links_elements[0].click()
                        else:
                            print("没有找到下一页按钮或页码链接")
                            break
                    else:
                        # 点击下一页
                        next_buttons[0].click()

                    # 等待页面加载
                    time.sleep(3)

                    # 提取当前页的新闻
                    page_source = self.driver.page_source
                    page_links = self.extract_news_links_from_html(page_source)

                    if not page_links:
                        print(f"第{page_num}页没有找到新闻，停止分页")
                        break

                    all_news_links.extend(page_links)
                    print(f"第{page_num}页找到 {len(page_links)} 条新闻")

                except NoSuchElementException:
                    print(f"第{page_num}页不存在或无法访问")
                    break
                except Exception as e:
                    print(f"获取第{page_num}页时出错: {e}")
                    break

            # 去重
            seen_urls = set()
            unique_links = []
            for link in all_news_links:
                if link['url'] not in seen_urls:
                    seen_urls.add(link['url'])
                    unique_links.append(link)

            print(f"总共找到 {len(unique_links)} 条去重后的新闻")
            return unique_links

        except Exception as e:
            print(f"处理分页时出错: {e}")
            return all_news_links

    def extract_article_content(self, html_content, article_url):
        """提取文章内容"""
        soup = BeautifulSoup(html_content, 'html.parser')
        article_data = {
            'url': article_url,
            'title': '',
            'publish_time': '',
            'content': '',
            'images': []
        }

        try:
            # 提取标题
            title_selectors = [
                'h2',
                '.content_kk h2',
                '.content_warp h2',
                '[class*="title"]',
                'h1'
            ]

            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    article_data['title'] = title_elem.get_text(strip=True)
                    print(f"找到标题: {article_data['title']}")
                    break

            # 提取发布时间
            time_patterns = [
                r'发布时间[：:\s]*(\d{4}-\d{2}-\d{2})',
                r'时间[：:\s]*(\d{4}-\d{2}-\d{2})',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{4}/\d{2}/\d{2})',
                r'(\d{4}\.\d{2}\.\d{2})'
            ]

            page_text = soup.get_text()
            for pattern in time_patterns:
                match = re.search(pattern, page_text)
                if match:
                    article_data['publish_time'] = match.group(1)
                    print(f"找到发布时间: {article_data['publish_time']}")
                    break

            # 提取正文内容 - 改进的内容提取策略
            content_selectors = [
                '.acc_content',
                '.content_warp',
                '.content_kk',
                '[class*="content"]',
                '.article-content'
            ]

            content_found = False
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    print(f"使用选择器找到内容容器: {selector}")

                    # 移除脚本和样式标签
                    for script in content_elem(['script', 'style']):
                        script.decompose()

                    # 提取所有文本内容
                    text_content = []

                    # 方法1: 按段落提取
                    for elem in content_elem.find_all(['p', 'div', 'span'], recursive=True):
                        text = elem.get_text(strip=True)
                        if text and len(text) > 15:  # 只保留较长的文本
                            # 避免重复内容
                            if text not in text_content:
                                text_content.append(text)

                    # 方法2: 如果段落提取失败，直接提取所有文本
                    if not text_content:
                        all_text = content_elem.get_text()
                        # 按行分割并过滤
                        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                        text_content = [line for line in lines if len(line) > 15]

                    if text_content:
                        article_data['content'] = '\n\n'.join(text_content)
                        content_found = True
                        print(f"提取到内容长度: {len(article_data['content'])} 字符")
                        break

            # 如果还是没有找到内容，使用最后的备选方案
            if not content_found:
                print("使用备选内容提取方案...")
                # 移除导航、头部、脚部等元素
                for unwanted in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                    unwanted.decompose()

                # 查找包含最多文本的div
                divs = soup.find_all('div')
                best_div = None
                max_text_length = 0

                for div in divs:
                    text = div.get_text(strip=True)
                    if len(text) > max_text_length and len(text) > 100:
                        max_text_length = len(text)
                        best_div = div

                if best_div:
                    text_lines = [line.strip() for line in best_div.get_text().split('\n')
                                  if line.strip() and len(line.strip()) > 15]
                    article_data['content'] = '\n\n'.join(text_lines[:10])  # 限制行数
                    print(f"备选方案提取到内容长度: {len(article_data['content'])} 字符")

            # 提取图片
            img_selectors = [
                '.acc_content img',
                '.content_warp img',
                '.content_kk img',
                'img'
            ]

            images_found = []
            for selector in img_selectors:
                images = soup.select(selector)
                if images:
                    images_found = images
                    break

            print(f"找到 {len(images_found)} 张图片")

            for i, img in enumerate(images_found):
                img_src = img.get('src')
                if img_src:
                    # 处理相对路径
                    if img_src.startswith('/'):
                        img_src = self.base_url + img_src
                    elif not img_src.startswith('http'):
                        img_src = urljoin(article_url, img_src)

                    # 生成文件名
                    article_id = re.search(r'contentid=(\d+)', article_url)
                    if article_id:
                        filename = f"article_{article_id.group(1)}_img_{i + 1}.jpg"
                    else:
                        filename = f"article_img_{int(time.time())}_{i + 1}.jpg"

                    # 下载图片
                    downloaded_filename = self.download_image(img_src, filename)
                    if downloaded_filename:
                        article_data['images'].append({
                            'original_src': img_src,
                            'filename': downloaded_filename,
                            'alt': img.get('alt', '')
                        })

            return article_data

        except Exception as e:
            print(f"提取文章内容失败: {e}")
            import traceback
            traceback.print_exc()
            return article_data

    def download_image(self, img_url, filename):
        """下载图片"""
        try:
            # 处理相对URL
            if img_url.startswith('/'):
                img_url = self.base_url + img_url
            elif not img_url.startswith('http'):
                img_url = urljoin(self.base_url, img_url)

            response = self.session.get(img_url, timeout=10)
            response.raise_for_status()

            filepath = os.path.join('news_data/images', filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"图片下载成功: {filename}")
            return filename
        except Exception as e:
            print(f"图片下载失败: {img_url}, 错误: {e}")
            return None

    def save_article(self, article_data):
        """保存文章数据"""
        try:
            # 生成文件名
            title = re.sub(r'[^\w\s-]', '', article_data['title'])[:50]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{title}.json"

            filepath = os.path.join('news_data/articles', filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False, indent=2)

            print(f"文章保存成功: {filename}")
            return filename

        except Exception as e:
            print(f"保存文章失败: {e}")
            return None

    def scrape_all_news(self, max_pages=3, delay=3):
        """爬取所有新闻"""
        all_articles = []

        print("开始获取新闻链接...")

        # 获取所有新闻链接（包括分页）
        news_links = self.handle_pagination(max_pages)

        if not news_links:
            print("未找到新闻链接，请检查页面结构或网络连接")
            return []

        print(f"开始爬取 {len(news_links)} 篇文章...")

        # 爬取每篇文章
        for i, news_item in enumerate(news_links):
            print(f"\n正在处理第 {i + 1}/{len(news_links)} 篇文章...")
            print(f"标题: {news_item['title']}")
            print(f"URL: {news_item['url']}")

            # 获取文章页面
            article_html = self.get_page_content(news_item['url'])
            if not article_html:
                print("获取文章页面失败")
                continue

            # 提取文章内容
            article_data = self.extract_article_content(article_html, news_item['url'])

            if not article_data['title']:
                article_data['title'] = news_item['title']

            # 保存文章
            if article_data['content'] or article_data['images']:
                saved_filename = self.save_article(article_data)
                if saved_filename:
                    all_articles.append(article_data)
                    print(f"文章处理完成，包含 {len(article_data['images'])} 张图片")
            else:
                print("文章内容为空，跳过保存")

            # 延时避免过于频繁的请求
            if i < len(news_links) - 1:
                time.sleep(delay)

        return all_articles


        print(f"\n爬取完成！")
        print(f"共爬取文章: {summary['total_articles']} 篇")
        print(f"共下载图片: {summary['total_images']} 张")
        print(f"数据保存在 news_data 目录中")

        return summary

    def __del__(self):
        """析构函数，关闭WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                print("WebDriver 已关闭")
            except:
                pass


def main():
    """主函数"""
    print("海信新闻爬虫启动...")

    scraper = HisenseNewsScraperImproved()

    try:
        # 开始爬取 (最多3页，每次请求间隔3秒)
        articles = scraper.scrape_all_news(max_pages=3, delay=3)

    except KeyboardInterrupt:
        print("\n用户中断爬取")
    except Exception as e:
        print(f"爬取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保清理资源
        if hasattr(scraper, 'driver') and scraper.driver:
            scraper.driver.quit()


if __name__ == "__main__":
    main()