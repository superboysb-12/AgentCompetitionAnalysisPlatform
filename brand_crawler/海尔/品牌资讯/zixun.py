import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time
import json
import os
import re
from urllib.parse import urljoin
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('haier_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class HaierNewsScraper:
    def __init__(self):
        self.base_url = "https://www.haier.com/about_haier/xinwen/?spm=cn.home_pc.news_20250101.6"
        self.output_dir = "haier_news_output"
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_driver(self):
        """配置并启动Chrome浏览器"""
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        # chrome_options.add_argument('--headless')

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.implicitly_wait(10)
            logging.info("Chrome浏览器启动成功")
        except Exception as e:
            logging.error(f"启动浏览器失败: {e}")
            raise

    def get_news_links_from_page(self):
        """从当前页面获取所有资讯链接"""
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "newslist"))
            )
            time.sleep(2)  # 等待JS渲染完成

            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            news_links = []
            newslist = soup.find('ul', class_='newslist js_newslist')

            if newslist:
                news_items = newslist.find_all('li', class_='item')
                for item in news_items:
                    title_link = item.find('a', class_='title js_newslist_title')
                    if title_link and title_link.get('href'):
                        href = title_link.get('href')
                        full_url = urljoin('https://www.haier.com', href)

                        title = title_link.get_text(strip=True)
                        date_elem = item.find('div', class_='date')
                        date = date_elem.get_text(strip=True) if date_elem else "未知日期"

                        news_links.append({
                            'title': title,
                            'url': full_url,
                            'date': date
                        })
                logging.info(f"当前页面找到 {len(news_links)} 条资讯")

            return news_links

        except Exception as e:
            logging.error(f"获取资讯链接失败: {e}")
            return []

    def get_news_content(self, news_url):
        """获取单个资讯的详细内容（包括文本和图片链接）"""
        try:
            response = self.session.get(news_url, timeout=30)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            details = {}

            title_elem = soup.find('h1', class_='title sd_text')
            details['title'] = title_elem.get_text(strip=True) if title_elem else "无标题"

            date_elem = soup.find('div', class_='date sd_date')
            details['date'] = date_elem.get_text(strip=True) if date_elem else "未知日期"

            article_content = []
            image_urls = []
            content_div = soup.find('div', class_='newsdetail_content sd_news clearfix')

            if content_div:
                paragraphs = content_div.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text:
                        article_content.append(text)

                images = content_div.find_all('img')
                for img in images:
                    src = img.get('src')
                    if src:
                        full_img_url = urljoin(news_url, src)
                        image_urls.append(full_img_url)

            details['content'] = '\n'.join(article_content)
            details['image_urls'] = image_urls
            details['url'] = news_url

            logging.info(f"成功获取资讯内容: {details['title'][:30]}...")
            return details

        except Exception as e:
            logging.error(f"获取资讯内容失败 {news_url}: {e}")
            return None

    def save_article_to_folder(self, article_data):
        """将单个资讯的内容和图片保存到独立文件夹中"""
        try:
            folder_name = re.sub(r'[\\/*?:"<>|]', "", article_data['title'])
            article_path = os.path.join(self.output_dir, folder_name)
            os.makedirs(article_path, exist_ok=True)
            logging.info(f"为文章 '{folder_name}' 创建目录: {article_path}")

            json_content = {
                'title': article_data['title'],
                'date': article_data['date'],
                'url': article_data['url'],
                'content': article_data['content']
            }
            json_filepath = os.path.join(article_path, 'content.json')
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, ensure_ascii=False, indent=4)
            logging.info(f"文字内容已保存到: {json_filepath}")

            for i, img_url in enumerate(article_data['image_urls']):
                try:
                    img_response = self.session.get(img_url, timeout=20, stream=True)
                    if img_response.status_code == 200:
                        img_name = os.path.basename(img_url.split('?')[0])
                        if not img_name:
                            img_name = f"image_{i + 1}.jpg"

                        img_filepath = os.path.join(article_path, img_name)
                        with open(img_filepath, 'wb') as f:
                            for chunk in img_response.iter_content(1024):
                                f.write(chunk)
                        logging.info(f"图片已保存: {img_filepath}")
                    else:
                        logging.warning(f"下载图片失败，状态码: {img_response.status_code}, URL: {img_url}")
                except Exception as e:
                    logging.error(f"下载图片时出错 {img_url}: {e}")
        except Exception as e:
            logging.error(f"保存文章到文件夹失败: {e}")

    def get_current_page_number(self):
        """获取当前页码"""
        try:
            current_page_elem = self.driver.find_element(By.CSS_SELECTOR, "a.js_pageI.cur")
            return int(current_page_elem.text)
        except:
            return 1

    def click_next_page(self):
        """
        针对海尔网站特定分页结构的翻页逻辑
        基于HTML结构：<a href="javascript:;" class="next">下一页</a>
        """
        try:
            current_page = self.get_current_page_number()
            expected_next_page = current_page + 1

            logging.info(f"当前页码: {current_page}, 尝试跳转到页码: {expected_next_page}")

            # 记录跳转前的文章信息用于验证
            old_articles = self.get_page_articles_for_comparison()
            old_url = self.driver.current_url

            # 查找"下一页"按钮
            next_button = None
            try:
                # 根据HTML结构，查找class="next"的按钮
                next_button = self.driver.find_element(By.CSS_SELECTOR, "a.next")

                # 检查按钮状态 - 如果没有"lose"类，说明可以点击
                button_classes = next_button.get_attribute("class")
                if "lose" in button_classes:
                    logging.info("已到达最后一页（按钮包含'lose'类）")
                    return False

                logging.info(f"找到下一页按钮，class: {button_classes}")

            except Exception as e:
                logging.error(f"未找到下一页按钮: {e}")
                return False

            # 确保按钮在视口中
            try:
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});",
                                           next_button)
                time.sleep(1.5)  # 等待滚动完成
            except Exception as e:
                logging.warning(f"滚动到按钮位置失败: {e}")

            # 尝试多种点击方式
            click_success = False

            # 方法1: 等待按钮可点击后使用ActionChains
            try:
                wait = WebDriverWait(self.driver, 10)
                wait.until(EC.element_to_be_clickable(next_button))

                # 使用ActionChains模拟真实用户点击
                actions = ActionChains(self.driver)
                actions.move_to_element(next_button).pause(0.5).click().perform()

                logging.info("方法1: ActionChains点击成功")
                click_success = True

            except Exception as e:
                logging.warning(f"方法1: ActionChains点击失败: {e}")

                # 方法2: 直接点击
                try:
                    next_button.click()
                    logging.info("方法2: 直接点击成功")
                    click_success = True

                except Exception as e2:
                    logging.warning(f"方法2: 直接点击失败: {e2}")

                    # 方法3: JavaScript点击
                    try:
                        self.driver.execute_script("arguments[0].click();", next_button)
                        logging.info("方法3: JavaScript点击成功")
                        click_success = True

                    except Exception as e3:
                        logging.warning(f"方法3: JavaScript点击失败: {e3}")

                        # 方法4: 触发JavaScript事件
                        try:
                            # 由于href是"javascript:;"，尝试直接触发onclick事件
                            self.driver.execute_script("""
                                var element = arguments[0];
                                var event = new MouseEvent('click', {
                                    view: window,
                                    bubbles: true,
                                    cancelable: true
                                });
                                element.dispatchEvent(event);
                            """, next_button)
                            logging.info("方法4: 触发click事件成功")
                            click_success = True

                        except Exception as e4:
                            logging.error(f"方法4: 触发事件失败: {e4}")

            if not click_success:
                logging.error("所有点击方法都失败了")
                return False

            # 等待页面更新
            return self.wait_for_page_update(expected_next_page, old_articles, old_url)

        except Exception as e:
            logging.error(f"翻页操作失败: {e}")
            return False

    def get_page_articles_for_comparison(self):
        """获取当前页面的文章信息用于比较"""
        try:
            articles = []
            news_items = self.driver.find_elements(By.CSS_SELECTOR, ".js_newslist li.item")
            for item in news_items[:3]:  # 只取前3个用于比较
                try:
                    title_elem = item.find_element(By.CSS_SELECTOR, "a.title")
                    articles.append(title_elem.text.strip())
                except:
                    pass
            return articles
        except:
            return []

    def wait_for_page_update(self, expected_page_number, old_articles, old_url, timeout=25):
        """
        专门针对海尔网站AJAX翻页的等待逻辑
        """
        try:
            logging.info(f"等待页面更新到第 {expected_page_number} 页...")

            # 第一阶段：等待AJAX请求完成（短暂等待）
            time.sleep(2)

            # 第二阶段：循环检测页面变化
            max_attempts = 12  # 最多检测12次，每次2秒

            for attempt in range(max_attempts):
                try:
                    # 检测方法1：检查页码高亮状态
                    try:
                        current_active = self.driver.find_element(By.CSS_SELECTOR, "a.js_pageI.cur")
                        current_page_text = current_active.text.strip()
                        if current_page_text == str(expected_page_number):
                            logging.info(f"✓ 页码高亮检测成功：第 {expected_page_number} 页")
                            return True
                    except Exception as e:
                        logging.debug(f"页码高亮检测失败: {e}")

                    # 检测方法2：检查新闻列表内容是否变化
                    try:
                        # 等待新闻列表重新加载
                        WebDriverWait(self.driver, 3).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".js_newslist li.item"))
                        )

                        new_articles = self.get_page_articles_for_comparison()
                        if new_articles and new_articles != old_articles:
                            logging.info("✓ 新闻内容变化检测成功")
                            # 再次确认页码
                            final_page = self.get_current_page_number()
                            if final_page == expected_page_number:
                                logging.info(f"✓ 页码确认成功：第 {final_page} 页")
                                return True
                            else:
                                logging.info(f"内容已变化但页码不匹配，当前页码: {final_page}")
                    except Exception as e:
                        logging.debug(f"内容变化检测失败: {e}")

                    # 检测方法3：检查URL变化（某些网站会改变URL参数）
                    try:
                        current_url = self.driver.current_url
                        if current_url != old_url:
                            logging.info(f"✓ URL变化检测成功: {current_url}")
                            return True
                    except Exception as e:
                        logging.debug(f"URL变化检测失败: {e}")

                    # 检测方法4：检查分页器状态变化
                    try:
                        # 检查当前页码按钮是否有cur类
                        expected_current = self.driver.find_element(
                            By.XPATH, f"//a[contains(@class, 'js_pageI') and text()='{expected_page_number}']"
                        )
                        if "cur" in expected_current.get_attribute("class"):
                            logging.info(f"✓ 分页器状态检测成功：第 {expected_page_number} 页已激活")
                            return True
                    except Exception as e:
                        logging.debug(f"分页器状态检测失败: {e}")

                    logging.info(f"第 {attempt + 1}/{max_attempts} 次检测：页面更新中...")
                    time.sleep(2)

                except Exception as e:
                    logging.warning(f"第 {attempt + 1} 次检测出错: {e}")
                    time.sleep(2)

            # 最终检查：即使检测失败，也尝试获取当前状态
            try:
                final_page = self.get_current_page_number()
                final_articles = self.get_page_articles_for_comparison()

                logging.warning(f"所有检测超时，最终状态检查：")
                logging.warning(f"  - 当前页码: {final_page} (期望: {expected_page_number})")
                logging.warning(f"  - 内容是否变化: {final_articles != old_articles}")

                # 如果页码正确或内容发生了变化，仍然认为翻页成功
                if final_page == expected_page_number or (final_articles and final_articles != old_articles):
                    logging.info("✓ 最终检查判定翻页成功")
                    return True

            except Exception as e:
                logging.error(f"最终状态检查失败: {e}")

            logging.warning(f"⚠ 翻页可能失败，当前页码可能仍为: {self.get_current_page_number()}")
            return False

        except Exception as e:
            logging.error(f"等待页面更新时发生错误: {e}")
            return False

    def run(self, max_pages=5):
        """运行爬虫"""
        try:
            self.setup_driver()
            self.driver.get(self.base_url)

            # 等待页面完全加载
            time.sleep(3)

            logging.info(f"开始爬取海尔资讯，最多爬取 {max_pages} 页")
            page_count = 0
            total_articles_processed = 0
            processed_urls = set()

            while page_count < max_pages:
                page_count += 1
                current_page = self.get_current_page_number()
                logging.info(f"正在爬取第 {page_count} 页（实际页码: {current_page}）")

                # 等待新闻列表出现
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".js_newslist li.item"))
                    )
                    time.sleep(2)  # 额外等待确保内容完全加载
                except Exception as e:
                    logging.warning(f"未能找到新闻条目元素，可能已结束: {e}")
                    break

                news_links = self.get_news_links_from_page()

                if not news_links:
                    logging.warning("当前页面没有找到资讯链接")
                    # 尝试继续翻页，可能是动态加载问题
                    if page_count >= 2:  # 如果不是第一页且找不到内容，则停止
                        break

                new_articles_found_on_page = 0
                for i, news_item in enumerate(news_links, 1):
                    if news_item['url'] in processed_urls:
                        logging.info(f"跳过已处理的资讯: {news_item['title']}")
                        continue

                    new_articles_found_on_page += 1
                    logging.info(f"正在处理第 {page_count} 页的新资讯 {i}/{len(news_links)}: {news_item['title']}")

                    article_details = self.get_news_content(news_item['url'])

                    if article_details:
                        self.save_article_to_folder(article_details)
                        total_articles_processed += 1
                        processed_urls.add(news_item['url'])

                    time.sleep(0.5)

                logging.info(f"第 {page_count} 页处理完成，新处理文章数: {new_articles_found_on_page}")

                if new_articles_found_on_page == 0 and page_count > 1:
                    logging.info("页面上没有发现新的文章，可能已到达末尾。")
                    break

                # 尝试翻页
                if page_count < max_pages:
                    logging.info(f"准备翻页到第 {page_count + 1} 页")

                    if not self.click_next_page():
                        logging.info("翻页失败，爬取结束")
                        break

                    time.sleep(1)  # 翻页后短暂等待

            logging.info(f"爬取完成，共处理 {total_articles_processed} 条资讯")

        except Exception as e:
            logging.error(f"爬虫运行出错: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                logging.info("浏览器已关闭")
            return total_articles_processed


def main():
    """主函数"""
    scraper = HaierNewsScraper()

    try:
        max_pages_input = input("请输入要爬取的最大页数 (默认5): ")
        max_pages = int(max_pages_input) if max_pages_input.isdigit() else 5
    except ValueError:
        max_pages = 5
        print("输入无效，将使用默认值5页。")

    total_count = scraper.run(max_pages)

    print("\n-------------------------------------------")
    print(f"爬取任务完成！共获取并保存了 {total_count} 条资讯。")
    print(f"数据已保存在脚本所在目录下的 '{scraper.output_dir}' 文件夹中。")
    print("-------------------------------------------")


if __name__ == "__main__":
    main()