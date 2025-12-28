import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json
import time

# 导入 Selenium 相关模块
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
# 主页 URL
base_url = "https://www.gree.com"
news_list_url = urljoin(base_url, "/about/news")

# 主保存目录
base_save_dir = "gree_news"
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)


def get_all_news_links_with_selenium(url, max_clicks=None):
    """
    使用 Selenium 自动点击"查看更多"并获取所有新闻链接。
    max_clicks: 限制点击"查看更多"按钮的次数。
    """
    print("正在使用 Selenium 自动加载所有新闻...")
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # 启用无头模式，不显示浏览器界面
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # 自动管理驱动
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    links = set()  # 使用集合来存储链接，自动去重
    click_count = 0

    try:
        driver.get(url)
        print("页面加载中，等待初始内容...")

        # 等待页面加载
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "newInfo-list-inner"))
        )
        print("初始页面加载完成。")

        # 添加一个初始等待，确保页面完全渲染
        time.sleep(2)

        while True:
            # 检查是否达到最大点击次数
            if max_clicks is not None and click_count >= max_clicks:
                print(f"已达到最大点击次数 {max_clicks}，停止加载更多新闻。")
                break

            try:
                # 1. 点击前，先获取当前新闻数量和所有链接
                news_elements = driver.find_elements(By.XPATH, "//div[@id='newInfo-list-inner']//li")
                initial_count = len(news_elements)
                print(f"点击前新闻数量: {initial_count}")

                # 收集当前已有的链接
                current_links = set()
                for li in news_elements:
                    a_tags = li.find_elements(By.TAG_NAME, "a")
                    for a_tag in a_tags:
                        href = a_tag.get_attribute("href")
                        if href:
                            current_links.add(href)

                # 2. 查找"查看更多"按钮
                more_buttons = driver.find_elements(By.CLASS_NAME, "view-more")
                if not more_buttons:
                    print("未找到查看更多按钮，可能所有内容已加载完毕。")
                    break

                more_button = more_buttons[0]

                # 检查按钮是否可见和可点击
                if not more_button.is_displayed():
                    print("查看更多按钮不可见，可能所有内容已加载完毕。")
                    break

                # 滚动到按钮位置
                driver.execute_script("arguments[0].scrollIntoView(true);", more_button)
                time.sleep(1)

                print(f"第 {click_count + 1} 次点击查看更多按钮...")

                # 3. 点击按钮
                try:
                    # 首先尝试普通点击
                    more_button.click()
                except:
                    # 如果普通点击失败，使用 JavaScript 点击
                    driver.execute_script("arguments[0].click();", more_button)

                click_count += 1
                print(f"点击成功，等待新内容加载...")

                # 4. 等待新内容加载 - 使用多种策略
                load_success = False

                # 策略1: 等待新闻数量增加
                try:
                    WebDriverWait(driver, 8).until(
                        lambda d: len(d.find_elements(By.XPATH, "//div[@id='newInfo-list-inner']//li")) > initial_count
                    )
                    load_success = True
                    print("检测到新闻数量增加")
                except TimeoutException:
                    print("等待新闻数量增加超时")

                # 策略2: 等待新链接出现
                if not load_success:
                    try:
                        def new_links_appeared(driver):
                            current_elements = driver.find_elements(By.XPATH, "//div[@id='newInfo-list-inner']//li")
                            new_links = set()
                            for li in current_elements:
                                a_tags = li.find_elements(By.TAG_NAME, "a")
                                for a_tag in a_tags:
                                    href = a_tag.get_attribute("href")
                                    if href:
                                        new_links.add(href)
                            return len(new_links) > len(current_links)

                        WebDriverWait(driver, 8).until(new_links_appeared)
                        load_success = True
                        print("检测到新链接出现")
                    except TimeoutException:
                        print("等待新链接出现超时")

                # 策略3: 简单的时间等待
                if not load_success:
                    print("使用时间等待策略...")
                    time.sleep(3)

                # 更新后的新闻数量
                updated_news_elements = driver.find_elements(By.XPATH, "//div[@id='newInfo-list-inner']//li")
                updated_count = len(updated_news_elements)
                print(f"更新后新闻数量: {updated_count}")

                # 如果数量没有增加，可能已经到底了
                if updated_count <= initial_count:
                    print("新闻数量未增加，可能已经加载完所有内容。")
                    # 但仍然继续尝试，以防页面有延迟
                    time.sleep(2)
                    final_elements = driver.find_elements(By.XPATH, "//div[@id='newInfo-list-inner']//li")
                    if len(final_elements) <= initial_count:
                        print("确认没有新内容，停止加载。")
                        break

            except (TimeoutException, NoSuchElementException) as e:
                print(f"查找或点击按钮时出现异常: {e}")
                print("可能已经没有更多内容了。")
                break
            except WebDriverException as e:
                print(f"Selenium 操作出错: {e}")
                print("尝试继续...")
                continue

        # 最终提取所有新闻链接
        print("正在提取所有新闻链接...")
        try:
            all_news_ul = driver.find_element(By.ID, "newInfo-list-inner")
            a_tags = all_news_ul.find_elements(By.TAG_NAME, "a")

            for a_tag in a_tags:
                href = a_tag.get_attribute("href")
                if href and href.strip():
                    full_url = urljoin(base_url, href)
                    links.add(full_url)

            print(f"总共找到 {len(links)} 个新闻链接。")

        except Exception as e:
            print(f"提取链接时出错: {e}")

    except Exception as e:
        print(f"Selenium 爬取过程中出错: {e}")

    finally:
        print("关闭浏览器...")
        driver.quit()  # 确保浏览器关闭

    return list(links)


def download_image(image_url, save_dir, index):
    """下载图片并保存到指定目录"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()

        # 获取文件扩展名
        file_name = os.path.basename(image_url.split('?')[0])
        if not file_name or '.' not in file_name:
            # 尝试从Content-Type获取扩展名
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            else:
                ext = '.jpg'  # 默认
            file_name = f"image_{index}{ext}"

        save_path = os.path.join(save_dir, file_name)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"图片下载成功: {save_path}")
        return file_name

    except requests.exceptions.RequestException as e:
        print(f"图片下载失败: {image_url}, 错误: {e}")
        return None
    except Exception as e:
        print(f"图片处理时发生未知错误: {e}")
        return None


def get_news_details(url):
    """爬取单个新闻详情页的内容并保存为JSON"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        content_div = soup.find('div', id='gree-newsDetails')
        if not content_div:
            print(f"未能找到新闻详情内容 div: {url}")
            return None

        # 获取标题
        title = content_div.find('div', class_='newsDetails-title')
        title_text = title.get_text(strip=True) if title else "无标题"
        # 清理文件名，移除特殊字符
        safe_title = "".join(c for c in title_text if c.isalnum() or c in (' ', '_', '-')).strip()
        if len(safe_title) > 100:  # 限制文件名长度
            safe_title = safe_title[:100]
        if not safe_title:
            safe_title = f"news_{int(time.time())}"

        # 创建新闻目录
        news_dir = os.path.join(base_save_dir, safe_title)
        if not os.path.exists(news_dir):
            os.makedirs(news_dir)

        # 获取发布时间
        time_tag = content_div.find('div', class_='newsDetails-time')
        time_text = time_tag.get_text(strip=True) if time_tag else "无发布时间"

        # 获取正文内容
        info_div = content_div.find('div', class_='newsDetails-information')
        body_text = info_div.get_text(separator='\n', strip=True) if info_div else "无正文"

        # 下载图片
        image_urls = []
        downloaded_image_names = []
        images = content_div.find_all('img')

        for index, img_tag in enumerate(images):
            src = img_tag.get('src')
            if src:
                full_src = urljoin(base_url, src)
                image_urls.append(full_src)
                print(f"正在下载图片 {index + 1}/{len(images)}: {full_src}")

                image_name = download_image(full_src, news_dir, index)
                if image_name:
                    downloaded_image_names.append(os.path.join(safe_title, image_name))

        # 构建新闻数据
        news_data = {
            "title": title_text,
            "publish_time": time_text,
            "content": body_text,
            "images": downloaded_image_names,
            "image_urls": image_urls,
            "source_url": url,
            "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存为JSON文件
        json_file_path = os.path.join(news_dir, f"{safe_title}.json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(news_data, f, ensure_ascii=False, indent=4)
        print(f"新闻内容已保存到: {json_file_path}")
        print("-" * 50)

        return news_data

    except requests.exceptions.RequestException as e:
        print(f"请求新闻详情页时出错: {url}, 错误: {e}")
        return None
    except Exception as e:
        print(f"处理新闻详情时发生未知错误: {url}, 错误: {e}")
        return None


def get_all_news_links_with_selenium_improved(url, max_clicks=None):
    """
    改进版本：使用 Selenium 自动点击"查看更多"并获取所有新闻链接。
    正确处理初始列表和动态加载的列表。
    """
    print("正在使用 Selenium 自动加载所有新闻...")
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # 可以启用无头模式
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    links = set()
    click_count = 0

    def get_all_current_news_count():
        """获取当前页面所有新闻的总数（包括初始和动态加载的）"""
        # 初始新闻列表
        initial_elements = driver.find_elements(By.XPATH, "//ul[@id='newInfo-list-inner']//li")
        # 动态加载的新闻列表
        more_elements = driver.find_elements(By.XPATH, "//ul[@id='newInfo-list-more']//li")
        total_count = len(initial_elements) + len(more_elements)
        return total_count, len(initial_elements), len(more_elements)

    try:
        driver.get(url)
        print("页面加载中...")

        # 等待页面初始加载
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "newInfo-list-inner"))
        )
        print("初始页面加载完成。")
        time.sleep(3)  # 给页面更多时间来完全渲染

        # 获取初始状态
        total_before, initial_before, more_before = get_all_current_news_count()
        print(f"初始状态 - 初始列表: {initial_before}, 动态列表: {more_before}, 总计: {total_before}")

        while True:
            if max_clicks is not None and click_count >= max_clicks:
                print(f"已达到最大点击次数 {max_clicks}，停止加载更多新闻。")
                break

            # 查找"查看更多"按钮
            more_buttons = driver.find_elements(By.CLASS_NAME, "view-more")

            if not more_buttons:
                print("未找到查看更多按钮，停止加载。")
                break

            more_button = more_buttons[0]

            # 检查按钮状态
            if not more_button.is_displayed() or not more_button.is_enabled():
                print("查看更多按钮不可用，停止加载。")
                break

            # 检查按钮文本，如果显示"没有更多"之类的，则停止
            button_text = more_button.text.strip()
            if any(keyword in button_text for keyword in ['没有更多', '无更多', '已全部加载']):
                print(f"按钮显示{button_text}，停止加载。")
                break

            try:
                # 滚动到按钮位置
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                                      more_button)
                time.sleep(2)

                print(f"第 {click_count + 1} 次点击查看更多按钮...")

                # 点击按钮
                driver.execute_script("arguments[0].click();", more_button)
                click_count += 1

                # 等待新内容加载 - 检查 newInfo-list-more 容器
                print("等待新内容加载...")
                load_success = False
                max_wait_time = 15
                wait_interval = 1
                waited_time = 0

                while waited_time < max_wait_time:
                    time.sleep(wait_interval)
                    waited_time += wait_interval

                    # 检查总新闻数量是否增加
                    total_current, initial_current, more_current = get_all_current_news_count()

                    if total_current > total_before:
                        print(f"新内容已加载！")
                        print(f"  初始列表: {initial_current} (之前: {initial_before})")
                        print(f"  动态列表: {more_current} (之前: {more_before})")
                        print(f"  总计: {total_current} (增加了 {total_current - total_before} 条)")
                        load_success = True
                        total_before = total_current  # 更新基准值
                        break

                    print(f"等待新内容加载... ({waited_time}s) - 总数: {total_current}")

                if not load_success:
                    print("等待超时，新内容可能没有加载，或者已经没有更多内容。")
                    # 再次检查，可能是网络延迟
                    time.sleep(3)
                    final_total, final_initial, final_more = get_all_current_news_count()
                    if final_total <= total_before:
                        print("确认没有新内容加载，停止点击。")
                        break
                    else:
                        print(f"延迟检测到新内容: {final_total} 条")
                        total_before = final_total

            except Exception as e:
                print(f"点击按钮时出现异常: {e}")
                break

        # 最终提取所有链接（从两个容器中）
        print("正在提取所有新闻链接...")
        try:
            # 从初始列表提取链接
            try:
                initial_ul = driver.find_element(By.ID, "newInfo-list-inner")
                initial_li_elements = initial_ul.find_elements(By.TAG_NAME, "li")
                print(f"从初始列表找到 {len(initial_li_elements)} 个新闻项")

                for li in initial_li_elements:
                    a_tags = li.find_elements(By.TAG_NAME, "a")
                    for a_tag in a_tags:
                        href = a_tag.get_attribute("href")
                        if href and href.strip():
                            full_url = urljoin(base_url, href)
                            links.add(full_url)
            except NoSuchElementException:
                print("未找到初始新闻列表")

            # 从动态加载列表提取链接
            try:
                more_ul = driver.find_element(By.ID, "newInfo-list-more")
                more_li_elements = more_ul.find_elements(By.TAG_NAME, "li")
                print(f"从动态列表找到 {len(more_li_elements)} 个新闻项")

                for li in more_li_elements:
                    a_tags = li.find_elements(By.TAG_NAME, "a")
                    for a_tag in a_tags:
                        href = a_tag.get_attribute("href")
                        if href and href.strip():
                            full_url = urljoin(base_url, href)
                            links.add(full_url)
            except NoSuchElementException:
                print("未找到动态新闻列表（可能没有点击查看更多）")

            print(f"总共找到 {len(links)} 个独特的新闻链接。")

        except Exception as e:
            print(f"提取链接时出错: {e}")

    except Exception as e:
        print(f"Selenium 爬取过程中出现严重错误: {e}")

    finally:
        print("关闭浏览器...")
        try:
            driver.quit()
        except:
            pass

    return list(links)


# --- 主程序 ---
if __name__ == "__main__":
    # 设置期望的点击次数
    desired_clicks = 10

    print(f"开始爬取格力新闻，最多点击 {desired_clicks} 次查看更多按钮...")
    print("=" * 60)

    # 使用改进版本的函数
    all_links = get_all_news_links_with_selenium_improved(news_list_url, max_clicks=desired_clicks)

    if all_links:
        print("=" * 60)
        print(f"开始爬取 {len(all_links)} 篇新闻详情页...")

        success_count = 0
        for i, link in enumerate(all_links, 1):
            print(f"\n处理第 {i}/{len(all_links)} 篇新闻: {link}")
            result = get_news_details(link)
            if result:
                success_count += 1

            # 添加延迟避免请求过于频繁
            if i < len(all_links):
                time.sleep(1)

        print("=" * 60)
        print(f"爬取完成！成功处理 {success_count}/{len(all_links)} 篇新闻。")
    else:
        print("未找到任何新闻链接。")