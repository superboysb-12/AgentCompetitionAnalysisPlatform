import os
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from scrape_sub_products import scrape_all_sub_products, save_sub_product_tables_as_csv # 导入 save_sub_product_tables_as_csv


BASE_URL = "https://www.daikin-china.com.cn/newha/products/"
OUTPUT_DIR = "daikin_product"
DEBUG_DETAIL_PAGES_DIR = os.path.join(OUTPUT_DIR, "debug_detail_pages")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DETAIL_PAGES_DIR, exist_ok=True)


def initialize_driver():
    """初始化Chrome WebDriver，增强稳定性配置"""
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-crash-reporter")
    options.add_argument("--disable-in-process-stack-traces")
    options.add_argument("--disable-logging")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-backgrounding-occluded-windows")

    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.6478.183 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(60)
    driver.implicitly_wait(3)

    return driver


def safe_get_url(driver, url, max_retries=3):
    """安全地加载URL，带重试机制"""
    for attempt in range(max_retries):
        try:
            print(f"正在加载URL (尝试 {attempt + 1}/{max_retries}): {url}")
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)
            return True
        except TimeoutException:
            print(f"加载URL超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                try:
                    driver.execute_script("window.stop();")
                except:
                    pass
                time.sleep(2)
        except WebDriverException as e:
            print(f"WebDriver错误: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return False
        except Exception as e:
            print(f"加载URL时发生错误: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
    return False


def safe_click(driver, element, element_name="element"):
    """尝试常规点击，失败则尝试JavaScript点击"""
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable(element))
        element.click()
        return True
    except StaleElementReferenceException:
        print(f"警告: {element_name} 元素已过时，尝试重新查找并点击...")
        # 实际应用中，这里需要根据情况重新定位元素，如果只是一个简单的tab点击，JS点击可能足够
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception as js_e:
            print(f"JS点击 {element_name} (处理过时元素) 失败: {js_e}")
            return False
    except Exception as e:
        print(f"常规点击 {element_name} 失败: {e}. 尝试JS点击...")
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception as js_e:
            print(f"JS点击 {element_name} 失败: {js_e}")
            return False


def close_extra_windows(driver, main_handle):
    """关闭所有非主窗口"""
    try:
        current_handles = driver.window_handles
        if len(current_handles) > 1:
            for handle in current_handles:
                if handle != main_handle:
                    driver.switch_to.window(handle)
                    driver.close()
            driver.switch_to.window(main_handle)
            return True
        else:
            return True
    except Exception as e:
        print(f"关闭额外窗口时出错: {e}")
        try:
            driver.switch_to.window(main_handle)
        except:
            pass
        return False



def get_product_image_url(driver):
    """尝试获取产品海报URL"""
    selectors = [
        (By.CSS_SELECTOR, "#tab-0 img"),
        (By.CSS_SELECTOR, ".product-image img"),
        (By.CSS_SELECTOR, ".poster img"),
    ]

    for selector in selectors:
        try:
            img_element = WebDriverWait(driver, 3).until(EC.presence_of_element_located(selector))
            url = img_element.get_attribute("src")
            if url:
                return url
        except:
            continue

    return None


def get_all_product_introductions(driver):
    """获取所有产品简介"""
    introductions = []
    selectors = [
        (By.CSS_SELECTOR, "div.pageContainer.products div.contentView div.right div.swiperBox div.info div.label p"),
        (By.CSS_SELECTOR, "div.pageContainer.products div.contentView div.right div.swiperBox div.info div.label"),
        (By.CSS_SELECTOR, ".product-intro p"),
        (By.CSS_SELECTOR, ".product-intro"),
    ]

    for selector in selectors:
        try:
            elements = driver.find_elements(*selector)
            for elem in elements:
                text = elem.text.strip()
                if text and len(text) > 5 and text not in introductions:
                    introductions.append(text)
            if introductions:
                break
        except:
            continue

    return introductions if introductions else ["N/A"]


def get_all_product_features(driver):
    """获取所有产品特点"""
    features_container_selector = (
        By.CSS_SELECTOR,
        "div.pageContainer.products div.contentView div.right div.swiperBox div.info div.intro"
    )
    features_list = []

    try:
        features_container = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(features_container_selector)
        )
        full_text = features_container.text.strip()

        if full_text:
            cleaned_features = [line.strip() for line in full_text.split('\n') if
                                line.strip() and len(line.strip()) > 3]
            if cleaned_features:
                features_list.extend(cleaned_features)
            else:
                features_list.append(full_text)
    except TimeoutException:
        pass
    except Exception as e:
        print(f"获取产品特点时发生错误: {e}")

    return features_list if features_list else ["N/A"]


def scrape_single_product(driver, product_card, index, total, main_window_handle):
    """抓取单个产品系列及其子产品的详情"""
    product_card_name = "未知产品"
    # 从产品卡片上获取名称
    name_selectors = [
        (By.CSS_SELECTOR, "div.name"),
        (By.CSS_SELECTOR, "h3.product-title"),
        (By.CSS_SELECTOR, ".product-info .title"),
    ]
    for selector in name_selectors:
        try:
            name_element = product_card.find_element(*selector)
            product_card_name = name_element.text.strip()
            if product_card_name:
                break
        except:
            pass

    print(f"\n{'=' * 60}")
    print(f"处理第 {index + 1}/{total} 个产品系列: '{product_card_name}'")
    print(f"{'=' * 60}")

    button_selectors = [
        (By.CSS_SELECTOR, "div > div.button > button:nth-child(1)"),
        (By.CSS_SELECTOR, ".button button"),
        (By.XPATH, ".//button[contains(text(), '了解详情')]"),
    ]

    learn_more_button = None
    for selector in button_selectors:
        try:
            learn_more_button = product_card.find_element(*selector)
            if learn_more_button and learn_more_button.is_displayed() and learn_more_button.is_enabled():
                break
        except:
            pass

    if not learn_more_button:
        print(f"无法找到'{product_card_name}'的'了解详情'按钮，跳过")
        return False

    original_windows = set(driver.window_handles)

    if not safe_click(driver, learn_more_button, f"'{product_card_name}'的'了解详情'按钮"):
        print(f"点击失败，跳过")
        return False

    new_tab_opened = False
    for _ in range(10): # 等待新标签页打开
        time.sleep(1)
        current_windows = set(driver.window_handles)
        if len(current_windows) > len(original_windows):
            new_tab_opened = True
            break

    if not new_tab_opened:
        print(f"新标签页未打开，跳过")
        return False

    new_window = (set(driver.window_handles) - original_windows).pop()
    driver.switch_to.window(new_window)

    # 等待详情页加载
    detail_loaded = False
    detail_name_selectors = [
        (By.CSS_SELECTOR, "div.pageContainer.products div.contentView div.right div.swiperBox div.info div.name"),
        (By.CSS_SELECTOR, ".info .name"),
        (By.CSS_SELECTOR, ".product-detail"),
    ]

    for selector in detail_name_selectors:
        try:
            WebDriverWait(driver, 15).until(EC.presence_of_element_located(selector))
            detail_loaded = True
            break
        except TimeoutException:
            continue
        except Exception as e:
            print(f"等待详情页元素时发生错误: {e}")
            continue

    if not detail_loaded:
        print("详情页加载失败或关键元素未出现")
        driver.close()
        driver.switch_to.window(main_window_handle)
        return False

    time.sleep(3)

    detail_data = {
        "产品名称": product_card_name,
        "详情页URL": driver.current_url
    }

    for selector in detail_name_selectors:
        try:
            detail_name_elem = driver.find_element(*selector)
            detail_name = detail_name_elem.text.strip()
            if detail_name and len(detail_name) > len(product_card_name):
                detail_data["产品名称"] = detail_name
                print(f"详情页产品名称 (已更新): {detail_name}")
                break
        except:
            pass

    # 抓取主产品系列信息
    detail_data["产品简介"] = get_all_product_introductions(driver)
    detail_data["产品特点"] = get_all_product_features(driver)
    detail_data["产品海报URL"] = get_product_image_url(driver)

    # 使用独立模块抓取子产品信息
    detail_data["子产品"] = scrape_all_sub_products(driver)

    safe_json_name = "".join(c for c in detail_data["产品名称"] if c.isalnum() or c in (' ', '_', '-', '+', '.'))
    if not safe_json_name or safe_json_name == "未知产品":
        safe_json_name = f"product_detail_{index}"

    product_json_dir = os.path.join(OUTPUT_DIR, safe_json_name)
    os.makedirs(product_json_dir, exist_ok=True)
    json_file_path = os.path.join(product_json_dir, f"{safe_json_name}.json")

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(detail_data, f, ensure_ascii=False, indent=4)
    print(f"✓ 产品系列信息已保存到: {json_file_path}")

    save_sub_product_tables_as_csv(detail_data, OUTPUT_DIR)


    driver.close()
    driver.switch_to.window(main_window_handle)
    time.sleep(1)

    return True


def scrape_products(driver):
    """主抓取函数"""
    if not safe_get_url(driver, BASE_URL):
        print("无法加载产品列表页，退出")
        return

    main_window_handle = driver.current_window_handle

    PRODUCT_LIST_CONTAINER_SELECTOR = (
        By.CSS_SELECTOR,
        "#app2 > div > div > div.pageContainer.products > div.slotBox_ > div.slotBox_.contentView > div.right > div"
    )

    try:
        WebDriverWait(driver, 30).until(EC.presence_of_element_located(PRODUCT_LIST_CONTAINER_SELECTOR))
    except TimeoutException:
        print("产品列表容器加载超时")
        return

    # 获取产品系列的总数，用于进度显示
    try:
        product_list_container = driver.find_element(*PRODUCT_LIST_CONTAINER_SELECTOR)
        total_product_cards_on_page = product_list_container.find_elements(By.XPATH, "./div")
        product_count = len(total_product_cards_on_page)
        print(f"找到 {product_count} 个产品系列")
    except NoSuchElementException:
        print("未找到产品系列容器或产品卡片。")
        return

    success_count = 0
    fail_count = 0

    for i in range(product_count):
        try:
            # 每次循环都重新查找产品卡片列表，以应对 StaleElementReferenceException
            product_list_container = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(PRODUCT_LIST_CONTAINER_SELECTOR)
            )
            product_cards = product_list_container.find_elements(By.XPATH, "./div")

            if i >= len(product_cards):
                print(f"警告: 索引 {i} 超出当前产品列表范围，可能页面内容已变化或元素过时。跳过。")
                fail_count += 1
                continue

            current_product_card = product_cards[i]

            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", current_product_card)
            WebDriverWait(current_product_card, 5).until(
                EC.presence_of_element_located((By.XPATH, ".//button[contains(text(), '了解详情')]"))
            )


            if scrape_single_product(driver, current_product_card, i, product_count, main_window_handle):
                success_count += 1
            else:
                fail_count += 1

            # 确保每次循环结束都返回主窗口，并处理可能残留的额外窗口
            if driver.current_window_handle != main_window_handle:
                close_extra_windows(driver, main_window_handle)

            time.sleep(2)

        except StaleElementReferenceException:
            print(f"✗ 处理第 {i + 1} 个产品系列时发生 StaleElementReferenceException，尝试重新加载列表页并继续。")
            fail_count += 1
            # 尝试重新加载列表页
            try:
                close_extra_windows(driver, main_window_handle)
                if not safe_get_url(driver, BASE_URL):
                    print("无法重新加载列表页，终止爬取")
                    break
                main_window_handle = driver.current_window_handle # 更新主窗口句柄
                WebDriverWait(driver, 15).until(EC.presence_of_element_located(PRODUCT_LIST_CONTAINER_SELECTOR))
                # 由于重新加载了页面，i 需要保持当前值，下一次循环会重新获取产品卡片
            except Exception as recovery_error:
                print(f"StaleElementReferenceException 恢复失败: {recovery_error}")
                break
        except Exception as e:
            print(f"✗ 处理第 {i + 1} 个产品系列时发生错误: {e}")
            fail_count += 1

            try:
                print("尝试恢复爬取...")
                close_extra_windows(driver, main_window_handle)
                if not safe_get_url(driver, BASE_URL):
                    print("无法重新加载列表页，终止爬取")
                    break
                main_window_handle = driver.current_window_handle # 更新主窗口句柄
                WebDriverWait(driver, 15).until(EC.presence_of_element_located(PRODUCT_LIST_CONTAINER_SELECTOR))
                print("已恢复到产品列表页。")
            except Exception as recovery_error:
                print(f"恢复失败: {recovery_error}")
                break

    print(f"\n{'=' * 60}")
    print(f"爬取完成!")
    print(f"成功: {success_count} 个产品系列")
    print(f"失败: {fail_count} 个产品系列")
    print(f"总计: {product_count} 个产品系列")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    driver = None
    try:
        driver = initialize_driver()
        scrape_products(driver)
    except Exception as e:
        print(f"脚本执行过程中发生致命错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if driver:
            print("关闭浏览器...")
            try:
                driver.quit()
            except:
                pass