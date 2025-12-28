import time
import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException


def _safe_click(driver, element, element_name="element"):
    """尝试常规点击，失败则尝试JavaScript点击"""
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable(element))
        element.click()
        return True
    except StaleElementReferenceException:
        print(f"警告: {element_name} 元素已过时，尝试重新查找并点击...")
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


def _get_available_tabs(driver):
    """
    检测产品详情页中实际存在的tab选项卡，动态确定tab ID
    Returns:
        list: 包含可用tab信息的列表，每个元素是字典 {'id': 'tab-X', 'name': '室内机/室外机/选配件', 'element': WebElement}
    """
    available_tabs = []

    tab_container_selectors = [
        "#app2 > div > div > div.pageContainer.products > div.slotBox_ > div.slotBox_.contentView > div.right > div > div.tabBox.tabBox-fixed > div.tab",
        "div.tabBox.tabBox-fixed > div.tab",
        "div.right > div > div.tabBox > div.tab",
        "#app2 div.tabBox.tabBox-fixed div.tab",
        "div.contentView div.tabBox div.tab"
    ]

    tab_container = None
    for selector in tab_container_selectors:
        try:
            tab_container = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            break
        except TimeoutException:
            continue

    if not tab_container:
        print("所有tab容器选择器均未找到元素")
        return []

    tab_items = tab_container.find_elements(By.CSS_SELECTOR, "div.item")

    if len(tab_items) == 0:
        print("未在tab容器中找到任何tab项。")
        return []

    print(f"检测到 {len(tab_items)} 个tab选项")

    for i, tab_item in enumerate(tab_items):
        try:
            name_elem = tab_item.find_element(By.TAG_NAME, "span")
            tab_name = name_elem.text.strip()

            tab_target_id = None

            tab_target_id = tab_item.get_attribute("data-tab-id")

            if not tab_target_id:
                href = tab_item.get_attribute("href")
                if href and "#" in href:
                    tab_target_id = href.split("#")[-1]

            if not tab_target_id:
                tab_target_id = f"tab-{i}"

            is_active = "active" in tab_item.get_attribute("class")

            available_tabs.append({
                'id': tab_target_id,
                'name': tab_name,
                'element': tab_item,
                'is_active': is_active
            })
        except NoSuchElementException:
            print(f"  - 警告: 无法从tab项 {i + 1} 获取名称。跳过此项。")
        except Exception as e:
            print(f"  - 处理tab项 {i + 1} 时发生未知错误: {e}")

    return available_tabs


def _extract_installation_effects_from_container(driver, container, sub_product_name):  # 传入 driver
    """
    直接从子产品容器中提取安装效果内容（不点击按钮），需要处理隐藏的对话框
    Args:
        driver: WebDriver实例 (新增参数)
        container: 子产品容器元素 (div.item)
        sub_product_name: 子产品名称
    Returns:
        list: 包含安装效果的列表
    """
    installation_elements = []
    dialog_element = None
    original_display_style = None

    try:
        dialog_selectors = [
            "div[role='dialog'][aria-modal='true']",
            "div.el-dialog[aria-modal='true']",
            "div.el-dialog"
        ]

        for selector in dialog_selectors:
            try:
                dialog_element = container.find_element(By.CSS_SELECTOR, selector)
                break
            except NoSuchElementException:
                continue

        if dialog_element:
            original_display_style = dialog_element.value_of_css_property("display")

            # 使用JavaScript强制显示对话框
            print(f"  - 强制显示 '{sub_product_name}' 的对话框 (原始 display: {original_display_style})...")
            # 确保 display 不是 none，并且可见性为 visible
            driver.execute_script(
                "arguments[0].style.display='block'; arguments[0].style.visibility='visible'; arguments[0].style.opacity='1';",
                dialog_element)
            time.sleep(1)

        detail_box_selectors = [
            "div.el-dialog__body > div.detail-box",
            "div.detail-box",
            "div[data-v][class='detail-box']"
        ]

        detail_box = None
        for selector in detail_box_selectors:
            try:
                if dialog_element:
                    detail_box = dialog_element.find_element(By.CSS_SELECTOR, selector)
                    detail_box = container.find_element(By.CSS_SELECTOR, selector)

                break
            except NoSuchElementException:
                continue
            except Exception as e:
                print(f"  - 选择器 '{selector}' 查找时出错: {e}")

        if not detail_box:
            print(f"  - 未能找到 '{sub_product_name}' 的安装效果内容（即使尝试显示对话框）")
            return []

        # 提取所有图片
        try:
            img_elements = detail_box.find_elements(By.TAG_NAME, "img")
            print(f"  - 找到 {len(img_elements)} 张图片")
            for img in img_elements:
                src = img.get_attribute("src")
                alt = img.get_attribute("alt") or ""
                if src:
                    installation_elements.append({"type": "image", "src": src, "alt": alt})
        except Exception as e:
            print(f"  - 提取图片时出错: {e}")

        # 提取所有文本内容
        try:
            all_text_elements = detail_box.find_elements(
                By.XPATH,
                ".//*[not(self::script) and not(self::style) and not(self::img) and string-length(normalize-space(.)) > 2]"
            )

            seen_texts = set()
            for elem in all_text_elements:
                try:
                    text = elem.text.strip()
                    if text and len(text) > 2:
                        is_duplicate_or_subsumed = False
                        texts_to_remove = set()

                        for seen in list(seen_texts):
                            if text == seen:
                                is_duplicate_or_subsumed = True
                                break
                            if text in seen and len(text) < len(seen):
                                is_duplicate_or_subsumed = True
                                break
                            if seen in text and len(seen) < len(text):
                                texts_to_remove.add(seen)

                        for t_remove in texts_to_remove:
                            seen_texts.discard(t_remove)
                            installation_elements[:] = [
                                e for e in installation_elements
                                if not (e.get('type') == 'text' and e.get('content') == t_remove)
                            ]

                        if not is_duplicate_or_subsumed:
                            installation_elements.append({"type": "text", "content": text})
                            seen_texts.add(text)
                except StaleElementReferenceException:
                    continue
                except Exception as elem_e:
                    print(f"  - 处理文本元素时出错: {elem_e}")

            print(f"  - 提取了 {len([e for e in installation_elements if e['type'] == 'text'])} 条唯一文本")
        except Exception as e:
            print(f"  - 提取文本时出错: {e}")

    finally:
        # 恢复对话框的原始样式
        if dialog_element and original_display_style is not None:
            try:
                driver.execute_script(f"arguments[0].style.display='{original_display_style}';", dialog_element)
                if 'visibility' in dialog_element.get_attribute('style'):
                    driver.execute_script("arguments[0].style.visibility='';", dialog_element)
                if 'opacity' in dialog_element.get_attribute('style'):
                    driver.execute_script("arguments[0].style.opacity='';", dialog_element)
            except Exception as e:
                print(f"  - 恢复对话框样式时发生错误: {e}")

    return installation_elements


def _scrape_sub_products_from_tab(driver, tab_info):
    """
    从指定的tab中爬取子产品信息
    Args:
        driver: WebDriver实例
        tab_info: tab信息字典 {'id': 'tab-1', 'name': '室内机', 'element': WebElement, 'is_active': bool}
    Returns:
        子产品数据列表
    """
    sub_products_data = []
    tab_id = tab_info['id']
    tab_name = tab_info['name']

    print(f"--- 正在抓取 {tab_name} ({tab_id}) 中的子产品 ---")

    if not tab_info['is_active']:
        try:
            print(f"点击切换到 {tab_name} ({tab_id}) 标签页")
            _safe_click(driver, tab_info['element'], f"{tab_name} 标签页按钮")
            time.sleep(3)
        except Exception as e:
            print(f"点击 {tab_name} 标签页按钮时发生错误: {e}")
            return []
    else:
        print(f"{tab_name} ({tab_id}) 标签页已激活")

    tab_content_container = None
    try:
        tab_content_container = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"#{tab_id}"))
        )
    except TimeoutException:
        print(f"等待 {tab_name} 内容加载超时，内容区域ID #{tab_id} 未找到。")
        print("尝试使用更通用的选择器定位活动tab内容区域...")
        general_content_selectors = [
            "div.tab-pane.active",
            "div.content-box.active",
            "div.right > div > div.tabBox + div > div.tab-pane.active",
            "div.right > div > div.tabBox + div > div.content-wrapper",
            "div.right > div > div:nth-child(5) > div",
            "div.right > div > div.detail-container > div.product-list-wrapper",
        ]

        for selector_str in general_content_selectors:
            try:
                tab_content_container = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector_str))
                )
                break
            except TimeoutException:
                continue
            except Exception as e:
                print(f"尝试通用选择器 '{selector_str}' 时发生错误: {e}")

    if not tab_content_container:
        print(f"在 {tab_name} ({tab_id}) 中未能定位到任何内容区域。")
        return []

    sub_product_list_selectors = [
        f"#{tab_id} div.item[data-v-4801ea40]",
        f"#{tab_id} div.item",
        "div.item[data-v-4801ea40]",
        "div.item",
        ".item[data-v-4801ea40]",
        ".item"
    ]

    sub_product_containers = []
    found_selector = None
    for selector_str in sub_product_list_selectors:
        try:
            if tab_content_container:
                WebDriverWait(tab_content_container, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector_str))
                )
                containers = tab_content_container.find_elements(By.CSS_SELECTOR, selector_str)
                if containers:
                    sub_product_containers = containers
                    found_selector = selector_str

                    break
            else:
                break
        except TimeoutException:
            print(f"等待子产品容器超时，选择器 '{selector_str}' 未找到。")
            continue
        except Exception as e:
            print(f"尝试选择器 '{selector_str}' 时发生错误: {e}")

    if not sub_product_containers:
        print(f"在 {tab_name} ({tab_id}) 中未能找到任何子产品容器。")
        return []

    for i, container in enumerate(sub_product_containers):
        sub_product = {}

        name_selector = (By.CSS_SELECTOR, "div.params > div.info > div.name")
        try:
            name_elem = container.find_element(*name_selector)
            sub_product_name = name_elem.text.strip()

            # 跳过名称为空的子产品
            if not sub_product_name:
                print(f"  - 跳过 {tab_name} 子产品 {i + 1}: 名称为空")
                continue

            sub_product["名称"] = sub_product_name
            print(f"  - 抓取 {tab_name} 子产品 {i + 1}: {sub_product['名称']}")
        except NoSuchElementException:
            print(f"  - 跳过 {tab_name} 子产品 {i + 1}: 无法获取名称")
            continue
        except Exception as e:
            print(f"  - 跳过 {tab_name} 子产品 {i + 1}: 获取名称时发生错误 {e}")
            continue

        # 提取参数表格（仅用于保存CSV，不存入JSON）
        params_table_selector = (By.CSS_SELECTOR, "div.paramsImage > table")
        params_table_html = None
        try:
            table_elem = container.find_element(*params_table_selector)
            params_table_html = table_elem.get_attribute('outerHTML').strip()
        except NoSuchElementException:
            print(f"  - 无法获取 {tab_name} {sub_product['名称']} 参数表格")
        except Exception as e:
            print(f"  - 获取 {tab_name} {sub_product['名称']} 参数表格时发生错误: {e}")

        installation_elements = _extract_installation_effects_from_container(driver, container, sub_product['名称'])
        sub_product["安装效果"] = installation_elements

        sub_product["_params_table_html_temp"] = params_table_html

        sub_products_data.append(sub_product)
        time.sleep(0.5)

    return sub_products_data


def scrape_all_sub_products(driver):
    """
    爬取所有子产品（室内机、室外机、选配件）
    Args:
        driver: WebDriver实例
    Returns:
        dict: 包含所有子产品的字典，key为tab名称，value为子产品列表
    """
    sub_products_dict = {}

    available_tabs = _get_available_tabs(driver)

    if not available_tabs:
        print("该产品系列没有子产品分类tab或检测失败。")
        sub_products_dict["说明"] = "该产品系列没有子产品分类"
    else:
        skip_tabs = ["系列介绍", "产品介绍", "简介", "概述"]

        for tab_info in available_tabs:
            tab_name = tab_info['name']

            if tab_name in skip_tabs:
                print(f"跳过 {tab_name}（介绍页面，不包含子产品）")
                continue

            sub_products = _scrape_sub_products_from_tab(driver, tab_info)
            sub_products_dict[tab_name] = sub_products
            print(f"已抓取 {tab_name}: {len(sub_products)} 个子产品")

    return sub_products_dict


def _html_table_to_dataframe(html_table_string):
    """
    将HTML表格字符串解析为Pandas DataFrame。
    Args:
        html_table_string (str): 包含HTML表格的字符串。
    Returns:
        pd.DataFrame: 解析后的表格数据，如果失败则返回None。
    """
    try:
        soup = BeautifulSoup(html_table_string, 'lxml')
        table = soup.find('table')
        if not table:
            print("HTML中未找到表格。")
            return None

        rows = table.find_all('tr')
        if not rows:
            print("表格中未找到行。")
            return None

        data = []
        for r_idx, row in enumerate(rows):
            cols = row.find_all(['th', 'td'])
            if not cols:
                continue

            row_data = []
            for c_idx, col in enumerate(cols):
                text = col.get_text(strip=True)
                row_data.append(text)
            data.append(row_data)

        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print(f"解析HTML表格时发生错误: {e}")
        return None


def save_sub_product_tables_as_csv(product_data, base_output_dir="daikin_product"):
    """
    将产品数据中的子产品参数表格保存为CSV文件。
    Args:
        product_data (dict): 包含产品系列信息的字典。
        base_output_dir (str): 保存所有产品数据的根目录。
    """
    product_series_name = product_data.get("产品名称", "未知产品系列")
    print(f"\n--- 正在处理产品系列: {product_series_name} 的子产品表格 ---")

    product_series_dir = os.path.join(base_output_dir, product_series_name)
    if not os.path.exists(product_series_dir):
        os.makedirs(product_series_dir)

    sub_products = product_data.get("子产品", {})
    if not sub_products:
        print(f"产品系列 '{product_series_name}' 没有子产品数据。")
        return

    for sub_category_name, sub_product_list in sub_products.items():
        if not sub_product_list or sub_category_name == "说明":
            continue

        sub_category_dir = os.path.join(product_series_dir, sub_category_name)
        if not os.path.exists(sub_category_dir):
            os.makedirs(sub_category_dir)

        for sub_product in sub_product_list:
            sub_product_name = sub_product.get("名称", "未知子产品")

            # 从临时字段获取HTML
            html_table = sub_product.get("_params_table_html_temp")

            if html_table and html_table != "N/A":
                print(f"  - 正在为子产品 '{sub_product_name}' (分类: {sub_category_name}) 解析并保存参数表格...")
                df = _html_table_to_dataframe(html_table)
                if df is not None and not df.empty:
                    safe_sub_product_name = "".join(
                        c for c in sub_product_name if c.isalnum() or c in (' ', '.', '_')).strip()
                    csv_filename = f"{safe_sub_product_name}.csv"
                    csv_filepath = os.path.join(sub_category_dir, csv_filename)

                    try:
                        df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
                    except Exception as e:
                        print(f"    保存表格到 {csv_filepath} 失败: {e}")
                else:
                    print(f"    子产品 '{sub_product_name}' (分类: {sub_category_name}) 的参数表格解析为空或失败。")

            # 清理临时字段
            if "_params_table_html_temp" in sub_product:
                del sub_product["_params_table_html_temp"]

