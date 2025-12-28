import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, unquote
import re
import os


class CarrierProductScraper:
    def __init__(self):
        self.base_url = "https://www.carrier.com"
        # 空气端产品
        # 空气处理机组
        self.list_url = "https://www.carrier.com/commercial/zh/cn/products/commercial-products/air-side/air-handlers/"
        # 变风量空气末端
        # "https://www.carrier.com/commercial/zh/cn/products/commercial-products/air-side/air-terminals/"
        # 风机盘管
        # "https://www.carrier.com/commercial/zh/cn/products/commercial-products/air-side/fan-coils/"

        # 冷水机组/热泵机组
        # "https://www.carrier.com/commercial/zh/cn/products/commercial-products/chillers/"

        # 控制
        # ”https://www.carrier.com/commercial/zh/cn/products/commercial-products/controls/“
        # 产品详情页信息名称需更换json中对应标签
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        self.products_data = []
        self.output_dir = "carrier_products"

    def get_page(self, url):
        """获取页面内容"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30, verify=False)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"获取页面失败 {url}: {str(e)}")
            return None

    def get_product_links_with_names(self):
        """从列表页获取产品链接和名称"""
        print(f"正在获取产品列表: {self.list_url}")
        html = self.get_page(self.list_url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        products = []

        product_cards = soup.select('#product-view > div.row.product-list.grid-view > div')
        if not product_cards:
            product_cards = soup.select('div.product-list.grid-view > div')
        if not product_cards:
            product_cards = soup.select('div[class*="product-list"] > div')
        if not product_cards:
            product_cards = soup.select('div.card')

        if product_cards:
            seen_urls = set()

            for card in product_cards:
                link_elem = None

                link_elem = card.select_one('div.product-titles a')
                if not link_elem:
                    link_elem = card.select_one('div.card-title a')
                if not link_elem:
                    all_card_links = card.select('a[href*="/air-handlers/"]')
                    for a in all_card_links:
                        text = a.get_text(strip=True)
                        if text and text.lower() not in ['details', '了解更多', 'learn more', '查看详情']:
                            link_elem = a
                            break
                    if not link_elem and all_card_links:
                        link_elem = all_card_links[0]

                if link_elem and link_elem.get('href'):
                    product_url = urljoin(self.base_url, link_elem['href'])

                    if product_url in seen_urls:
                        continue

                    if '/air-handlers/' in product_url and product_url != self.list_url:
                        seen_urls.add(product_url)

                        product_name = link_elem.get_text(strip=True)
                        if not product_name or product_name.lower() in ['details', '了解更多']:
                            name_elem = (card.select_one('div.card-title a') or
                                         card.select_one('div.product-titles a') or
                                         card.select_one('h3') or
                                         card.select_one('h4'))
                            if name_elem:
                                product_name = name_elem.get_text(strip=True)

                        if not product_name:
                            product_name = os.path.basename(product_url.rstrip('/')).upper()

                        products.append({
                            'url': product_url,
                            'name': product_name
                        })
                        print(f"  ✓ {product_name} → {product_url}")
        else:
            print("[警告] 未找到产品卡片，尝试直接从链接提取...")
            product_url_prefix = "/commercial/zh/cn/products/commercial-products/air-side/air-handlers/"

            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link.get('href')
                if href and product_url_prefix in href:
                    full_url = urljoin(self.base_url, href)
                    path_segments = [s for s in full_url.replace(self.base_url, '').split('/') if s]

                    if len(path_segments) > len([s for s in product_url_prefix.split('/') if s]):
                        product_name = link.get_text(strip=True) or os.path.basename(full_url.rstrip('/')).upper()

                        if not any(p['url'] == full_url for p in products):
                            products.append({
                                'url': full_url,
                                'name': product_name
                            })
                            print(f"  ✓ {product_name} → {full_url}")

        print(f"共找到 {len(products)} 个产品")
        return products

    def download_image(self, image_url, save_path):
        """下载图片"""
        try:
            response = requests.get(image_url, stream=True, headers=self.headers, timeout=30, verify=False)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            print(f"    ✗ 图片下载失败: {str(e)}")
            return False

    def find_product_description(self, soup):
        """查找产品描述 - 完整提取包括列表内容"""
        desc_container = soup.select_one('#main-content div.ct-product-details div.col-md-7 div.col-12')
        if not desc_container:
            desc_container = soup.select_one('div.ct-product-details section div.col-12')
        if not desc_container:
            desc_container = soup.select_one('div.container.ct-product-details div.col-12')

        if desc_container:
            for br in desc_container.find_all('br'):
                br.replace_with('\n')

            description_parts = []
            for child in desc_container.children:
                if hasattr(child, 'name'):
                    if child.name == 'h3':
                        title_text = child.get_text(strip=False).strip()
                        if title_text:
                            description_parts.append(f"\n{title_text}\n")
                    elif child.name == 'h4':
                        subtitle_text = child.get_text(strip=False).strip()
                        if subtitle_text:
                            description_parts.append(f"{subtitle_text}\n")
                    elif child.name == 'ul':
                        list_items = child.find_all('li', recursive=False)
                        if list_items:
                            for li in list_items:
                                for br in li.find_all('br'):
                                    br.replace_with('\n')
                                li_text = li.get_text(strip=False).strip()
                                if li_text:
                                    description_parts.append(f"• {li_text}")
                            description_parts.append('')
                    elif child.name == 'p':
                        for br in child.find_all('br'):
                            br.replace_with('\n')
                        p_text = child.get_text(strip=False).strip()
                        if p_text:
                            description_parts.append(f"{p_text}\n")
                    elif child.name == 'div':
                        div_text = child.get_text(strip=False).strip()
                        if div_text and len(div_text) > 10:
                            description_parts.append(div_text)
                elif hasattr(child, 'strip'):
                    text = child.strip()
                    if text and len(text) > 3:
                        description_parts.append(text)

            if description_parts:
                full_description = '\n'.join(part for part in description_parts if part).strip()
                full_description = re.sub(r'\n{3,}', '\n\n', full_description)
                return full_description

        product_detail = soup.select_one('div.ct-product-details')
        if product_detail:
            for br in product_detail.find_all('br'):
                br.replace_with('\n')

            text_parts = []
            for elem in product_detail.find_all(['h3', 'h4', 'p', 'ul']):
                if elem.name == 'ul':
                    for li in elem.find_all('li'):
                        li_text = li.get_text(strip=True)
                        if li_text:
                            text_parts.append(f"• {li_text}")
                else:
                    text = elem.get_text(strip=False).strip()
                    if text and len(text) > 5:
                        text_parts.append(text)

            if text_parts:
                full_text = '\n'.join(text_parts[:20])
                if len(full_text) > 30:
                    return full_text
        return ""

    def extract_product_details(self, product_info):
        """提取单个产品的详细信息"""
        url = product_info['url']
        product_name = product_info['name']

        print(f"正在爬取产品: {product_name} ({url})")

        html = self.get_page(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        product_data = {
            'url': url,
            'name': product_name,
            'images': [],
            'description': '',
            'features': [],
            'specifications': []
        }

        clean_product_name = re.sub(r'[\\/:*?"<>|]', '_', product_name).strip()
        if not clean_product_name:
            clean_product_name = os.path.basename(unquote(url.rstrip('/'))) or f"product_{int(time.time())}"

        image_urls = []
        images = soup.select('#gallery-wrapper > li > div > img')

        if images:
            for img in images:
                img_url = img.get('src')
                if img_url and not img_url.startswith('data:'):
                    if not any(kw in img_url.lower() for kw in ['icon', 'logo', 'sprite', 'placeholder']):
                        full_img_url = urljoin(self.base_url, img_url)
                        if full_img_url not in image_urls:
                            image_urls.append(full_img_url)

        if not image_urls:
            img_selectors = [
                '#gallery-wrapper img',
                'div[class*="gallery"] img',
                'div[class*="product-image"] img',
                'div.product-details img',
            ]

            for selector in img_selectors:
                images = soup.select(selector)
                if images:
                    for img in images:
                        img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                        if img_url and not img_url.startswith('data:'):
                            if not any(kw in img_url.lower() for kw in ['icon', 'logo', 'sprite', 'placeholder']):
                                full_img_url = urljoin(self.base_url, img_url)
                                if full_img_url not in image_urls:
                                    image_urls.append(full_img_url)
                    if image_urls:
                        break

        product_data['images'] = image_urls
        product_data['description'] = self.find_product_description(soup)

        feature_selectors = [
            '#tab-collapse-0 ul li',
            'div[id*="feature"] ul li',
            'div[class*="feature"] ul li',
            'section[aria-labelledby*="feature"] ul li',
            'div[role="tabpanel"] ul li',
        ]

        for selector in feature_selectors:
            features = soup.select(selector)
            if features and len(features) > 0:
                for feature in features[:30]:
                    text = feature.get_text(strip=True)
                    if text and len(text) > 3:
                        product_data['features'].append(text)
                if product_data['features']:
                    break

        spec_selectors = [
            '#tab-collapse-1 ul li',
            'div[id*="spec"] ul li',
            'div[class*="spec"] ul li',
            'div[id*="param"] ul li',
            'section[aria-labelledby*="spec"] ul li',
            'table.specifications tr',
            'dl.specifications dt, dl.specifications dd',
        ]

        for selector in spec_selectors:
            specs = soup.select(selector)
            if specs and len(specs) > 0:
                for spec in specs[:50]:
                    text = spec.get_text(strip=True)
                    if text and len(text) > 3:
                        product_data['specifications'].append(text)
                if product_data['specifications']:
                    break

        return product_data, image_urls, clean_product_name

    def scrape_all_products(self):
        """爬取所有产品并保存"""
        print("开始爬取 Carrier 产品列表")

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"数据将保存到: {os.path.abspath(self.output_dir)}\n")

        products = self.get_product_links_with_names()
        if not products:
            print("未找到产品。")
            return

        success_count = 0
        for i, product_info in enumerate(products, 1):
            print(f"\n--- 处理产品 [{i}/{len(products)}]: {product_info['name']} ---")

            result = self.extract_product_details(product_info)

            if result:
                product_data, image_urls, clean_product_name = result
                success_count += 1

                product_folder = os.path.join(self.output_dir, clean_product_name)
                os.makedirs(product_folder, exist_ok=True)

                json_path = os.path.join(product_folder, f"{clean_product_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(product_data, f, ensure_ascii=False, indent=2)
                print(f"  产品信息已保存: {os.path.basename(json_path)}")

                if image_urls:
                    print(f"  开始下载 {len(image_urls)} 张图片...")
                    downloaded_paths = []
                    for idx, img_url in enumerate(image_urls, 1):
                        img_name = f"image_{idx}_{os.path.basename(unquote(img_url))}"
                        img_name = re.sub(r'[\\/:*?"<>|]', '_', img_name)
                        if len(img_name) > 100:
                            ext = os.path.splitext(img_name)[1]
                            img_name = f"image_{idx}{ext}"
                        save_path = os.path.join(product_folder, img_name)
                        if self.download_image(img_url, save_path):
                            downloaded_paths.append(save_path)
                    product_data['images_local_paths'] = downloaded_paths
                else:
                    print("  未找到图片可供下载。")
                self.products_data.append(product_data)
            else:
                print(f"  跳过产品: {product_info['name']} (详情页获取失败)")

            if i < len(products):
                time.sleep(2)

        print("\n" + "=" * 70)
        print(f"爬取完成！成功处理 {success_count} / {len(products)} 个产品。")
        print("=" * 70)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

    scraper = CarrierProductScraper()
    scraper.scrape_all_products()