import requests
from bs4 import BeautifulSoup
import os
import json
import re


def crawl_mcquay_products():
    """
    Crawls the McQuay China website for product information,
    extracting details for each product series and the individual
    products within them. Saves the data into JSON files and downloads
    associated images.
    """

    # base_url = "https://www.mcquay.com.cn/jycp/list_19.aspx?lcid=&typeid=33"
    base_url = "https://www.mcquay.com.cn/jycp/list_19.aspx?lcid=&typeid=34"
    company_name = "mcquay_jy"

    if not os.path.exists(company_name):
        os.makedirs(company_name)

    session = requests.Session()
    session.verify = False  # Disables SSL certificate verification
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = session.get(base_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to request product list page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    product_series_items = soup.select('body > section > div > ul > li')

    if not product_series_items:
        print("No product series list items found. Please check the selector.")
        return

    for i, series_item in enumerate(product_series_items):
        series_link_tag = series_item.select_one('div > a')
        series_name_tag = series_item.select_one('div > a > p')

        if series_link_tag and series_name_tag:
            series_url = "https://www.mcquay.com.cn" + series_link_tag['href']
            series_name = re.sub(r'[\\/*?:"<>|]', '_', series_name_tag.get_text(strip=True))

            print(f"Processing product series: {series_name} ({series_url})")

            series_folder = os.path.join(company_name, series_name)
            if not os.path.exists(series_folder):
                os.makedirs(series_folder)

            products_in_series = []
            series_info = {'series_name': series_name, 'series_url': series_url}

            try:
                detail_response = session.get(series_url, headers=headers)
                detail_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Failed to request product series detail page: {series_url} - {e}")
                continue

            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

            series_poster_img_tag = detail_soup.select_one('div.tab.tab-active img')
            if series_poster_img_tag and 'src' in series_poster_img_tag.attrs:
                series_poster_url = series_poster_img_tag['src']
                if not series_poster_url.startswith("http"):
                    series_poster_url = "https://www.mcquay.com.cn" + series_poster_url

                series_info['series_poster_url'] = series_poster_url
                try:
                    img_data = session.get(series_poster_url, headers=headers).content
                    series_poster_filename = f"{series_name}_poster.jpg"
                    series_poster_path = os.path.join(series_folder, series_poster_filename)

                    if not os.path.exists(series_poster_path):
                        with open(series_poster_path, 'wb') as handler:
                            handler.write(img_data)
                    else:
                        print(f"  Series poster already exists, skipping: {series_poster_path}")
                    series_info['series_poster_local_path'] = series_poster_path
                except requests.exceptions.RequestException as e:
                    print(f"  Failed to download series poster: {series_poster_url} - {e}")
                    series_info['series_poster_local_path'] = "Download failed"
            else:
                series_info['series_poster_url'] = "Not found"
                series_info['series_poster_local_path'] = "Not found"

            product_details_list = detail_soup.select('#imageGallery > li')

            if not product_details_list:
                print(f"  No product details found in series {series_name}.")
            else:
                for idx, product_item in enumerate(product_details_list):
                    product_info = {}

                    product_name = product_item.get('titles', '').strip()
                    if not product_name:
                        img_tag = product_item.select_one('img')
                        product_name = img_tag.get('alt', f"Unknown Product_{idx + 1}").strip()
                    product_info['product_name'] = product_name
                    print(f"    Processing product: {product_name}")

                    poster_img_tag = product_item.select_one('img')
                    if poster_img_tag and 'src' in poster_img_tag.attrs:
                        poster_url = poster_img_tag['src']
                        if not poster_url.startswith("http"):
                            poster_url = "https://www.mcquay.com.cn" + poster_url
                        product_info['product_poster_url'] = poster_url

                        try:
                            img_data = session.get(poster_url, headers=headers).content
                            cleaned_name = re.sub(r'[\\/*?:"<>|]', '_', product_name)
                            img_filename = f"{cleaned_name[:100]}.jpg"
                            img_path = os.path.join(series_folder, img_filename)

                            if not os.path.exists(img_path):
                                with open(img_path, 'wb') as handler:
                                    handler.write(img_data)
                            else:
                                print(f"      Product poster already exists, skipping: {img_path}")
                            product_info['product_poster_local_path'] = img_path
                        except requests.exceptions.RequestException as e:
                            print(f"      Failed to download product poster: {poster_url} - {e}")
                            product_info['product_poster_local_path'] = "Download failed"
                    else:
                        product_info['product_poster_url'] = "Not found"
                        product_info['product_poster_local_path'] = "Not found"

                    for attr in ['range', 'area']:
                        if attr in product_item.attrs and product_item[attr].strip():
                            soup_attr = BeautifulSoup(product_item[attr].strip(), 'html.parser')
                            product_info[attr] = soup_attr.get_text(strip=True)
                            print(f"      {attr.capitalize()}: {product_info[attr]}")

                    if 'models' in product_item.attrs and product_item['models'].strip():
                        models_soup = BeautifulSoup(product_item['models'].strip(), 'html.parser')
                        for p_tag in models_soup.select('p'):
                            p_text = p_tag.get_text(strip=True)
                            if not p_text:
                                continue

                            match = re.match(r'^(.*?)[：:](.*)', p_text)
                            if match:
                                key = match.group(1).strip()
                                value = match.group(2).strip()
                                if key and value:
                                    product_info[key] = value
                                    print(f"      {key}: {value}")
                            elif 'description' not in product_info:
                                product_info['description'] = p_text

                    products_in_series.append(product_info)

            series_data_to_save = {
                'series_details': series_info,
                'products': products_in_series
            }

            json_filename = f"{series_name}.json"
            series_json_path = os.path.join(series_folder, json_filename)
            try:
                with open(series_json_path, 'w', encoding='utf-8') as f:
                    json.dump(series_data_to_save, f, ensure_ascii=False, indent=4)
                print(f"  Series data saved successfully: {series_json_path}\n")
            except IOError as e:
                print(f"  Failed to save series data: {series_json_path} - {e}\n")

        else:
            print(f"Could not find a valid link or name for list item {i + 1}.")


if __name__ == "__main__":
    crawl_mcquay_products()