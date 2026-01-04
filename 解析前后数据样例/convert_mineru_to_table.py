import json
import os
import csv
from pathlib import Path
from html.parser import HTMLParser
from html import unescape
import re

class TableHTMLParser(HTMLParser):
    """解析表格HTML，提取表格数据"""
    def __init__(self):
        super().__init__()
        self.tables = []
        self.current_table = []
        self.current_row = []
        self.in_td = False
        self.in_th = False

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.current_row = []
        elif tag in ['td', 'th']:
            self.in_td = True
            self.in_th = (tag == 'th')

    def handle_endtag(self, tag):
        if tag == 'tr' and self.current_row:
            self.current_table.append(self.current_row)
        elif tag in ['td', 'th']:
            self.in_td = False
            self.in_th = False

    def handle_data(self, data):
        if self.in_td or self.in_th:
            self.current_row.append(data.strip())

    def get_table(self):
        if self.current_table:
            return self.current_table
        return []

def extract_text_from_html(html_content):
    """从HTML中提取纯文本"""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', html_content)
    # 解码HTML实体
    text = unescape(text)
    # 清理空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_table_from_html(html_content):
    """解析表格HTML，返回二维数组"""
    parser = TableHTMLParser()
    parser.feed(html_content)
    return parser.get_table()

def process_sample(sample_path, output_dir):
    """处理单个样本文件夹"""
    sample_name = os.path.basename(sample_path)
    results = []

    # 查找所有JSON结果文件
    json_files = list(Path(sample_path).glob("*_result.json"))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for page_info in data.get('pdf_info', []):
            page_idx = page_info.get('page_idx', 0)

            for block in page_info.get('para_blocks', []):
                block_type = block.get('type', '')
                bbox = block.get('bbox', [])

                result = {
                    'sample': sample_name,
                    'file': json_file.name,
                    'page': page_idx,
                    'type': block_type,
                    'bbox': str(bbox),
                    'content': '',
                    'table_data': '',
                    'image_path': ''
                }

                # 处理表格类型
                if block_type == 'table':
                    blocks = block.get('blocks', [])
                    for b in blocks:
                        lines = b.get('lines', [])
                        for line in lines:
                            spans = line.get('spans', [])
                            for span in spans:
                                if span.get('type') == 'table':
                                    html = span.get('html', '')
                                    result['content'] = extract_text_from_html(html)
                                    # 解析表格数据
                                    table_data = parse_table_from_html(html)
                                    if table_data:
                                        # 转换为可读格式
                                        table_str = '\n'.join([' | '.join(row) for row in table_data])
                                        result['table_data'] = table_str
                                    result['image_path'] = span.get('image_path', '')

                # 处理文本类型
                elif block_type in ['text', 'title']:
                    lines = block.get('lines', [])
                    texts = []
                    for line in lines:
                        spans = line.get('spans', [])
                        for span in spans:
                            if span.get('type') == 'text':
                                texts.append(span.get('content', ''))
                    result['content'] = ' '.join(texts)

                # 处理图片类型
                elif block_type == 'image':
                    blocks = block.get('blocks', [])
                    for b in blocks:
                        lines = b.get('lines', [])
                        for line in lines:
                            spans = line.get('spans', [])
                            for span in spans:
                                if span.get('type') == 'image':
                                    result['image_path'] = span.get('image_path', '')
                                    result['content'] = '[图片]'

                if result['content'] or result['table_data']:
                    results.append(result)

            # 处理discarded_blocks（页眉页脚等）
            for block in page_info.get('discarded_blocks', []):
                block_type = block.get('type', '')
                if block_type in ['header', 'footer']:
                    result = {
                        'sample': sample_name,
                        'file': json_file.name,
                        'page': page_idx,
                        'type': f'discarded_{block_type}',
                        'bbox': str(block.get('bbox', [])),
                        'content': '',
                        'table_data': '',
                        'image_path': ''
                    }
                    lines = block.get('lines', [])
                    texts = []
                    for line in lines:
                        spans = line.get('spans', [])
                        for span in spans:
                            if span.get('type') == 'text':
                                texts.append(span.get('content', ''))
                    result['content'] = ' '.join(texts)
                    results.append(result)

    return results

def save_to_csv(data, output_file):
    """保存到CSV文件"""
    if not data:
        print(f"没有数据可保存到 {output_file}")
        return

    keys = data[0].keys()
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data:
            # 处理表格数据中的换行符
            row_copy = row.copy()
            row_copy['table_data'] = row_copy['table_data'].replace('\n', ' \\n ')
            writer.writerow(row_copy)
    print(f"已保存 {len(data)} 条记录到 {output_file}")

def save_tables_to_separate_csv(data, output_dir):
    """将表格数据单独保存为CSV文件"""
    tables_only = [row for row in data if row['type'] == 'table' and row['table_data']]

    for idx, row in enumerate(tables_only):
        sample_name = row['sample']
        table_data = row['table_data']
        lines = table_data.split('\n')
        rows = [line.split(' | ') for line in lines if line.strip()]

        if rows:
            output_file = os.path.join(output_dir, f"table_{sample_name}_{idx}.csv")
            with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"已保存表格到 {output_file}")

def main():
    # 数据根目录
    base_dir = r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\解析前后数据样例\解析前后数据样例"
    output_dir = r"C:\Users\19501\Desktop\AgentCompetitionAnalysisPlatform\解析前后数据样例\output"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 三个子目录
    categories = ['表格', '非表格', '有表格有文字']

    all_results = []

    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            continue

        print(f"\n处理目录: {category}")
        print("=" * 50)

        # 遍历该目录下的所有样本文件夹
        sample_folders = [f for f in os.listdir(category_path)
                         if os.path.isdir(os.path.join(category_path, f))]

        for sample_name in sample_folders:
            sample_path = os.path.join(category_path, sample_name)
            print(f"处理样本: {sample_name}")

            results = process_sample(sample_path, output_dir)
            all_results.extend(results)

    # 保存所有数据到CSV
    output_csv = os.path.join(output_dir, 'all_data.csv')
    save_to_csv(all_results, output_csv)

    # 单独保存表格数据
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    save_tables_to_separate_csv(all_results, tables_dir)

    # 打印统计信息
    print("\n" + "=" * 50)
    print("处理完成！统计信息：")
    print(f"总记录数: {len(all_results)}")
    print(f"表格类型: {sum(1 for r in all_results if r['type'] == 'table')}")
    print(f"文本类型: {sum(1 for r in all_results if r['type'] == 'text')}")
    print(f"标题类型: {sum(1 for r in all_results if r['type'] == 'title')}")
    print(f"图片类型: {sum(1 for r in all_results if r['type'] == 'image')}")
    print(f"\n输出文件:")
    print(f"  - {output_csv}")
    print(f"  - {tables_dir}/ (表格CSV文件)")

if __name__ == '__main__':
    main()
