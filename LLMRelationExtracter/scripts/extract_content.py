import json
import os
from datetime import datetime

def extract_title_content(results_dir):
    """从爬虫结果目录中提取所有JSON文件的title和content

    Args:
        results_dir: 爬虫结果文件所在目录路径

    Returns:
        提取的文档数据列表
    """
    extracted_data = []

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    for file_name in json_files:
        file_path = os.path.join(results_dir, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            task_name = data.get('task_name', 'unknown')
            results = data.get('results', [])

            print(f"处理文件: {file_name}, 包含 {len(results)} 条数据")

            for i, result in enumerate(results):
                content_data = result.get('content', {})

                title_list = content_data.get('title', [])
                title = title_list[0] if title_list else ""

                content_list = content_data.get('content', [])
                content = '\n'.join(content_list) if content_list else ""

                if title or content:
                    extracted_data.append({
                        'source_file': file_name,
                        'source_task': task_name,
                        'url': result.get('url', ''),
                        'title': title,
                        'content': content,
                        'doc_id': f"{task_name}_{i+1}"
                    })

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    return extracted_data

def save_extracted_data(data, output_path):
    """保存提取的数据为JSON格式

    Args:
        data: 提取的文档数据列表
        output_path: 输出文件路径
    """
    output_data = {
        'extraction_timestamp': datetime.now().isoformat(),
        'total_documents': len(data),
        'documents': data
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(data)} 条数据到: {output_path}")

if __name__ == "__main__":
    results_dir = r"C:\Users\19501\Desktop\知识图谱课程\crawl\results"
    output_path = r"C:\Users\19501\Desktop\知识图谱课程\crawl\extracted_content.json"

    print("开始提取title和content...")
    extracted_data = extract_title_content(results_dir)

    save_extracted_data(extracted_data, output_path)

    print(f"\n提取完成！")
    print(f"总共提取了 {len(extracted_data)} 条文档")
    print(f"输出文件: {output_path}")