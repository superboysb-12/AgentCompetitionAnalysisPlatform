#!/usr/bin/env python3
"""
检测哪些文档/文本段没有成功提取到关系三元组
参照原始输入数据 (data/input/extracted_content.json) 进行统计
"""

import json
import argparse
import os
from typing import Dict, List, Set
from collections import defaultdict


def load_json_file(file_path: str) -> Dict:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_doc_ids_from_triplets(kg_data: Dict) -> Set[str]:
    """从知识图谱结果中提取所有有三元组的文档ID"""
    doc_ids_with_triplets = set()

    # 提取所有三元组
    triplets = []
    if 'triplets' in kg_data:
        if isinstance(kg_data['triplets'], list):
            triplets = kg_data['triplets']
        elif isinstance(kg_data['triplets'], dict):
            triplets = kg_data['triplets'].get('all', [])
    elif isinstance(kg_data, list):
        triplets = kg_data

    # 收集所有出现三元组的文档ID
    for triplet in triplets:
        doc_id = triplet.get('doc_id', '')
        source_url = triplet.get('source_url', '')

        # 优先使用doc_id，如果没有则用source_url
        doc_key = doc_id if doc_id else source_url
        if doc_key:
            doc_ids_with_triplets.add(doc_key)

    return doc_ids_with_triplets


def analyze_extraction_coverage(input_data: Dict, kg_data: Dict) -> Dict:
    """分析文档提取覆盖情况"""

    # 从输入数据获取所有文档
    all_documents = input_data.get('documents', [])
    total_docs = len(all_documents)

    # 从知识图谱结果中获取有三元组的文档ID
    doc_ids_with_triplets = extract_doc_ids_from_triplets(kg_data)

    # 统计每个文档的三元组数量
    doc_triplet_count = defaultdict(int)
    triplets = []
    if 'triplets' in kg_data:
        if isinstance(kg_data['triplets'], list):
            triplets = kg_data['triplets']
        elif isinstance(kg_data['triplets'], dict):
            triplets = kg_data['triplets'].get('all', [])
    elif isinstance(kg_data, list):
        triplets = kg_data

    for triplet in triplets:
        doc_id = triplet.get('doc_id', '')
        source_url = triplet.get('source_url', '')
        doc_key = doc_id if doc_id else source_url
        if doc_key:
            doc_triplet_count[doc_key] += 1

    # 分类文档
    docs_with_extraction = []
    docs_without_extraction = []

    for doc in all_documents:
        doc_id = doc.get('doc_id', '')
        source_url = doc.get('url', '')
        doc_key = doc_id if doc_id else source_url

        doc_info = {
            'doc_id': doc_id,
            'source_url': source_url,
            'title': doc.get('title', ''),
            'content': doc.get('content', ''),
            'content_length': len(doc.get('content', '')),
            'source_file': doc.get('source_file', ''),
            'source_task': doc.get('source_task', '')
        }

        if doc_key in doc_ids_with_triplets:
            doc_info['triplet_count'] = doc_triplet_count[doc_key]
            docs_with_extraction.append(doc_info)
        else:
            docs_without_extraction.append(doc_info)

    # 计算统计数据
    return {
        'total_documents': total_docs,
        'docs_with_extraction': len(docs_with_extraction),
        'docs_without_extraction': len(docs_without_extraction),
        'coverage_rate': f"{len(docs_with_extraction) / total_docs * 100:.1f}%" if total_docs > 0 else "N/A",
        'total_triplets': len(triplets),
        'avg_triplets_per_doc': len(triplets) / len(docs_with_extraction) if docs_with_extraction else 0,
        'docs_with_extraction_details': docs_with_extraction,
        'docs_without_extraction_details': docs_without_extraction,
        'doc_triplet_count': dict(doc_triplet_count)
    }


def print_analysis_report(analysis: Dict, show_content: bool = False, max_content_length: int = 500):
    """打印分析报告"""
    print("\n" + "=" * 80)
    print("文档提取覆盖情况分析报告")
    print("=" * 80)

    print(f"\n【统计摘要】")
    print(f"  总文档数: {analysis['total_documents']}")
    print(f"  成功提取文档数: {analysis['docs_with_extraction']}")
    print(f"  未提取文档数: {analysis['docs_without_extraction']}")
    print(f"  提取覆盖率: {analysis['coverage_rate']}")
    print(f"  总三元组数: {analysis['total_triplets']}")
    print(f"  平均每文档三元组数: {analysis['avg_triplets_per_doc']:.2f}")

    # 成功提取的文档
    if analysis['docs_with_extraction_details']:
        print(f"\n【成功提取的文档】({analysis['docs_with_extraction']} 个)")
        print(f"{'序号':<6} {'文档ID':<15} {'标题':<35} {'字数':<8} {'三元组数':<10} {'来源任务':<15}")
        print("-" * 110)
        for i, doc in enumerate(analysis['docs_with_extraction_details'][:50], 1):  # 只显示前50个
            doc_id = doc['doc_id'][:13] if len(doc['doc_id']) > 13 else doc['doc_id']
            title = doc['title'][:33] if len(doc['title']) > 33 else doc['title']
            source_task = doc['source_task'][:13] if len(doc['source_task']) > 13 else doc['source_task']
            print(f"{i:<6} {doc_id:<15} {title:<35} {doc['content_length']:<8} {doc['triplet_count']:<10} {source_task:<15}")

        if len(analysis['docs_with_extraction_details']) > 50:
            print(f"... (还有 {len(analysis['docs_with_extraction_details']) - 50} 个文档未显示)")

    # 未提取的文档
    if analysis['docs_without_extraction_details']:
        print(f"\n【未成功提取三元组的文档】({analysis['docs_without_extraction']} 个)")
        print(f"{'序号':<6} {'文档ID':<15} {'标题':<35} {'字数':<8} {'来源任务':<15}")
        print("-" * 100)
        for i, doc in enumerate(analysis['docs_without_extraction_details'], 1):
            doc_id = doc['doc_id'][:13] if len(doc['doc_id']) > 13 else doc['doc_id']
            title = doc['title'][:33] if len(doc['title']) > 33 else doc['title']
            source_task = doc['source_task'][:13] if len(doc['source_task']) > 13 else doc['source_task']
            print(f"{i:<6} {doc_id:<15} {title:<35} {doc['content_length']:<8} {source_task:<15}")

            # 如果启用了显示内容
            if show_content:
                content = doc['content']
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                print(f"       内容预览: {content}")
                print(f"       来源URL: {doc['source_url']}")
                print(f"       来源文件: {doc['source_file']}")
                print()
    else:
        print(f"\n【未成功提取三元组的文档】")
        print("  [提示] 所有文档都成功提取到了三元组！")

    # 提取效率分布
    if analysis['docs_with_extraction_details']:
        print(f"\n【提取效率分布】")

        # 按三元组数量分组
        distribution = {
            '0个三元组': 0,
            '1个三元组': 0,
            '2-5个三元组': 0,
            '6-10个三元组': 0,
            '11-20个三元组': 0,
            '20+个三元组': 0
        }

        distribution['0个三元组'] = analysis['docs_without_extraction']

        for doc in analysis['docs_with_extraction_details']:
            count = doc['triplet_count']
            if count == 1:
                distribution['1个三元组'] += 1
            elif 2 <= count <= 5:
                distribution['2-5个三元组'] += 1
            elif 6 <= count <= 10:
                distribution['6-10个三元组'] += 1
            elif 11 <= count <= 20:
                distribution['11-20个三元组'] += 1
            else:
                distribution['20+个三元组'] += 1

        total_for_percentage = analysis['total_documents']
        for range_name, count in distribution.items():
            percentage = count / total_for_percentage * 100 if total_for_percentage else 0
            bar = '█' * int(percentage / 2)
            print(f"  {range_name:<15}: {count:>4} 个文档 ({percentage:>5.1f}%) {bar}")

    print("\n" + "=" * 80 + "\n")


def save_no_extraction_docs(analysis: Dict, output_path: str, include_content: bool = True, format_for_reextraction: bool = False):
    """保存未提取到三元组的文档到JSON文件

    Args:
        analysis: 分析结果
        output_path: 输出文件路径
        include_content: 是否包含文档内容
        format_for_reextraction: 是否格式化为可重新提取的格式（与extracted_content.json格式一致）
    """
    if format_for_reextraction:
        # 格式化为与extracted_content.json一致的格式，可直接用于关系提取
        import datetime

        output_data = {
            'extraction_timestamp': datetime.datetime.now().isoformat(),
            'total_documents': analysis['docs_without_extraction'],
            'documents': []
        }

        for doc in analysis['docs_without_extraction_details']:
            doc_data = {
                'source_file': doc['source_file'],
                'source_task': doc['source_task'],
                'url': doc['source_url'],
                'title': doc['title'],
                'content': doc['content'],
                'doc_id': doc['doc_id']
            }
            output_data['documents'].append(doc_data)
    else:
        # 原有格式，包含统计信息
        output_data = {
            'statistics': {
                'total_documents': analysis['total_documents'],
                'docs_with_extraction': analysis['docs_with_extraction'],
                'docs_without_extraction': analysis['docs_without_extraction'],
                'coverage_rate': analysis['coverage_rate']
            },
            'no_extraction_documents': []
        }

        for doc in analysis['docs_without_extraction_details']:
            doc_data = {
                'doc_id': doc['doc_id'],
                'source_url': doc['source_url'],
                'title': doc['title'],
                'content_length': doc['content_length'],
                'source_file': doc['source_file'],
                'source_task': doc['source_task']
            }

            if include_content:
                doc_data['content'] = doc['content']

            output_data['no_extraction_documents'].append(doc_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 未提取文档列表已保存到: {output_path}")


def save_full_analysis(analysis: Dict, output_path: str):
    """保存完整分析结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"[完成] 完整分析结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='检测哪些文档/文本段没有成功提取到关系三元组',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python check_no_extraction.py

  # 指定文件路径
  python check_no_extraction.py -i data/input/extracted_content.json -k knowledge_graph.json

  # 显示未提取文档的内容预览
  python check_no_extraction.py --show-content

  # 调整内容预览长度
  python check_no_extraction.py --show-content --max-length 1000

  # 保存未提取文档列表到JSON（统计格式）
  python check_no_extraction.py -o no_extraction_docs.json

  # 保存为可重新提取的格式（可直接用于main.py）
  python check_no_extraction.py -o no_extraction_docs_for_reextraction.json --format-for-reextraction

  # 保存完整分析结果
  python check_no_extraction.py --full-output full_analysis.json
        """
    )

    parser.add_argument('-i', '--input',
                        default='data/input/extracted_content.json',
                        help='原始输入文档路径 (默认: data/input/extracted_content.json)')

    parser.add_argument('-k', '--kg',
                        default='knowledge_graph.json',
                        help='知识图谱提取结果文件路径 (默认: knowledge_graph.json)')

    parser.add_argument('-o', '--output',
                        help='保存未提取文档列表到JSON文件')

    parser.add_argument('--full-output',
                        help='保存完整分析结果到JSON文件')

    parser.add_argument('--show-content',
                        action='store_true',
                        help='显示未提取文档的内容预览')

    parser.add_argument('--max-length',
                        type=int,
                        default=500,
                        help='内容预览最大长度 (默认: 500)')

    parser.add_argument('--no-content-in-output',
                        action='store_true',
                        help='输出JSON时不包含文档内容（仅元数据）')

    parser.add_argument('--format-for-reextraction',
                        action='store_true',
                        help='输出为可重新提取的格式（与extracted_content.json格式一致，可直接用于main.py）')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"[错误] 原始输入文件不存在: {args.input}")
        return 1

    if not os.path.exists(args.kg):
        print(f"[错误] 知识图谱结果文件不存在: {args.kg}")
        return 1

    try:
        # 加载文件
        print(f"[加载] 原始输入文档: {args.input}")
        input_data = load_json_file(args.input)

        print(f"[加载] 知识图谱结果: {args.kg}")
        kg_data = load_json_file(args.kg)

        # 分析
        print("[分析] 文档提取覆盖情况...")
        analysis = analyze_extraction_coverage(input_data, kg_data)

        # 先保存文件（避免打印时编码错误导致无法保存）
        if args.output:
            save_no_extraction_docs(
                analysis,
                args.output,
                include_content=not args.no_content_in_output,
                format_for_reextraction=args.format_for_reextraction
            )

        if args.full_output:
            save_full_analysis(analysis, args.full_output)

        # 最后打印报告
        try:
            print_analysis_report(analysis, show_content=args.show_content, max_content_length=args.max_length)
        except UnicodeEncodeError as e:
            print(f"\n[警告] 打印报告时遇到编码问题，但文件已成功保存")
            print(f"[统计] 总文档: {analysis['total_documents']}, 成功提取: {analysis['docs_with_extraction']}, 未提取: {analysis['docs_without_extraction']}")

        return 0

    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
