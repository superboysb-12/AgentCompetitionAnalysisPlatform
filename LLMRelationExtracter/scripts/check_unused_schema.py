#!/usr/bin/env python3
"""
检测配置中定义但未在提取结果中使用的实体类型和关系类型
"""

import json
import yaml
import argparse
import os
from typing import Dict, Set, List
from collections import Counter


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_extraction_results(result_path: str) -> Dict:
    """加载三元组提取结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_used_types(results: Dict) -> tuple[Set[str], Set[str]]:
    """从提取结果中提取已使用的实体类型和关系类型"""
    used_entity_types = set()
    used_relation_types = set()

    # 尝试不同的结果格式
    triplets = []

    # 格式1: 直接是triplets数组
    if 'triplets' in results:
        if isinstance(results['triplets'], list):
            triplets = results['triplets']
        elif isinstance(results['triplets'], dict):
            # 格式2: triplets是字典，包含in_config/out_of_config/all
            triplets = results['triplets'].get('all', [])

    # 格式3: 直接是数组
    if isinstance(results, list):
        triplets = results

    for triplet in triplets:
        used_entity_types.add(triplet.get('subject_type', ''))
        used_entity_types.add(triplet.get('object_type', ''))
        used_relation_types.add(triplet.get('relation', ''))

    # 移除空字符串
    used_entity_types.discard('')
    used_relation_types.discard('')

    return used_entity_types, used_relation_types


def analyze_schema_coverage(config: Dict, results: Dict) -> Dict:
    """分析Schema覆盖情况"""
    # 获取配置中定义的类型
    config_entity_types = set(config.get('entity_types', {}).keys())
    config_relation_types = set(config.get('relation_types', {}).keys())

    # 获取实际使用的类型
    used_entity_types, used_relation_types = extract_used_types(results)

    # 计算未使用的类型
    unused_entity_types = config_entity_types - used_entity_types
    unused_relation_types = config_relation_types - used_relation_types

    # 计算配置外的类型（新发现的类型）
    new_entity_types = used_entity_types - config_entity_types
    new_relation_types = used_relation_types - config_relation_types

    # 统计每个类型的使用频率
    entity_type_counts = Counter()
    relation_type_counts = Counter()

    triplets = []
    if 'triplets' in results:
        if isinstance(results['triplets'], list):
            triplets = results['triplets']
        elif isinstance(results['triplets'], dict):
            triplets = results['triplets'].get('all', [])
    if isinstance(results, list):
        triplets = results

    for triplet in triplets:
        entity_type_counts[triplet.get('subject_type', '')] += 1
        entity_type_counts[triplet.get('object_type', '')] += 1
        relation_type_counts[triplet.get('relation', '')] += 1

    return {
        'config_entity_types': config_entity_types,
        'config_relation_types': config_relation_types,
        'used_entity_types': used_entity_types,
        'used_relation_types': used_relation_types,
        'unused_entity_types': unused_entity_types,
        'unused_relation_types': unused_relation_types,
        'new_entity_types': new_entity_types,
        'new_relation_types': new_relation_types,
        'entity_type_counts': dict(entity_type_counts),
        'relation_type_counts': dict(relation_type_counts),
        'statistics': {
            'total_config_entities': len(config_entity_types),
            'total_config_relations': len(config_relation_types),
            'used_entities': len(used_entity_types),
            'used_relations': len(used_relation_types),
            'unused_entities': len(unused_entity_types),
            'unused_relations': len(unused_relation_types),
            'new_entities': len(new_entity_types),
            'new_relations': len(new_relation_types),
            'entity_coverage': f"{len(used_entity_types & config_entity_types) / len(config_entity_types) * 100:.1f}%" if config_entity_types else "N/A",
            'relation_coverage': f"{len(used_relation_types & config_relation_types) / len(config_relation_types) * 100:.1f}%" if config_relation_types else "N/A"
        }
    }


def print_analysis_report(analysis: Dict, verbose: bool = False):
    """打印分析报告"""
    print("\n" + "=" * 80)
    print("Schema 使用情况分析报告")
    print("=" * 80)

    stats = analysis['statistics']
    print(f"\n【统计摘要】")
    print(f"  配置实体类型数: {stats['total_config_entities']}")
    print(f"  配置关系类型数: {stats['total_config_relations']}")
    print(f"  实际使用实体类型数: {stats['used_entities']}")
    print(f"  实际使用关系类型数: {stats['used_relations']}")
    print(f"  实体类型覆盖率: {stats['entity_coverage']}")
    print(f"  关系类型覆盖率: {stats['relation_coverage']}")
    print(f"  未使用实体类型数: {stats['unused_entities']}")
    print(f"  未使用关系类型数: {stats['unused_relations']}")
    print(f"  新发现实体类型数: {stats['new_entities']}")
    print(f"  新发现关系类型数: {stats['new_relations']}")

    # 未使用的实体类型
    if analysis['unused_entity_types']:
        print(f"\n【未使用的实体类型】({len(analysis['unused_entity_types'])} 个)")
        for entity_type in sorted(analysis['unused_entity_types']):
            print(f"  ❌ {entity_type}")
    else:
        print(f"\n【未使用的实体类型】")
        print("  ✅ 所有配置的实体类型都已使用")

    # 未使用的关系类型
    if analysis['unused_relation_types']:
        print(f"\n【未使用的关系类型】({len(analysis['unused_relation_types'])} 个)")
        for relation_type in sorted(analysis['unused_relation_types']):
            print(f"  ❌ {relation_type}")
    else:
        print(f"\n【未使用的关系类型】")
        print("  ✅ 所有配置的关系类型都已使用")

    # 新发现的类型
    if analysis['new_entity_types']:
        print(f"\n【新发现的实体类型】({len(analysis['new_entity_types'])} 个)")
        for entity_type in sorted(analysis['new_entity_types']):
            count = analysis['entity_type_counts'].get(entity_type, 0)
            print(f"  ⭐ {entity_type} (出现 {count} 次)")

    if analysis['new_relation_types']:
        print(f"\n【新发现的关系类型】({len(analysis['new_relation_types'])} 个)")
        for relation_type in sorted(analysis['new_relation_types']):
            count = analysis['relation_type_counts'].get(relation_type, 0)
            print(f"  ⭐ {relation_type} (出现 {count} 次)")

    # 详细使用频率（如果启用verbose）
    if verbose:
        print(f"\n【实体类型使用频率】(Top 20)")
        entity_counts = analysis['entity_type_counts']
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for entity_type, count in sorted_entities:
            status = "✓" if entity_type in analysis['config_entity_types'] else "⭐"
            print(f"  {status} {entity_type}: {count} 次")

        print(f"\n【关系类型使用频率】(Top 20)")
        relation_counts = analysis['relation_type_counts']
        sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for relation_type, count in sorted_relations:
            status = "✓" if relation_type in analysis['config_relation_types'] else "⭐"
            print(f"  {status} {relation_type}: {count} 次")

    print("\n" + "=" * 80)
    print("说明:")
    print("  ❌ - 配置中定义但未使用的类型")
    print("  ✅ - 已使用的配置类型")
    print("  ⭐ - 新发现的类型(不在配置中)")
    print("=" * 80 + "\n")


def save_analysis_to_json(analysis: Dict, output_path: str):
    """保存分析结果到JSON文件"""
    # 转换set为list以便JSON序列化
    serializable_analysis = {
        'config_entity_types': sorted(list(analysis['config_entity_types'])),
        'config_relation_types': sorted(list(analysis['config_relation_types'])),
        'used_entity_types': sorted(list(analysis['used_entity_types'])),
        'used_relation_types': sorted(list(analysis['used_relation_types'])),
        'unused_entity_types': sorted(list(analysis['unused_entity_types'])),
        'unused_relation_types': sorted(list(analysis['unused_relation_types'])),
        'new_entity_types': sorted(list(analysis['new_entity_types'])),
        'new_relation_types': sorted(list(analysis['new_relation_types'])),
        'entity_type_counts': analysis['entity_type_counts'],
        'relation_type_counts': analysis['relation_type_counts'],
        'statistics': analysis['statistics']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_analysis, f, ensure_ascii=False, indent=2)

    print(f"✓ 分析结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='检测配置中定义但未在提取结果中使用的Schema类型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python check_unused_schema.py -c config.yaml -r knowledge_graph.json

  # 显示详细使用频率
  python check_unused_schema.py -c config.yaml -r knowledge_graph.json -v

  # 保存分析结果到JSON
  python check_unused_schema.py -c config.yaml -r knowledge_graph.json -o analysis.json
        """
    )

    parser.add_argument('-c', '--config',
                        default='config.yaml',
                        help='配置文件路径 (默认: config.yaml)')

    parser.add_argument('-r', '--results',
                        default='knowledge_graph.json',
                        help='三元组提取结果文件路径 (默认: knowledge_graph.json)')

    parser.add_argument('-o', '--output',
                        help='保存分析结果到JSON文件')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='显示详细的使用频率统计')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        return 1

    if not os.path.exists(args.results):
        print(f"❌ 错误: 结果文件不存在: {args.results}")
        return 1

    try:
        # 加载文件
        print(f"📖 加载配置文件: {args.config}")
        config = load_config(args.config)

        print(f"📖 加载提取结果: {args.results}")
        results = load_extraction_results(args.results)

        # 分析
        print("🔍 分析Schema使用情况...")
        analysis = analyze_schema_coverage(config, results)

        # 打印报告
        print_analysis_report(analysis, verbose=args.verbose)

        # 保存到文件（如果指定）
        if args.output:
            save_analysis_to_json(analysis, args.output)

        return 0

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
