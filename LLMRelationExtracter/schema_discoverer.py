"""Schema发现器模块，用于收集和分析配置外的实体类型和关系类型"""

import csv
import json
import os
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


class SchemaDiscoverer:
    """Schema发现器，负责收集和分析未在配置中定义的实体类型和关系类型"""

    def __init__(self, output_dir: str = "data/output"):
        """初始化Schema发现器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.unknown_entities = []  # 未知实体类型的实例
        self.unknown_relations = []  # 未知关系类型的实例

        # 统计计数器
        self.entity_type_counter = Counter()
        self.relation_type_counter = Counter()
        self.entity_examples = {}  # {entity_type: [examples]}
        self.relation_examples = {}  # {relation_type: [(subject, object, evidence)]}

    def collect_from_triplet(self, triplet: Any, config_entity_types: Set[str],
                            config_relation_types: Set[str]):
        """从三元组中收集未在配置中定义的类型

        Args:
            triplet: 关系三元组对象
            config_entity_types: 配置中定义的实体类型集合
            config_relation_types: 配置中定义的关系类型集合
        """
        # 收集未知关系类型
        if not triplet.is_relation_in_config:
            self.unknown_relations.append({
                'relation_type': triplet.relation,
                'subject': triplet.subject,
                'subject_type': triplet.subject_type,
                'object': triplet.object,
                'object_type': triplet.object_type,
                'confidence': triplet.confidence,
                'evidence': triplet.evidence,
                'source_url': triplet.source_url,
                'timestamp': datetime.now().isoformat()
            })

            # 统计和示例
            self.relation_type_counter[triplet.relation] += 1
            if triplet.relation not in self.relation_examples:
                self.relation_examples[triplet.relation] = []
            if len(self.relation_examples[triplet.relation]) < 5:  # 只保留前5个示例
                self.relation_examples[triplet.relation].append({
                    'subject': triplet.subject,
                    'object': triplet.object,
                    'evidence': triplet.evidence,
                    'confidence': triplet.confidence
                })

        # 收集未知实体类型
        for entity, entity_type, is_in_config in [
            (triplet.subject, triplet.subject_type, triplet.is_subject_type_in_config),
            (triplet.object, triplet.object_type, triplet.is_object_type_in_config)
        ]:
            if not is_in_config:
                self.unknown_entities.append({
                    'entity_type': entity_type,
                    'entity_value': entity,
                    'relation': triplet.relation,
                    'confidence': triplet.confidence,
                    'evidence': triplet.evidence,
                    'source_url': triplet.source_url,
                    'timestamp': datetime.now().isoformat()
                })

                # 统计和示例
                self.entity_type_counter[entity_type] += 1
                if entity_type not in self.entity_examples:
                    self.entity_examples[entity_type] = []
                if entity not in self.entity_examples[entity_type] and len(self.entity_examples[entity_type]) < 10:
                    self.entity_examples[entity_type].append(entity)

    def export_to_csv(self, prefix: str = "schema_discovery"):
        """导出收集的数据到CSV文件

        Args:
            prefix: 文件名前缀

        Returns:
            导出的文件路径列表
        """
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = []

        # 1. 导出未知实体类型汇总
        if self.entity_type_counter:
            entity_summary_file = os.path.join(
                self.output_dir,
                f"{prefix}_unknown_entities_{timestamp}.csv"
            )

            with open(entity_summary_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    '实体类型', '出现次数', '示例1', '示例2', '示例3',
                    '示例4', '示例5', '建议描述'
                ])

                for entity_type, count in self.entity_type_counter.most_common():
                    examples = self.entity_examples.get(entity_type, [])
                    row = [
                        entity_type,
                        count,
                        examples[0] if len(examples) > 0 else '',
                        examples[1] if len(examples) > 1 else '',
                        examples[2] if len(examples) > 2 else '',
                        examples[3] if len(examples) > 3 else '',
                        examples[4] if len(examples) > 4 else '',
                        f'[待补充] {entity_type}的描述'
                    ]
                    writer.writerow(row)

            exported_files.append(entity_summary_file)
            logger.info(f"未知实体类型汇总已导出: {entity_summary_file}")

        # 2. 导出未知关系类型汇总
        if self.relation_type_counter:
            relation_summary_file = os.path.join(
                self.output_dir,
                f"{prefix}_unknown_relations_{timestamp}.csv"
            )

            with open(relation_summary_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    '关系类型', '出现次数', '示例主体', '示例客体',
                    '示例证据', '置信度', '建议描述', '建议主体类型', '建议客体类型'
                ])

                for relation_type, count in self.relation_type_counter.most_common():
                    examples = self.relation_examples.get(relation_type, [])
                    if examples:
                        example = examples[0]  # 使用第一个示例
                        row = [
                            relation_type,
                            count,
                            example['subject'],
                            example['object'],
                            example['evidence'][:50] + '...' if len(example['evidence']) > 50 else example['evidence'],
                            f"{example['confidence']:.2f}",
                            f'[待补充] {relation_type}的描述',
                            '[待分析]',
                            '[待分析]'
                        ]
                    else:
                        row = [relation_type, count, '', '', '', '', f'[待补充] {relation_type}的描述', '[待分析]', '[待分析]']
                    writer.writerow(row)

            exported_files.append(relation_summary_file)
            logger.info(f"未知关系类型汇总已导出: {relation_summary_file}")

        # 3. 导出未知实体详细列表
        if self.unknown_entities:
            entity_detail_file = os.path.join(
                self.output_dir,
                f"{prefix}_unknown_entities_detail_{timestamp}.csv"
            )

            with open(entity_detail_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    '实体类型', '实体值', '所在关系', '置信度',
                    '证据', '来源URL', '时间戳'
                ])

                for entity in self.unknown_entities:
                    writer.writerow([
                        entity['entity_type'],
                        entity['entity_value'],
                        entity['relation'],
                        f"{entity['confidence']:.2f}",
                        entity['evidence'][:100] + '...' if len(entity['evidence']) > 100 else entity['evidence'],
                        entity['source_url'],
                        entity['timestamp']
                    ])

            exported_files.append(entity_detail_file)
            logger.info(f"未知实体详细列表已导出: {entity_detail_file}")

        # 4. 导出未知关系详细列表
        if self.unknown_relations:
            relation_detail_file = os.path.join(
                self.output_dir,
                f"{prefix}_unknown_relations_detail_{timestamp}.csv"
            )

            with open(relation_detail_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    '关系类型', '主体', '主体类型', '客体', '客体类型',
                    '置信度', '证据', '来源URL', '时间戳'
                ])

                for relation in self.unknown_relations:
                    writer.writerow([
                        relation['relation_type'],
                        relation['subject'],
                        relation['subject_type'],
                        relation['object'],
                        relation['object_type'],
                        f"{relation['confidence']:.2f}",
                        relation['evidence'][:100] + '...' if len(relation['evidence']) > 100 else relation['evidence'],
                        relation['source_url'],
                        relation['timestamp']
                    ])

            exported_files.append(relation_detail_file)
            logger.info(f"未知关系详细列表已导出: {relation_detail_file}")

        # 5. 生成Schema建议JSON
        schema_suggestions = self.generate_schema_suggestions()
        if schema_suggestions['entity_types'] or schema_suggestions['relation_types']:
            suggestion_file = os.path.join(
                self.output_dir,
                f"{prefix}_schema_suggestions_{timestamp}.json"
            )

            with open(suggestion_file, 'w', encoding='utf-8') as f:
                json.dump(schema_suggestions, f, ensure_ascii=False, indent=2)

            exported_files.append(suggestion_file)
            logger.info(f"Schema建议已导出: {suggestion_file}")

        return exported_files

    def generate_schema_suggestions(self) -> Dict[str, Any]:
        """生成Schema配置建议

        Returns:
            包含实体类型和关系类型建议的字典
        """
        suggestions = {
            'generated_at': datetime.now().isoformat(),
            'entity_types': {},
            'relation_types': {},
            'summary': {
                'unknown_entity_types_count': len(self.entity_type_counter),
                'unknown_relation_types_count': len(self.relation_type_counter),
                'total_unknown_entity_instances': len(self.unknown_entities),
                'total_unknown_relation_instances': len(self.unknown_relations)
            }
        }

        # 生成实体类型建议
        for entity_type, count in self.entity_type_counter.most_common():
            examples = self.entity_examples.get(entity_type, [])
            suggestions['entity_types'][entity_type] = {
                'description': f'[待补充] {entity_type}的描述',
                'examples': examples[:5],
                'occurrence_count': count,
                'suggestion': '请根据实际含义补充描述和更多示例'
            }

        # 生成关系类型建议
        for relation_type, count in self.relation_type_counter.most_common():
            examples_data = self.relation_examples.get(relation_type, [])

            # 分析可能的主体和客体类型
            subject_types = set()
            object_types = set()
            example_triplets = []

            for ex in examples_data[:3]:
                example_triplets.append(f"{ex['subject']} {relation_type} {ex['object']}")

            suggestions['relation_types'][relation_type] = {
                'description': f'[待补充] {relation_type}的描述',
                'examples': example_triplets,
                'occurrence_count': count,
                'subject_types': '[待分析]',
                'object_types': '[待分析]',
                'suggestion': '请根据实际三元组补充描述、主体类型和客体类型约束'
            }

        return suggestions

    def get_summary(self) -> Dict[str, Any]:
        """获取收集摘要统计信息

        Returns:
            包含各类统计数据的摘要字典
        """
        return {
            'unknown_entity_types': len(self.entity_type_counter),
            'unknown_relation_types': len(self.relation_type_counter),
            'total_unknown_entities': len(self.unknown_entities),
            'total_unknown_relations': len(self.unknown_relations),
            'top_unknown_entity_types': self.entity_type_counter.most_common(10),
            'top_unknown_relation_types': self.relation_type_counter.most_common(10)
        }

    def print_summary(self):
        """打印收集摘要到控制台"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("Schema发现摘要")
        print("="*60)

        print(f"\n发现未知实体类型: {summary['unknown_entity_types']} 种")
        print(f"发现未知关系类型: {summary['unknown_relation_types']} 种")
        print(f"未知实体实例总数: {summary['total_unknown_entities']}")
        print(f"未知关系实例总数: {summary['total_unknown_relations']}")

        if summary['top_unknown_entity_types']:
            print("\n前10个未知实体类型:")
            for entity_type, count in summary['top_unknown_entity_types']:
                examples = ', '.join(self.entity_examples.get(entity_type, [])[:3])
                print(f"  • {entity_type}: {count}次 (示例: {examples})")

        if summary['top_unknown_relation_types']:
            print("\n前10个未知关系类型:")
            for relation_type, count in summary['top_unknown_relation_types']:
                print(f"  • {relation_type}: {count}次")

        print("="*60 + "\n")
