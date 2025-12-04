#!/usr/bin/env python3
"""
模型性能自动化评估脚本
支持横向对比多个模型的关系抽取性能，无需人工标注
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    """模型性能自动化评估器"""

    def __init__(self, output_dir: str = "evaluation_results"):
        """初始化评估器

        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.models_data = {}
        self.evaluation_results = {}

    def load_model_output(self, model_name: str, kg_file_path: str):
        """加载模型输出的知识图谱文件

        Args:
            model_name: 模型名称 (如 "deepseek-v3", "gemini-2.5", "gpt-5")
            kg_file_path: 知识图谱JSON文件路径
        """
        print(f"加载模型 [{model_name}] 的输出: {kg_file_path}")

        try:
            with open(kg_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.models_data[model_name] = data
            print(f"  ✓ 成功加载，三元组数量: {len(data.get('triplets', {}).get('all', []))}")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            raise

    def evaluate_all_models(self) -> Dict[str, Any]:
        """评估所有已加载的模型"""
        print("\n" + "="*80)
        print("开始自动化评估...")
        print("="*80)

        results = {}

        for model_name, data in self.models_data.items():
            print(f"\n评估模型: {model_name}")
            print("-"*80)

            model_result = self._evaluate_single_model(model_name, data)
            results[model_name] = model_result

            self._print_model_summary(model_name, model_result)

        self.evaluation_results = results
        return results

    def _evaluate_single_model(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个模型的输出"""

        triplets = data.get('triplets', {}).get('all', [])
        metadata = data.get('metadata', {})
        stats = metadata.get('statistics', {})

        result = {
            'model_name': model_name,

            # 1. 基础统计
            'basic_stats': self._compute_basic_stats(triplets, stats),

            # 2. 质量评分
            'quality_scores': self._compute_quality_scores(triplets, data),

            # 3. Schema符合度
            'schema_compliance': self._compute_schema_compliance(triplets, data),

            # 4. 一致性评分
            'consistency_scores': self._compute_consistency_scores(triplets),

            # 5. 多样性评分
            'diversity_scores': self._compute_diversity_scores(triplets),

            # 6. Evidence质量
            'evidence_quality': self._compute_evidence_quality(triplets),

            # 7. 性能指标
            'performance_metrics': self._compute_performance_metrics(stats),

            # 8. 成本效益
            'cost_efficiency': self._compute_cost_efficiency(stats, triplets),
        }

        # 计算综合得分
        result['overall_score'] = self._compute_overall_score(result)

        return result

    def _compute_basic_stats(self, triplets: List[Dict], stats: Dict) -> Dict[str, Any]:
        """计算基础统计信息"""
        return {
            'total_triplets': len(triplets),
            'total_documents': stats.get('processing_summary', {}).get('total_documents', 0),
            'avg_triplets_per_doc': len(triplets) / max(stats.get('processing_summary', {}).get('total_documents', 1), 1),
            'total_processing_time': stats.get('processing_summary', {}).get('total_processing_time', 0),
            'avg_time_per_doc': stats.get('processing_summary', {}).get('avg_time_per_document', 0),
        }

    def _compute_quality_scores(self, triplets: List[Dict], data: Dict) -> Dict[str, float]:
        """计算质量评分

        基于以下维度:
        1. 平均置信度
        2. 高置信度(>0.9)三元组比例
        3. 低置信度(<0.7)三元组比例
        4. 置信度标准差(越小越稳定)
        """
        confidences = [t.get('confidence', 0.0) for t in triplets]

        if not confidences:
            return {
                'avg_confidence': 0.0,
                'high_confidence_ratio': 0.0,
                'low_confidence_ratio': 0.0,
                'confidence_std': 0.0,
                'confidence_score': 0.0
            }

        avg_conf = np.mean(confidences)
        high_conf_ratio = sum(1 for c in confidences if c > 0.9) / len(confidences)
        low_conf_ratio = sum(1 for c in confidences if c < 0.7) / len(confidences)
        conf_std = np.std(confidences)

        # 综合置信度评分 (0-100)
        # 高平均置信度 + 高比例高置信度 - 低比例低置信度 - 高标准差(不稳定)
        confidence_score = (
            avg_conf * 50 +                    # 平均置信度贡献50分
            high_conf_ratio * 30 -             # 高置信度比例贡献30分
            low_conf_ratio * 20 -              # 低置信度比例扣20分
            (conf_std / 0.3) * 10              # 标准差过大扣分
        )
        confidence_score = max(0, min(100, confidence_score))

        return {
            'avg_confidence': round(avg_conf, 3),
            'high_confidence_ratio': round(high_conf_ratio, 3),
            'low_confidence_ratio': round(low_conf_ratio, 3),
            'confidence_std': round(conf_std, 3),
            'confidence_score': round(confidence_score, 2)
        }

    def _compute_schema_compliance(self, triplets: List[Dict], data: Dict) -> Dict[str, Any]:
        """计算Schema符合度评分

        基于:
        1. 配置内三元组比例(越高越好)
        2. 实体类型符合度
        3. 关系类型符合度
        """
        if not triplets:
            return {
                'in_config_ratio': 0.0,
                'relation_compliance': 0.0,
                'entity_compliance': 0.0,
                'schema_score': 0.0
            }

        # 统计符合度
        in_config_count = sum(1 for t in triplets
                             if t.get('is_relation_in_config', False) and
                                t.get('is_subject_type_in_config', False) and
                                t.get('is_object_type_in_config', False))

        relation_in_config = sum(1 for t in triplets if t.get('is_relation_in_config', False))
        subject_in_config = sum(1 for t in triplets if t.get('is_subject_type_in_config', False))
        object_in_config = sum(1 for t in triplets if t.get('is_object_type_in_config', False))

        total = len(triplets)

        in_config_ratio = in_config_count / total
        relation_compliance = relation_in_config / total
        entity_compliance = (subject_in_config + object_in_config) / (2 * total)

        # Schema符合度评分 (0-100)
        # 高配置内比例是好事，但也要考虑模型的发现能力
        # 100%配置内不一定最好，可能遗漏了有价值的新关系
        schema_score = (
            in_config_ratio * 60 +           # 配置内比例贡献60分
            relation_compliance * 20 +       # 关系符合度贡献20分
            entity_compliance * 20           # 实体符合度贡献20分
        )

        return {
            'in_config_ratio': round(in_config_ratio, 3),
            'relation_compliance': round(relation_compliance, 3),
            'entity_compliance': round(entity_compliance, 3),
            'schema_score': round(schema_score, 2)
        }

    def _compute_consistency_scores(self, triplets: List[Dict]) -> Dict[str, Any]:
        """计算一致性评分

        基于:
        1. 实体命名一致性(同一实体是否使用相同名称)
        2. 关系使用一致性(相似关系是否合并)
        3. 类型标注一致性
        """
        if not triplets:
            return {
                'entity_name_consistency': 0.0,
                'relation_consistency': 0.0,
                'consistency_score': 0.0
            }

        # 1. 实体命名一致性
        # 统计每个实体的不同变体数量
        entity_variants = defaultdict(set)
        for t in triplets:
            subj = t.get('subject', '').lower().strip()
            obj = t.get('object', '').lower().strip()
            if subj:
                # 使用简化版本作为key，检查是否有多种写法
                key = re.sub(r'[^\w\s]', '', subj)[:10]
                entity_variants[key].add(subj)
            if obj:
                key = re.sub(r'[^\w\s]', '', obj)[:10]
                entity_variants[key].add(obj)

        # 计算平均每个实体有多少变体(越接近1越好)
        avg_variants = np.mean([len(variants) for variants in entity_variants.values()]) if entity_variants else 1.0
        entity_consistency = 1.0 / avg_variants

        # 2. 关系使用一致性
        # 统计关系类型的使用频率分布，评估是否过于分散
        relation_counter = Counter(t.get('relation', '') for t in triplets)
        total_relations = len(relation_counter)

        if total_relations > 0:
            # 计算Gini系数衡量集中度
            relation_freqs = sorted(relation_counter.values())
            n = len(relation_freqs)
            cumsum = np.cumsum(relation_freqs)
            gini = (2 * sum((i+1) * freq for i, freq in enumerate(relation_freqs))) / (n * sum(relation_freqs)) - (n + 1) / n
            # Gini越高说明分布越集中，适度集中是好的
            relation_consistency = gini
        else:
            relation_consistency = 0.0

        # 综合一致性评分 (0-100)
        consistency_score = (
            entity_consistency * 50 +         # 实体一致性贡献50分
            relation_consistency * 50         # 关系一致性贡献50分
        )

        return {
            'entity_name_consistency': round(entity_consistency, 3),
            'relation_consistency': round(relation_consistency, 3),
            'avg_entity_variants': round(avg_variants, 2),
            'consistency_score': round(consistency_score, 2)
        }

    def _compute_diversity_scores(self, triplets: List[Dict]) -> Dict[str, Any]:
        """计算多样性评分

        基于:
        1. 关系类型多样性
        2. 实体类型多样性
        3. 信息覆盖广度
        """
        if not triplets:
            return {
                'relation_type_count': 0,
                'entity_type_count': 0,
                'unique_entities_count': 0,
                'diversity_score': 0.0
            }

        # 统计类型数量
        relations = set(t.get('relation', '') for t in triplets)
        subject_types = set(t.get('subject_type', '') for t in triplets)
        object_types = set(t.get('object_type', '') for t in triplets)
        entity_types = subject_types.union(object_types)

        # 统计唯一实体数量
        entities = set()
        for t in triplets:
            entities.add(t.get('subject', ''))
            entities.add(t.get('object', ''))

        relation_count = len(relations)
        entity_type_count = len(entity_types)
        unique_entities = len(entities)

        # 计算多样性指数
        # 使用香农熵衡量分布均匀度
        relation_counter = Counter(t.get('relation', '') for t in triplets)
        relation_entropy = self._compute_entropy([v for v in relation_counter.values()])

        entity_type_counter = Counter()
        for t in triplets:
            entity_type_counter[t.get('subject_type', '')] += 1
            entity_type_counter[t.get('object_type', '')] += 1
        entity_type_entropy = self._compute_entropy([v for v in entity_type_counter.values()])

        # 多样性评分 (0-100)
        # 类型数量多 + 分布均匀 = 高多样性
        diversity_score = (
            min(relation_count / 20, 1.0) * 30 +      # 关系类型数量(上限20个给满分)
            min(entity_type_count / 15, 1.0) * 30 +   # 实体类型数量(上限15个给满分)
            (relation_entropy / 3.0) * 20 +           # 关系分布均匀度
            (entity_type_entropy / 3.0) * 20          # 实体类型分布均匀度
        )

        return {
            'relation_type_count': relation_count,
            'entity_type_count': entity_type_count,
            'unique_entities_count': unique_entities,
            'relation_entropy': round(relation_entropy, 3),
            'entity_type_entropy': round(entity_type_entropy, 3),
            'diversity_score': round(diversity_score, 2)
        }

    def _compute_entropy(self, counts: List[int]) -> float:
        """计算香农熵"""
        if not counts:
            return 0.0
        total = sum(counts)
        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log2(p) for p in probs)

    def _compute_evidence_quality(self, triplets: List[Dict]) -> Dict[str, Any]:
        """计算Evidence质量评分

        基于:
        1. evidence_spans完整性
        2. evidence长度合理性
        3. spans数量合理性
        """
        if not triplets:
            return {
                'spans_coverage': 0.0,
                'avg_evidence_length': 0.0,
                'avg_spans_per_triplet': 0.0,
                'evidence_score': 0.0
            }

        # 统计evidence_spans
        with_spans = sum(1 for t in triplets if t.get('evidence_spans') and len(t['evidence_spans']) > 0)
        spans_coverage = with_spans / len(triplets)

        # 统计evidence长度
        evidence_lengths = [len(t.get('evidence', '')) for t in triplets if t.get('evidence')]
        avg_evidence_length = np.mean(evidence_lengths) if evidence_lengths else 0

        # 统计每个三元组的spans数量
        spans_counts = [len(t.get('evidence_spans', [])) for t in triplets]
        avg_spans = np.mean(spans_counts) if spans_counts else 0

        # Evidence质量评分 (0-100)
        # 高覆盖率 + 合理长度 + 适量spans
        evidence_score = (
            spans_coverage * 50 +                           # spans覆盖率贡献50分
            min(avg_evidence_length / 50, 1.0) * 30 +       # 平均长度合理性(50字符为满分)
            min(avg_spans / 2, 1.0) * 20                    # 平均spans数量(2个为满分)
        )

        return {
            'spans_coverage': round(spans_coverage, 3),
            'avg_evidence_length': round(avg_evidence_length, 2),
            'avg_spans_per_triplet': round(avg_spans, 2),
            'evidence_score': round(evidence_score, 2)
        }

    def _compute_performance_metrics(self, stats: Dict) -> Dict[str, Any]:
        """计算性能指标"""
        processing = stats.get('processing_summary', {})

        return {
            'total_processing_time': processing.get('total_processing_time', 0),
            'avg_time_per_doc': processing.get('avg_time_per_document', 0),
            'total_tokens': processing.get('total_tokens_used', 0),
            'speed_score': self._compute_speed_score(processing.get('avg_time_per_document', 0))
        }

    def _compute_speed_score(self, avg_time: float) -> float:
        """计算速度评分 (0-100)

        基准: 2秒/文档 = 100分, 10秒/文档 = 50分
        """
        if avg_time <= 0:
            return 100.0

        # 指数衰减评分
        score = 100 * np.exp(-avg_time / 5)
        return round(max(0, min(100, score)), 2)

    def _compute_cost_efficiency(self, stats: Dict, triplets: List[Dict]) -> Dict[str, Any]:
        """计算成本效益

        考虑:
        1. 每个三元组的Token成本
        2. 每个文档的Token成本
        3. Token效率(三元组数/Token数)
        """
        processing = stats.get('processing_summary', {})
        total_tokens = processing.get('total_tokens_used', 0)
        total_docs = processing.get('total_documents', 1)
        total_triplets = len(triplets)

        if total_tokens == 0:
            return {
                'tokens_per_triplet': 0,
                'tokens_per_doc': 0,
                'triplets_per_1k_tokens': 0,
                'cost_efficiency_score': 0
            }

        tokens_per_triplet = total_tokens / max(total_triplets, 1)
        tokens_per_doc = total_tokens / max(total_docs, 1)
        triplets_per_1k_tokens = (total_triplets / total_tokens) * 1000

        # 成本效益评分 (0-100)
        # 高三元组密度 = 高效率
        cost_efficiency_score = min(triplets_per_1k_tokens / 2, 1.0) * 100

        return {
            'tokens_per_triplet': round(tokens_per_triplet, 2),
            'tokens_per_doc': round(tokens_per_doc, 2),
            'triplets_per_1k_tokens': round(triplets_per_1k_tokens, 3),
            'cost_efficiency_score': round(cost_efficiency_score, 2)
        }

    def _compute_overall_score(self, result: Dict[str, Any]) -> float:
        """计算综合得分

        加权平均各项评分
        """
        weights = {
            'quality_scores': 0.35,        # 质量最重要
            'schema_compliance': 0.25,     # Schema符合度
            'consistency_scores': 0.20,    # 一致性
            'diversity_scores': 0.20,      # 多样性
            # 移除了 evidence_quality, cost_efficiency, speed
        }

        score = 0.0
        for key, weight in weights.items():
            score_key = f"{key.replace('_scores', '')}_score" if key.endswith('_scores') else f"{key}_score"
            sub_result = result.get(key, {})

            if score_key in sub_result:
                score += sub_result[score_key] * weight

        return round(score, 2)

    def _print_model_summary(self, model_name: str, result: Dict[str, Any]):
        """打印模型评估摘要"""
        print(f"\n【{model_name}】评估摘要:")
        print(f"  综合得分: {result['overall_score']}/100")
        print(f"  - 质量评分: {result['quality_scores']['confidence_score']}/100")
        print(f"  - Schema符合度: {result['schema_compliance']['schema_score']}/100")
        print(f"  - 一致性评分: {result['consistency_scores']['consistency_score']}/100")
        print(f"  - 多样性评分: {result['diversity_scores']['diversity_score']}/100")
        print(f"\n  基础统计:")
        print(f"  - 三元组数: {result['basic_stats']['total_triplets']}")
        print(f"  - 平均置信度: {result['quality_scores']['avg_confidence']}")
        print(f"  - 配置内比例: {result['schema_compliance']['in_config_ratio']*100:.1f}%")
        print(f"  - 处理时间: {result['basic_stats']['total_processing_time']:.2f}s")

    def generate_comparison_report(self):
        """生成对比报告"""
        if not self.evaluation_results:
            print("没有评估结果，请先运行 evaluate_all_models()")
            return

        print("\n" + "="*80)
        print("生成对比报告...")
        print("="*80)

        # 1. 生成Excel报告
        excel_path = self._generate_excel_report()
        print(f"✓ Excel报告: {excel_path}")

        # 2. 生成JSON报告
        json_path = self._generate_json_report()
        print(f"✓ JSON报告: {json_path}")

        # 3. 生成可视化图表
        chart_paths = self._generate_charts()
        print(f"✓ 可视化图表: {len(chart_paths)} 个")
        for path in chart_paths:
            print(f"  - {path}")

        # 4. 生成Markdown报告
        md_path = self._generate_markdown_report()
        print(f"✓ Markdown报告: {md_path}")

        print("\n" + "="*80)
        print("评估完成！")
        print("="*80)

    def _generate_excel_report(self) -> str:
        """生成Excel格式的详细对比报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: 综合对比
            self._write_overview_sheet(writer)

            # Sheet 2: 质量指标
            self._write_quality_sheet(writer)

            # Sheet 3: Schema符合度
            self._write_schema_sheet(writer)

            # Sheet 4: 性能与成本
            self._write_performance_sheet(writer)

            # Sheet 5: 详细指标
            self._write_detailed_sheet(writer)

        return excel_path

    def _write_overview_sheet(self, writer):
        """写入概览sheet"""
        data = []
        for model_name, result in self.evaluation_results.items():
            row = {
                '模型': model_name,
                '综合得分': result['overall_score'],
                '质量评分': result['quality_scores']['confidence_score'],
                'Schema符合度': result['schema_compliance']['schema_score'],
                '一致性评分': result['consistency_scores']['consistency_score'],
                '多样性评分': result['diversity_scores']['diversity_score'],
                '三元组总数': result['basic_stats']['total_triplets'],
                '处理时间(s)': result['basic_stats']['total_processing_time'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('综合得分', ascending=False)
        df.to_excel(writer, sheet_name='综合对比', index=False)

    def _write_quality_sheet(self, writer):
        """写入质量指标sheet"""
        data = []
        for model_name, result in self.evaluation_results.items():
            quality = result['quality_scores']
            row = {
                '模型': model_name,
                '平均置信度': quality['avg_confidence'],
                '高置信度比例': quality['high_confidence_ratio'],
                '低置信度比例': quality['low_confidence_ratio'],
                '置信度标准差': quality['confidence_std'],
                '质量评分': quality['confidence_score'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='质量指标', index=False)

    def _write_schema_sheet(self, writer):
        """写入Schema符合度sheet"""
        data = []
        for model_name, result in self.evaluation_results.items():
            schema = result['schema_compliance']
            row = {
                '模型': model_name,
                '配置内比例': schema['in_config_ratio'],
                '关系符合度': schema['relation_compliance'],
                '实体符合度': schema['entity_compliance'],
                'Schema评分': schema['schema_score'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Schema符合度', index=False)

    def _write_performance_sheet(self, writer):
        """写入性能与成本sheet"""
        data = []
        for model_name, result in self.evaluation_results.items():
            perf = result['performance_metrics']
            cost = result['cost_efficiency']
            row = {
                '模型': model_name,
                '总处理时间(s)': perf['total_processing_time'],
                '平均时间/文档(s)': perf['avg_time_per_doc'],
                '总Token数': perf['total_tokens'],
                'Token/三元组': cost['tokens_per_triplet'],
                'Token/文档': cost['tokens_per_doc'],
                '三元组/1K Tokens': cost['triplets_per_1k_tokens'],
                '速度评分': perf['speed_score'],
                '成本效益评分': cost['cost_efficiency_score'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='性能与成本', index=False)

    def _write_detailed_sheet(self, writer):
        """写入详细指标sheet"""
        data = []
        for model_name, result in self.evaluation_results.items():
            row = {'模型': model_name}

            # 展平所有指标
            for category, metrics in result.items():
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str)):
                            row[f"{category}_{key}"] = value

            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='详细指标', index=False)

    def _generate_json_report(self) -> str:
        """生成JSON格式报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.json")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)

        return json_path

    def _generate_charts(self) -> List[str]:
        """生成可视化图表"""
        chart_paths = []

        # 设置图表风格
        sns.set_style("whitegrid")

        # 1. 综合雷达图
        radar_path = self._plot_radar_chart()
        chart_paths.append(radar_path)

        # 2. 各维度对比柱状图
        bar_path = self._plot_comparison_bars()
        chart_paths.append(bar_path)

        # 3. 性能-质量散点图
        scatter_path = self._plot_performance_quality_scatter()
        chart_paths.append(scatter_path)

        # 4. 置信度分布对比
        conf_path = self._plot_confidence_distributions()
        chart_paths.append(conf_path)

        return chart_paths

    def _plot_radar_chart(self) -> str:
        """绘制综合雷达图"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(self.output_dir, f"radar_comparison_{timestamp}.png")

        # 准备数据
        categories = ['质量', 'Schema', '一致性', '多样性']
        models = list(self.evaluation_results.keys())

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        for model_name in models:
            result = self.evaluation_results[model_name]
            values = [
                result['quality_scores']['confidence_score'],
                result['schema_compliance']['schema_score'],
                result['consistency_scores']['consistency_score'],
                result['diversity_scores']['diversity_score'],
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('模型综合性能对比雷达图', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        return chart_path

    def _plot_comparison_bars(self) -> str:
        """绘制各维度对比柱状图"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(self.output_dir, f"bar_comparison_{timestamp}.png")

        models = list(self.evaluation_results.keys())
        metrics = {
            '质量': [self.evaluation_results[m]['quality_scores']['confidence_score'] for m in models],
            'Schema': [self.evaluation_results[m]['schema_compliance']['schema_score'] for m in models],
            '一致性': [self.evaluation_results[m]['consistency_scores']['consistency_score'] for m in models],
            '多样性': [self.evaluation_results[m]['diversity_scores']['diversity_score'] for m in models],
        }

        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (metric_name, values) in enumerate(metrics.items()):
            offset = width * (i - len(metrics) / 2)
            ax.bar(x + offset, values, width, label=metric_name)

        ax.set_xlabel('模型')
        ax.set_ylabel('评分')
        ax.set_title('各维度性能对比', size=16)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        return chart_path

    def _plot_performance_quality_scatter(self) -> str:
        """绘制性能-质量散点图"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(self.output_dir, f"scatter_perf_quality_{timestamp}.png")

        models = list(self.evaluation_results.keys())

        speeds = [self.evaluation_results[m]['performance_metrics']['speed_score'] for m in models]
        qualities = [self.evaluation_results[m]['quality_scores']['confidence_score'] for m in models]
        sizes = [self.evaluation_results[m]['basic_stats']['total_triplets'] / 10 for m in models]

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(speeds, qualities, s=sizes, alpha=0.6, c=range(len(models)), cmap='viridis')

        for i, model in enumerate(models):
            ax.annotate(model, (speeds[i], qualities[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)

        ax.set_xlabel('速度评分', fontsize=12)
        ax.set_ylabel('质量评分', fontsize=12)
        ax.set_title('性能-质量权衡分析\n(气泡大小=三元组数量)', size=16)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        return chart_path

    def _plot_confidence_distributions(self) -> str:
        """绘制置信度分布对比"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(self.output_dir, f"confidence_dist_{timestamp}.png")

        fig, axes = plt.subplots(1, len(self.models_data), figsize=(6*len(self.models_data), 5))

        if len(self.models_data) == 1:
            axes = [axes]

        for i, (model_name, data) in enumerate(self.models_data.items()):
            triplets = data.get('triplets', {}).get('all', [])
            confidences = [t.get('confidence', 0) for t in triplets]

            axes[i].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_xlabel('置信度')
            axes[i].set_ylabel('三元组数量')
            axes[i].set_title(f'{model_name}\n平均: {np.mean(confidences):.3f}')
            axes[i].grid(axis='y', alpha=0.3)

        plt.suptitle('置信度分布对比', size=16)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        return chart_path

    def _generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.md")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 模型性能对比评估报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. 综合排名
            f.write("## 1. 综合排名\n\n")
            ranking = sorted(self.evaluation_results.items(),
                           key=lambda x: x[1]['overall_score'], reverse=True)

            f.write("| 排名 | 模型 | 综合得分 | 三元组数 | 处理时间 |\n")
            f.write("|------|------|----------|----------|----------|\n")
            for rank, (model, result) in enumerate(ranking, 1):
                f.write(f"| {rank} | **{model}** | {result['overall_score']}/100 | "
                       f"{result['basic_stats']['total_triplets']} | "
                       f"{result['basic_stats']['total_processing_time']:.2f}s |\n")

            # 2. 各维度冠军
            f.write("\n## 2. 各维度最佳模型\n\n")

            dimensions = {
                '质量': ('quality_scores', 'confidence_score'),
                'Schema符合度': ('schema_compliance', 'schema_score'),
                '一致性': ('consistency_scores', 'consistency_score'),
                '多样性': ('diversity_scores', 'diversity_score'),
            }

            for dim_name, (cat, key) in dimensions.items():
                best_model = max(self.evaluation_results.items(),
                               key=lambda x: x[1][cat][key])
                score = best_model[1][cat][key]
                f.write(f"- **{dim_name}**: {best_model[0]} ({score}/100)\n")

            # 3. 详细分析
            f.write("\n## 3. 详细分析\n\n")
            for model_name, result in ranking:
                f.write(f"### {model_name}\n\n")
                f.write(f"**综合得分**: {result['overall_score']}/100\n\n")

                f.write("**优势**:\n")
                # 找出得分>80的维度
                high_scores = []
                for dim_name, (cat, key) in dimensions.items():
                    if result[cat][key] > 80:
                        high_scores.append(f"{dim_name} ({result[cat][key]}/100)")

                if high_scores:
                    for item in high_scores:
                        f.write(f"- {item}\n")
                else:
                    f.write("- 各维度表现均衡\n")

                f.write("\n**基础统计**:\n")
                f.write(f"- 三元组总数: {result['basic_stats']['total_triplets']}\n")
                f.write(f"- 平均置信度: {result['quality_scores']['avg_confidence']}\n")
                f.write(f"- 配置内比例: {result['schema_compliance']['in_config_ratio']*100:.1f}%\n")
                f.write(f"- 处理时间: {result['basic_stats']['total_processing_time']:.2f}s\n")
                f.write(f"- Token使用: {result['performance_metrics']['total_tokens']}\n")
                f.write("\n")

        return md_path


def main():
    """主函数 - 使用示例"""
    print("="*80)
    print("模型性能自动化评估系统")
    print("="*80)

    # 创建评估器
    evaluator = ModelEvaluator(output_dir="evaluation_results")

    # 加载模型输出（请根据实际情况修改路径）
    models_to_evaluate = {
        "deepseek-v3": "data/output/knowledge_graph_deepseek.json",
        "gemini-2.5-flash": "data/output/knowledge_graph_gemini-2.5-flash.json",
        "gpt-5": "data/output/knowledge_graph_gpt-5.json",
    }

    # 检查文件是否存在并加载
    loaded_models = []
    for model_name, file_path in models_to_evaluate.items():
        if os.path.exists(file_path):
            evaluator.load_model_output(model_name, file_path)
            loaded_models.append(model_name)
        else:
            print(f"⚠️  文件不存在: {file_path}")

    if not loaded_models:
        print("\n❌ 没有找到任何模型输出文件，请检查路径配置")
        print("请修改 models_to_evaluate 字典中的文件路径")
        return

    print(f"\n成功加载 {len(loaded_models)} 个模型的输出")

    # 执行评估
    results = evaluator.evaluate_all_models()

    # 生成对比报告
    evaluator.generate_comparison_report()

    print("\n✅ 评估完成！请查看 evaluation_results 目录下的报告文件")


if __name__ == "__main__":
    main()
