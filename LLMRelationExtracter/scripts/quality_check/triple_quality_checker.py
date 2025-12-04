#!/usr/bin/env python3
"""
中文三元组质量评估器（核心模块）

实现两个核心指标：
1. support_score: 证据支持度（0-1），判断三元组是否被来源句支持
2. consistency_score: 稳健一致性（0-1），评估抽取结果的稳定性
"""

import json
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

# 尝试导入可选依赖，缺失则自动降级
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# 尝试导入句向量模型（可选）
SENTENCE_TRANSFORMER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TripleQualityChecker:
    """中文三元组质量评估器（无参照）"""

    def __init__(self, output_dir: str = "data/output/quality_check_results"):
        """初始化评估器

        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 评估权重
        self.alpha = 0.6  # support_score权重
        self.beta = 0.4   # consistency_score权重
        self.fuzzy_threshold = 80

        # 初始化句向量模型（可选）
        self.sentence_model = None
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                logger.info("尝试加载句向量模型...")
                self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
                logger.info("✓ 句向量模型加载成功")
            except Exception as e:
                logger.warning(f"句向量模型加载失败，将使用TF-IDF: {e}")

        # TF-IDF向量化器（备用）
        self.tfidf = None
        if SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(
                analyzer='char',
                ngram_range=(1, 3),
                max_features=1000
            )

        # 内置谓词同义词表
        self.predicate_synonyms = self._build_predicate_synonyms()

        self.kg_data = {}
        logger.info("✓ TripleQualityChecker初始化完成")

    def _build_predicate_synonyms(self) -> Dict[str, List[str]]:
        """构建谓词同义词表"""
        return {
            '拥有': ['有', '持有', '具有', '拥有'],
            '位于': ['在', '处于', '坐落', '位于'],
            '生产': ['制造', '生产', '出产', '研发'],
            '研发': ['开发', '研制', '研发', '设计'],
            '竞争': ['对手', '竞争', '角逐', 'PK'],
            '合作': ['协作', '合作', '联合', '携手'],
            '收购': ['并购', '收购', '购买', '兼并'],
            '投资': ['注资', '投资', '入股', '融资'],
            '增长': ['提升', '增加', '增长', '上升'],
            '下降': ['减少', '下降', '降低', '下跌'],
            '发布': ['推出', '发布', '公布', '宣布'],
            '应用': ['使用', '应用', '采用', '运用'],
        }

    def load_kg_output(self, kg_name: str, kg_file_path: str):
        """加载知识图谱输出文件

        Args:
            kg_name: 知识图谱名称（如 "deepseek", "gpt-4"）
            kg_file_path: 知识图谱JSON文件路径
        """
        print(f"加载知识图谱 [{kg_name}] 的输出: {kg_file_path}")

        try:
            with open(kg_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            triplets = data.get('triplets', {}).get('all', [])
            self.kg_data[kg_name] = {
                'data': data,
                'triplets': triplets
            }
            print(f"  ✓ 成功加载，三元组数量: {len(triplets)}")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            raise

    def _fuzzy_match(self, text: str, target: str, threshold: int = 80) -> bool:
        """模糊匹配两个字符串"""
        if not text or not target:
            return False

        text = text.lower().strip()
        target = target.lower().strip()

        if target in text:
            return True

        if RAPIDFUZZ_AVAILABLE:
            try:
                score = fuzz.partial_ratio(text, target)
                return score >= threshold
            except:
                pass

        # 降级：简单字符重叠率
        target_chars = set(target)
        text_chars = set(text)
        overlap = len(target_chars & text_chars)
        ratio = (overlap / len(target_chars)) * 100 if target_chars else 0
        return ratio >= threshold

    def _compute_semantic_similarity(self, premise: str, hypothesis: str) -> float:
        """计算两个句子的语义相似度"""
        if not premise or not hypothesis:
            return 0.0

        # 方法1：使用句向量（首选）
        if self.sentence_model is not None:
            try:
                embeddings = self.sentence_model.encode([premise, hypothesis])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(np.clip(similarity, 0, 1))
            except Exception as e:
                logger.warning(f"句向量计算失败，降级到TF-IDF: {e}")

        # 方法2：降级到TF-IDF
        if SKLEARN_AVAILABLE and self.tfidf is not None:
            try:
                vectors = self.tfidf.fit_transform([premise, hypothesis])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                return float(np.clip(similarity, 0, 1))
            except Exception as e:
                logger.warning(f"TF-IDF计算失败: {e}")

        # 方法3：最简单的Jaccard相似度
        set1 = set(premise)
        set2 = set(hypothesis)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _triplet_to_hypothesis(self, subject: str, predicate: str, obj: str) -> List[str]:
        """将三元组转换为中文假设句（三种模板）"""
        templates = [
            f"{subject}的{predicate}是{obj}。",
            f"{subject}{predicate}{obj}。",
            f"关于{subject}，可以说其{predicate}为{obj}。"
        ]
        return templates

    def _check_predicate_match(self, premise: str, predicate: str) -> bool:
        """检查谓词是否在证据中（使用同义词表）"""
        premise_lower = premise.lower()

        if predicate.lower() in premise_lower:
            return True

        for key, synonyms in self.predicate_synonyms.items():
            if predicate in synonyms or predicate == key:
                for syn in synonyms:
                    if syn in premise_lower:
                        return True

        return False

    def compute_support_score(
        self,
        subject: str,
        predicate: str,
        obj: str,
        evidence: str,
        subject_attributes: Dict[str, Any] = None,
        object_attributes: Dict[str, Any] = None
    ) -> float:
        """计算证据支持度（包含属性）

        Args:
            subject: 主体
            predicate: 谓词
            obj: 客体
            evidence: 证据句
            subject_attributes: 主体属性字典
            object_attributes: 客体属性字典

        Returns:
            support_score (0-1)
        """
        if not evidence:
            return 0.0

        # 1. 语义相似度（三模板取max）
        hypotheses = self._triplet_to_hypothesis(subject, predicate, obj)
        similarities = []

        for hyp in hypotheses:
            sim = self._compute_semantic_similarity(evidence, hyp)
            similarities.append(sim)

        max_sim = max(similarities) if similarities else 0.0

        # 2. 启发式匹配
        subject_match = self._fuzzy_match(evidence, subject, self.fuzzy_threshold)
        object_match = self._fuzzy_match(evidence, obj, self.fuzzy_threshold)
        predicate_match = self._check_predicate_match(evidence, predicate)

        # 启发式得分
        heuristic_score = 0.0
        if subject_match and object_match:
            heuristic_score = 0.5
            if predicate_match:
                heuristic_score += 0.3
        elif subject_match or object_match:
            heuristic_score = 0.2

        # 3. 属性支持度评估
        attribute_score = self._compute_attribute_support(
            evidence,
            subject_attributes or {},
            object_attributes or {}
        )

        # 4. 综合评分：0.5*语义 + 0.3*启发式 + 0.2*属性
        support_score = 0.5 * max_sim + 0.3 * heuristic_score + 0.2 * attribute_score
        support_score = float(np.clip(support_score, 0, 1))

        return support_score

    def _compute_attribute_support(
        self,
        evidence: str,
        subject_attributes: Dict[str, Any],
        object_attributes: Dict[str, Any]
    ) -> float:
        """计算属性的证据支持度

        Args:
            evidence: 证据句
            subject_attributes: 主体属性
            object_attributes: 客体属性

        Returns:
            属性支持度 (0-1)
        """
        all_attributes = {}
        all_attributes.update(subject_attributes)
        all_attributes.update(object_attributes)

        if not all_attributes:
            return 1.0  # 无属性时返回满分

        matched_count = 0
        total_count = len(all_attributes)

        for attr_key, attr_value in all_attributes.items():
            # 检查属性键是否在证据中
            key_match = self._fuzzy_match(evidence, str(attr_key), threshold=70)

            # 检查属性值是否在证据中
            value_str = str(attr_value)
            value_match = self._fuzzy_match(evidence, value_str, threshold=70)

            # 如果键或值匹配，则认为该属性被支持
            if key_match or value_match:
                matched_count += 1

        # 计算匹配率
        attribute_support = matched_count / total_count if total_count > 0 else 1.0
        return attribute_support

    def compute_consistency_score(self, triplet: Dict[str, Any]) -> float:
        """计算稳健一致性

        Args:
            triplet: 三元组字典

        Returns:
            consistency_score (0-1)
        """
        # 代理指标1：置信度
        confidence = triplet.get('confidence', 0.5)

        # 代理指标2：证据完整性
        evidence = triplet.get('evidence', '')
        evidence_spans = triplet.get('evidence_spans', [])

        has_evidence = len(evidence) > 0
        has_spans = len(evidence_spans) > 0
        evidence_quality = 0.5

        if has_evidence and has_spans:
            evidence_quality = 1.0
        elif has_evidence:
            evidence_quality = 0.7

        # 代理指标3：类型符合性
        is_relation_in_config = triplet.get('is_relation_in_config', False)
        is_subject_in_config = triplet.get('is_subject_type_in_config', False)
        is_object_in_config = triplet.get('is_object_type_in_config', False)

        type_compliance = sum([
            is_relation_in_config,
            is_subject_in_config,
            is_object_in_config
        ]) / 3.0

        # 综合评分：置信度50% + 证据质量30% + 类型符合性20%
        consistency_score = (
            0.5 * confidence +
            0.3 * evidence_quality +
            0.2 * type_compliance
        )
        consistency_score = float(np.clip(consistency_score, 0, 1))

        return consistency_score

    def evaluate_single_kg(self, kg_name: str) -> Dict[str, Any]:
        """评估单个知识图谱

        Args:
            kg_name: 知识图谱名称

        Returns:
            评估结果字典
        """
        print(f"\n评估知识图谱: {kg_name}")
        print("-"*80)

        kg_info = self.kg_data[kg_name]
        triplets = kg_info['triplets']
        data = kg_info['data']

        evaluated_triplets = []
        support_scores = []
        consistency_scores = []
        overall_scores = []

        # 属性统计
        total_subject_attrs = 0
        total_object_attrs = 0
        triplets_with_subject_attrs = 0
        triplets_with_object_attrs = 0

        # 逐个评估三元组
        for i, triplet in enumerate(triplets):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(triplets)}")

            try:
                # 统计属性
                subject_attrs = triplet.get('subject_attributes', {})
                object_attrs = triplet.get('object_attributes', {})

                if subject_attrs:
                    triplets_with_subject_attrs += 1
                    total_subject_attrs += len(subject_attrs)

                if object_attrs:
                    triplets_with_object_attrs += 1
                    total_object_attrs += len(object_attrs)

                # 计算support_score（包含属性评估）
                support_score = self.compute_support_score(
                    triplet.get('subject', ''),
                    triplet.get('relation', ''),
                    triplet.get('object', ''),
                    triplet.get('evidence', ''),
                    subject_attributes=subject_attrs,
                    object_attributes=object_attrs
                )

                # 计算consistency_score
                consistency_score = self.compute_consistency_score(triplet)

                # 综合质量分
                overall_quality = self.alpha * support_score + self.beta * consistency_score

                # 保存评分（内部字段）
                triplet_copy = triplet.copy()
                triplet_copy['_qc_support_score'] = round(support_score, 3)
                triplet_copy['_qc_consistency_score'] = round(consistency_score, 3)
                triplet_copy['_qc_overall'] = round(overall_quality, 3)

                evaluated_triplets.append(triplet_copy)
                support_scores.append(support_score)
                consistency_scores.append(consistency_score)
                overall_scores.append(overall_quality)

            except Exception as e:
                logger.warning(f"评估三元组失败: {e}")
                # 保守评分
                triplet_copy = triplet.copy()
                triplet_copy['_qc_support_score'] = 0.5
                triplet_copy['_qc_consistency_score'] = 0.5
                triplet_copy['_qc_overall'] = 0.5
                evaluated_triplets.append(triplet_copy)
                support_scores.append(0.5)
                consistency_scores.append(0.5)
                overall_scores.append(0.5)

        # 统计结果
        result = {
            'kg_name': kg_name,
            'total_triplets': len(triplets),
            'support_score': {
                'mean': round(float(np.mean(support_scores)), 3),
                'std': round(float(np.std(support_scores)), 3),
                'min': round(float(np.min(support_scores)), 3),
                'max': round(float(np.max(support_scores)), 3),
                'median': round(float(np.median(support_scores)), 3),
            },
            'consistency_score': {
                'mean': round(float(np.mean(consistency_scores)), 3),
                'std': round(float(np.std(consistency_scores)), 3),
                'min': round(float(np.min(consistency_scores)), 3),
                'max': round(float(np.max(consistency_scores)), 3),
                'median': round(float(np.median(consistency_scores)), 3),
            },
            'overall_quality': {
                'mean': round(float(np.mean(overall_scores)), 3),
                'std': round(float(np.std(overall_scores)), 3),
                'min': round(float(np.min(overall_scores)), 3),
                'max': round(float(np.max(overall_scores)), 3),
                'median': round(float(np.median(overall_scores)), 3),
            },
            'quality_distribution': {
                'high_quality': sum(1 for s in overall_scores if s >= 0.7),
                'medium_quality': sum(1 for s in overall_scores if 0.4 <= s < 0.7),
                'low_quality': sum(1 for s in overall_scores if s < 0.4),
            },
            'attribute_statistics': {
                'triplets_with_subject_attrs': triplets_with_subject_attrs,
                'triplets_with_object_attrs': triplets_with_object_attrs,
                'total_subject_attrs': total_subject_attrs,
                'total_object_attrs': total_object_attrs,
                'avg_subject_attrs_per_triplet': round(total_subject_attrs / len(triplets), 2) if triplets else 0,
                'avg_object_attrs_per_triplet': round(total_object_attrs / len(triplets), 2) if triplets else 0,
                'subject_attr_coverage': round(triplets_with_subject_attrs / len(triplets) * 100, 1) if triplets else 0,
                'object_attr_coverage': round(triplets_with_object_attrs / len(triplets) * 100, 1) if triplets else 0,
            },
            'evaluated_triplets': evaluated_triplets
        }

        self._print_kg_summary(kg_name, result)
        return result

    def _print_kg_summary(self, kg_name: str, result: Dict[str, Any]):
        """打印知识图谱评估摘要"""
        print(f"\n【{kg_name}】评估摘要:")
        print(f"  三元组总数: {result['total_triplets']}")
        print(f"  平均支持度: {result['support_score']['mean']:.3f} (±{result['support_score']['std']:.3f})")
        print(f"  平均一致性: {result['consistency_score']['mean']:.3f} (±{result['consistency_score']['std']:.3f})")
        print(f"  平均综合质量: {result['overall_quality']['mean']:.3f}")
        print(f"  质量分布:")
        total = result['total_triplets']
        print(f"    高质量(≥0.7): {result['quality_distribution']['high_quality']} ({result['quality_distribution']['high_quality']/total*100:.1f}%)")
        print(f"    中等质量(0.4-0.7): {result['quality_distribution']['medium_quality']} ({result['quality_distribution']['medium_quality']/total*100:.1f}%)")
        print(f"    低质量(<0.4): {result['quality_distribution']['low_quality']} ({result['quality_distribution']['low_quality']/total*100:.1f}%)")

        # 打印属性统计
        attr_stats = result['attribute_statistics']
        print(f"  属性统计:")
        print(f"    主体属性覆盖率: {attr_stats['subject_attr_coverage']:.1f}% ({attr_stats['triplets_with_subject_attrs']}/{total})")
        print(f"    客体属性覆盖率: {attr_stats['object_attr_coverage']:.1f}% ({attr_stats['triplets_with_object_attrs']}/{total})")
        print(f"    平均主体属性数: {attr_stats['avg_subject_attrs_per_triplet']:.2f}")
        print(f"    平均客体属性数: {attr_stats['avg_object_attrs_per_triplet']:.2f}")

    def evaluate_all_kgs(self) -> Dict[str, Any]:
        """评估所有已加载的知识图谱"""
        print("\n" + "="*80)
        print("开始质量评估...")
        print("="*80)

        results = {}

        for kg_name in self.kg_data.keys():
            result = self.evaluate_single_kg(kg_name)
            results[kg_name] = result

        return results

    def generate_quality_report(self, results: Dict[str, Any]):
        """生成质量评估报告"""
        print("\n" + "="*80)
        print("生成评估报告...")
        print("="*80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存JSON报告
        json_path = os.path.join(self.output_dir, f"quality_check_report_{timestamp}.json")

        report_data = {
            'metadata': {
                'evaluation_time': datetime.now().isoformat(),
                'evaluator': 'TripleQualityChecker',
                'version': '1.0',
                'total_kgs': len(results)
            },
            'results': results
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"✓ JSON报告: {json_path}")

        # 2. 生成Markdown摘要
        md_path = os.path.join(self.output_dir, f"quality_check_summary_{timestamp}.md")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 三元组质量评估报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 综合排名
            f.write("## 综合质量排名\n\n")
            ranking = sorted(results.items(),
                           key=lambda x: x[1]['overall_quality']['mean'], reverse=True)

            f.write("| 排名 | 知识图谱 | 综合质量 | 支持度 | 一致性 | 三元组数 |\n")
            f.write("|------|----------|----------|--------|--------|----------|\n")
            for rank, (kg_name, result) in enumerate(ranking, 1):
                f.write(f"| {rank} | **{kg_name}** | "
                       f"{result['overall_quality']['mean']:.3f} | "
                       f"{result['support_score']['mean']:.3f} | "
                       f"{result['consistency_score']['mean']:.3f} | "
                       f"{result['total_triplets']} |\n")

            # 各维度最佳
            f.write("\n## 各维度最佳知识图谱\n\n")
            best_support = max(results.items(), key=lambda x: x[1]['support_score']['mean'])
            best_consistency = max(results.items(), key=lambda x: x[1]['consistency_score']['mean'])
            best_overall = max(results.items(), key=lambda x: x[1]['overall_quality']['mean'])

            f.write(f"- **证据支持度**: {best_support[0]} ({best_support[1]['support_score']['mean']:.3f})\n")
            f.write(f"- **稳健一致性**: {best_consistency[0]} ({best_consistency[1]['consistency_score']['mean']:.3f})\n")
            f.write(f"- **综合质量**: {best_overall[0]} ({best_overall[1]['overall_quality']['mean']:.3f})\n")

            # 详细分析
            f.write("\n## 详细分析\n\n")
            for kg_name, result in ranking:
                f.write(f"### {kg_name}\n\n")
                f.write(f"- **三元组总数**: {result['total_triplets']}\n")
                f.write(f"- **平均支持度**: {result['support_score']['mean']:.3f}\n")
                f.write(f"- **平均一致性**: {result['consistency_score']['mean']:.3f}\n")
                f.write(f"- **平均综合质量**: {result['overall_quality']['mean']:.3f}\n")
                f.write(f"- **质量分布**:\n")
                total = result['total_triplets']
                f.write(f"  - 高质量: {result['quality_distribution']['high_quality']} ({result['quality_distribution']['high_quality']/total*100:.1f}%)\n")
                f.write(f"  - 中等质量: {result['quality_distribution']['medium_quality']} ({result['quality_distribution']['medium_quality']/total*100:.1f}%)\n")
                f.write(f"  - 低质量: {result['quality_distribution']['low_quality']} ({result['quality_distribution']['low_quality']/total*100:.1f}%)\n")

                # 添加属性统计
                attr_stats = result.get('attribute_statistics', {})
                if attr_stats:
                    f.write(f"- **属性抽取情况**:\n")
                    f.write(f"  - 主体属性覆盖率: {attr_stats.get('subject_attr_coverage', 0):.1f}%\n")
                    f.write(f"  - 客体属性覆盖率: {attr_stats.get('object_attr_coverage', 0):.1f}%\n")
                    f.write(f"  - 平均主体属性数: {attr_stats.get('avg_subject_attrs_per_triplet', 0):.2f}\n")
                    f.write(f"  - 平均客体属性数: {attr_stats.get('avg_object_attrs_per_triplet', 0):.2f}\n")

                f.write("\n")

        print(f"✓ Markdown摘要: {md_path}")

        # 3. 保存详细的三元组评估结果
        for kg_name, result in results.items():
            detail_path = os.path.join(self.output_dir, f"{kg_name}_quality_details_{timestamp}.json")
            with open(detail_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'kg_name': kg_name,
                    'statistics': {
                        'support_score': result['support_score'],
                        'consistency_score': result['consistency_score'],
                        'overall_quality': result['overall_quality'],
                        'quality_distribution': result['quality_distribution'],
                        'attribute_statistics': result.get('attribute_statistics', {})
                    },
                    'evaluated_triplets': result['evaluated_triplets']
                }, f, ensure_ascii=False, indent=2)
            print(f"✓ {kg_name} 详细结果: {detail_path}")

        print("\n" + "="*80)
        print("评估报告生成完成！")
        print("="*80)

        # 返回JSON报告路径供可视化使用
        return json_path
