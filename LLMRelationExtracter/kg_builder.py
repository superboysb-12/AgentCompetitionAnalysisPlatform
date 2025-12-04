import json
import csv
import os
import hashlib
import logging
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import threading

try:
    from .kg_extractor import KnowledgeGraphExtractor, ExtractionResult, Triplet
    from .schema_discoverer import SchemaDiscoverer
except ImportError:
    from kg_extractor import KnowledgeGraphExtractor, ExtractionResult, Triplet
    from schema_discoverer import SchemaDiscoverer

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """知识图谱构建器，负责批量处理文档和输出管理"""

    def __init__(self, config_path: str, checkpoint_name: str = None):
        """初始化知识图谱构建器

        Args:
            config_path: 配置文件路径
            checkpoint_name: 自定义checkpoint文件名（可选）
        """
        self.extractor = KnowledgeGraphExtractor(config_path)
        self.config = self.extractor.config
        self.all_triplets = []
        self.all_results = []

        # 保存自定义checkpoint名称
        self.custom_checkpoint_name = checkpoint_name

        # 初始化Schema发现器（如果启用）
        self.enable_schema_discovery = self.config.get('schema_discovery', {}).get('enabled', True)
        if self.enable_schema_discovery:
            output_dir = self.config.get('schema_discovery', {}).get('output_dir', 'data/output')
            self.schema_discoverer = SchemaDiscoverer(output_dir)
            logger.info("Schema发现功能已启用")
        else:
            self.schema_discoverer = None
            logger.info("Schema发现功能未启用")

        # 初始化断点续传
        self.checkpoint_dir = "data/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.processed_docs = set()  # 已处理的文档ID集合

        # 添加停止标志（用于余额不足等情况）
        self.should_stop = False
        self.stop_reason = ""
        self.stop_lock = threading.Lock()  # 线程锁保护停止标志

    def _get_checkpoint_path(self, input_file: str) -> str:
        """获取checkpoint文件路径

        Args:
            input_file: 输入文件路径

        Returns:
            checkpoint文件的完整路径
        """
        # 如果指定了自定义checkpoint名称，使用自定义名称
        if self.custom_checkpoint_name:
            checkpoint_file = self.custom_checkpoint_name
            # 确保文件名以.json结尾
            if not checkpoint_file.endswith('.json'):
                checkpoint_file += '.json'
            return os.path.join(self.checkpoint_dir, checkpoint_file)

        # 否则使用输入文件的hash作为checkpoint文件名
        file_hash = hashlib.md5(input_file.encode()).hexdigest()
        return os.path.join(self.checkpoint_dir, f"checkpoint_{file_hash}.json")

    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载checkpoint数据

        Args:
            checkpoint_path: checkpoint文件路径

        Returns:
            checkpoint数据字典，如果加载失败则返回空字典
        """
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                logger.info(f"加载checkpoint: 已处理 {len(checkpoint.get('processed_docs', []))} 个文档")
                return checkpoint
            except Exception as e:
                logger.warning(f"加载checkpoint失败: {e}, 将从头开始")
                return {}
        return {}

    def _save_checkpoint(self, checkpoint_path: str, processed_docs: List[str],
                        results: List[ExtractionResult]):
        """保存checkpoint数据

        Args:
            checkpoint_path: checkpoint文件路径
            processed_docs: 已处理的文档ID列表
            results: 抽取结果列表
        """
        try:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'processed_docs': processed_docs,
                'results_count': len(results),
                'triplets_count': sum(len(r.triplets) for r in results)
            }
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            logger.debug(f"Checkpoint已保存: {len(processed_docs)} 个文档")
        except Exception as e:
            logger.warning(f"保存checkpoint失败: {e}")

    def _save_intermediate_triplets(self, checkpoint_path: str, results: List[ExtractionResult]):
        """保存中间三元组结果到文件

        Args:
            checkpoint_path: checkpoint文件路径
            results: 抽取结果列表
        """
        try:
            triplets_path = checkpoint_path.replace('.json', '_triplets.json')
            all_triplets = []
            for result in results:
                all_triplets.extend([t.to_dict() for t in result.triplets])

            with open(triplets_path, 'w', encoding='utf-8') as f:
                json.dump(all_triplets, f, ensure_ascii=False, indent=2)
            logger.debug(f"中间结果已保存: {len(all_triplets)} 个三元组")
        except Exception as e:
            logger.warning(f"保存中间结果失败: {e}")

    def _deduplicate_triplets(self, triplets: List[Triplet]) -> List[Triplet]:
        """去除重复的关系三元组

        Args:
            triplets: 原始三元组列表

        Returns:
            去重后的三元组列表
        """
        if not self.config['output']['deduplicate']:
            return triplets

        seen = set()
        deduplicated = []

        for triplet in triplets:
            # 创建唯一标识
            key = (triplet.subject.lower(), triplet.relation, triplet.object.lower())
            if key not in seen:
                seen.add(key)
                deduplicated.append(triplet)

        removed_count = len(triplets) - len(deduplicated)
        if removed_count > 0:
            logger.info(f"去重完成: 移除了 {removed_count} 个重复三元组")

        return deduplicated

    def process_documents(self, documents: List[Dict[str, Any]], checkpoint_path: str = None,
                         enable_checkpoint: bool = True) -> List[ExtractionResult]:
        """批量处理文档列表，支持断点续传

        Args:
            documents: 待处理的文档列表
            checkpoint_path: checkpoint文件路径（可选）
            enable_checkpoint: 是否启用断点续传

        Returns:
            所有文档的抽取结果列表
        """
        batch_size = self.config['processing']['batch_size']
        enable_parallel = self.config['processing']['enable_parallel']
        max_workers = self.config['processing']['max_workers']

        results = []

        # 加载checkpoint
        processed_doc_ids = set()
        if enable_checkpoint and checkpoint_path:
            checkpoint = self._load_checkpoint(checkpoint_path)
            processed_doc_ids = set(checkpoint.get('processed_docs', []))

            # 尝试加载之前的中间结果
            triplets_path = checkpoint_path.replace('.json', '_triplets.json')
            if os.path.exists(triplets_path):
                try:
                    with open(triplets_path, 'r', encoding='utf-8') as f:
                        saved_triplets = json.load(f)
                    logger.info(f"✓ 加载了 {len(saved_triplets)} 个已保存的三元组")
                except Exception as e:
                    logger.warning(f"加载中间结果失败: {e}")

        # 过滤出未处理的文档
        pending_documents = [doc for doc in documents
                           if doc.get('doc_id', '') not in processed_doc_ids]

        if processed_doc_ids:
            logger.info(f"跳过 {len(processed_doc_ids)} 个已处理的文档，剩余 {len(pending_documents)} 个待处理")

        if not pending_documents:
            logger.info("所有文档都已处理完成")
            return results

        if enable_parallel and len(pending_documents) > 1:
            # 并行处理
            logger.info(f"启用并行处理，进程数: {max_workers}")
            processed_count = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_doc = {
                    executor.submit(self._process_single_document, doc): doc
                    for doc in pending_documents
                }

                # 收集结果
                for future in tqdm(as_completed(future_to_doc), total=len(pending_documents), desc="处理文档"):
                    # 检查停止标志
                    with self.stop_lock:
                        if self.should_stop:
                            logger.error(f"检测到停止信号，取消剩余任务")
                            # 取消所有未完成的任务
                            for f in future_to_doc:
                                if not f.done():
                                    f.cancel()
                            # 保存当前进度
                            if enable_checkpoint and checkpoint_path:
                                self._save_checkpoint(checkpoint_path, list(processed_doc_ids), results)
                                self._save_intermediate_triplets(checkpoint_path, results)
                            logger.error(f"已保存当前进度，请充值后重新运行继续处理")
                            break

                    try:
                        result = future.result()
                        doc = future_to_doc[future]

                        if result:
                            results.append(result)
                            processed_doc_ids.add(doc.get('doc_id', ''))
                            processed_count += 1

                            # 定期保存checkpoint（每10个文档）
                            if enable_checkpoint and checkpoint_path and processed_count % 10 == 0:
                                self._save_checkpoint(checkpoint_path, list(processed_doc_ids), results)
                                self._save_intermediate_triplets(checkpoint_path, results)

                    except RuntimeError as e:
                        # 余额不足等严重错误
                        if "余额不足" in str(e) or "quota_not_enough" in str(e):
                            with self.stop_lock:
                                self.should_stop = True
                                self.stop_reason = str(e)
                            logger.error(f"遇到严重错误: {e}")
                            # 取消剩余任务
                            for f in future_to_doc:
                                if not f.done():
                                    f.cancel()
                            # 保存进度
                            if enable_checkpoint and checkpoint_path:
                                self._save_checkpoint(checkpoint_path, list(processed_doc_ids), results)
                                self._save_intermediate_triplets(checkpoint_path, results)
                            logger.error(f"已保存当前进度，请处理问题后重新运行")
                            break
                        else:
                            doc = future_to_doc[future]
                            logger.error(f"处理文档失败: {doc.get('doc_id', 'unknown')}, 错误: {e}")

                    except Exception as e:
                        doc = future_to_doc[future]
                        logger.error(f"处理文档失败: {doc.get('doc_id', 'unknown')}, 错误: {e}")

        else:
            # 顺序处理
            logger.info("启用顺序处理")
            for idx, doc in enumerate(tqdm(pending_documents, desc="处理文档")):
                # 检查停止标志
                with self.stop_lock:
                    if self.should_stop:
                        logger.error(f"检测到停止信号: {self.stop_reason}")
                        # 保存当前进度
                        if enable_checkpoint and checkpoint_path:
                            self._save_checkpoint(checkpoint_path, list(processed_doc_ids), results)
                            self._save_intermediate_triplets(checkpoint_path, results)
                        logger.error(f"已保存当前进度，请充值后重新运行继续处理")
                        break

                try:
                    result = self._process_single_document(doc)
                    if result:
                        results.append(result)
                        processed_doc_ids.add(doc.get('doc_id', ''))

                        # 定期保存checkpoint（每5个文档）
                        if enable_checkpoint and checkpoint_path and (idx + 1) % 5 == 0:
                            self._save_checkpoint(checkpoint_path, list(processed_doc_ids), results)
                            self._save_intermediate_triplets(checkpoint_path, results)

                except RuntimeError as e:
                    # 余额不足等严重错误
                    if "余额不足" in str(e) or "quota_not_enough" in str(e):
                        with self.stop_lock:
                            self.should_stop = True
                            self.stop_reason = str(e)
                        logger.error(f"遇到严重错误: {e}")
                        # 保存进度
                        if enable_checkpoint and checkpoint_path:
                            self._save_checkpoint(checkpoint_path, list(processed_doc_ids), results)
                            self._save_intermediate_triplets(checkpoint_path, results)
                        logger.error(f"已保存当前进度，请处理问题后重新运行")
                        break
                    else:
                        logger.error(f"处理文档失败: {doc.get('doc_id', 'unknown')}, 错误: {e}")

                except Exception as e:
                    logger.error(f"处理文档失败: {doc.get('doc_id', 'unknown')}, 错误: {e}")

        # 最终保存checkpoint
        if enable_checkpoint and checkpoint_path:
            self._save_checkpoint(checkpoint_path, list(processed_doc_ids), results)
            self._save_intermediate_triplets(checkpoint_path, results)

        return results

    def _process_single_document(self, document: Dict[str, Any]) -> ExtractionResult:
        """处理单个文档

        Args:
            document: 文档字典，包含title、content、doc_id等字段

        Returns:
            该文档的抽取结果，处理失败返回None
        """
        # 检查是否需要停止
        with self.stop_lock:
            if self.should_stop:
                logger.warning(f"检测到停止信号: {self.stop_reason}，跳过文档处理")
                return None

        title = document.get('title', '')
        content = document.get('content', '')
        doc_id = document.get('doc_id', '')
        source_url = document.get('url', '')  # 获取source_url

        # 合并标题和内容
        text = f"{title}\n{content}" if title else content

        if not text.strip():
            logger.warning(f"文档 {doc_id} 内容为空，跳过")
            return None

        logger.info(f"处理文档: {doc_id}")

        try:
            result = self.extractor.extract_from_text(text)

            # 为每个三元组添加文档信息和source_url
            for triplet in result.triplets:
                triplet.doc_id = doc_id
                triplet.source_url = source_url

            return result

        except RuntimeError as e:
            # 捕获余额不足等严重错误
            if "余额不足" in str(e) or "quota_not_enough" in str(e):
                with self.stop_lock:
                    self.should_stop = True
                    self.stop_reason = str(e)
                logger.error(f"设置停止标志: {self.stop_reason}")
                raise  # 重新抛出异常
            else:
                raise

    def build_knowledge_graph(self, input_file: str, enable_checkpoint: bool = True) -> Dict[str, Any]:
        """从输入文件构建知识图谱，支持断点续传

        Args:
            input_file: 输入JSON文件路径
            enable_checkpoint: 是否启用断点续传功能

        Returns:
            包含三元组、统计信息和输出路径的结果字典
        """
        logger.info(f"开始构建知识图谱，输入文件: {input_file}")

        # 获取checkpoint路径
        checkpoint_path = self._get_checkpoint_path(input_file) if enable_checkpoint else None

        if enable_checkpoint and checkpoint_path:
            logger.info(f"断点续传已启用，checkpoint文件: {checkpoint_path}")

        # 加载文档
        documents = self._load_documents(input_file)
        logger.info(f"加载了 {len(documents)} 个文档")

        # 批量处理（支持断点续传）
        results = self.process_documents(documents, checkpoint_path, enable_checkpoint)

        # 收集所有三元组
        all_triplets = []
        for result in results:
            all_triplets.extend(result.triplets)
            self.all_results.append(result)

        # 去重
        all_triplets = self._deduplicate_triplets(all_triplets)
        self.all_triplets = all_triplets

        # 收集未知实体和关系（如果启用Schema发现）
        if self.enable_schema_discovery and self.schema_discoverer:
            logger.info("开始收集未知实体和关系...")
            config_entity_types = set(self.config.get('entity_types', {}).keys())
            config_relation_types = set(self.config.get('relation_types', {}).keys())

            for triplet in all_triplets:
                self.schema_discoverer.collect_from_triplet(
                    triplet,
                    config_entity_types,
                    config_relation_types
                )

            # 打印收集摘要
            self.schema_discoverer.print_summary()

            # 导出到CSV
            exported_files = self.schema_discoverer.export_to_csv(prefix="schema_discovery")
            logger.info(f"Schema发现结果已导出，共 {len(exported_files)} 个文件")

        # 统计信息
        stats = self._compute_statistics(results, all_triplets)

        # 保存结果
        output_path = self._save_results(all_triplets, stats)

        # 清理checkpoint（处理完成后）
        if enable_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                triplets_path = checkpoint_path.replace('.json', '_triplets.json')
                if os.path.exists(triplets_path):
                    os.remove(triplets_path)
                logger.info("已清理checkpoint文件")
            except Exception as e:
                logger.warning(f"清理checkpoint失败: {e}")

        logger.info(f"知识图谱构建完成！")
        logger.info(f"共提取 {len(all_triplets)} 个三元组")
        logger.info(f"结果保存至: {output_path}")

        return {
            'triplets': [t.to_dict() for t in all_triplets],
            'statistics': stats,
            'output_path': output_path
        }

    def _load_documents(self, input_file: str) -> List[Dict[str, Any]]:
        """从JSON文件加载文档列表

        Args:
            input_file: 输入JSON文件路径

        Returns:
            文档字典列表
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('documents', [])

    def _compute_statistics(self, results: List[ExtractionResult], triplets: List[Triplet]) -> Dict[str, Any]:
        """计算抽取结果的统计信息

        Args:
            results: 所有文档的抽取结果列表
            triplets: 所有三元组列表

        Returns:
            包含各类统计指标的字典
        """
        if not results:
            return {}

        # 基本统计
        total_docs = len(results)
        total_triplets = len(triplets)
        total_time = sum(r.processing_time for r in results)
        avg_time_per_doc = total_time / total_docs if total_docs > 0 else 0

        # 分类三元组统计
        in_config_triplets = []
        out_of_config_triplets = []
        mixed_triplets = []  # 部分在配置中的三元组

        for triplet in triplets:
            if (triplet.is_relation_in_config and
                triplet.is_subject_type_in_config and
                triplet.is_object_type_in_config):
                in_config_triplets.append(triplet)
            elif (not triplet.is_relation_in_config and
                  not triplet.is_subject_type_in_config and
                  not triplet.is_object_type_in_config):
                out_of_config_triplets.append(triplet)
            else:
                mixed_triplets.append(triplet)

        # 关系类型统计
        relation_counts = {}
        in_config_relation_counts = {}
        out_of_config_relation_counts = {}

        for triplet in triplets:
            relation_counts[triplet.relation] = relation_counts.get(triplet.relation, 0) + 1
            if triplet.is_relation_in_config:
                in_config_relation_counts[triplet.relation] = in_config_relation_counts.get(triplet.relation, 0) + 1
            else:
                out_of_config_relation_counts[triplet.relation] = out_of_config_relation_counts.get(triplet.relation, 0) + 1

        # 实体类型统计
        entity_type_counts = {}
        in_config_entity_counts = {}
        out_of_config_entity_counts = {}

        for triplet in triplets:
            # 主实体类型
            entity_type_counts[triplet.subject_type] = entity_type_counts.get(triplet.subject_type, 0) + 1
            if triplet.is_subject_type_in_config:
                in_config_entity_counts[triplet.subject_type] = in_config_entity_counts.get(triplet.subject_type, 0) + 1
            else:
                out_of_config_entity_counts[triplet.subject_type] = out_of_config_entity_counts.get(triplet.subject_type, 0) + 1

            # 客实体类型
            entity_type_counts[triplet.object_type] = entity_type_counts.get(triplet.object_type, 0) + 1
            if triplet.is_object_type_in_config:
                in_config_entity_counts[triplet.object_type] = in_config_entity_counts.get(triplet.object_type, 0) + 1
            else:
                out_of_config_entity_counts[triplet.object_type] = out_of_config_entity_counts.get(triplet.object_type, 0) + 1

        # Token使用统计
        total_tokens = 0
        for result in results:
            if result.token_usage:
                total_tokens += result.token_usage.get('total_tokens', 0)

        # 置信度分布
        confidence_scores = [t.confidence for t in triplets]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        min_confidence = min(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0

        return {
            'processing_summary': {
                'total_documents': total_docs,
                'total_triplets': total_triplets,
                'total_processing_time': round(total_time, 2),
                'avg_time_per_document': round(avg_time_per_doc, 2),
                'total_tokens_used': total_tokens
            },
            'classification_summary': {
                'fully_in_config': len(in_config_triplets),
                'fully_out_of_config': len(out_of_config_triplets),
                'mixed_config': len(mixed_triplets),
                'in_config_percentage': round(len(in_config_triplets) / total_triplets * 100, 2) if total_triplets > 0 else 0,
                'out_of_config_percentage': round(len(out_of_config_triplets) / total_triplets * 100, 2) if total_triplets > 0 else 0
            },
            'relation_distribution': {
                'all': dict(sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)),
                'in_config': dict(sorted(in_config_relation_counts.items(), key=lambda x: x[1], reverse=True)),
                'out_of_config': dict(sorted(out_of_config_relation_counts.items(), key=lambda x: x[1], reverse=True))
            },
            'entity_type_distribution': {
                'all': dict(sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True)),
                'in_config': dict(sorted(in_config_entity_counts.items(), key=lambda x: x[1], reverse=True)),
                'out_of_config': dict(sorted(out_of_config_entity_counts.items(), key=lambda x: x[1], reverse=True))
            },
            'confidence_stats': {
                'average': round(avg_confidence, 3),
                'minimum': round(min_confidence, 3),
                'maximum': round(max_confidence, 3)
            }
        }

    def _save_results(self, triplets: List[Triplet], stats: Dict[str, Any]) -> str:
        """保存抽取结果到文件

        Args:
            triplets: 三元组列表
            stats: 统计信息字典

        Returns:
            输出文件路径
        """
        output_format = self.config['output']['format']
        output_path = self.config['output']['output_path']
        save_intermediate = self.config['output']['save_intermediate']

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # 分类三元组：配置内vs配置外
        in_config_triplets = []
        out_of_config_triplets = []

        for triplet in triplets:
            if (triplet.is_relation_in_config and
                triplet.is_subject_type_in_config and
                triplet.is_object_type_in_config):
                in_config_triplets.append(triplet)
            else:
                out_of_config_triplets.append(triplet)

        if output_format == 'json':
            output_data = {
                'metadata': {
                    'extraction_timestamp': pd.Timestamp.now().isoformat(),
                    'config_used': self.config,
                    'statistics': stats
                },
                'triplets': {
                    'in_config': [t.to_dict() for t in in_config_triplets],
                    'out_of_config': [t.to_dict() for t in out_of_config_triplets],
                    'all': [t.to_dict() for t in triplets]
                },
                'summary': {
                    'total_triplets': len(triplets),
                    'in_config_count': len(in_config_triplets),
                    'out_of_config_count': len(out_of_config_triplets)
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        elif output_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for triplet in triplets:
                    f.write(json.dumps(triplet.to_dict(), ensure_ascii=False) + '\n')

        elif output_format == 'csv':
            df = pd.DataFrame([t.to_dict() for t in triplets])
            df.to_csv(output_path, index=False, encoding='utf-8-sig')

        elif output_format == 'neo4j':
            self._save_neo4j_format(triplets, output_path)

        # 保存中间结果
        if save_intermediate:
            intermediate_path = output_path.replace('.', '_detailed.')
            self._save_intermediate_results(intermediate_path)

        # 保存分类结果
        self._save_classified_results(in_config_triplets, out_of_config_triplets, output_path)

        return output_path

    def _save_neo4j_format(self, triplets: List[Triplet], output_path: str):
        """将三元组保存为Neo4j导入格式

        Args:
            triplets: 三元组列表
            output_path: 输出文件路径
        """
        # 节点文件
        nodes_path = output_path.replace('.json', '_nodes.csv')
        edges_path = output_path.replace('.json', '_edges.csv')

        # 收集所有节点
        nodes = set()
        edges = []

        for triplet in triplets:
            nodes.add((triplet.subject, triplet.subject_type))
            nodes.add((triplet.object, triplet.object_type))
            edges.append({
                'source': triplet.subject,
                'target': triplet.object,
                'relation': triplet.relation,
                'confidence': triplet.confidence,
                'evidence': triplet.evidence
            })

        # 保存节点
        nodes_df = pd.DataFrame(list(nodes), columns=['name', 'type'])
        nodes_df.to_csv(nodes_path, index=False, encoding='utf-8-sig')

        # 保存边
        edges_df = pd.DataFrame(edges)
        edges_df.to_csv(edges_path, index=False, encoding='utf-8-sig')

        logger.info(f"Neo4j格式文件保存至: {nodes_path}, {edges_path}")

    def _save_classified_results(self, in_config_triplets: List[Triplet], out_of_config_triplets: List[Triplet], base_output_path: str):
        """保存分类后的三元组到单独文件

        Args:
            in_config_triplets: 配置内的三元组列表
            out_of_config_triplets: 配置外的三元组列表
            base_output_path: 基础输出文件路径
        """
        base_name = base_output_path.replace('.json', '')

        # 保存配置内的三元组
        in_config_path = f"{base_name}_in_config.json"
        with open(in_config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'description': '配置内的三元组（关系和实体类型都在预定义配置中）',
                    'count': len(in_config_triplets)
                },
                'triplets': [t.to_dict() for t in in_config_triplets]
            }, f, ensure_ascii=False, indent=2)

        # 保存配置外的三元组
        out_config_path = f"{base_name}_out_of_config.json"
        with open(out_config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'description': '配置外的三元组（包含非预定义的关系或实体类型）',
                    'count': len(out_of_config_triplets)
                },
                'triplets': [t.to_dict() for t in out_of_config_triplets]
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"分类结果保存至:")
        logger.info(f"  配置内三元组: {in_config_path} ({len(in_config_triplets)} 个)")
        logger.info(f"  配置外三元组: {out_config_path} ({len(out_of_config_triplets)} 个)")

    def _save_intermediate_results(self, output_path: str):
        """保存详细的中间处理结果

        Args:
            output_path: 输出文件路径
        """
        detailed_results = []

        for result in self.all_results:
            detailed_results.append({
                'text_preview': result.text[:200] + '...' if len(result.text) > 200 else result.text,
                'text_length': len(result.text),
                'processing_time': result.processing_time,
                'token_usage': result.token_usage,
                'triplets_count': len(result.triplets),
                'triplets': [t.to_dict() for t in result.triplets]
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        logger.info(f"中间结果保存至: {output_path}")