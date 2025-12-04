#!/usr/bin/env python3
"""知识图谱三元组抽取主程序，支持Few-shot learning、Self-consistency、批量处理、并行处理"""

import argparse
import sys
import os
import logging
from typing import Dict, Any

# 添加当前目录到Python路径以支持导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入我们的模块
from kg_builder import KnowledgeGraphBuilder
from few_shot_manager import FewShotManager
import kg_extractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kg_extraction.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主函数，解析命令行参数并执行知识图谱抽取任务"""
    parser = argparse.ArgumentParser(description="知识图谱三元组抽取工具")

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入数据文件路径 (JSON格式)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出文件路径 (覆盖配置文件中的设置)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式：只处理前5个文档'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='指定checkpoint文件名（可选，默认根据输入文件自动生成）'
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    # 检查配置文件
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)

    try:
        # 创建知识图谱构建器（few-shot功能在构建器初始化时自动启用）
        logger.info("初始化知识图谱构建器...")
        builder = KnowledgeGraphBuilder(args.config, checkpoint_name=args.checkpoint)

        # 动态调整配置
        if args.output:
            builder.config['output']['output_path'] = args.output

        # 测试模式
        if args.test:
            logger.info("测试模式：只处理前5个文档")
            import json
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['documents'] = data['documents'][:5]

            test_input_path = args.input.replace('.json', '_test.json')
            with open(test_input_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            args.input = test_input_path

        # 构建知识图谱
        logger.info("开始构建知识图谱...")
        result = builder.build_knowledge_graph(args.input)

        # 输出结果统计
        stats = result['statistics']
        logger.info("\n" + "="*50)
        logger.info("知识图谱构建完成！")
        logger.info("="*50)

        if 'processing_summary' in stats:
            summary = stats['processing_summary']
            logger.info(f"处理文档数: {summary['total_documents']}")
            logger.info(f"提取三元组数: {summary['total_triplets']}")
            logger.info(f"总处理时间: {summary['total_processing_time']} 秒")
            logger.info(f"平均每文档时间: {summary['avg_time_per_document']} 秒")
            logger.info(f"使用Token总数: {summary.get('total_tokens_used', 'N/A')}")

        if 'classification_summary' in stats:
            class_summary = stats['classification_summary']
            logger.info(f"\n三元组分类统计:")
            logger.info(f"  完全符合配置: {class_summary['fully_in_config']} ({class_summary['in_config_percentage']}%)")
            logger.info(f"  完全不符合配置: {class_summary['fully_out_of_config']} ({class_summary['out_of_config_percentage']}%)")
            logger.info(f"  部分符合配置: {class_summary['mixed_config']}")

        if 'relation_distribution' in stats:
            logger.info("\n关系类型分布:")
            rel_dist = stats['relation_distribution']
            logger.info("  总体分布 (前10个):")
            for relation, count in list(rel_dist['all'].items())[:10]:
                config_status = "[配置内]" if relation in builder.config['relation_types'] else "[配置外]"
                logger.info(f"    {config_status} {relation}: {count}")

            if rel_dist['out_of_config']:
                logger.info("  配置外关系类型:")
                for relation, count in list(rel_dist['out_of_config'].items())[:5]:
                    logger.info(f"    [配置外] {relation}: {count}")

        if 'entity_type_distribution' in stats:
            entity_dist = stats['entity_type_distribution']
            if entity_dist['out_of_config']:
                logger.info("\n配置外实体类型 (前5个):")
                for entity_type, count in list(entity_dist['out_of_config'].items())[:5]:
                    logger.info(f"  [配置外] {entity_type}: {count}")

        if 'confidence_stats' in stats:
            conf_stats = stats['confidence_stats']
            logger.info(f"\n置信度统计:")
            logger.info(f"  平均: {conf_stats['average']}")
            logger.info(f"  最小: {conf_stats['minimum']}")
            logger.info(f"  最大: {conf_stats['maximum']}")

        logger.info(f"\n结果文件: {result['output_path']}")
        logger.info("="*50)

    except KeyboardInterrupt:
        logger.info("\n用户中断程序执行")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def demo():
    """演示函数，展示如何使用API进行知识图谱抽取"""
    print("知识图谱抽取演示")
    print("="*50)

    # 示例文本
    demo_text = """
    格力电器2025年上半年业绩报告显示，空调业务营收同比增长12%，达到156亿元。
    公司在变频技术方面取得重大突破，新一代直流变频压缩机能效比提升至5.2。
    美的集团与海尔集团在智能家居领域展开激烈竞争，双方都加大了对物联网技术的投资。
    """

    try:
        # 创建抽取器（few-shot功能在抽取器初始化时自动启用）
        extractor = kg_extractor.KnowledgeGraphExtractor('config.yaml')

        # 进行抽取
        print("正在抽取三元组...")
        result = extractor.extract_from_text(demo_text)

        print(f"\n抽取完成！共找到 {len(result.triplets)} 个三元组：")
        print("-"*50)

        for i, triplet in enumerate(result.triplets, 1):
            print(f"{i}. ({triplet.subject}, {triplet.relation}, {triplet.object})")
            print(f"   类型: ({triplet.subject_type}, {triplet.object_type})")
            print(f"   置信度: {triplet.confidence}")
            print(f"   证据: {triplet.evidence[:50]}...")
            print()

        print(f"处理时间: {result.processing_time:.2f} 秒")
        if result.token_usage:
            print(f"Token使用: {result.token_usage['total_tokens']}")

    except Exception as e:
        print(f"演示出错: {e}")
        print("请检查配置文件和API设置")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有参数，运行演示
        demo()
    else:
        # 否则运行主程序
        main()