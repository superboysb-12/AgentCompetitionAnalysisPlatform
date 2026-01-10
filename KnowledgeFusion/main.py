"""
知识融合主程序
整合指代融合、品牌推断、知识融合的完整流程

运行方式: python main.py
"""

import os
import sys
import time
from pathlib import Path

# 必须在导入langchain之前设置环境变量
os.environ.setdefault("LANGCHAIN_VERBOSE", "false")

# 设置UTF-8编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger, log_section
from config import load_all_configs
from data_loader import load_and_prepare_data, filter_rare_brands
from data_saver import save_entities_original_format, save_combined_log
from alias_fusion import perform_alias_fusion
from brand_inference import perform_brand_inference


def main():
    """主函数"""
    # 初始化日志
    logger = get_logger(log_dir=Path("output"))

    log_section(logger, "知识融合系统 - 启动")

    # 加载配置
    logger.info("加载配置...")
    configs = load_all_configs()

    llm_config = configs["llm"]
    inference_config = configs["brand_inference"]
    fusion_config = configs["fusion"]
    data_config = configs["data"]

    logger.info(f"  输入数据: {data_config['input_path']}")
    logger.info(f"  输出目录: {data_config['output_dir']}")

    # 选择测试规模
    print("\n请选择测试规模:")
    print("1. 完整数据集")
    print("2. 小规模测试 (前50个实体)")
    print("3. 极小规模测试 (前10个实体)")

    choice = input("\n请输入选项 (1/2/3, 默认3): ").strip() or '3'

    # ==================== 阶段0: 数据加载 ====================
    log_section(logger, "阶段0: 数据加载")

    entities, original_data = load_and_prepare_data(
        data_config['input_path'],
        logger
    )

    # 根据选择截取数据
    if choice == '3':
        entities = entities[:10]
        logger.info("使用极小规模数据集: 10 个实体")
    elif choice == '2':
        entities = entities[:50]
        logger.info("使用小规模数据集: 50 个实体")
    else:
        logger.info(f"使用完整数据集: {len(entities)} 个实体")
        confirm = input("确认要运行完整测试吗？(y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("已取消")
            return

    # 过滤稀有品牌（如果启用）
    filtered_entities = []
    if data_config.get('filter_rare_brands', False):
        logger.info("\n过滤稀有品牌...")
        entities, filtered_entities = filter_rare_brands(
            entities,
            min_frequency_ratio=data_config.get('min_brand_frequency_ratio', 0.01),
            logger=logger
        )

    # 收集所有日志
    all_logs = {
        'filtered_brands': {
            'count': len(filtered_entities),
            'entities': [
                {
                    'brand': e.get('brand'),
                    'product_model': e.get('product_model'),
                    'category': e.get('category')
                }
                for e in filtered_entities[:100]  # 只保存前100个
            ] if filtered_entities else []
        }
    }

    # ==================== 阶段1: 指代融合 ====================
    log_section(logger, "阶段1: 指代融合 (别名统一)")

    start_time = time.time()
    entities, alias_map, alias_llm_results = perform_alias_fusion(
        entities,
        llm_config,
        logger
    )
    alias_time = time.time() - start_time

    logger.info(f"指代融合完成，耗时 {alias_time:.2f} 秒")

    # 始终保存指代融合日志（即使没有合并），方便调试
    if alias_map or alias_llm_results:
        all_logs['alias_fusion'] = {
            'alias_map': alias_map,
            'llm_results': alias_llm_results
        }

    # ==================== 阶段2: 品牌推断 ====================
    log_section(logger, "阶段2: 品牌推断")

    start_time = time.time()
    entities, inference_results = perform_brand_inference(
        entities,
        llm_config,
        inference_config,
        logger
    )
    inference_time = time.time() - start_time

    logger.info(f"品牌推断完成，耗时 {inference_time:.2f} 秒")

    if inference_results:
        all_logs['brand_inference'] = inference_results

    # ==================== 阶段3: 保存结果 ====================
    log_section(logger, "阶段3: 保存结果")

    output_dir = Path(data_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始格式
    original_path = output_dir / "fused_entities_original_format.json"
    save_entities_original_format(entities, original_data, str(original_path), logger)

    # 保存合并的日志
    log_path = output_dir / "fusion_logs.json"
    save_combined_log(all_logs, str(log_path), logger)

    saved_files = {
        'entities': str(original_path),
        'logs': str(log_path)
    }

    # ==================== 总结 ====================
    log_section(logger, "执行总结")

    total_time = alias_time + inference_time

    logger.info(f"\n各阶段耗时:")
    logger.info(f"  指代融合: {alias_time:.2f} 秒")
    logger.info(f"  品牌推断: {inference_time:.2f} 秒")
    logger.info(f"  总耗时: {total_time:.2f} 秒")

    # 统计品牌覆盖率
    total_entities = len(entities)
    with_brand = sum(1 for e in entities if e.get('brand'))
    without_brand = total_entities - with_brand

    logger.info(f"\n品牌覆盖情况:")
    logger.info(f"  总实体数: {total_entities}")
    logger.info(f"  有品牌: {with_brand} ({with_brand/total_entities*100:.1f}%)")
    logger.info(f"  无品牌: {without_brand} ({without_brand/total_entities*100:.1f}%)")

    logger.info(f"\n输出文件:")
    for name, path in saved_files.items():
        logger.info(f"  {name}: {path}")

    log_section(logger, "知识融合系统 - 完成")

    return entities, all_logs


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
