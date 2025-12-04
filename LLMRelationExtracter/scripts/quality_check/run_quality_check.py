#!/usr/bin/env python3
"""
运行三元组质量评估的便捷脚本

使用方法:
  cd LLMRelationExtracter
  python scripts/quality_check/run_quality_check.py
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from scripts.quality_check.triple_quality_checker import TripleQualityChecker

def main():
    """主函数"""
    print("="*80)
    print("🚀 启动三元组质量评估")
    print("="*80)

    # 切换到项目根目录
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}\n")

    # 创建评估器，输出到 data/output/quality_check_results
    checker = TripleQualityChecker(output_dir="data/output/quality_check_results")

    # 配置要评估的知识图谱
    # 根据你的实际输出文件配置
    kgs_to_check = {
        "full": "data/output/knowledge_graph_gpt-4.1_final.json",
    }

    print(f"\n📁 准备评估以下知识图谱:")
    for kg_name in kgs_to_check.keys():
        print(f"  - {kg_name}")

    # 检查文件是否存在并加载
    print(f"\n📥 加载知识图谱输出文件...")
    loaded_kgs = []
    for kg_name, file_path in kgs_to_check.items():
        if os.path.exists(file_path):
            try:
                checker.load_kg_output(kg_name, file_path)
                loaded_kgs.append(kg_name)
            except Exception as e:
                print(f"  ❌ 加载 {kg_name} 失败: {e}")
        else:
            print(f"  ⚠️  文件不存在: {file_path}")

    if not loaded_kgs:
        print("\n❌ 没有找到任何有效的知识图谱输出文件")
        print("请确保:")
        print("  1. 已经运行过 main.py 生成知识图谱")
        print("  2. 文件路径配置正确")
        print(f"  3. 在项目根目录运行: cd {project_root}")
        return 1

    print(f"\n✅ 成功加载 {len(loaded_kgs)} 个知识图谱的输出:")
    for kg in loaded_kgs:
        print(f"  ✓ {kg}")

    # 执行评估
    print(f"\n{'='*80}")
    print("📊 开始质量评估...")
    print(f"{'='*80}")

    results = checker.evaluate_all_kgs()

    # 生成评估报告
    print(f"\n{'='*80}")
    print("📝 生成评估报告...")
    print(f"{'='*80}")

    report_path = checker.generate_quality_report(results)

    # 生成可视化图表
    print(f"\n{'='*80}")
    print("📊 生成可视化图表...")
    print(f"{'='*80}\n")

    try:
        # 尝试导入并运行可视化
        from scripts.quality_check.visualize_quality import QualityVisualizer

        visualizer = QualityVisualizer(report_path)
        viz_results = visualizer.generate_all()

        # 统计生成的文件
        chart_count = len(viz_results.get('charts', []))
        has_excel = bool(viz_results.get('excel'))

    except ImportError as e:
        print(f"⚠️  可视化依赖未安装: {e}")
        print("提示: pip install matplotlib seaborn pandas openpyxl")
        print("可稍后手动运行: python scripts/quality_check/visualize_quality.py")
    except Exception as e:
        print(f"⚠️  可视化生成失败: {e}")
        print("可稍后手动运行: python scripts/quality_check/visualize_quality.py")

    # 输出快速摘要
    print(f"\n{'='*80}")
    print("🏆 质量评估结果摘要")
    print(f"{'='*80}\n")

    # 按综合质量排序
    ranking = sorted(results.items(),
                    key=lambda x: x[1]['overall_quality']['mean'],
                    reverse=True)

    print("综合质量排名:")
    for rank, (kg_name, result) in enumerate(ranking, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"  {medal} {kg_name:20s} - 综合质量: {result['overall_quality']['mean']:.3f}")

    print(f"\n各维度最佳:")

    # 找出各维度最佳
    best_support = max(results.items(), key=lambda x: x[1]['support_score']['mean'])
    best_consistency = max(results.items(), key=lambda x: x[1]['consistency_score']['mean'])
    best_overall = max(results.items(), key=lambda x: x[1]['overall_quality']['mean'])

    print(f"  🌟 证据支持度  : {best_support[0]:20s} ({best_support[1]['support_score']['mean']:.3f})")
    print(f"  🌟 稳健一致性  : {best_consistency[0]:20s} ({best_consistency[1]['consistency_score']['mean']:.3f})")
    print(f"  🌟 综合质量    : {best_overall[0]:20s} ({best_overall[1]['overall_quality']['mean']:.3f})")

    print(f"\n{'='*80}")
    print("✅ 质量评估完成！")
    print(f"{'='*80}")
    print(f"\n📂 详细报告已保存到: data/output/quality_check_results/")
    print(f"  - JSON报告 (quality_check_report_*.json)")
    print(f"  - Markdown摘要 (quality_check_summary_*.md)")
    print(f"  - 详细结果 (*_quality_details_*.json)")

    # 如果生成了可视化
    try:
        if 'viz_results' in locals():
            if viz_results.get('charts'):
                print(f"  - 可视化图表 ({len(viz_results['charts'])} 个PNG图片)")
            if viz_results.get('excel'):
                print(f"  - Excel表格 (quality_comparison_*.xlsx)")
    except:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
