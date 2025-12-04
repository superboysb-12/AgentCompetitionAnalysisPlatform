#!/usr/bin/env python3
"""
运行模型评估的便捷脚本

使用方法:
  cd LLMRelationExtracter
  python scripts/evaluation/run_evaluation.py
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from scripts.evaluation.model_evaluator import ModelEvaluator

def main():
    """主函数"""
    print("="*80)
    print("🚀 启动模型性能自动化评估")
    print("="*80)

    # 切换到项目根目录
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}\n")

    # 创建评估器，输出到 data/output/evaluation_results
    evaluator = ModelEvaluator(output_dir="data/output/evaluation_results")

    # 配置要评估的模型
    # 根据你的实际输出文件配置
    models_to_evaluate = {
        "deepseek": "data/output/knowledge_graph_deepseek.json",
        "gemini-2.5-flash": "data/output/knowledge_graph_gemini-2.5-flash.json",
        "gpt-5": "data/output/knowledge_graph_gpt-5.json",
    }

    print(f"\n📁 准备评估以下模型:")
    for model_name in models_to_evaluate.keys():
        print(f"  - {model_name}")

    # 检查文件是否存在并加载
    print(f"\n📥 加载模型输出文件...")
    loaded_models = []
    for model_name, file_path in models_to_evaluate.items():
        if os.path.exists(file_path):
            try:
                evaluator.load_model_output(model_name, file_path)
                loaded_models.append(model_name)
            except Exception as e:
                print(f"  ❌ 加载 {model_name} 失败: {e}")
        else:
            print(f"  ⚠️  文件不存在: {file_path}")

    if not loaded_models:
        print("\n❌ 没有找到任何有效的模型输出文件")
        print("请确保:")
        print("  1. 已经运行过 main.py 生成知识图谱")
        print("  2. 文件路径配置正确")
        print(f"  3. 在项目根目录运行: cd {project_root}")
        return 1

    print(f"\n✅ 成功加载 {len(loaded_models)} 个模型的输出:")
    for model in loaded_models:
        print(f"  ✓ {model}")

    # 执行评估
    print(f"\n{'='*80}")
    print("📊 开始自动化评估...")
    print(f"{'='*80}")

    results = evaluator.evaluate_all_models()

    # 生成对比报告
    print(f"\n{'='*80}")
    print("📝 生成对比报告...")
    print(f"{'='*80}")

    evaluator.generate_comparison_report()

    # 输出快速摘要
    print(f"\n{'='*80}")
    print("🏆 评估结果摘要")
    print(f"{'='*80}\n")

    # 按综合得分排序
    ranking = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)

    print("综合排名:")
    for rank, (model, result) in enumerate(ranking, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"  {medal} {model:20s} - 综合得分: {result['overall_score']}/100")

    print(f"\n各维度最佳:")

    dimensions = {
        '质量': ('quality_scores', 'confidence_score'),
        'Schema符合度': ('schema_compliance', 'schema_score'),
        '一致性': ('consistency_scores', 'consistency_score'),
        '多样性': ('diversity_scores', 'diversity_score'),
        'Evidence质量': ('evidence_quality', 'evidence_score'),
        '成本效益': ('cost_efficiency', 'cost_efficiency_score'),
        '速度': ('performance_metrics', 'speed_score'),
    }

    for dim_name, (cat, key) in dimensions.items():
        best_model = max(results.items(), key=lambda x: x[1][cat][key])
        score = best_model[1][cat][key]
        print(f"  🌟 {dim_name:12s}: {best_model[0]:20s} ({score:.1f}/100)")

    print(f"\n{'='*80}")
    print("✅ 评估完成！")
    print(f"{'='*80}")
    print(f"\n📂 详细报告已保存到: data/output/evaluation_results/")
    print(f"  - Excel报告 (model_comparison_*.xlsx)")
    print(f"  - Markdown报告 (model_comparison_*.md)")
    print(f"  - 可视化图表 (*.png)")
    print(f"  - JSON数据 (model_comparison_*.json)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
