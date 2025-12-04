#!/usr/bin/env python3
"""
三元组质量评估可视化工具

功能：
- 生成对比柱状图
- 生成雷达图
- 生成质量分布图
- 生成散点图
- 导出Excel表格

使用方法:
  cd LLMRelationExtracter
  python scripts/quality_check/visualize_quality.py
"""

import os
import sys
import json
import glob
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 尝试导入可视化依赖
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️  matplotlib/seaborn未安装，无法生成图表")
    print("请运行: pip install matplotlib seaborn")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas未安装，无法生成Excel表格")

# 设置中文字体
if PLOT_AVAILABLE:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


class QualityVisualizer:
    """质量评估可视化工具"""

    def __init__(self, report_path: str, output_dir: str = None):
        """初始化可视化工具

        Args:
            report_path: 质量评估报告JSON文件路径
            output_dir: 输出目录（默认与报告同目录）
        """
        self.report_path = report_path
        self.output_dir = output_dir or os.path.dirname(report_path)

        # 加载报告数据
        with open(report_path, 'r', encoding='utf-8') as f:
            self.report_data = json.load(f)

        self.results = self.report_data.get('results', {})
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"✓ 加载报告: {report_path}")
        print(f"✓ 发现 {len(self.results)} 个知识图谱的评估结果")

    def plot_comparison_bars(self) -> str:
        """绘制对比柱状图"""
        if not PLOT_AVAILABLE:
            return ""

        kg_names = list(self.results.keys())
        support_scores = [self.results[kg]['support_score']['mean'] for kg in kg_names]
        consistency_scores = [self.results[kg]['consistency_score']['mean'] for kg in kg_names]
        overall_scores = [self.results[kg]['overall_quality']['mean'] for kg in kg_names]

        x = np.arange(len(kg_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width, support_scores, width, label='证据支持度', color='#3498db')
        bars2 = ax.bar(x, consistency_scores, width, label='稳健一致性', color='#2ecc71')
        bars3 = ax.bar(x + width, overall_scores, width, label='综合质量', color='#e74c3c')

        ax.set_xlabel('知识图谱', fontsize=12)
        ax.set_ylabel('评分', fontsize=12)
        ax.set_title('三元组质量评估对比', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(kg_names, rotation=15, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'quality_comparison_bars_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 柱状图: {output_path}")
        return output_path

    def plot_radar_chart(self) -> str:
        """绘制雷达图"""
        if not PLOT_AVAILABLE:
            return ""

        categories = ['证据支持度', '稳健一致性', '综合质量']
        kg_names = list(self.results.keys())

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set3(np.linspace(0, 1, len(kg_names)))

        for i, kg_name in enumerate(kg_names):
            values = [
                self.results[kg_name]['support_score']['mean'],
                self.results[kg_name]['consistency_score']['mean'],
                self.results[kg_name]['overall_quality']['mean'],
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=kg_name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_title('三元组质量评估雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'quality_radar_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 雷达图: {output_path}")
        return output_path

    def plot_quality_distribution(self) -> str:
        """绘制质量分布图"""
        if not PLOT_AVAILABLE:
            return ""

        kg_names = list(self.results.keys())
        n_kgs = len(kg_names)

        fig, axes = plt.subplots(1, n_kgs, figsize=(6*n_kgs, 5))
        if n_kgs == 1:
            axes = [axes]

        for i, kg_name in enumerate(kg_names):
            dist = self.results[kg_name]['quality_distribution']
            total = self.results[kg_name]['total_triplets']

            categories = ['高质量\n(≥0.7)', '中等质量\n(0.4-0.7)', '低质量\n(<0.4)']
            counts = [dist['high_quality'], dist['medium_quality'], dist['low_quality']]
            percentages = [c/total*100 for c in counts]
            colors = ['#2ecc71', '#f39c12', '#e74c3c']

            axes[i].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            axes[i].set_ylabel('三元组数量', fontsize=11)
            axes[i].set_title(f'{kg_name}\n平均质量: {self.results[kg_name]["overall_quality"]["mean"]:.3f}',
                            fontsize=12, fontweight='bold')
            axes[i].grid(axis='y', alpha=0.3)

            # 添加数值和百分比标签
            for j, (count, pct) in enumerate(zip(counts, percentages)):
                axes[i].text(j, count + max(counts)*0.02, f'{count}\n({pct:.1f}%)',
                           ha='center', va='bottom', fontsize=9)

        plt.suptitle('三元组质量分布', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'quality_distribution_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 质量分布图: {output_path}")
        return output_path

    def plot_scatter(self) -> str:
        """绘制支持度-一致性散点图"""
        if not PLOT_AVAILABLE:
            return ""

        kg_names = list(self.results.keys())
        support_scores = [self.results[kg]['support_score']['mean'] for kg in kg_names]
        consistency_scores = [self.results[kg]['consistency_score']['mean'] for kg in kg_names]
        sizes = [self.results[kg]['total_triplets'] / 10 for kg in kg_names]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(kg_names)))
        scatter = ax.scatter(support_scores, consistency_scores, s=sizes, alpha=0.6,
                           c=range(len(kg_names)), cmap='viridis', edgecolors='black', linewidth=1)

        for i, kg_name in enumerate(kg_names):
            ax.annotate(kg_name, (support_scores[i], consistency_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))

        ax.set_xlabel('证据支持度', fontsize=12)
        ax.set_ylabel('稳健一致性', fontsize=12)
        ax.set_title('支持度-一致性散点图\n(气泡大小=三元组数量)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)

        # 添加对角线参考线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'quality_scatter_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 散点图: {output_path}")
        return output_path

    def plot_score_heatmap(self) -> str:
        """绘制评分热力图"""
        if not PLOT_AVAILABLE or not PANDAS_AVAILABLE:
            return ""

        kg_names = list(self.results.keys())
        metrics = ['证据支持度', '稳健一致性', '综合质量']

        data = []
        for kg in kg_names:
            data.append([
                self.results[kg]['support_score']['mean'],
                self.results[kg]['consistency_score']['mean'],
                self.results[kg]['overall_quality']['mean']
            ])

        df = pd.DataFrame(data, index=kg_names, columns=metrics)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': '评分'}, linewidths=0.5,
                   vmin=0, vmax=1.0, ax=ax)

        ax.set_title('质量评估热力图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'quality_heatmap_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 热力图: {output_path}")
        return output_path

    def export_excel(self) -> str:
        """导出Excel表格"""
        if not PANDAS_AVAILABLE:
            return ""

        excel_path = os.path.join(self.output_dir, f'quality_comparison_{self.timestamp}.xlsx')

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: 综合对比
            self._write_overview_sheet(writer)

            # Sheet 2: 详细统计
            self._write_detailed_sheet(writer)

            # Sheet 3: 质量分布
            self._write_distribution_sheet(writer)

        print(f"✓ Excel表格: {excel_path}")
        return excel_path

    def _write_overview_sheet(self, writer):
        """写入综合对比sheet"""
        data = []
        for kg_name, result in self.results.items():
            row = {
                '知识图谱': kg_name,
                '证据支持度': result['support_score']['mean'],
                '稳健一致性': result['consistency_score']['mean'],
                '综合质量': result['overall_quality']['mean'],
                '三元组总数': result['total_triplets'],
                '高质量数量': result['quality_distribution']['high_quality'],
                '中等质量数量': result['quality_distribution']['medium_quality'],
                '低质量数量': result['quality_distribution']['low_quality'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('综合质量', ascending=False)
        df.to_excel(writer, sheet_name='综合对比', index=False)

    def _write_detailed_sheet(self, writer):
        """写入详细统计sheet"""
        data = []
        for kg_name, result in self.results.items():
            row = {
                '知识图谱': kg_name,
                '支持度-平均': result['support_score']['mean'],
                '支持度-标准差': result['support_score']['std'],
                '支持度-最小': result['support_score']['min'],
                '支持度-最大': result['support_score']['max'],
                '支持度-中位数': result['support_score']['median'],
                '一致性-平均': result['consistency_score']['mean'],
                '一致性-标准差': result['consistency_score']['std'],
                '一致性-最小': result['consistency_score']['min'],
                '一致性-最大': result['consistency_score']['max'],
                '一致性-中位数': result['consistency_score']['median'],
                '综合质量-平均': result['overall_quality']['mean'],
                '综合质量-标准差': result['overall_quality']['std'],
                '综合质量-最小': result['overall_quality']['min'],
                '综合质量-最大': result['overall_quality']['max'],
                '综合质量-中位数': result['overall_quality']['median'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='详细统计', index=False)

    def _write_distribution_sheet(self, writer):
        """写入质量分布sheet"""
        data = []
        for kg_name, result in self.results.items():
            total = result['total_triplets']
            dist = result['quality_distribution']
            row = {
                '知识图谱': kg_name,
                '总数': total,
                '高质量数量': dist['high_quality'],
                '高质量占比': f"{dist['high_quality']/total*100:.1f}%",
                '中等质量数量': dist['medium_quality'],
                '中等质量占比': f"{dist['medium_quality']/total*100:.1f}%",
                '低质量数量': dist['low_quality'],
                '低质量占比': f"{dist['low_quality']/total*100:.1f}%",
            }
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('高质量数量', ascending=False)
        df.to_excel(writer, sheet_name='质量分布', index=False)

    def generate_all(self):
        """生成所有可视化"""
        print("\n" + "="*80)
        print("📊 开始生成可视化图表...")
        print("="*80 + "\n")

        chart_paths = []

        if PLOT_AVAILABLE:
            chart_paths.append(self.plot_comparison_bars())
            chart_paths.append(self.plot_radar_chart())
            chart_paths.append(self.plot_quality_distribution())
            chart_paths.append(self.plot_scatter())
            chart_paths.append(self.plot_score_heatmap())
        else:
            print("❌ 无法生成图表，请安装: pip install matplotlib seaborn")

        excel_path = ""
        if PANDAS_AVAILABLE:
            excel_path = self.export_excel()
        else:
            print("❌ 无法生成Excel，请安装: pip install pandas openpyxl")

        print("\n" + "="*80)
        print("✅ 可视化生成完成！")
        print("="*80)

        return {
            'charts': [p for p in chart_paths if p],
            'excel': excel_path
        }


def main():
    """主函数"""
    print("="*80)
    print("📊 三元组质量评估可视化工具")
    print("="*80)
    print()

    # 切换到项目根目录
    os.chdir(project_root)

    # 查找最新的质量评估报告
    report_dir = "data/output/quality_check_results"

    if not os.path.exists(report_dir):
        print(f"❌ 未找到评估结果目录: {report_dir}")
        print("请先运行: python scripts/quality_check/run_quality_check.py")
        return 1

    # 查找所有报告文件
    report_files = glob.glob(os.path.join(report_dir, "quality_check_report_*.json"))

    if not report_files:
        print(f"❌ 未找到评估报告文件")
        print("请先运行: python scripts/quality_check/run_quality_check.py")
        return 1

    # 使用最新的报告
    latest_report = max(report_files, key=os.path.getmtime)
    print(f"📂 使用报告: {os.path.basename(latest_report)}\n")

    # 创建可视化工具
    visualizer = QualityVisualizer(latest_report)

    # 生成所有可视化
    results = visualizer.generate_all()

    # 输出总结
    if results['charts']:
        print(f"\n📈 生成了 {len(results['charts'])} 个图表:")
        for chart in results['charts']:
            print(f"  - {os.path.basename(chart)}")

    if results['excel']:
        print(f"\n📊 生成了Excel表格:")
        print(f"  - {os.path.basename(results['excel'])}")

    print(f"\n📂 所有文件保存在: {report_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
