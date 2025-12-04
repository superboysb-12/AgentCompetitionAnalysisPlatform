# Quality Check Module

中文三元组质量评估模块（无参照评估）

## 📋 功能说明

本模块实现了基于两个核心指标的三元组质量评估：

### 核心指标

1. **support_score（证据支持度，0-1）**
   - 判断三元组 (S, P, O) 是否被来源句支持
   - 方法：
     - 将三元组转为3种中文假设句模板
     - 计算与原句的语义相似度（句向量/TF-IDF/Jaccard自动降级）
     - 启发式验证：S和O模糊匹配 + P同义词匹配
     - 综合评分：0.6×语义相似度 + 0.4×启发式匹配

2. **consistency_score（稳健一致性，0-1）**
   - 评估抽取结果的稳定性
   - 方法（代理指标）：
     - 置信度50% + 证据质量30% + Schema符合度20%

### 综合质量分
- `overall_quality = 0.6 × support_score + 0.4 × consistency_score`

## 🚀 使用方法

### 1. 环境检查
```bash
cd LLMRelationExtracter
python scripts/quality_check/test_quality_check.py
```

### 2. 运行质量评估
```bash
cd LLMRelationExtracter
python scripts/quality_check/run_quality_check.py
```

脚本会自动：
- 从 `data/output/` 目录加载知识图谱文件
- 评估所有三元组的质量
- 生成详细报告到 `data/output/quality_check_results/`
- **自动生成可视化图表和Excel表格** ✨

### 3. 手动生成可视化（可选）
如果需要单独生成可视化：
```bash
cd LLMRelationExtracter
python scripts/quality_check/visualize_quality.py
```

会自动使用最新的评估报告生成图表。

### 3. 配置要评估的知识图谱

编辑 `run_quality_check.py` 中的 `kgs_to_check` 字典：

```python
kgs_to_check = {
    "deepseek": "data/output/knowledge_graph_deepseek.json",
    "gemini-2.5-flash": "data/output/knowledge_graph_gemini-2.5-flash.json",
    "gpt-4.1": "data/output/knowledge_graph_gpt-4.1.json",
}
```

## 📊 输出说明

评估完成后，会在 `data/output/quality_check_results/` 目录生成：

### 1. 文本报告
- `quality_check_report_YYYYMMDD_HHMMSS.json` - 完整的JSON报告
- `quality_check_summary_YYYYMMDD_HHMMSS.md` - Markdown格式摘要

### 2. 可视化图表 ✨
- `quality_comparison_bars_YYYYMMDD_HHMMSS.png` - 对比柱状图
- `quality_radar_YYYYMMDD_HHMMSS.png` - 雷达图
- `quality_distribution_YYYYMMDD_HHMMSS.png` - 质量分布图
- `quality_scatter_YYYYMMDD_HHMMSS.png` - 支持度-一致性散点图
- `quality_heatmap_YYYYMMDD_HHMMSS.png` - 评分热力图

### 3. Excel表格 ✨
- `quality_comparison_YYYYMMDD_HHMMSS.xlsx` - 包含3个sheet：
  - **综合对比**: 各知识图谱的主要指标
  - **详细统计**: 完整的统计数据（平均值、标准差、最大最小值等）
  - **质量分布**: 高中低质量三元组的分布

### 4. 详细结果（每个知识图谱）
- `{kg_name}_quality_details_YYYYMMDD_HHMMSS.json` - 包含每个三元组的评分

### 输出格式示例

**JSON报告结构**：
```json
{
  "metadata": {
    "evaluation_time": "2025-10-11T12:00:00",
    "evaluator": "TripleQualityChecker",
    "version": "1.0",
    "total_kgs": 3
  },
  "results": {
    "deepseek": {
      "kg_name": "deepseek",
      "total_triplets": 1234,
      "support_score": {
        "mean": 0.785,
        "std": 0.123,
        "min": 0.234,
        "max": 0.987,
        "median": 0.812
      },
      "consistency_score": {
        "mean": 0.856,
        "std": 0.098,
        ...
      },
      "overall_quality": {
        "mean": 0.814,
        ...
      },
      "quality_distribution": {
        "high_quality": 890,
        "medium_quality": 300,
        "low_quality": 44
      }
    },
    ...
  }
}
```

**每个三元组的评分字段**（内部字段，带 `_qc_` 前缀）：
```json
{
  "subject": "格力电器",
  "relation": "生产",
  "object": "空调",
  "evidence": "格力电器是中国最大的空调生产企业...",
  "confidence": 0.95,

  "_qc_support_score": 0.876,
  "_qc_consistency_score": 0.912,
  "_qc_overall": 0.890
}
```

## 📈 评估结果示例

运行完成后会在终端显示：

```
================================================================================
🏆 质量评估结果摘要
================================================================================

综合质量排名:
  🥇 deepseek            - 综合质量: 0.814
  🥈 gpt-4.1             - 综合质量: 0.789
  🥉 gemini-2.5-flash    - 综合质量: 0.756

各维度最佳:
  🌟 证据支持度  : deepseek            (0.785)
  🌟 稳健一致性  : gpt-4.1             (0.867)
  🌟 综合质量    : deepseek            (0.814)

================================================================================
✅ 质量评估完成！
================================================================================

📂 详细报告已保存到: data/output/quality_check_results/
  - JSON报告 (quality_check_report_*.json)
  - Markdown摘要 (quality_check_summary_*.md)
  - 详细结果 (*_quality_details_*.json)
```

## 🔧 技术细节

### 自动降级机制
1. **语义相似度计算**：
   - 首选：Sentence-BERT 句向量（如果安装）
   - 降级：TF-IDF + 余弦相似度
   - 最终降级：Jaccard字符集相似度

2. **模糊匹配**：
   - 首选：rapidfuzz 模糊匹配（阈值80）
   - 降级：简单字符重叠率

### 依赖包
- **必需**：
  - `numpy>=1.24.0`
  - `scikit-learn>=1.3.0`
  - `rapidfuzz>=3.0.0`

- **可视化（强烈推荐）** ✨：
  - `matplotlib>=3.7.0` - 图表绘制
  - `seaborn>=0.12.0` - 高级可视化
  - `pandas>=2.0.0` - 数据处理
  - `openpyxl>=3.1.0` - Excel支持

- **可选（提升性能）**：
  - `sentence-transformers` - 句向量模型

**安装所有依赖：**
```bash
pip install -r requirements.txt
```

## 📝 与模型评估的对比

| 特性 | 模型评估 (evaluation) | 质量评估 (quality_check) |
|------|----------------------|-------------------------|
| 目的 | 横向对比多个模型性能 | 评估三元组内在质量 |
| 评估维度 | 8维度（质量、Schema、一致性、多样性等） | 2核心指标（支持度、一致性） |
| 输入 | 多个知识图谱文件 | 多个知识图谱文件 |
| 输出位置 | `data/output/evaluation_results/` | `data/output/quality_check_results/` |
| 报告类型 | Excel + Markdown + 图表 + JSON | JSON + Markdown |
| 适用场景 | 模型选型、性能对比 | 质量把关、数据清洗 |

## 💡 使用建议

1. **首次使用**：先运行 `test_quality_check.py` 检查环境
2. **批量评估**：编辑 `run_quality_check.py` 配置所有要评估的知识图谱
3. **结果分析**：查看 Markdown 摘要快速了解整体情况，JSON详细结果用于深入分析
4. **低质量过滤**：可根据 `_qc_overall < 0.4` 过滤低质量三元组

## 🔍 质量分布说明

- **高质量 (≥0.7)**：证据充分，抽取可靠
- **中等质量 (0.4-0.7)**：基本可用，可能需要人工复核
- **低质量 (<0.4)**：建议过滤或重新抽取

## 📚 相关文档

- [主README](../../README.md)
- [模型评估模块](../evaluation/)
- [Scripts目录说明](../README.md)
