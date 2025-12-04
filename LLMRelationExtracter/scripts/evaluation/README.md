# 模型性能自动化评估系统

完全自动化的LLM关系抽取模型评估系统，无需人工标注，通过8个维度对比不同模型的性能。

## 📂 文件结构

```
scripts/evaluation/
├── model_evaluator.py           # 核心评估器(800+行)
├── run_evaluation.py            # 一键运行脚本
├── test_evaluation.py           # 环境自检脚本
├── requirements_evaluation.txt  # 依赖包
├── README.md                    # 本文件
└── EVALUATION_GUIDE.md         # 完整使用文档
```

## 🚀 快速开始 (3步)

### 步骤1: 安装依赖

```bash
cd LLMRelationExtracter
pip install -r scripts/evaluation/requirements_evaluation.txt
```

### 步骤2: 运行自检

```bash
python scripts/evaluation/test_evaluation.py
```

### 步骤3: 运行评估

```bash
python scripts/evaluation/run_evaluation.py
```

## 📊 评估维度

| 维度 | 权重 | 说明 |
|------|------|------|
| **质量评分** | 30% | 置信度、稳定性 |
| **Schema符合度** | 20% | 配置内比例、规范性 |
| **一致性** | 15% | 命名统一、分布均匀 |
| **多样性** | 15% | 信息覆盖广度 |
| **Evidence质量** | 10% | 证据完整性 |
| **成本效益** | 10% | Token使用效率 |
| **速度** | - | 处理性能 |
| **综合得分** | - | 0-100分 |

## 📈 输出报告

评估结果保存在 `data/output/evaluation_results/`:

- **Excel报告** - 5个sheet详细对比
- **Markdown报告** - 排名和分析
- **可视化图表** - 雷达图、柱状图、散点图、分布图
- **JSON数据** - 完整评估数据

## 📖 详细文档

查看 [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) 获取:
- 评估指标详解
- 报告解读指南
- 自定义配置
- 常见问题FAQ

## 💡 使用示例

### 基础用法

```bash
# 在LLMRelationExtracter目录下运行
cd LLMRelationExtracter
python scripts/evaluation/run_evaluation.py
```

### Python API

```python
from scripts.evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(output_dir="data/output/evaluation_results")
evaluator.load_model_output("model-name", "path/to/kg.json")
results = evaluator.evaluate_all_models()
evaluator.generate_comparison_report()
```

## 🔧 配置模型

编辑 `scripts/evaluation/run_evaluation.py`:

```python
models_to_evaluate = {
    "deepseek-v3": "data/output/knowledge_graph_deepseek.json",
    "gemini-2.5-flash": "data/output/knowledge_graph_gemini-2.5-flash.json",
    "gpt-5": "data/output/knowledge_graph_gpt-5.json",
}
```

## ✅ 特点

- ✅ **完全自动化** - 无需人工标注
- ✅ **多维度评估** - 8个维度全面对比
- ✅ **可视化报告** - 图表直观易懂
- ✅ **可定制化** - 权重、指标可调整
- ✅ **标准化流程** - 可重复评估

## 🐛 常见问题

### Q: 依赖包安装失败？
```bash
pip install numpy pandas matplotlib seaborn openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 找不到模型输出文件？
确保在项目根目录运行，且已经用 `main.py` 生成了知识图谱。

### Q: 如何修改评估权重？
编辑 `model_evaluator.py` 中的 `_compute_overall_score` 函数。

---

**版本**: v1.0
**最后更新**: 2025-10-10
