# Scripts Directory

此目录包含辅助脚本和工具，用于数据处理、分析和维护。

## 📁 脚本列表

### 数据处理脚本

#### `extract_content.py`
- **功能**: 从爬虫结果中提取标题和内容
- **用途**: 将爬虫输出转换为知识图谱提取所需的格式
- **使用**:
```bash
python scripts/extract_content.py --input ../crawl/results/your_data.json \
                                  --output data/input/processed_data.json
```

#### `schema_discoverer.py`
- **功能**: 自动发现文本中的实体和关系类型
- **用途**: 帮助构建和优化Schema定义
- **使用**:
```bash
python scripts/schema_discoverer.py --input data/input/your_data.json
```

#### `test_schema_discovery.py`
- **功能**: 测试Schema发现功能
- **用途**: 验证Schema发现器的工作状态

---

### 分析和检查脚本

#### `check_no_extraction.py`
- **功能**: 检查没有提取到三元组的文档
- **用途**: 识别需要改进Schema或调整参数的文档
- **使用**:
```bash
python scripts/check_no_extraction.py --input knowledge_graph.json
```

#### `check_unused_schema.py`
- **功能**: 检查配置中未被使用的Schema
- **用途**: 优化Schema定义，移除无用的实体和关系类型
- **使用**:
```bash
python scripts/check_unused_schema.py --kg knowledge_graph.json \
                                      --config config.yaml
```

---

## 🚀 常用工作流程

### 1. 数据预处理流程
```bash
# 步骤1: 提取爬虫内容
python scripts/extract_content.py \
    --input ../crawl/results/your_crawl.json \
    --output data/input/processed.json

# 步骤2: 运行知识图谱提取
python main.py --input data/input/processed.json
```

### 2. Schema优化流程
```bash
# 步骤1: 自动发现Schema
python scripts/schema_discoverer.py --input data/input/your_data.json

# 步骤2: 检查未使用的Schema
python scripts/check_unused_schema.py \
    --kg knowledge_graph.json \
    --config config.yaml

# 步骤3: 根据结果优化config.yaml中的Schema定义
```

### 3. 质量检查流程
```bash
# 步骤1: 检查无提取结果的文档
python scripts/check_no_extraction.py --input knowledge_graph.json

# 步骤2: 分析并改进Schema或调整参数
# 步骤3: 重新运行提取
```

### 4. 质量评估流程（三元组质量评估）
```bash
# 步骤1: 环境检查
python scripts/quality_check/test_quality_check.py

# 步骤2: 运行质量评估
python scripts/quality_check/run_quality_check.py

# 步骤3: 查看评估报告
# 报告位于：data/output/quality_check_results/
```

---

## 📂 目录结构

### `evaluation/`
模型性能自动化评估模块
- `model_evaluator.py` - 模型评估器（8维度评估）
- `run_evaluation.py` - 运行评估脚本
- `test_evaluation.py` - 环境检查脚本

### `quality_check/`
三元组质量评估模块（无参照评估）
- `triple_quality_checker.py` - 质量评估器（2核心指标）
- `run_quality_check.py` - 运行质量评估脚本
- `test_quality_check.py` - 环境检查脚本
- `README.md` - 模块详细说明

---

## 📝 脚本参数说明

### extract_content.py
| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--input` | 输入爬虫结果文件 | 是 | - |
| `--output` | 输出处理后的文件 | 否 | `data/input/extracted_content.json` |

### schema_discoverer.py
| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--input` | 输入数据文件 | 是 | - |
| `--output` | 输出Schema文件 | 否 | `discovered_schema.yaml` |
| `--sample-size` | 采样文档数量 | 否 | 100 |

### check_no_extraction.py
| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--input` | 知识图谱结果文件 | 是 | - |
| `--output` | 输出无提取文档列表 | 否 | `no_extraction_docs.json` |

### check_unused_schema.py
| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--kg` | 知识图谱结果文件 | 是 | - |
| `--config` | 配置文件路径 | 是 | - |
| `--report` | 输出分析报告 | 否 | `schema_usage_report.txt` |

---

## 💡 使用建议

1. **数据预处理**: 始终先运行`extract_content.py`确保数据格式正确
2. **Schema优化**: 定期运行`check_unused_schema.py`清理无用的Schema定义
3. **质量监控**: 每次大批量提取后运行`check_no_extraction.py`检查质量
4. **Schema发现**: 处理新领域数据时，先运行`schema_discoverer.py`快速构建Schema

---

## 🔧 开发新脚本

如需添加新的辅助脚本，请遵循以下规范：

1. **命名**: 使用小写字母和下划线，如`new_script.py`
2. **文档**: 在脚本开头添加docstring说明功能和用法
3. **参数**: 使用argparse处理命令行参数
4. **日志**: 使用logging模块记录关键信息
5. **更新**: 在本README中添加新脚本的说明

---

## 📚 相关文档

- [主README](../README.md) - 项目整体说明
- [配置指南](../docs/README_NEW.md) - 详细配置说明
- [Schema指南](../docs/SCHEMA_GUIDE.md) - Schema定义指南
