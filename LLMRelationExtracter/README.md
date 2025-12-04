# LLM知识图谱关系抽取系统

一个基于大语言模型的知识图谱三元组抽取系统，支持灵活的实体和关系类型定义、JSON Schema约束、证据位置标注和来源追踪。

## 🌟 主要特性

- ✅ **灵活的Schema定义**: 支持自定义实体类型和关系类型，带描述、约束和示例
- ✅ **JSON Schema约束**: 确保LLM输出结构化和一致性
- ✅ **证据位置标注**: 提取`evidence_spans`字段，标注证据在原文中的精确位置
- ✅ **来源追踪**: 每个三元组包含`source_url`，方便审查和溯源
- ✅ **Few-shot学习**: 支持动态Few-shot示例提升抽取质量
- ✅ **批量处理**: 支持并行处理大量文档
- ✅ **多格式输出**: JSON、JSONL、CSV、Neo4j导入格式
- ✅ **配置内外分类**: 自动区分符合/不符合配置的三元组
- ✅ **多模型支持**: OpenAI、Azure OpenAI、GLM、Gemini、DeepSeek等主流LLM

## 📁 项目结构

```
LLMRelationExtracter/
├── README.md                  # 本文档
├── config.yaml                # 实际配置文件（不提交Git）
├── main.py                    # 主程序入口
├── kg_extractor.py            # 核心三元组抽取器
├── kg_builder.py              # 批量处理和知识图谱构建器
├── few_shot_manager.py        # Few-shot示例管理器
├── schema_discoverer.py       # Schema发现工具
├── __init__.py                # 包初始化文件
│
├── config/                    # 配置文件目录
│   ├── README.md              # 配置说明文档
│   ├── config.yaml            # OpenAI兼容配置（Gemini/DeepSeek等）
│   ├── config.azure.yaml      # Azure OpenAI专用配置
│   └── config.glm.yaml        # 智谱AI GLM专用配置
│
├── data/                      # 数据目录
│   ├── input/                 # 输入数据（*.json）
│   ├── output/                # 输出结果（知识图谱）
│   └── checkpoints/           # 处理检查点
│
├── scripts/                   # 工具脚本
│   ├── extract_content.py     # 数据预处理工具
│   ├── check_no_extraction.py # 检查无提取结果的文件
│   ├── check_unused_schema.py # 检查未使用的Schema元素
│   ├── convert_to_entity_attributes.py # 格式转换工具
│   ├── schema_discoverer.py   # Schema自动发现
│   ├── test_schema_discovery.py # Schema发现测试
│   ├── evaluation/            # 评估脚本
│   └── quality_check/         # 质量检查工具
│
├── tests/                     # 测试脚本
│   ├── test_extraction.py     # 功能测试脚本
│   └── test_entity_extraction.py # 实体抽取测试
│
├── assets/                    # 资源文件
│   ├── schema.docx            # Schema定义文档
│   └── schema_diagram.jpg     # Schema结构图
│
└── logs/                      # 日志文件
    └── kg_extraction.log
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
cd LLMRelationExtracter

# 安装依赖
pip install -r requirements.txt
```

**依赖包：**
- `openai` - OpenAI API客户端
- `zhipuai` - 智谱AI GLM客户端（可选）
- `pyyaml` - YAML配置解析
- `tiktoken` - Token计数
- `pandas` - 数据处理
- `tqdm` - 进度条显示

### 2. 配置设置

**选择配置文件：**
- **OpenAI/Gemini/DeepSeek等**: 使用 `config/config.yaml`
- **Azure OpenAI**: 使用 `config/config.azure.yaml`
- **智谱AI GLM**: 使用 `config/config.glm.yaml`

```bash
# 方式1: 使用OpenAI兼容接口（推荐Gemini 2.5 Flash）
cp config/config.yaml config.yaml

# 方式2: 使用Azure OpenAI（企业级，稳定性高）
cp config/config.azure.yaml config.yaml

# 方式3: 使用智谱AI GLM-4（免费）
cp config/config.glm.yaml config.yaml
```

**编辑配置文件：**

```yaml
# OpenAI/Gemini/DeepSeek等
model:
  provider: "openai"              # 或 "zhipuai"
  model_name: "gemini-2.5-flash"  # 或 "glm-4-flash"
  api_key: "YOUR-API-KEY-HERE"    # 填入你的API密钥
  api_base: "https://api.example.com/v1/"
  max_tokens: 16000
  temperature: 0.1
  timeout: 60

# Azure OpenAI
model:
  provider: "azure"
  azure_endpoint: "https://YOUR-RESOURCE-NAME.openai.azure.com/"
  api_key: "YOUR-AZURE-API-KEY"
  api_version: "2024-02-15-preview"
  deployment_name: "gpt-4o"       # 部署名称
  max_tokens: 16000
  temperature: 0.1
  timeout: 300
```

**获取API密钥：**
- **OpenAI**: https://platform.openai.com/
- **Azure OpenAI**: https://portal.azure.com/ （需要Azure订阅）
- **智谱AI**: https://open.bigmodel.cn/ （新用户免费额度，glm-4-flash永久免费）
- **Gemini**: https://aistudio.google.com/
- **DeepSeek**: https://platform.deepseek.com/

### 3. 运行测试

```bash
# 运行功能测试，验证系统正常工作
python tests/test_extraction.py
```

### 4. 执行抽取

```bash
# 测试模式（只处理前5个文档）
python main.py --input data/input/your_data.json --test

# 完整运行
python main.py --input data/input/your_data.json

# 启用并行处理
python main.py --input data/input/your_data.json --parallel

# 指定输出路径
python main.py --input data/input/your_data.json --output data/output/my_kg.json

# 详细输出模式
python main.py --input data/input/your_data.json -v
```

## 💡 核心功能详解

### 1. 灵活的Schema定义

在配置文件中自定义实体和关系类型：

```yaml
entity_types:
  公司:
    description: "企业、公司、集团等商业组织"
    examples: ["格力", "美的", "海尔"]
    attributes:
      企业规模:
        value_type: "文本"
        description: "企业的规模描述"
      市场份额:
        value_type: "数值"
        unit: "%"
        description: "市场占有率"

relation_types:
  制造:
    description: "主体生产、制造客体"
    subject_types: ["公司"]
    object_types: ["产品"]
    examples: ["格力制造空调"]
```

**空调行业预定义Schema：**
- **18种实体类型**: 品牌、系列、品类、产品型号、制造商、零部件、技术、制冷剂、能效等级、性能参数、产品功能、工程便利性、认证机构、地区、政策、市场、数值、时间
- **27种关系类型**: 属于品牌、属于系列、制造、供应、竞争、合作、采用技术、使用制冷剂、符合能效等

详细Schema说明请参考配置文件中的注释。

### 2. 证据位置标注

每个三元组包含精确的证据位置，支持溯源验证：

```json
{
  "subject": "格力",
  "relation": "制造",
  "object": "空调",
  "evidence": "格力电器空调销量",
  "evidence_spans": [
    {
      "start": 0,
      "end": 10,
      "text": "格力电器空调销量"
    }
  ],
  "source_url": "https://example.com/article1"
}
```

### 3. JSON Schema结构化输出

启用后强制LLM按预定义格式输出，大幅提高质量：

```yaml
json_schema:
  enabled: true  # 启用（需要模型支持）
  schema:
    type: "object"
    properties:
      triplets:
        type: "array"
        items:
          # 详细schema定义
```

**支持的模型：**
- ✅ OpenAI: GPT-4o, GPT-4o-mini
- ✅ 智谱AI: GLM-4, GLM-4-plus, GLM-4-flash, GLM-4-air
- ❌ OpenAI: GPT-4, GPT-3.5（不支持Structured Outputs）

### 4. Few-shot学习

自动添加高质量示例提升抽取准确性：

```yaml
advanced_techniques:
  enable_few_shot: true
  few_shot_count: 2  # 示例数量
```

### 5. 输出格式

系统生成三个JSON文件：
- `knowledge_graph.json` - 完整结果（包含元数据和统计）
- `knowledge_graph_in_config.json` - 配置内三元组（完全符合Schema）
- `knowledge_graph_out_of_config.json` - 配置外三元组（包含未定义类型）

**输出示例：**

```json
{
  "metadata": {
    "extraction_timestamp": "2025-10-13T14:00:00",
    "model": "gemini-2.5-flash",
    "statistics": {
      "processing_summary": {
        "total_documents": 100,
        "total_triplets": 450,
        "total_processing_time": 120.5,
        "avg_time_per_document": 1.2
      },
      "classification_summary": {
        "fully_in_config": 380,
        "fully_out_of_config": 45,
        "in_config_percentage": 84.4
      }
    }
  },
  "triplets": {
    "in_config": [...],
    "out_of_config": [...],
    "all": [...]
  }
}
```

## 📖 使用示例

### Python API

```python
from kg_extractor import KnowledgeGraphExtractor
from kg_builder import KnowledgeGraphBuilder

# 单文本提取
extractor = KnowledgeGraphExtractor('config.yaml')
result = extractor.extract_from_text("格力电器2025年空调销量增长15%")

for triplet in result.triplets:
    print(f"({triplet.subject}, {triplet.relation}, {triplet.object})")
    print(f"  来源: {triplet.source_url}")
    print(f"  证据: {triplet.evidence}")
    print(f"  位置: {triplet.evidence_spans}")

# 批量处理
builder = KnowledgeGraphBuilder('config.yaml')
result = builder.build_knowledge_graph('data/input/documents.json')
print(f"提取了 {len(result['triplets'])} 个三元组")
```

### 命令行

```bash
# 基本用法
python main.py -i data/input/documents.json

# 详细输出
python main.py -i data/input/documents.json -v

# 自定义批处理大小
python main.py -i data/input/documents.json -b 20

# 组合使用
python main.py -i data/input/documents.json -o output.json --parallel -v
```

## ⚙️ 配置选项

### 模型配置

```yaml
model:
  provider: "openai"           # openai/zhipuai/azure/deepseek等
  model_name: "gpt-4o"         # 模型名称
  api_key: "sk-..."            # API密钥
  api_base: "https://..."      # API端点
  max_tokens: 2000             # 最大输出token
  temperature: 0.1             # 温度参数（0-1）
  timeout: 60                  # 超时时间（秒）
```

**推荐模型：**
- **免费**: GLM-4-flash（永久免费，性能优秀）
- **性价比**: Gemini-2.5-flash、DeepSeek-V3
- **高质量**: GPT-4o、Claude-3.5-Sonnet

### 高级技巧

```yaml
advanced_techniques:
  enable_few_shot: true          # 启用few-shot
  few_shot_count: 2              # 示例数量
  enable_self_consistency: false # 自我一致性（多次采样）
  consistency_count: 3           # 一致性检查次数
  enable_verification: true      # 结果验证
```

### 处理配置

```yaml
processing:
  batch_size: 10              # 批处理大小
  max_text_length: 8000       # 最大文本长度（字符）
  max_retries: 3              # 最大重试次数
  retry_delay: 1              # 重试间隔（秒）
  enable_parallel: true       # 并行处理
  max_workers: 10             # 并行进程数
```

**性能调优：**
- **OpenAI**: max_workers: 10-20
- **GLM**: max_workers: 5-10（避免QPS限制）
- **大文本**: 增加 max_text_length 和 timeout

### 输出配置

```yaml
output:
  format: "json"              # 输出格式: json/jsonl/csv/neo4j
  output_path: "knowledge_graph.json"
  save_intermediate: true     # 保存中间结果
  deduplicate: true           # 去重
  confidence_threshold: 0.7   # 置信度阈值
```

## 🔧 工具脚本

### 数据预处理

```bash
# 提取和清理内容
python scripts/extract_content.py --input raw_data.json --output clean_data.json
```

### 质量检查

```bash
# 检查无提取结果的文件
python scripts/check_no_extraction.py --kg knowledge_graph.json

# 检查未使用的Schema元素
python scripts/check_unused_schema.py --kg knowledge_graph.json --config config.yaml

# 转换为实体-属性格式
python scripts/convert_to_entity_attributes.py --input kg.json --output entities.json
```

### Schema发现

```bash
# 自动发现新的实体和关系类型
python schema_discoverer.py --input data/input/documents.json --output discovered_schema.yaml

# 测试Schema发现
python scripts/test_schema_discovery.py
```

## 🎓 Schema定义指南

### 实体类型示例

空调行业包含18种预定义实体类型：

| 实体类型 | 说明 | 示例 |
|---------|------|------|
| **品牌** | 空调产品品牌 | 格力、美的、海尔、大金 |
| **系列** | 品牌下的产品线 | 海信耀享系列、格力T爽系列 |
| **产品型号** | 具体型号 | 美的观酷 KFR-35GW/N8XHC1 |
| **制造商** | 生产企业 | 格力电器、美的集团 |
| **技术** | 采用的技术 | 变频技术、热泵技术 |
| **制冷剂** | 制冷剂类型 | R410A、R32、R290 |
| **能效等级** | 能效标准 | 一级能效、新国标一级 |
| **性能参数** | 性能指标 | 制冷量、APF、噪音值 |

### 关系类型示例

27种预定义关系类型，覆盖产品全生命周期：

| 关系类型 | 说明 | 示例 |
|---------|------|------|
| **属于品牌** | 产品归属品牌 | 美的观酷属于品牌美的 |
| **属于系列** | 型号归属系列 | KFR-35GW属于系列观酷 |
| **制造** | 生产关系 | 格力制造空调 |
| **采用技术** | 技术应用 | 产品采用变频技术 |
| **使用制冷剂** | 制冷剂使用 | 产品使用R32制冷剂 |
| **符合能效** | 能效认证 | 产品符合一级能效 |

### 典型三元组示例

**输入文本：**
```
美的观酷 KFR-35GW/N8XHC1是一款壁挂式空调，采用变频技术，
制冷量为3500W，能效等级达到新国标一级。
```

**提取三元组：**
1. (美的观酷, 属于品牌, 美的)
2. (KFR-35GW/N8XHC1, 属于系列, 观酷)
3. (KFR-35GW/N8XHC1, 属于品类, 壁挂式空调)
4. (KFR-35GW/N8XHC1, 采用技术, 变频技术)
5. (KFR-35GW/N8XHC1, 具有参数, 制冷量)
6. (制冷量, 参数值为, 3500W)
7. (KFR-35GW/N8XHC1, 符合能效, 新国标一级)

## 🐛 常见问题

### Q: 如何提高抽取质量？

1. **调整Few-shot示例**：增加高质量示例数量
2. **优化提示词**：在配置文件中修改 `prompts` 部分
3. **提高置信度阈值**：设置 `confidence_threshold: 0.8`
4. **启用JSON Schema**：确保输出格式规范
5. **使用更强大的模型**：如GPT-4o、GLM-4-plus

### Q: evidence_spans不准确？

这取决于LLM的能力，可以：
1. 在提示词中强调位置准确性
2. 使用更强大的模型（如GPT-4o）
3. 增加Few-shot示例的位置标注质量
4. 启用JSON Schema强制格式约束

### Q: 如何处理大量文档？

```bash
# 启用并行处理，增加worker数量
python main.py -i data/input/large_dataset.json --parallel

# 或在config.yaml中设置
processing:
  enable_parallel: true
  max_workers: 20  # 根据API限制调整
  batch_size: 50   # 增大批处理
```

### Q: API返回错误或超时？

1. **增加超时时间**：`timeout: 120`
2. **降低并发数**：`max_workers: 5`（特别是GLM）
3. **减少max_tokens**：`max_tokens: 4000`
4. **检查API密钥**：确保密钥有效且有余额
5. **查看日志**：`logs/kg_extraction.log`

### Q: JSON Schema报错？

```
错误: "This model does not support structured outputs"
```

**解决方案：**
1. 检查模型是否支持（GPT-4o+, GLM-4+）
2. 在配置文件中设置 `json_schema.enabled: false`
3. 升级到支持的模型版本

### Q: 如何添加自定义实体/关系类型？

编辑配置文件：

```yaml
entity_types:
  新实体类型:
    description: "详细描述"
    examples: ["示例1", "示例2"]

relation_types:
  新关系:
    description: "关系说明"
    subject_types: ["允许的主体类型"]
    object_types: ["允许的客体类型"]
    examples: ["示例三元组"]
```

### Q: 支持哪些LLM模型？

**完全支持（含JSON Schema）：**
- ✅ OpenAI: GPT-4o, GPT-4o-mini
- ✅ Azure OpenAI: GPT-4o, GPT-4o-mini（需要Azure订阅）
- ✅ 智谱AI: GLM-4, GLM-4-plus, GLM-4-flash, GLM-4-air
- ✅ Anthropic: Claude-3.5-Sonnet (需要配置)

**基本支持（无JSON Schema）：**
- ✅ OpenAI: GPT-4, GPT-3.5-turbo
- ✅ Gemini: Gemini-2.5-flash, Gemini-1.5-pro
- ✅ DeepSeek: DeepSeek-V3, DeepSeek-Chat
- ✅ 其他OpenAI兼容接口

**Azure OpenAI特点：**
- 企业级稳定性和SLA保证
- 支持虚拟网络和私有部署
- 数据驻留和合规性保证
- 与Azure生态系统集成

## 📝 注意事项

1. **API密钥安全**
   - `config.yaml` 包含API密钥，已添加到 `.gitignore`
   - 不要将实际配置文件提交到Git
   - 生产环境建议使用环境变量

2. **成本控制**
   - 监控API调用量和Token消耗
   - 使用 `--test` 模式进行小规模测试
   - 优先选择性价比高的模型（GLM-4-flash免费）

3. **性能优化**
   - 大文件建议使用 `--parallel` 并行处理
   - 根据API限制调整 `max_workers`
   - 启用 `save_intermediate` 支持断点续传

4. **输出质量**
   - 证据位置准确性取决于模型能力
   - 建议人工抽查配置外三元组
   - 使用质量检查脚本验证结果

## 🤝 贡献

欢迎提交Issue和Pull Request！

**贡献方向：**
- 新的实体/关系类型定义
- 更好的提示词模板
- 质量检查工具改进
- 新模型集成
- 性能优化

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请通过GitHub Issue联系。

---

**版本**: v2.0
**最后更新**: 2025-10-13
**行业**: 制冷空调
**适用场景**: 知识图谱构建、关系抽取、信息提取、竞争分析
