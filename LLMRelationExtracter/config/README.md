# 配置文件说明

本目录包含LLMRelationExtracter系统的配置文件。

## 配置文件

### config.yaml
主配置文件，使用OpenAI兼容的API接口（包括Gemini、DeepSeek等）。

**使用方法：**
```bash
# 复制为实际配置文件
cp config/config.yaml config.yaml

# 编辑并填入你的API密钥
# 然后运行
python main.py --input data/input/your_data.json
```

### config.glm.yaml
智谱AI GLM模型配置文件，专门针对GLM系列模型优化。

**使用方法：**
```bash
# 复制为实际配置文件
cp config/config.glm.yaml config.yaml

# 编辑并填入你的GLM API密钥
# 然后运行
python main.py --input data/input/your_data.json
```

### config.azure.yaml
Azure OpenAI服务配置文件，使用Microsoft Azure上托管的OpenAI模型。

**使用方法：**
```bash
# 复制为实际配置文件
cp config/config.azure.yaml config.yaml

# 编辑并填入你的Azure OpenAI配置
# 需要配置: azure_endpoint, api_key, api_version, deployment_name
# 然后运行
python main.py --input data/input/your_data.json
```

**Azure OpenAI配置说明：**
- `provider`: 设置为 `"azure"`
- `azure_endpoint`: Azure OpenAI端点，格式如 `https://YOUR-RESOURCE-NAME.openai.azure.com/`
- `api_key`: Azure OpenAI API密钥（在Azure Portal中获取）
- `api_version`: API版本，推荐使用 `"2024-02-15-preview"`
- `deployment_name`: 在Azure Portal中创建的部署名称（如 `gpt-4o`）

## 配置项说明

### 模型配置 (model)
- `provider`: 模型提供商（openai/azure/zhipuai等）
- `model_name`: 模型名称（非Azure时使用）
- `deployment_name`: 部署名称（Azure OpenAI专用）
- `api_key`: API密钥（必填）
- `api_base`: API地址（OpenAI兼容接口使用）
- `azure_endpoint`: Azure OpenAI端点（Azure专用）
- `api_version`: API版本（Azure专用）
- `max_tokens`: 最大输出token数
- `temperature`: 温度参数（0-1）
- `timeout`: 超时时间（秒）

### 实体类型 (entity_types)
定义可识别的实体类型，每个实体类型包含：
- `description`: 类型描述
- `examples`: 示例列表
- `attributes`: 实体属性定义（可选）

### 关系类型 (relation_types)
定义可识别的关系类型，每个关系包含：
- `description`: 关系描述
- `subject_types`: 允许的主体类型
- `object_types`: 允许的客体类型
- `examples`: 示例三元组

### JSON Schema (json_schema)
- `enabled`: 是否启用JSON Schema结构化输出
- `schema`: JSON Schema定义

**注意：** 只有支持的模型才能启用此功能（GPT-4o+, GLM-4+）

### 提示词配置 (prompts)
- `system_prompt`: 系统提示词
- `task_prompt`: 任务提示词
- `few_shot_examples`: Few-shot示例

### 高级技巧 (advanced_techniques)
- `enable_few_shot`: 启用Few-shot学习
- `few_shot_count`: Few-shot示例数量
- `enable_self_consistency`: 启用自我一致性
- `consistency_count`: 一致性检查次数

### 输出配置 (output)
- `format`: 输出格式（json/jsonl/csv/neo4j）
- `output_path`: 输出文件路径
- `confidence_threshold`: 置信度阈值
- `deduplicate`: 是否去重

### 处理配置 (processing)
- `batch_size`: 批处理大小
- `max_text_length`: 最大文本长度
- `max_retries`: 最大重试次数
- `retry_delay`: 重试间隔（秒）
- `enable_parallel`: 启用并行处理
- `max_workers`: 并行进程数

## 快速开始

1. **选择配置文件**
   - 使用OpenAI/Gemini/DeepSeek等：选择 `config.yaml`
   - 使用Azure OpenAI：选择 `config.azure.yaml`
   - 使用智谱AI GLM：选择 `config.glm.yaml`

2. **复制并配置**
   ```bash
   # OpenAI兼容接口
   cp config/config.yaml config.yaml

   # 或者使用Azure OpenAI
   # cp config/config.azure.yaml config.yaml

   # 或者使用智谱AI
   # cp config/config.glm.yaml config.yaml

   # 编辑config.yaml，填入API密钥和相关配置
   ```

3. **运行系统**
   ```bash
   python main.py --input data/input/documents.json
   ```

## 注意事项

1. **API密钥安全**
   - 生产环境的 `config.yaml` 已在 `.gitignore` 中
   - 不要将包含真实API密钥的配置文件提交到Git

2. **模型选择**
   - **免费/试用**：GLM-4-flash（永久免费，性能优秀）
   - **Azure OpenAI**：企业级，稳定性高，支持私有部署
   - **OpenAI官方**：GPT-4o（强大但较贵）
   - **性价比**：Gemini-2.5-flash、DeepSeek-V3

3. **Azure OpenAI特别说明**
   - 需要在Azure Portal中创建OpenAI资源
   - 需要创建模型部署（deployment）
   - 支持虚拟网络和私有端点
   - 计费方式与OpenAI官方不同

4. **并发控制**
   - OpenAI官方：可设置较高并发（10-20）
   - Azure OpenAI：根据配额调整（通常5-15）
   - GLM：建议降低并发（5-10）避免触发QPS限制

5. **JSON Schema**
   - 提高输出质量，但需要模型支持
   - 支持的模型：GPT-4o+, GLM-4+, Azure OpenAI (GPT-4o+)
   - 不支持时设置 `enabled: false`

## 高级自定义

### 添加新实体类型

```yaml
entity_types:
  新实体类型:
    description: "详细描述"
    examples: ["示例1", "示例2"]
    attributes:
      属性名:
        value_type: "数值/文本/枚举"
        description: "属性说明"
```

### 添加新关系类型

```yaml
relation_types:
  新关系:
    description: "关系说明"
    subject_types: ["主体类型1", "主体类型2"]
    object_types: ["客体类型1", "客体类型2"]
    examples: ["示例三元组"]
```

## 更多信息

详细文档请参考主 README.md 文件。
