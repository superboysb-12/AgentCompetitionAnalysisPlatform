# 配置文件说明 (config.yaml)

本文档详细说明了关系提取器的配置文件结构和使用方法。

## 📋 目录

- [配置文件结构](#配置文件结构)
- [Schema配置](#schema配置)
- [预定义配置](#预定义配置)
- [如何使用配置](#如何使用配置)
- [自定义配置](#自定义配置)
- [配置参数说明](#配置参数说明)

## 🏗️ 配置文件结构

```yaml
# config.yaml 基本结构
schemas:          # Schema定义区域
  schema_name:    # Schema名称
    实体类型: [属性1, 属性2, ...]

configs:          # 配置定义区域
  config_name:    # 配置名称
    schema_key: "schema名称"
    model: {...}           # 模型配置
    text_processing: {...} # 文本处理配置
    logging: {...}         # 日志配置
```

## 🎯 Schema配置

### 内置Schema类型

#### 1. general (通用信息抽取)
```yaml
general:
  人物: ["头衔", "所属机构", "观点"]
  产品: ["型号", "技术亮点", "应用场景"]
  活动: ["时间", "地点", "主办方"]
```
**适用场景**: 一般性文本信息提取

#### 2. business (商业分析)
```yaml
business:
  公司: ["名称", "业务范围", "产品", "市场地位"]
  高管: ["姓名", "职位", "公司", "主要观点"]
  产品: ["名称", "特点", "技术优势", "目标市场"]
  合作: ["合作方", "合作内容", "合作时间"]
```
**适用场景**: 企业新闻、商业报告分析

#### 3. news (新闻事件)
```yaml
news:
  事件: ["名称", "时间", "地点", "参与方"]
  人物: ["姓名", "身份", "言论", "行为"]
  机构: ["名称", "类型", "作用", "态度"]
```
**适用场景**: 新闻文本、事件报道分析

#### 4. technology (技术分析)
```yaml
technology:
  技术: ["名称", "类型", "特点", "应用领域"]
  产品: ["名称", "技术规格", "性能指标", "创新点"]
  公司: ["名称", "技术实力", "研发方向", "产品线"]
```
**适用场景**: 技术文档、产品说明分析

## ⚙️ 预定义配置

### 1. standard (标准配置)
- **用途**: 日常使用的平衡配置
- **特点**: 中等批次大小，标准精度
- **配置**: batch_size=8, precision=bfloat16

### 2. high_performance (高性能配置)
- **用途**: 需要快速处理大量文本
- **特点**: 大批次处理，更多线程
- **配置**: batch_size=16, max_workers=8

### 3. memory_optimized (内存优化配置)
- **用途**: 内存受限环境
- **特点**: 小批次处理，短文本块
- **配置**: batch_size=4, chunk_max=500

### 4. business_analysis (商业分析配置)
- **用途**: 商业文本分析
- **特点**: 使用business schema，中等批次
- **配置**: schema_key=business, batch_size=12

### 5. news_analysis (新闻分析配置)
- **用途**: 新闻文本分析
- **特点**: 使用news schema，更大重叠窗口
- **配置**: schema_key=news, stride_ratio=0.3

### 6. tech_analysis (技术分析配置)
- **用途**: 技术文档分析
- **特点**: 使用technology schema
- **配置**: schema_key=technology, batch_size=10

## 🚀 如何使用配置

### 1. Python代码中使用

```python
from async_relation_extractor import create_extractor, extract_relations_async

# 使用预定义配置
async with create_extractor("high_performance") as extractor:
    result = await extractor.extract_async(text)

# 使用便捷函数
result = await extract_relations_async(text, config_name="business_analysis")
```

### 2. 查看可用配置

```python
from config_loader import list_available_configs, list_available_schemas

print("可用配置:", list_available_configs())
print("可用Schema:", list_available_schemas())
```

## 🎨 自定义配置

### 1. 添加自定义Schema

在config.yaml中添加新的schema：

```yaml
schemas:
  my_custom_schema:
    实体类型1: ["属性1", "属性2"]
    实体类型2: ["属性3", "属性4"]
```

### 2. 创建自定义配置

```yaml
configs:
  my_config:
    schema_key: "my_custom_schema"
    model:
      name: "paddlenlp/PP-UIE-0.5B"
      batch_size: 6
      precision: "bfloat16"
    text_processing:
      chunk_max: 600
      stride_ratio: 0.2
```

### 3. 运行时自定义

```python
# 使用自定义schema
custom_schema = {
    "机构": ["名称", "类型", "职能"],
    "政策": ["名称", "发布时间", "适用范围"]
}

async with create_extractor("standard", schema=custom_schema) as extractor:
    result = await extractor.extract_async(text)

# 参数覆盖
async with create_extractor("standard", batch_size=16, chunk_max=1000) as extractor:
    result = await extractor.extract_async(text)
```

## 📖 配置参数说明

### 模型配置 (model)

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| name | 模型名称 | paddlenlp/PP-UIE-0.5B | - |
| batch_size | 批处理大小 | 8 | 1-32 |
| precision | 精度设置 | bfloat16 | float16/float32/float64/bfloat16 |
| schema_lang | Schema语言 | zh | zh/en |
| max_workers | 最大线程数 | 4 | 1-16 |

### 文本处理配置 (text_processing)

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| chunk_min | 文本块最小长度 | 300 | 100-500 |
| chunk_max | 文本块最大长度 | 800 | 500-2000 |
| stride_ratio | 滑窗重叠比例 | 0.25 | 0.1-0.5 |
| dedup_min_length | 去重最小长度 | 20 | 10-100 |
| fingerprint_length | 指纹长度 | 80 | 50-200 |

### 日志配置 (logging)

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| level | 日志级别 | INFO | DEBUG/INFO/WARNING/ERROR |

## 💡 最佳实践

### 1. 场景选择

- **大批量处理**: 使用 `high_performance` 配置
- **内存受限**: 使用 `memory_optimized` 配置
- **商业文本**: 使用 `business_analysis` 配置
- **新闻文本**: 使用 `news_analysis` 配置

### 2. 性能优化

- 根据GPU显存调整 `batch_size`
- 长文本使用更大的 `chunk_max`
- 重要信息多的文本增大 `stride_ratio`

### 3. 准确性优化

- 针对特定领域创建专门的schema
- 调整文本分块参数适应文本特点
- 使用合适的precision平衡速度和精度

## 🔧 故障排除

### 常见问题

1. **内存不足**: 降低 `batch_size` 和 `chunk_max`
2. **精度要求高**: 使用 `float32` 精度
3. **处理速度慢**: 增大 `batch_size` 和 `max_workers`
4. **提取结果不准确**: 调整schema定义或文本分块参数

### 日志调试

设置日志级别为DEBUG查看详细信息：

```yaml
logging:
  level: "DEBUG"
```

## 📝 配置模板

复制以下模板创建自定义配置：

```yaml
configs:
  my_custom_config:
    schema_key: "general"  # 或自定义schema名
    model:
      name: "paddlenlp/PP-UIE-0.5B"
      batch_size: 8
      precision: "bfloat16"
      schema_lang: "zh"
      max_workers: 4
    text_processing:
      chunk_min: 300
      chunk_max: 800
      stride_ratio: 0.25
      dedup_min_length: 20
      fingerprint_length: 80
    logging:
      level: "INFO"
```