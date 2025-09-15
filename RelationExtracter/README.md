# RelationExtracter - 异步关系提取器

基于PaddleNLP的高性能异步关系提取器，专为后端服务设计，支持多种配置和自定义Schema。

## 🌟 特性

- ⚡ **异步处理**: 基于asyncio的异步架构，避免阻塞事件循环
- 🎯 **多种Schema**: 预定义4种专业Schema（通用、商业、新闻、技术）
- ⚙️ **灵活配置**: YAML配置文件，6种预设配置适应不同场景
- 🔧 **易于集成**: 简洁API设计，支持FastAPI、Django等Web框架
- 📦 **批量处理**: 支持文本批量提取，提升处理效率
- 🎨 **高度可定制**: 支持自定义Schema和运行时参数覆盖
- 📊 **健康监控**: 内置健康检查和日志记录功能

## 🚀 快速开始

### 安装依赖

```bash
pip install paddlenlp paddlepaddle-gpu pyyaml
# 或者 CPU 版本
pip install paddlenlp paddlepaddle pyyaml
```

### 基本使用

```python
import asyncio
from async_relation_extractor import extract_relations_async

async def main():
    text = \"\"\"
    海信集团副总裁胡剑涌在2025年中国制冷展上发表重要讲话。
    他表示，海信中央空调将继续致力于技术创新。
    \"\"\"

    result = await extract_relations_async(text)
    print(result)

asyncio.run(main())
```

## 📁 项目结构

```
RelationExtracter/
├── async_relation_extractor.py    # 异步关系提取器核心
├── config_loader.py               # 配置文件加载器
├── config.yaml                    # YAML配置文件
├── test.py                        # 原始同步版本（参考）
├── test_async_extractor.py        # 异步版本测试文件
├── CONFIG_README.md               # 配置文件详细说明
└── README.md                      # 项目说明文档
```

## 🎯 核心组件

### 1. AsyncRelationExtractor (异步关系提取器)

主要的异步关系提取类，支持多种初始化方式：

```python
from async_relation_extractor import AsyncRelationExtractor, create_extractor

# 方式1: 使用预定义配置
async with create_extractor("high_performance") as extractor:
    result = await extractor.extract_async(text)

# 方式2: 自定义配置
async with create_extractor("standard", batch_size=16) as extractor:
    result = await extractor.extract_async(text)

# 方式3: 完全自定义
extractor = AsyncRelationExtractor(
    config_name="business_analysis",
    schema=custom_schema
)
```

### 2. 配置管理系统

基于YAML的配置管理，支持：
- 4种预定义Schema
- 6种预设配置
- 运行时参数覆盖
- 自定义Schema添加

```python
from config_loader import list_available_configs, get_config

print("可用配置:", list_available_configs())
config = get_config("high_performance")
```

### 3. 便捷函数

提供简化的API接口：

```python
from async_relation_extractor import extract_relations_async

# 使用默认配置
result = await extract_relations_async(text)

# 指定配置
result = await extract_relations_async(text, config_name="business_analysis")

# 使用特定Schema
result = await extract_relations_async(text, schema_key="technology")
```

## ⚙️ 配置说明

### 预定义配置

| 配置名称 | 适用场景 | 特点 |
|----------|----------|------|
| `standard` | 日常使用 | 平衡的性能和资源使用 |
| `high_performance` | 大量文本处理 | 高批次大小，多线程 |
| `memory_optimized` | 内存受限环境 | 小批次，短文本块 |
| `business_analysis` | 商业文本分析 | 商业Schema，优化参数 |
| `news_analysis` | 新闻文本分析 | 新闻Schema，大重叠窗口 |
| `tech_analysis` | 技术文档分析 | 技术Schema，适中配置 |

### Schema类型

| Schema | 实体类型 | 适用场景 |
|--------|----------|----------|
| `general` | 人物、产品、活动 | 通用文本信息提取 |
| `business` | 公司、高管、产品、合作 | 企业新闻、商业报告 |
| `news` | 事件、人物、机构 | 新闻文本、事件报道 |
| `technology` | 技术、产品、公司 | 技术文档、产品介绍 |

详细配置说明请参考 [CONFIG_README.md](CONFIG_README.md)

## 🔧 API 参考

### AsyncRelationExtractor

#### 初始化参数

```python
AsyncRelationExtractor(
    config_name: str = "standard",     # 配置名称
    schema: Optional[Dict] = None,     # 自定义Schema
    schema_key: Optional[str] = None,  # Schema键名
    **overrides                        # 参数覆盖
)
```

#### 主要方法

```python
# 异步初始化
await extractor.initialize()

# 单文本提取
result = await extractor.extract_async(text)

# 批量提取
results = await extractor.batch_extract_async([text1, text2, ...])

# 健康检查
is_healthy = await extractor.health_check()

# 资源清理
await extractor.close()
```

### 便捷函数

```python
# 异步提取
extract_relations_async(
    text: str,
    config_name: str = "standard",
    schema: Optional[Dict] = None,
    schema_key: Optional[str] = None
) -> List[Dict]

# 创建提取器
create_extractor(
    config_name: str = "standard",
    **overrides
) -> AsyncRelationExtractor
```

## 📊 性能测试

在我们的测试环境中（GTX 显卡，1020字符文本）：

- **处理时间**: ~13.55秒
- **提取准确率**: 良好，能准确识别主要实体和关系
- **内存使用**: 根据配置可调，支持低内存模式
- **并发支持**: 基于asyncio，支持高并发处理

## 🚀 后端集成示例

### FastAPI 集成

```python
from fastapi import FastAPI
from async_relation_extractor import AsyncRelationExtractor

app = FastAPI()

# 全局提取器实例
extractor = None

@app.on_event("startup")
async def startup():
    global extractor
    extractor = AsyncRelationExtractor("high_performance")
    await extractor.initialize()

@app.on_event("shutdown")
async def shutdown():
    await extractor.close()

@app.post("/extract")
async def extract_relations(text: str):
    result = await extractor.extract_async(text)
    return {"result": result}
```

### Django Async Views

```python
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

# 初始化提取器
extractor = AsyncRelationExtractor("business_analysis")

@csrf_exempt
@require_http_methods(["POST"])
async def extract_relations(request):
    data = json.loads(request.body)
    text = data.get("text", "")

    result = await extractor.extract_async(text)
    return JsonResponse({"result": result})
```

## 🧪 测试

运行完整测试套件：

```bash
python test_async_extractor.py
```

测试包括：
- 配置信息显示
- 基础提取功能
- 多配置切换
- 批量处理
- 自定义Schema
- 参数覆盖
- 性能测试

## 🔍 使用场景

### 1. 企业情报分析
```python
# 使用商业分析配置
result = await extract_relations_async(
    business_news_text,
    config_name="business_analysis"
)
```

### 2. 新闻事件监控
```python
# 使用新闻分析配置
result = await extract_relations_async(
    news_article,
    config_name="news_analysis"
)
```

### 3. 技术文档解析
```python
# 使用技术分析配置
result = await extract_relations_async(
    technical_document,
    config_name="tech_analysis"
)
```

### 4. 大规模文本处理
```python
# 高性能批量处理
async with create_extractor("high_performance") as extractor:
    results = await extractor.batch_extract_async(text_list)
```

## ⚡ 性能优化建议

### 1. 模型加载优化
- 建议在应用启动时初始化提取器
- 避免重复加载模型实例
- 使用单例模式管理模型

### 2. 批量处理优化
- 大量文本使用 `batch_extract_async`
- 根据显存调整 `batch_size`
- 使用 `high_performance` 配置

### 3. 内存优化
- 内存受限时使用 `memory_optimized` 配置
- 调整 `chunk_max` 控制内存使用
- 及时调用 `close()` 释放资源

## 🐛 常见问题

### Q: 模型加载慢怎么办？
A: 首次运行会下载模型，之后会使用缓存。建议在部署时预下载模型。

### Q: 内存不足怎么处理？
A: 使用 `memory_optimized` 配置，或手动降低 `batch_size` 和 `chunk_max`。

### Q: 提取结果不准确？
A: 尝试使用更适合的Schema类型，或创建针对性的自定义Schema。

### Q: 如何提升处理速度？
A: 使用 `high_performance` 配置，增大 `batch_size` 和 `max_workers`。

## 📋 TODO

- [ ] 添加模型单例管理避免重复加载
- [ ] 支持更多预训练模型
- [ ] 添加结果后处理和过滤功能
- [ ] 支持流式处理大文件
- [ ] 添加模型热更新功能
- [ ] 优化GPU显存使用

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目基于MIT许可证开源。

## 📞 支持

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 查看文档和示例代码
- 参考配置说明文档