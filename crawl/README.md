# 智能网页爬虫

基于 Python 和 Playwright 构建的现代化智能爬虫，专为爬取动态 JavaScript 网站设计，采用清洁架构，支持多种爬取模式。

## 功能特性

多种爬取模式：
- 直接模式 (Direct)：直接爬取页面内容，无需按钮交互
- 间接模式 (Indirect)：点击按钮后爬取跳转页面内容
- 多步模式 (Multistep)：执行复杂操作序列（点击、提取、等待）

智能按钮发现：
- 智能元素检测与去重
- 基于 CSS 选择器的精准定位
- 自动 URL 解析（相对路径转绝对路径）

灵活的点击策略：
- 直接导航：通过 href 属性快速导航
- 新标签页监听：处理弹窗和新标签页
- 同页导航：跟踪页面跳转
- 直接点击：处理 AJAX 和单页应用
- 内部链接：专门处理文章卡片

页面加载检测：
- DOM 内容加载检测
- 网络空闲监控
- JavaScript 执行等待
- 动态内容稳定性检查
- 快速模式选项

内容提取：
- 基于 CSS 选择器提取
- 支持文本、链接和图片
- 自动 URL 解析
- 可自定义字段映射

## 项目架构

项目遵循清洁架构原则，职责分离清晰：

```
crawl/
├── crawler.py              # 主爬虫引擎
├── run.py                  # 程序入口
├── config/
│   └── manager.py         # 配置管理
├── core/
│   ├── browser.py         # 浏览器生命周期管理
│   ├── detector.py        # 页面加载检测
│   ├── discovery.py       # 按钮发现逻辑
│   ├── extractor.py       # 内容提取
│   ├── strategies.py      # 点击策略实现
│   ├── storage.py         # 数据存储
│   └── types.py           # 类型定义
├── task_config/           # 任务配置文件
└── results/               # 爬取结果输出
```

## 安装

### 环境要求

- Python 3.8+
- pip

### 安装步骤

1. 安装 Python 依赖：

```bash
pip install playwright pyyaml
```

2. 安装 Playwright 浏览器：

```bash
playwright install chromium
```

## 使用方法

### 基本用法

使用配置文件运行爬虫：

```bash
python run.py task_config/example.yaml
```

### 配置说明

创建 YAML 配置文件定义爬取任务：

#### 直接模式示例

```yaml
settings:
  browser_type: chromium
  headless: false
  output_dir: results

tasks:
  - name: example_direct
    mode: direct
    start_url: https://example.com
    browser:
      timeout: 180
      js_wait_time: 3
    content_extraction:
      title: h1
      content: .article-content
      links: a
```

#### 间接模式示例

```yaml
tasks:
  - name: example_indirect
    mode: indirect
    start_url: https://example.com
    browser:
      timeout: 180
      js_wait_time: 8
    button_discovery:
      selectors:
        - .article-card
        - .news-item
      max_buttons: 10
      deduplicate: true
      smart_discovery: true
    content_extraction:
      title: h1
      content: .article-body
      author: .author-name
      date: .publish-date
```

#### 多步模式示例

```yaml
tasks:
  - name: example_multistep
    mode: multistep
    start_url: https://example.com
    browser:
      timeout: 180
    operation_sequence:
      - step: 1
        action: click
        selector: .load-more-button
        description: "点击加载更多"
        wait_after: 2
      - step: 2
        action: extract
        description: "提取文章列表"
        save_id: articles_batch_1
      - step: 3
        action: wait
        wait_time: 3
```

### 配置项说明

#### 全局设置

- `browser_type`: 使用的浏览器（chromium, firefox, webkit）
- `headless`: 无头模式运行（true/false）
- `output_dir`: 结果保存目录
- `concurrent_limit`: 并发任务数
- `global_timeout`: 全局超时时间（秒）

#### 浏览器设置

- `timeout`: 页面加载超时时间（秒）
- `js_wait_time`: JavaScript 执行等待时间（秒）
- `max_retries`: 最大重试次数
- `fast_mode`: 启用快速加载模式
- `network_timeout`: 网络请求超时时间
- `wait_for_networkidle`: 等待网络空闲状态

#### 按钮发现

- `selectors`: 按钮的 CSS 选择器列表
- `max_buttons`: 最大处理按钮数量
- `deduplicate`: 启用元素去重
- `smart_discovery`: 启用智能选择器优化

#### 内容提取

定义字段名称及其对应的 CSS 选择器：

```yaml
content_extraction:
  title: h1, .title
  content: .article-body, article
  links: a[href]
  images: img[src]
```

多个选择器用逗号分隔（实现回退机制）。

## 输出格式

结果以 JSON 格式保存在配置的输出目录中：

```json
{
  "task_name": "example_task",
  "timestamp": "2025-10-16T10:30:00",
  "total_results": 10,
  "successful_results": 9,
  "results": [
    {
      "url": "https://example.com/article/1",
      "original_url": "https://example.com",
      "content": {
        "title": "文章标题",
        "content": "文章内容...",
        "author": "张三"
      },
      "timestamp": "2025-10-16T10:31:00",
      "new_tab": false,
      "button_info": {
        "text": "阅读更多",
        "selector": ".article-card",
        "href": "https://example.com/article/1"
      }
    }
  ]
}
```

## 高级特性

### 智能按钮发现

爬虫可以自动检测和去重相似元素：

- 基于位置的去重
- 文本内容比较
- URL 和属性匹配
- 选择器优化减少冗余

### 多种点击策略

系统自动为每个按钮选择最佳策略：

1. **直接导航**：最快，直接使用 href
2. **新标签页监听**：检测弹窗
3. **同页导航**：跟踪 URL 变化
4. **直接点击**：处理 AJAX，渐进式等待
5. **内部链接**：专门处理文章容器

### 页面加载检测

智能页面加载检测，多重检查：

- DOM 就绪状态
- 网络活动监控
- 内容稳定性验证
- 可配置超时和重试逻辑

## 日志

爬虫提供不同级别的详细日志：

- INFO: 一般进度和成功消息
- WARNING: 非关键问题和回退操作
- ERROR: 严重失败
- DEBUG: 详细调试信息

配置日志级别：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 故障排查

### 常见问题

**页面加载不完整**
- 增加 `js_wait_time` 配置值
- 启用 `wait_for_networkidle` 选项
- 调整 `timeout` 值

**按钮点击失败**
- 使用浏览器开发者工具验证 CSS 选择器
- 检查元素是否可见和可用
- 查看日志中的策略尝试信息

**出现重复结果**
- 在按钮发现中启用 `deduplicate`
- 检查日志中的 URL 模式
- 实现自定义去重逻辑

**性能慢**
- 启用 `fast_mode` 加快加载
- 减少 `js_wait_time` 值
- 使用无头模式
- 限制 `max_buttons` 数量

## 开发

### 项目结构

代码遵循清洁架构原则：

- **crawler.py**: 主要编排逻辑
- **config/**: 配置加载和验证
- **core/**: 核心爬取功能模块
- **task_config/**: 任务特定配置

### 扩展爬虫

添加新的点击策略：

1. 创建继承自 `ClickStrategyBase` 的类
2. 实现 `can_handle()` 和 `execute()` 方法
3. 添加到 `ClickStrategyManager.strategies` 列表

添加新的存储后端：

1. 创建实现 `Storage` 协议的类
2. 更新 `StorageFactory.create_storage()`

## 许可证

本项目用于教育和研究目的。

## 贡献

欢迎贡献代码！请确保代码遵循现有架构模式并包含适当的错误处理。
