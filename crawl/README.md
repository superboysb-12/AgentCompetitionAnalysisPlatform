# Smart Crawler

基于清洁架构的Python Web爬虫，专为处理JavaScript动态页面和新标签页跳转设计。

## ✨ 核心特性

- **🔗 智能标签页处理**: 自动检测并切换到新标签页
- **🎯 多策略点击**: 5种点击策略自适应不同网站
- **🧠 智能等待**: 动态检测页面加载完成状态
- **📦 模块化架构**: 基于SOLID原则的清洁架构
- **⚡ 异步高效**: 基于Playwright的异步处理

## 🚀 快速开始

### 安装依赖
```bash
pip install playwright pyyaml
playwright install
```

### 基本使用
```bash
python run.py config.yaml
```

## 📁 项目结构

```
crawler/
├── core/                 # 核心业务逻辑
│   ├── browser.py       # 浏览器管理
│   ├── discovery.py     # 按钮发现
│   ├── strategies.py    # 点击策略
│   ├── extractor.py     # 内容提取
│   ├── detector.py      # 页面检测
│   └── storage.py       # 数据存储
├── config/              # 配置管理
│   └── manager.py       # 配置处理
├── crawler.py           # 主爬虫引擎
├── config.yaml          # 配置文件
├── run.py              # 运行脚本
└── requirements.txt     # 依赖列表
```

## ⚙️ 配置示例

```yaml
tasks:
  - name: "example"
    start_url: "https://example.com"

    browser:
      timeout: 180
      js_wait_time: 8
      max_retries: 3

    button_discovery:
      selectors:
        - "div.article-card-container"
        - "a[href*='article']"
      max_buttons: 5

    content_extraction:
      title: "h1, .title"
      content: ".content, article"
      author: ".author"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "results"
```

## 🎯 适用场景

- 新闻网站文章爬取
- 电商产品信息提取
- 论坛帖子内容获取
- 任何需要点击跳转的网站

## 📊 输出格式

结果保存为JSON格式：

```json
{
  "task_name": "example",
  "total_results": 5,
  "results": [
    {
      "url": "https://example.com/article/123",
      "original_url": "https://example.com",
      "new_tab": true,
      "content": {
        "title": ["文章标题"],
        "content": ["文章内容..."]
      }
    }
  ]
}
```

## 🔧 扩展开发

### 添加新的点击策略
```python
from core.strategies import ClickStrategyBase

class CustomStrategy(ClickStrategyBase):
    def can_handle(self, button: ButtonInfo) -> bool:
        return True  # 判断逻辑

    async def execute(self, page, button, config):
        # 实现点击逻辑
        pass
```

### 自定义存储方式
```python
from core.storage import Storage

class CustomStorage:
    async def save(self, task_name: str, results: List[CrawlResult]):
        # 实现存储逻辑
        pass
```

## 📋 依赖

- Python 3.8+
- Playwright
- PyYAML

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License