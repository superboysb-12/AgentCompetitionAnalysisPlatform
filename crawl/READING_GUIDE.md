# 爬虫模块代码阅读指南

## 📚 阅读路径（按顺序）

### 🎯 第一阶段：理解整体流程（30分钟）

#### 1. **run.py** - 入口文件（26行）
**阅读重点**：
- 如何启动爬虫：`SmartCrawler(config_file)`
- 异步执行：`asyncio.run(main())`

**关键代码**：
```python
crawler = SmartCrawler(config_file)
await crawler.start()
```

---

#### 2. **crawler.py** - 主调度引擎（核心，~400行）
**阅读重点**：
- `SmartCrawler` 类的初始化流程
- `start()` 方法：总调度器
- 三种爬取模式的分发逻辑：
  - `_execute_direct_mode()` - 直接爬取
  - `_execute_indirect_mode()` - 间接爬取（点击按钮）
  - `_execute_multistep_mode()` - 多步骤爬取

**关键方法阅读顺序**：
```python
__init__()              # 1. 初始化：加载配置、创建管理器
  ↓
start()                 # 2. 启动：遍历任务
  ↓
_execute_indirect_mode() # 3. 最常用的间接爬取模式
  ↓
_click_and_extract()    # 4. 核心：点击按钮 + 提取内容
  ↓
_save_results()         # 5. 保存结果
```

**重点关注**：
- L67-L95: `start()` 方法 - 任务循环
- L130-L200: `_execute_indirect_mode()` - 间接爬取逻辑
- L250-L320: `_click_and_extract()` - 点击策略调用

---

#### 3. **core/types.py** - 数据结构定义（~100行）
**阅读重点**：
- `CrawlResult` - 爬取结果数据类
- `ClickStrategy` - 点击策略枚举
- `Protocol` 接口定义：
  - `BrowserManager` - 浏览器管理接口
  - `ContentExtractor` - 内容提取接口
  - `Storage` - 存储接口

**为什么先读这个**：
- 理解核心数据结构
- 熟悉接口定义（Protocol-based Design）
- 为后续阅读做铺垫

**关键代码**：
```python
@dataclass
class CrawlResult:
    url: str
    original_url: str
    content: dict
    timestamp: str
    new_tab: bool
    strategy_used: Optional[ClickStrategy]
    button_info: Optional[dict]
```

---

### 🔧 第二阶段：深入核心组件（1-2小时）

#### 4. **core/browser.py** - 浏览器管理（~200行）
**阅读重点**：
- `BrowserManager` 类
- Playwright 浏览器初始化：`initialize()`
- 页面创建与关闭：`create_page()`, `close_page()`
- 上下文管理：`__aenter__`, `__aexit__`

**重要方法**：
```python
async def initialize():           # 启动浏览器
async def create_page():          # 创建新标签页
async def close_page():           # 关闭标签页
async def close():                # 关闭浏览器
```

**设计模式**：Context Manager (上下文管理器)

---

#### 5. **core/strategies.py** - 点击策略（核心，~400行）
**阅读重点**：
⭐ **这是爬虫的核心逻辑！** ⭐

**5种点击策略**：
1. **DirectNavigationStrategy** (L50-L80)
   - 直接使用 `href` 跳转
   - 最快，但不触发JavaScript

2. **NewTabListenerStrategy** (L85-L150)
   - 监听 `window.open()` 打开的新标签
   - 适用于新窗口打开的链接

3. **SamePageNavigationStrategy** (L155-L210)
   - 跟踪 URL 变化
   - 适用于单页应用 (SPA)

4. **DirectClickStrategy** (L215-L280)
   - 直接点击，渐进式等待
   - 适用于 AJAX 加载内容

5. **InternalLinkStrategy** (L285-L340)
   - 特殊处理文章容器内的链接
   - 自动发现并点击内部链接

**阅读顺序**：
```
1. 先看 ClickStrategy 基类 (L20-L45)
2. 按难度从简单到复杂：
   - DirectNavigationStrategy (最简单)
   - DirectClickStrategy (理解等待逻辑)
   - NewTabListenerStrategy (理解标签监听)
   - SamePageNavigationStrategy (理解URL跟踪)
   - InternalLinkStrategy (最复杂)
3. 最后看策略选择逻辑 (crawler.py中)
```

**关键代码模式**：
```python
class ClickStrategy(Protocol):
    async def execute(
        self,
        page: Page,
        element: ElementHandle,
        config: dict
    ) -> tuple[bool, Optional[Page], ClickStrategy]:
        """返回: (成功?, 新页面?, 使用的策略)"""
```

---

#### 6. **core/detector.py** - 页面加载检测（~180行）
**阅读重点**：
- `PageLoadDetector` 类
- 三种检测机制：
  1. **DOM Ready** - 页面结构加载完成
  2. **Network Idle** - 网络请求静默
  3. **Content Stability** - 内容稳定（不再变化）

**重要方法**：
```python
async def wait_for_page_load():        # 综合等待
async def _wait_for_dom_ready():       # DOM检测
async def _wait_for_network_idle():    # 网络空闲检测
async def _check_content_stability():  # 内容稳定性检测
```

**为什么重要**：
- 解决动态加载问题
- 防止过早提取导致内容不完整

---

#### 7. **core/extractor.py** - 内容提取（~150行）
**阅读重点**：
- `ContentExtractor` 类
- CSS 选择器提取：`extract_content()`
- 智能回退机制：选择器失败时的处理
- `all_text` 提取：全页面文本提取

**关键方法**：
```python
async def extract_content(page, selectors):
    """根据配置的选择器提取内容"""
    # 1. 尝试配置的选择器
    # 2. 失败时回退到 all_text
    # 3. 清理和格式化文本
```

---

#### 8. **core/discovery.py** - 按钮发现（~200行）
**阅读重点**：
- `ButtonDiscovery` 类
- 智能按钮发现：`discover_buttons()`
- 去重逻辑：`_deduplicate_buttons()`
- 选择器优化：避免重复发现相同元素

**关键方法**：
```python
async def discover_buttons(page, config):
    """发现页面上的可点击按钮/链接"""
    # 1. 根据选择器查找元素
    # 2. 提取按钮信息 (text, href, selector)
    # 3. 去重
    # 4. 限制数量 (max_buttons)
```

---

#### 9. **core/storage.py** - 存储管理（~100行）
**阅读重点**：
- `JSONStorage` 类
- 结果保存：`save()`
- 文件命名：时间戳 + 任务名
- 数据清洗：转换为可序列化格式

**设计模式**：Factory Pattern (工厂模式)

---

### ⚙️ 第三阶段：配置管理（30分钟）

#### 10. **config/manager.py** - 配置管理器（~150行）
**阅读重点**：
- `ConfigManager` 类
- YAML 配置加载与验证
- 配置格式兼容性处理
- 默认值设置

**配置结构**：
```python
{
    'settings': {...},           # 全局设置
    'tasks': [                   # 任务列表
        {
            'name': '...',
            'mode': 'indirect',
            'start_url': '...',
            'browser': {...},
            'button_discovery': {...},
            'content_extraction': {...}
        }
    ]
}
```

---

## 🎓 实战示例阅读

### 示例1：间接爬取模式完整流程

**阅读路径**：
```
1. crawler.py: start()
   → 遍历任务

2. crawler.py: _execute_indirect_mode()
   → 打开起始页

3. core/discovery.py: discover_buttons()
   → 发现所有按钮

4. crawler.py: _click_and_extract()
   循环：
   ├─ core/strategies.py: 尝试5种策略
   ├─ core/detector.py: 等待页面加载
   ├─ core/extractor.py: 提取内容
   └─ 保存结果

5. core/storage.py: save()
   → 保存到JSON
```

### 示例2：点击策略选择逻辑

**查看位置**：`crawler.py` 中的 `_click_and_extract()` 方法

```python
# 策略尝试顺序（硬编码）
strategies = [
    DirectNavigationStrategy(),      # 1. 最快
    NewTabListenerStrategy(),        # 2. 处理弹窗
    SamePageNavigationStrategy(),    # 3. 单页应用
    DirectClickStrategy(),           # 4. AJAX
    InternalLinkStrategy()           # 5. 兜底
]

for strategy in strategies:
    success, new_page, used_strategy = await strategy.execute(...)
    if success:
        break  # 成功后停止尝试
```

---

## 📝 关键概念理解

### 1. 异步编程模式
**涉及文件**：所有 `.py` 文件

**关键点**：
- `async def` - 异步函数定义
- `await` - 等待异步操作完成
- `asyncio.run()` - 运行异步主函数

### 2. Strategy Pattern（策略模式）
**涉及文件**：`core/strategies.py`

**核心思想**：
- 定义接口：`ClickStrategy` Protocol
- 实现多个策略类
- 运行时动态选择策略

### 3. Protocol-based Design
**涉及文件**：`core/types.py`

**核心思想**：
- 使用 `typing.Protocol` 定义接口
- 不强制继承，鸭子类型
- 类型检查友好

### 4. Context Manager（上下文管理器）
**涉及文件**：`core/browser.py`

**使用方式**：
```python
async with browser_manager:
    # 自动初始化浏览器
    page = await browser_manager.create_page()
    # ...
# 自动关闭浏览器
```

---

## 🗂️ 文件清单速查

### 按重要性排序

| 优先级 | 文件 | 行数 | 难度 | 说明 |
|--------|------|------|------|------|
| ⭐⭐⭐ | `crawler.py` | ~400 | 中 | **主调度引擎，必读** |
| ⭐⭐⭐ | `core/strategies.py` | ~400 | 高 | **核心逻辑，重点** |
| ⭐⭐⭐ | `core/types.py` | ~100 | 低 | **数据结构，先读** |
| ⭐⭐ | `core/browser.py` | ~200 | 中 | 浏览器管理 |
| ⭐⭐ | `core/detector.py` | ~180 | 中 | 页面加载检测 |
| ⭐⭐ | `core/discovery.py` | ~200 | 中 | 按钮发现 |
| ⭐⭐ | `core/extractor.py` | ~150 | 低 | 内容提取 |
| ⭐ | `core/storage.py` | ~100 | 低 | 存储管理 |
| ⭐ | `config/manager.py` | ~150 | 低 | 配置管理 |
| ⭐ | `run.py` | 26 | 低 | 入口文件 |

### 按功能分类

**核心调度**：
- `run.py` - 入口
- `crawler.py` - 调度器

**爬取核心**：
- `core/strategies.py` - 点击策略 ⭐
- `core/detector.py` - 页面检测
- `core/discovery.py` - 按钮发现
- `core/extractor.py` - 内容提取

**基础设施**：
- `core/browser.py` - 浏览器管理
- `core/storage.py` - 存储
- `core/types.py` - 数据结构
- `config/manager.py` - 配置管理

---

## 🎯 不同场景的阅读建议

### 场景1：快速了解（30分钟）
**只读这3个**：
1. `run.py` - 如何启动
2. `core/types.py` - 数据结构
3. `crawler.py` 的 `start()` 和 `_execute_indirect_mode()` 方法

### 场景2：理解核心逻辑（2小时）
**阅读路径**：
1. `core/types.py` - 数据结构
2. `crawler.py` - 主流程
3. `core/strategies.py` - 点击策略（重点）
4. `core/detector.py` - 页面检测
5. `core/extractor.py` - 内容提取

### 场景3：修改或扩展功能（深入学习）
**全部阅读**，按本文档顺序

---

## 📌 阅读技巧

### 1. 使用IDE的导航功能
- **跳转到定义**: Ctrl+Click (VS Code)
- **查找引用**: Shift+F12
- **查看类层次**: Ctrl+H

### 2. 先看接口，再看实现
```python
# 先看 Protocol 定义
class ClickStrategy(Protocol):
    async def execute(...) -> ...:
        ...

# 再看具体实现
class DirectNavigationStrategy:
    async def execute(...) -> ...:
        # 实现细节
```

### 3. 追踪调用链
从 `crawler.py` 的 `start()` 开始，跟踪每个方法调用：
```
start()
  → _execute_indirect_mode()
    → _click_and_extract()
      → strategy.execute()
        → detector.wait_for_page_load()
          → extractor.extract_content()
```

### 4. 对照配置文件理解
打开一个配置文件（如 `task_config/chinaiol_indirect.yaml`），对照代码理解：
- 配置如何加载：`config/manager.py`
- 配置如何使用：`crawler.py`

---

## 🔍 调试建议

### 启用详细日志
```python
# 修改 crawler.py 的日志级别
logging.basicConfig(level=logging.DEBUG)
```

### 关闭无头模式
```yaml
# task_config/*.yaml
settings:
  headless: false  # 可以看到浏览器操作
```

### 减少爬取数量
```yaml
# task_config/*.yaml
button_discovery:
  max_buttons: 3  # 只爬3个链接，快速测试
```

---

## 📚 推荐学习资源

### Playwright 官方文档
- [Python Async API](https://playwright.dev/python/docs/api/class-playwright)
- [Page Object Model](https://playwright.dev/python/docs/pom)

### Python 异步编程
- [asyncio 官方文档](https://docs.python.org/3/library/asyncio.html)
- [Real Python: Async IO](https://realpython.com/async-io-python/)

### 设计模式
- Strategy Pattern
- Factory Pattern
- Protocol-based Design (PEP 544)

---

**最后建议**：
1. ⭐ 先快速浏览一遍所有文件，建立整体印象
2. ⭐ 重点精读 `crawler.py` 和 `core/strategies.py`
3. ⭐ 实际运行一次，对照日志理解流程
4. ⭐ 修改配置文件，测试不同场景

祝阅读愉快！🚀
