# Smart Crawler 配置文件指南 - 性能优化版本

本指南详细说明如何为 Smart Crawler 创建和配置 YAML 配置文件，支持两种爬取模式和最新的性能优化功能。

## 📋 配置文件结构概览

```yaml
tasks:
  - name: "任务名称"
    mode: "direct|indirect"     # 爬取模式
    start_url: "起始页面URL"
    browser: {...}              # 浏览器设置（含性能优化）
    button_discovery: {...}     # 按钮发现配置（仅indirect模式需要）
    content_extraction: {...}   # 内容提取配置

settings: {...}               # 全局设置
```

---

## 🎯 爬取模式 (mode)

### 1. 直接爬取模式 (direct)

直接爬取指定URL页面的内容，不进行任何点击操作。

```yaml
tasks:
  - name: "direct_crawl_example"
    mode: "direct"  # 关键配置
    start_url: "https://example.com/product-page"

    browser:
      timeout: 120
      js_wait_time: 3             # 优化：减少等待时间
      fast_mode: true             # 新增：启用快速模式
      check_content_stability: false  # 新增：禁用内容检测
      max_retries: 3

    # 注意：直接模式下不需要button_discovery配置

    content_extraction:
      title: "h1, .page-title"
      content: ".main-content, article"
      products: ".product-card"
```

**适用场景：**
- 单页面内容爬取
- 产品列表页面
- 详情页面直接爬取
- 不需要点击交互的页面

### 2. 间接爬取模式 (indirect)

点击页面中的指定按钮/链接，在跳转后的页面爬取内容。

```yaml
tasks:
  - name: "indirect_crawl_example"
    mode: "indirect"  # 可省略，默认值
    start_url: "https://example.com"

    browser:
      timeout: 180
      js_wait_time: 3             # 优化：减少等待时间
      fast_js_wait_time: 1        # 新增：快速模式JS等待
      fast_mode: true             # 新增：启用快速等待模式
      wait_for_networkidle: false # 新增：不等待网络空闲
      check_content_stability: false # 新增：禁用内容稳定性检测
      network_timeout: 10000      # 新增：网络超时10秒
      max_retries: 3

    button_discovery:
      selectors:
        - "a[href*='/product/']"  # 精确的产品链接
        - ".product-link"         # 产品链接类
      max_buttons: 10
      deduplicate: true

    content_extraction:
      title: "h1, .product-title"
      description: ".product-description"
```

**适用场景：**
- 需要点击进入详情页的场景
- 多级导航的网站
- 动态加载的内容页面

---

## ⚡ 性能优化配置 (browser)

新增多项性能优化配置，特别针对新标签页等待过慢的问题：

```yaml
browser:
  # 基础配置
  timeout: 240                    # 页面加载超时
  js_wait_time: 3                 # JavaScript等待时间（优化后）
  max_retries: 3                  # 重试次数

  # 🚀 性能优化配置
  fast_mode: true                 # 启用快速等待模式
  fast_js_wait_time: 1            # 快速模式下JS等待时间
  wait_for_networkidle: false     # 不等待网络空闲，加快速度
  check_content_stability: false  # 禁用内容稳定性检测
  network_timeout: 10000          # 网络超时（毫秒）
  network_wait_attempts: 2        # 网络等待重试次数
  stability_check_interval: 1     # 内容稳定性检查间隔（秒）

  # 传统配置
  wait_for_navigation: true       # 等待页面导航完成
  handle_popups: true             # 处理弹窗
```

### 性能优化说明

| 配置项 | 默认值 | 优化值 | 说明 |
|--------|--------|--------|------|
| `fast_mode` | `false` | `true` | 启用快速等待，跳过复杂检测 |
| `js_wait_time` | `5` | `3` | JavaScript等待时间（秒） |
| `fast_js_wait_time` | - | `1` | 快速模式JS等待时间 |
| `wait_for_networkidle` | `true` | `false` | 不等待网络空闲 |
| `check_content_stability` | `true` | `false` | 禁用内容稳定性检测 |
| `network_timeout` | `30000` | `10000` | 网络超时时间（毫秒） |

**性能提升：** 等待时间从 **55秒+** 减少到 **11秒左右**，提升约 **80%**

---

## 🔍 按钮发现配置 (button_discovery)

**仅在 indirect 模式下需要配置**

```yaml
button_discovery:
  selectors:                     # CSS选择器列表
    - "h3 a[href*='Details']"    # 推荐：精确的选择器
    - ".m0 h3 a"                 # 简洁的备用选择器
    - "#channelContent h3 a"     # 特定区域选择器
  max_buttons: 15                # 最大按钮数量
  deduplicate: true              # 启用去重
  smart_discovery: true          # 智能选择器优化
```

**选择器优化建议：**
- ✅ 使用简洁精确的选择器，如 `h3 a[href*='Details']`
- ✅ 指定特定的类名或ID，如 `.article-title a`
- ❌ 避免过于复杂的nth-child选择器
- ❌ 避免会点击到导航菜单、广告等不相关元素

---

## 📄 内容提取配置 (content_extraction)

优化后的内容提取配置，支持更多字段和更好的兜底策略：

```yaml
content_extraction:
  # 基本内容
  title: "h1.rich_media_title, h1.title, .rich_media_title, h1, .article-title"
  content: "#js_content p, .islock p, .article-content, .post-content, article p"
  summary: ".summary, .excerpt, .description, .intro"

  # 元数据
  author: ".author, .writer, .byline, .post-author"
  publish_date: ".publish-date, .post-date, .date, time, .timestamp"
  tags: ".tags, .tag, .category, .post-tags"

  # 媒体内容
  images: "img[src], .article-image img, .content img, #js_content img"
  videos: "video, .video-player"

  # 链接信息
  links: "a[href], .content a, .article-content a"

  # 兜底方案 - 确保有内容被提取
  all_text: "p, div, span, h1, h2, h3, h4, h5, h6, article, main, section"
```

---

## ⚙️ 全局设置 (settings)

```yaml
settings:
  browser_type: "chromium"         # 浏览器类型
  headless: false                  # 是否隐藏浏览器窗口
  output_dir: "results"            # 结果保存目录
  storage_type: "json"             # 存储格式
  global_timeout: 300              # 全局超时时间(秒)，优化后

  # 多跳转处理配置（优化）
  multi_hop:
    enable: true
    max_hops: 3                    # 减少跳转次数
    wait_between_hops: 2           # 减少跳转间等待时间

  # 新标签页处理
  tab_handling:
    auto_switch: true              # 自动切换到新标签页
    close_previous: false          # 保留之前的标签页用于返回
    wait_for_load: true            # 等待新页面完全加载
```

---

## 📝 完整配置示例

### 示例1：高性能直接爬取

```yaml
# 高性能直接爬取模式
tasks:
  - name: "fast_direct_crawl"
    mode: "direct"
    start_url: "https://example.com/articles"

    browser:
      timeout: 120
      js_wait_time: 2              # 快速模式
      fast_mode: true              # 启用快速等待
      check_content_stability: false
      wait_for_networkidle: false
      max_retries: 2

    content_extraction:
      title: "h1, .article-title, .page-title"
      content: ".article-content, .post-content, article, main"
      author: ".author, .writer"
      publish_date: ".date, time, .timestamp"
      images: "img[src], .content img"
      all_text: "p, div, span, h1, h2, h3, h4, h5, h6"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "fast_results"
  global_timeout: 180
```

### 示例2：优化的间接爬取（新标签页友好）

```yaml
# 针对新标签页优化的间接爬取
tasks:
  - name: "optimized_indirect_crawl"
    mode: "indirect"
    start_url: "https://news-site.com"

    browser:
      timeout: 240
      js_wait_time: 3              # 标准等待
      fast_js_wait_time: 1         # 新标签页快速等待
      fast_mode: true              # 启用快速模式
      wait_for_networkidle: false  # 不等待网络空闲
      check_content_stability: false # 禁用内容检测
      network_timeout: 10000       # 10秒网络超时
      network_wait_attempts: 2     # 减少重试次数
      max_retries: 3

    button_discovery:
      selectors:
        - "h3 a[href*='article']"  # 简洁精确的选择器
        - ".news-item a"           # 新闻项链接
        - ".article-title a"       # 文章标题链接
      max_buttons: 15
      deduplicate: true
      smart_discovery: true

    content_extraction:
      title: "h1, .article-title, .post-title, .news-title"
      content: ".article-content, .post-content, .news-content, article p"
      summary: ".summary, .excerpt, .description"
      author: ".author, .writer, .byline"
      publish_date: ".date, time, .publish-time"
      tags: ".tags, .category, .keywords"
      images: "img[src], .article-image img, .content img"
      links: "a[href], .content a"
      all_text: "p, div, span, h1, h2, h3, h4, h5, h6"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "optimized_results"
  global_timeout: 300

  multi_hop:
    enable: true
    max_hops: 3
    wait_between_hops: 2

  tab_handling:
    auto_switch: true
    close_previous: false
    wait_for_load: true
```

---

## 🚀 性能优化指南

### 新标签页等待慢的解决方案

如果遇到"新标签页已经完全加载但仍然很慢才爬取"的问题：

1. **启用快速模式**
   ```yaml
   browser:
     fast_mode: true
     fast_js_wait_time: 1
   ```

2. **禁用不必要的等待**
   ```yaml
   browser:
     wait_for_networkidle: false
     check_content_stability: false
   ```

3. **减少超时时间**
   ```yaml
   browser:
     network_timeout: 10000
     js_wait_time: 3
   ```

### 模式选择建议

| 场景 | 推荐模式 | 优化重点 |
|------|----------|----------|
| 单页面内容 | `direct` | 启用 `fast_mode` |
| 新标签页跳转 | `indirect` | 禁用 `check_content_stability` |
| 静态页面 | `direct` | 减少 `js_wait_time` |
| 动态页面 | `indirect` | 保持适当的 `js_wait_time` |

---

## 🔧 调试和优化技巧

### 1. 性能调试
```javascript
// 在浏览器控制台中测试页面加载状态
document.readyState  // "complete" 表示已加载
```

### 2. 选择器测试
```javascript
// 测试选择器是否有效
document.querySelectorAll('h3 a[href*="Details"]').length
```

### 3. 常见性能问题

**新标签页等待慢：**
- ✅ 启用 `fast_mode: true`
- ✅ 设置 `wait_for_networkidle: false`
- ✅ 禁用 `check_content_stability: false`

**内容提取为空：**
- ✅ 检查选择器是否正确
- ✅ 增加 `js_wait_time` 如果页面有动态加载
- ✅ 使用 `all_text` 作为兜底方案

**点击了错误的元素：**
- ✅ 使用更精确的选择器
- ✅ 启用 `smart_discovery: true`
- ✅ 测试选择器的匹配结果

---

## 🎯 快速开始模板

```yaml
# 高性能爬虫模板 - 复制并修改
tasks:
  - name: "my_optimized_task"
    mode: "indirect"  # 或 "direct"
    start_url: "https://your-website.com"

    browser:
      timeout: 180
      js_wait_time: 3
      fast_mode: true              # 🚀 性能优化
      fast_js_wait_time: 1
      wait_for_networkidle: false  # 🚀 不等待网络空闲
      check_content_stability: false # 🚀 禁用内容检测
      max_retries: 3

    # 间接模式需要配置按钮发现
    button_discovery:
      selectors:
        - "YOUR_PRECISE_SELECTOR_HERE"  # 如: "h3 a[href*='detail']"
      max_buttons: 10
      deduplicate: true
      smart_discovery: true

    content_extraction:
      title: "h1, .title, .article-title"
      content: ".content, article, .post-content"
      author: ".author, .writer"
      publish_date: ".date, time"
      images: "img[src], .content img"
      all_text: "p, div, span, h1, h2, h3, h4, h5, h6"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "results"
  global_timeout: 300
```

---

使用这些优化配置，您的爬虫性能将显著提升，特别是在处理新标签页跳转时！🚀