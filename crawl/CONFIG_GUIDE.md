# Smart Crawler 配置文件指南 - 更新版本

本指南详细说明如何为 Smart Crawler 创建和配置 YAML 配置文件，支持两种爬取模式。

## 📋 配置文件结构概览

```yaml
tasks:
  - name: "任务名称"
    mode: "direct|indirect"     # 新增：爬取模式
    start_url: "起始页面URL"
    browser: {...}              # 浏览器设置
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
      js_wait_time: 8
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
      js_wait_time: 10
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

## 🔍 按钮发现配置 (button_discovery)

**仅在 indirect 模式下需要配置**

```yaml
button_discovery:
  selectors:                     # CSS选择器列表
    - "a[href*='/product/']"     # 推荐：精确的选择器
    - ".product-card a"          # 产品卡片中的链接
    - "button[data-action='view']" # 特定功能按钮
  max_buttons: 15                # 最大按钮数量
  deduplicate: true              # 启用去重
  smart_discovery: true          # 智能选择器优化
```

**重要建议：**
- ✅ 使用精确的选择器，如 `a[href*='/product/']`
- ✅ 指定特定的类名，如 `.product-link`
- ❌ 避免过于宽泛的选择器，如 `a` 或 `button`
- ❌ 避免会点击到导航菜单、广告等不相关元素的选择器

---

## 📄 内容提取配置 (content_extraction)

两种模式都需要配置，用法相同：

```yaml
content_extraction:
  # 基本内容
  title: "h1, h2, .title"
  description: ".description, .summary"
  content: ".content, article, main"

  # 产品相关（适用于产品页面）
  price: ".price, .cost"
  specifications: ".specs, .technical-data"
  features: ".features, .benefits"

  # 媒体内容
  images: "img[src], .gallery img"
  videos: "video, .video-player"

  # 元数据
  author: ".author, .by"
  publish_date: ".date, time"
  tags: ".tags, .categories"

  # 兜底方案
  all_text: "p, div, span, h1, h2, h3, h4, h5, h6"
  links: "a[href]"
```

---

## ⚙️ 全局设置 (settings)

```yaml
settings:
  browser_type: "chromium"         # 浏览器类型
  headless: false                  # 是否隐藏浏览器窗口
  output_dir: "results"            # 结果保存目录
  storage_type: "json"             # 存储格式
  global_timeout: 300              # 全局超时时间(秒)
```

---

## 📝 完整配置示例

### 示例1：直接爬取产品列表页

```yaml
# 直接爬取产品列表页的所有产品信息
tasks:
  - name: "product_list_direct"
    mode: "direct"
    start_url: "https://shop.example.com/products"

    browser:
      timeout: 120
      js_wait_time: 8
      max_retries: 2

    content_extraction:
      page_title: "h1, .page-title"
      products: ".product-card, .product-item"
      product_names: ".product-title, .product-name"
      product_prices: ".price, .cost"
      product_links: "a[href*='/product/']"
      product_images: ".product-image img"
      all_text: "p, div, span, h1, h2, h3, h4, h5, h6"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "direct_results"
```

### 示例2：间接爬取产品详情页

```yaml
# 点击产品链接，爬取每个产品的详细信息
tasks:
  - name: "product_details_indirect"
    mode: "indirect"
    start_url: "https://shop.example.com/products"

    browser:
      timeout: 180
      js_wait_time: 10
      max_retries: 3

    button_discovery:
      selectors:
        - "a[href*='/product/']"     # 产品详情链接
        - ".product-card .view-btn"  # 查看按钮
      max_buttons: 20
      deduplicate: true
      smart_discovery: true

    content_extraction:
      title: "h1, .product-title"
      description: ".product-description, .product-summary"
      price: ".price, .current-price"
      specifications: ".specs, .product-specs"
      features: ".features, .highlights"
      images: ".product-gallery img, .product-image img"
      reviews: ".review, .customer-review"
      availability: ".stock, .availability"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "indirect_results"

  multi_hop:
    enable: true
    max_hops: 3
    wait_between_hops: 2
```

---

## 🎮 如何选择爬取模式

### 使用直接模式 (direct) 的情况：

- ✅ 目标内容都在同一个页面上
- ✅ 不需要点击操作
- ✅ 页面是静态的或已完全加载
- ✅ 爬取列表页、目录页、单个详情页

### 使用间接模式 (indirect) 的情况：

- ✅ 需要点击链接/按钮才能看到目标内容
- ✅ 内容分布在多个页面上
- ✅ 需要进入详情页获取完整信息
- ✅ 页面有动态加载或交互功能

---

## 🔧 调试和优化技巧

### 1. 调试选择器
```javascript
// 在浏览器控制台中测试选择器
document.querySelectorAll('your-selector').length
```

### 2. 模式选择建议
- 先尝试直接模式，看能否获取到需要的内容
- 如果内容需要点击才能看到，再使用间接模式

### 3. 常见问题
**直接模式问题：**
- 内容为空 → 检查选择器，确认内容已加载
- 页面加载慢 → 增加 `js_wait_time`

**间接模式问题：**
- 点击了不相关按钮 → 精确化选择器
- 重复爬取 → 启用 `deduplicate: true`

---

## 🚀 快速开始模板

```yaml
# 复制此模板并根据需求修改
tasks:
  - name: "my_crawl_task"
    mode: "direct"  # 或 "indirect"
    start_url: "https://your-website.com"

    browser:
      timeout: 120
      js_wait_time: 8
      max_retries: 3

    # 如果是间接模式，添加以下配置：
    # button_discovery:
    #   selectors:
    #     - "YOUR_BUTTON_SELECTOR_HERE"
    #   max_buttons: 10
    #   deduplicate: true

    content_extraction:
      title: "h1, .title"
      content: ".content, article, main"
      all_text: "p, div, span, h1, h2, h3, h4, h5, h6"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "results"
```

---

使用新的模式系统，您可以更精确地控制爬虫的行为，提高爬取效率和准确性！🎉