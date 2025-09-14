# Smart Crawler 配置文件指南

本指南详细说明如何为 Smart Crawler 创建和配置 YAML 配置文件，让您能够轻松爬取任何网站。

## 📋 配置文件结构概览

```yaml
tasks:
  - name: "任务名称"
    start_url: "起始页面URL"
    browser: {...}           # 浏览器设置
    button_discovery: {...}  # 按钮发现配置
    content_extraction: {...} # 内容提取配置

settings: {...}              # 全局设置
```

---

## 🎯 任务配置 (tasks)

### 基本信息

```yaml
tasks:
  - name: "my_website_crawler"      # 任务名称，用于结果文件命名
    start_url: "https://example.com" # 爬虫起始页面
```

**如何获取：**
- `name`: 自定义任务名称，建议使用网站名称或内容类型
- `start_url`: 要爬取的网站首页或列表页URL

---

## 🌐 浏览器配置 (browser)

```yaml
browser:
  timeout: 240                    # 页面加载超时(秒)
  js_wait_time: 15               # JavaScript执行等待时间(秒)
  max_retries: 5                 # 失败重试次数
  wait_for_navigation: true      # 是否等待页面导航完成
  handle_popups: true            # 是否处理弹窗
```

**参数说明：**
- `timeout`: 网站加载慢时增加此值（推荐180-300秒）
- `js_wait_time`: 有大量JavaScript的网站需要更长等待时间
- `max_retries`: 网络不稳定时增加重试次数
- `wait_for_navigation`: 页面跳转较多时建议启用
- `handle_popups`: 有弹窗广告的网站建议启用

---

## 🔍 按钮发现配置 (button_discovery)

这是最重要的配置部分，决定了爬虫要点击哪些元素。

```yaml
button_discovery:
  selectors:                     # CSS选择器列表
    - "div.article-card"         # 主要选择器
    - ".news-item"               # 备用选择器
    - "a[href*='article']"       # 链接选择器
  max_buttons: 20                # 最大按钮数量
  deduplicate: true              # 启用去重
  smart_discovery: true          # 智能选择器优化
```

### 🎯 如何获取CSS选择器

这是配置的核心！以下是获取选择器的详细步骤：

#### 方法1：Chrome开发者工具 (推荐)

1. **打开目标网站**
2. **按F12打开开发者工具**
3. **点击检查工具** (左上角箭头图标)
4. **点击要爬取的元素** (如文章卡片)
5. **右键选中的HTML代码**
6. **选择 "Copy" → "Copy selector"**

#### 方法2：手动分析HTML结构

查看页面源码，寻找包含目标内容的HTML标签：

```html
<!-- 示例：新闻卡片 -->
<div class="article-card-container">
  <h3 class="article-title">文章标题</h3>
  <a href="/article/123">阅读更多</a>
</div>
```

**可用选择器：**
- `div.article-card-container` - 选择卡片容器
- `.article-title` - 选择标题
- `a[href^='/article']` - 选择文章链接

#### 常见选择器模式

| 网站类型 | 常用选择器示例 |
|---------|---------------|
| 新闻网站 | `.article-item`, `.news-card`, `article` |
| 博客 | `.post`, `.entry`, `.blog-item` |
| 电商 | `.product-item`, `.goods-card` |
| 论坛 | `.topic-item`, `.thread` |
| 视频 | `.video-item`, `.movie-card` |

#### 选择器优先级建议

```yaml
selectors:
  - "div.specific-class"           # 1. 最具体的选择器
  - ".generic-class"               # 2. 通用类选择器
  - "tag.class"                    # 3. 标签+类选择器
  - "a[href*='keyword']"           # 4. 属性选择器
```

---

## 📄 内容提取配置 (content_extraction)

定义从目标页面提取哪些内容：

```yaml
content_extraction:
  # 文章标题
  title: "h1, h2, .article-title, .post-title, .title"

  # 正文内容
  content: ".article-content, .post-content, article, .content, main"

  # 文章摘要
  summary: ".summary, .description, .intro"

  # 作者信息
  author: ".author, .writer, .by-line"

  # 发布时间
  publish_date: ".date, .publish-date, time"

  # 标签分类
  tags: ".tags, .tag, .category"

  # 图片
  images: "img[src], .content img"

  # 链接
  links: "a[href]"

  # 兜底方案：所有文本
  all_text: "p, div, span, h1, h2, h3, h4, h5, h6"
```

### 🎯 如何获取内容选择器

#### 1. 打开一个目标文章页面
#### 2. 使用开发者工具检查每个内容区域
#### 3. 复制对应的CSS选择器

**选择器优先级说明：**
- 多个选择器用逗号分隔，按优先级排列
- 爬虫会依次尝试，使用第一个找到的元素
- 建议从具体到通用排列

---

## ⚙️ 全局设置 (settings)

```yaml
settings:
  browser_type: "chromium"         # 浏览器类型
  headless: false                  # 是否隐藏浏览器窗口
  output_dir: "results"            # 结果保存目录
  storage_type: "json"             # 存储格式
  global_timeout: 400              # 全局超时时间(秒)

  # 多跳转处理配置
  multi_hop:
    enable: true                   # 启用多跳转处理
    max_hops: 5                    # 最多跳转次数
    wait_between_hops: 3           # 跳转间等待时间(秒)

  # 新标签页处理
  tab_handling:
    auto_switch: true              # 自动切换到新标签页
    close_previous: false          # 是否关闭之前的标签页
    wait_for_load: true            # 等待新页面加载
```

---

## 📝 完整配置示例

### 示例1：新闻网站爬取

```yaml
# 爬取新闻网站文章
tasks:
  - name: "news_articles"
    start_url: "https://news-website.com"

    browser:
      timeout: 180
      js_wait_time: 10
      max_retries: 3

    button_discovery:
      selectors:
        - ".article-card"
        - ".news-item a"
      max_buttons: 15
      deduplicate: true

    content_extraction:
      title: "h1, .article-title"
      content: ".article-body, .content"
      author: ".author, .byline"
      publish_date: ".date, time"
      summary: ".summary, .lead"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "news_results"
```

### 示例2：博客爬取

```yaml
# 爬取个人博客
tasks:
  - name: "blog_posts"
    start_url: "https://myblog.com"

    browser:
      timeout: 120
      js_wait_time: 5
      max_retries: 2

    button_discovery:
      selectors:
        - ".post-title a"
        - "article h2 a"
      max_buttons: 10

    content_extraction:
      title: "h1, .post-title"
      content: ".post-content, .entry-content"
      tags: ".post-tags, .categories"
      publish_date: ".post-date"

settings:
  browser_type: "chromium"
  headless: true
  output_dir: "blog_results"
```

---

## 🔧 调试和优化技巧

### 1. 测试选择器
在浏览器控制台中测试选择器：
```javascript
// 测试选择器是否有效
document.querySelectorAll('div.article-card').length
```

### 2. 查看爬取日志
运行爬虫时观察日志输出：
```bash
python run.py your_config.yaml
```

### 3. 常见问题排查

**问题1：找不到按钮**
- 检查选择器是否正确
- 确认页面是否完全加载
- 增加 `js_wait_time`

**问题2：内容为空**
- 检查内容选择器
- 确认目标页面结构
- 查看 `all_text` 字段作为备用

**问题3：重复爬取**
- 启用 `deduplicate: true`
- 启用 `smart_discovery: true`
- 检查选择器是否重叠

---

## 🚀 快速开始模板

复制以下模板并根据您的目标网站修改：

```yaml
# 将此模板保存为 my_config.yaml
tasks:
  - name: "my_website"
    start_url: "https://your-website.com"

    browser:
      timeout: 180
      js_wait_time: 10
      max_retries: 3

    button_discovery:
      selectors:
        - "YOUR_BUTTON_SELECTOR_HERE"  # 替换为实际选择器
      max_buttons: 10
      deduplicate: true
      smart_discovery: true

    content_extraction:
      title: "h1, .title"
      content: ".content, article, main"
      all_text: "p, div, span"

settings:
  browser_type: "chromium"
  headless: false
  output_dir: "results"
  storage_type: "json"
  global_timeout: 300
```

---

## 💡 进阶技巧

### 1. 复杂选择器
```yaml
selectors:
  - "div[class*='article']:not(.advertisement)"  # 包含article但不是广告
  - "a[href^='/post/']:visible"                  # 以/post/开头的可见链接
  - ".container > .item:nth-child(odd)"          # 奇数位置的项目
```

### 2. 多任务配置
```yaml
tasks:
  - name: "task1"
    start_url: "https://site1.com"
    # ... 配置1

  - name: "task2"
    start_url: "https://site2.com"
    # ... 配置2
```

### 3. 条件配置
```yaml
button_discovery:
  selectors:
    - "div.article-card"           # 主选择器
    - ".fallback-selector"         # 备用选择器
  max_buttons: 50                  # 大型网站
  smart_discovery: true            # 自动优化
```

---

## 📞 需要帮助？

如果您在配置过程中遇到问题：

1. **查看日志输出** - 运行时的详细信息
2. **测试选择器** - 在浏览器控制台验证
3. **从简单开始** - 先用基本配置测试
4. **参考示例** - 查看 `avc_mr.yaml` 等示例文件

祝您爬取顺利！🎉