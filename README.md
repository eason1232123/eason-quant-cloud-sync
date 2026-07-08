# Eason Quant Cloud Sync v1

目标：部署一次，以后 ChatGPT 可以读取固定公开链接里的量化报告。

## 一次性部署步骤

### 1. 新建 GitHub 仓库

建议名字：

```text
eason-quant-cloud-sync
```

建议设为 Public，或者代码 Private + GitHub Pages Public。

### 2. 上传本项目所有文件

保持这个结构：

```text
.github/workflows/daily_quant.yml
scripts/build_report.py
config.py
requirements.txt
docs/
```

### 3. 添加 Tiingo Secret

GitHub 仓库页面：

```text
Settings
→ Secrets and variables
→ Actions
→ New repository secret
```

Name 填：

```text
TIINGO_API_KEY
```

Value 填你的 Tiingo API key。

### 4. 打开 GitHub Pages

```text
Settings
→ Pages
→ Build and deployment
→ Source: Deploy from a branch
→ Branch: main
→ Folder: /docs
→ Save
```

### 5. 手动跑第一次

```text
Actions
→ Eason Quant Daily
→ Run workflow
```

跑完后，打开：

```text
https://你的GitHub用户名.github.io/eason-quant-cloud-sync/market_report.json
```

如果能看到 JSON，就成功了。

## 以后怎么用

你只要告诉 ChatGPT 这个固定链接：

```text
以后请自动读取：
https://你的GitHub用户名.github.io/eason-quant-cloud-sync/market_report.json
```

以后你问：

```text
现在可以买SMH吗？
```

ChatGPT 就能先读取这个报告，再结合实时行情/新闻/宏观，给你完整判断。

## 安全说明

公开链接里不会包含 Tiingo API key。
API key 只保存在 GitHub Secrets。
公开的只有回测结果、技术指标、相关性和你的近似仓位结构。
