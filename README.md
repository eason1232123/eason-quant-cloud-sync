# Eason Quant Cloud Sync

目标：部署一次，以后 ChatGPT 可以读取固定公开链接里的量化报告。这个仓库不负责自动下单，只负责生成可核验的行情、技术指标、回测证据、信号候选和风险候选。

## 当前结构

```text
.github/workflows/daily-quant.yml
scripts/build_report.py
scripts/build_decision_report.py
config.py
requirements.txt
docs/
```

## 一次性部署步骤

### 1. 添加 Tiingo Secret

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

### 2. 打开 GitHub Pages

```text
Settings
→ Pages
→ Build and deployment
→ Source: Deploy from a branch
→ Branch: main
→ Folder: /docs
→ Save
```

### 3. 手动跑第一次

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

## 输出文件

### 核心报告

```text
docs/market_report.json
```

包含：数据新鲜度、技术指标、多规则回测、90日相关性、更新日志、错误信息。

### 回测汇总

```text
docs/backtest_summary.csv
```

包含每个 ticker、每个规则、每个 forward horizon 的样本数、胜率、平均/中位数收益、最大不利波动、与 QQQ/SPY/SMH/SOXX 同日期持有对比。

### 规则证据排名

```text
docs/rule_evidence_ranking.csv
```

用于看哪些规则历史证据更强，但这不等于直接买入。

### 决策层摘要

```text
docs/eason_signal.json
docs/latest_summary.json
docs/signal_candidates.csv
docs/risk_candidates.csv
```

`build_decision_report.py` 会把 `market_report.json` 再筛一层：

- 必须是最新交易日 active signal；
- 20日样本数必须 >= 20；
- 胜率必须 >= 55%；
- 平均收益和中位数收益必须 > 0；
- 相对基准 alpha 必须 > 0；
- worst MAE 不能差于 -15%；
- 最终下单前仍需要实时价格、新闻、财报、宏观和真实账户仓位确认。

所以输出里如果是 `NO_TRADE`，意思是量化证据还不够，不应该强行交易。

## ChatGPT 使用方式

以后你问：

```text
现在可以买 SMH 吗？
```

ChatGPT 应该优先读取：

```text
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/market_report.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/eason_signal.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/backtest_summary.csv
```

然后再结合实时行情、新闻、宏观、估值和你的真实账户仓位，给出最终判断。

## 安全说明

公开链接里不会包含 Tiingo API key。API key 只保存在 GitHub Secrets。

因为这个仓库是 public，不建议把真实现金、股数、账户净值硬编码进公开 JSON。公开报告最好只放量化证据、技术指标、信号候选和风险候选；真实仓位由你在 ChatGPT / IBKR / Finances 里单独确认。
