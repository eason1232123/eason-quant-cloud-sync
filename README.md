# Eason Quant Cloud Sync

目标：部署一次，以后 ChatGPT 可以读取固定公开链接里的量化报告。这个仓库不负责自动下单，只负责生成可核验的行情、技术指标、单信号回测、vectorbt 验证/证据层、组合级回测、样本外稳定性检查、交易复盘和风险候选。

## 当前 v4.3 结构

```text
.github/workflows/main.yml
scripts/build_report_safe.py
scripts/build_report.py
scripts/build_latest_summary.py
scripts/build_vectorbt_validation.py
scripts/build_vectorbt_backtest.py
scripts/build_portfolio_backtest.py
scripts/build_walk_forward_report.py
scripts/build_trade_review.py
scripts/build_decision_report.py
scripts/build_action_board_v3.py
data/trade_log.csv
config.py
requirements.txt
docs/
```

## 修复后的每日自动流程

```text
build_report_safe.py
→ build_latest_summary.py
→ build_vectorbt_validation.py
→ build_vectorbt_backtest.py
→ build_portfolio_backtest.py
→ build_walk_forward_report.py
→ build_trade_review.py
→ build_decision_report.py
→ build_action_board_v3.py
```

核心修复点：

```text
1. Workflow 实际路径是 .github/workflows/main.yml
2. build_decision_report.py 已加入每日链路
3. action_board.json 不再被 build_action_board_v3.py 读入，避免递归套娃
4. latest_summary.py 不再把全 false 的 active_signals dict 当成 active
5. vectorbt validation / evidence 采用下一根 bar 执行假设，避免同日收盘信号同日成交
6. portfolio backtest 使用前一交易日 regime 信号执行，避免同日信号/成交前视偏差
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

### 3. 手动运行

```text
Actions
→ Eason Quant Daily
→ Run workflow
```

跑完后，优先打开：

```text
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/action_board.json
```

其次检查：

```text
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/market_report.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/eason_signal.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/vectorbt_report.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/portfolio_backtest.json
```

如果 `market_report.json` 不是空文件，并且 `action_board.json` 能看到 `final_gate`、`signal_summary`、`vectorbt_evidence`、`portfolio_backtest`，就说明主链路成功。

## 重要限制

```text
GitHub 只负责数据、回测、证据、候选和风险提示。
它不是自动交易系统。
任何买卖前仍然必须做实时价格、新闻、宏观、估值、组合集中度、IBKR bid/ask 和人工确认。
```
