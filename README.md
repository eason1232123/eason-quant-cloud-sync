# Eason Quant Cloud Sync

目标：部署一次，以后 ChatGPT 可以读取固定公开链接里的量化报告。这个仓库不负责自动下单，只负责生成可核验的行情、技术指标、单信号回测、vectorbt 验证/证据层、组合级回测、样本外稳定性检查、交易复盘和风险候选。

## 当前 v4.4 结构

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

## v4.4 核心修复点

```text
1. Workflow 实际路径是 .github/workflows/main.yml
2. build_decision_report.py 已加入每日链路
3. action_board.json 不再被 build_action_board_v3.py 读入，避免递归套娃
4. active_signals dict 只有内部至少一个信号为 true 才算 active
5. vectorbt validation / evidence 采用下一根 bar 执行假设，避免同日收盘信号同日成交
6. portfolio backtest 使用前一交易日 regime 信号执行，避免同日信号/成交前视偏差
7. latest_summary.json 已退役，避免市场摘要和决策摘要命名冲突
```

## 摘要文件命名

```text
docs/latest_market_summary.json   # 市场/技术/信号轻量摘要，由 build_latest_summary.py 生成
docs/latest_market_summary.txt    # 同上，纯文本版
docs/latest_decision_summary.json # 决策层摘要，由 build_decision_report.py 生成
```

请不要再使用：

```text
docs/latest_summary.json
docs/latest_summary.txt
```

这两个旧名字已经移除，避免 ChatGPT 把市场摘要和决策摘要混在一起。

## API / 缓存策略

`build_report_safe.py` 是为了节省 Tiingo API 而设计的 cache-safe 入口：

```text
1. 读取 docs/*_daily.csv 作为本地价格缓存
2. 已有缓存的 ticker 只从最新日期的下一天开始增量拉取
3. 新 ticker 才做完整历史下载
4. 每次运行限制 Tiingo 请求数和新 ticker 全量下载数
5. 遇到 Tiingo 429 会打开 circuit breaker，后续 ticker 直接用缓存或 defer
6. append-only：已有历史行不覆盖，只追加新日期
```

所以大股票池不会每次全量重拉。API 不够时，系统会优先保留已有缓存，并在 `market_report.json` 的 `update_log` 和 `errors` 里说明哪些是 fresh、cache_only、deferred。

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
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/latest_market_summary.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/latest_decision_summary.json
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
