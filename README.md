# Eason Quant Cloud Sync

目标：部署一次，以后 ChatGPT 可以读取固定公开链接里的量化报告。这个仓库不负责自动下单，只负责生成可核验的行情、技术指标、单信号回测、vectorbt验证、组合级回测、样本外稳定性检查、交易复盘和风险候选。

## 当前 v3.5 结构

```text
.github/workflows/daily-quant.yml
scripts/build_report.py
scripts/build_vectorbt_validation.py
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

如果能看到 JSON，就成功了。

## v3.5 每日自动流程

```text
build_report.py
→ build_vectorbt_validation.py
→ build_portfolio_backtest.py
→ build_walk_forward_report.py
→ build_trade_review.py
→ build_decision_report.py
→ build_action_board_v3.py
→ commit docs
```

## 输出文件

### 1. 核心市场报告

```text
docs/market_report.json
```

包含：数据新鲜度、技术指标、多规则回测、90日相关性、更新日志、错误信息。

### 2. 单信号回测汇总

```text
docs/backtest_summary.csv
docs/rule_evidence_ranking.csv
```

包含每个 ticker、每个规则、每个 forward horizon 的样本数、胜率、平均/中位数收益、最大不利波动、与 QQQ/SPY/SMH/SOXX 同日期持有对比。

### 3. vectorbt 验证层 v3.5

```text
docs/vectorbt_validation.json
docs/vectorbt_signal_stats.csv
```

`build_vectorbt_validation.py` 使用 vectorbt 对核心规则做独立验证。它适合做高速、多ticker、多规则的 audit layer，帮助检查 pandas 单信号回测是否方向一致。

注意：vectorbt 验证层不是最终交易系统。它每个 ticker/rule 独立回测，不等于你的真实组合资金分配，也不替代组合级回测、walk-forward、实时行情、新闻和真实仓位检查。

### 4. 组合级回测 v2.0

```text
docs/portfolio_backtest.json
docs/portfolio_equity_curve.csv
docs/portfolio_trades.csv
docs/portfolio_vs_benchmark.csv
```

`build_portfolio_backtest.py` 会用模型组合权重回测：

```text
基础状态：QQQ 30%, SMH 25%, MSFT 20%, SPY 10%, CASH 15%
防守状态：QQQ 18%, SMH 12%, MSFT 16%, SPY 14%, CASH 40%
严重防守：QQQ 10%, SMH 8%, MSFT 12%, SPY 10%, CASH 60%
```

它会输出 CAGR、最大回撤、Sharpe、Sortino、Calmar、最终净值、交易次数、现金比例、科技/AI集中度、半导体仓位、MSFT仓位，并与买入持有 SPY / QQQ / SMH 对比。

注意：这是模型组合回测，不是你的真实账户交易记录。真实股数、现金、IBKR成交价仍然要在 ChatGPT 里单独确认。

### 5. 样本外 / 稳定性检查 v2.5

```text
docs/walk_forward_report.json
docs/market_regime_report.json
docs/overfitting_check.json
```

`build_walk_forward_report.py` 会把组合表现拆成：

```text
train_2005_2016
validation_2017_2021
test_2022_latest
```

目标是检查策略是不是只在某一段历史碰巧有效，而不是长期稳定有效。

### 6. 真实交易复盘 v3.0

```text
data/trade_log.csv
docs/trade_review.json
docs/trade_review.csv
docs/actual_vs_backtest.json
```

`data/trade_log.csv` 是空模板。你以后真实下单后，可以记录：

```text
date,ticker,action,shares,fill_price,fees,reason,signal_source,backtest_sample_count,buy_score,sell_risk_score,expected_thesis,invalidation_level,notes
```

`build_trade_review.py` 会自动计算每笔交易后 3 / 10 / 20 个交易日的结果，用来比较：

```text
实际交易表现 vs 回测预期
```

重要：这个仓库是 public 时，不建议记录真实敏感账户信息。如果要记录真实成交，最好改 private，或者只记录脱敏数据。

### 7. 总控板 / ChatGPT 主入口

```text
docs/action_board.json
docs/eason_master_status.json
docs/eason_signal.json
docs/latest_summary.json
docs/signal_candidates.csv
docs/risk_candidates.csv
```

最重要的是：

```text
docs/action_board.json
```

它会汇总：

```text
单信号回测
vectorbt 独立验证
组合级回测
walk-forward 稳定性
过拟合风险
市场状态
真实交易复盘
当前候选/风险
ChatGPT 必须复核的实时条件
```

所以：

```text
GitHub 通过 = 进入候选
ChatGPT 复核通过 = 才可能下单
IBKR 价格确认 + 人工确认 = 最终执行
```

## ChatGPT 使用方式

以后你问：

```text
现在可以买 SMH 吗？
```

ChatGPT 应该优先读取：

```text
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/action_board.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/eason_master_status.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/vectorbt_validation.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/eason_signal.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/portfolio_backtest.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/walk_forward_report.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/overfitting_check.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/trade_review.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/market_report.json
```

然后再结合实时行情、新闻、宏观、估值和你的真实账户仓位，给出最终判断。

## 安全说明

公开链接里不会包含 Tiingo API key。API key 只保存在 GitHub Secrets。

因为这个仓库是 public，不建议把真实现金、股数、账户净值、真实交易明细硬编码进公开 JSON。公开报告最好只放量化证据、技术指标、组合模型回测、信号候选和风险候选；真实仓位由你在 ChatGPT / IBKR / Finances 里单独确认。
