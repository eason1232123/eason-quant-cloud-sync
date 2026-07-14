# Eason Quant Cloud Sync

目标：部署一次，以后 ChatGPT 可以读取固定公开链接里的量化报告。这个仓库不负责自动下单，只负责生成可核验的行情、技术指标、单信号回测、vectorbt 验证/证据层、组合级回测、样本外稳定性检查、交易复盘和风险候选。

## 当前 v5.0 结构

```text
.github/workflows/main.yml
scripts/build_report_safe.py
scripts/build_report.py
scripts/market_clock.py
scripts/market_data_contract.py
scripts/build_latest_summary.py
scripts/build_vectorbt_validation.py
scripts/build_vectorbt_backtest.py
scripts/build_portfolio_backtest.py
scripts/build_walk_forward_report.py
scripts/build_trade_review.py
scripts/build_decision_report.py
scripts/build_action_board_v3.py
scripts/validate_decision_packet.py
scripts/validate_market_universe.py
schemas/decision_packet.schema.json
tests/test_decision_contract.py
data/trade_log.csv
config.py
requirements.txt
requirements-dev.txt
docs/
```

## 每日自动流程

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

## v5.0 决策契约

`docs/decision_packet.json` 是 GitHub 证据层交给 ChatGPT 的首要入口。它只包含公开模型组合的 ticker 范围，不包含真实股数、现金或账户价值。

```text
1. 只有公开模型组合（默认 QQQ/SMH/MSFT/SPY）的新鲜高风险信号可以封锁 GitHub 全局闸门
2. 非模型组合的观察池高风险只作为 advisory，必须由 ChatGPT/IBKR 确认真实持仓
3. 买入候选的技术日期和实际信号日期都必须与预期行情日完全一致；旧行情只进入 stale_excluded
4. 观察池落后超过 2 个工作日、长尾超过 5 个工作日时，跳过轮转并强制刷新
5. market_regime、buy_permission、data_status 和 portfolio_scope 不再为空
6. PR 与每日任务都会运行无网络单元测试，并验证 decision_packet JSON Schema
7. automatic_order_allowed 永远为 false
8. 每份核心报告显式记录数据源、America/New_York、数据日期、EOD 频率和复权策略；缺失时保留 null 并阻断
```

行情参考日和工作日差统一由 `scripts/market_clock.py` 提供；行情元信息契约统一由 `scripts/market_data_contract.py` 提供；Schema 与跨字段不变量统一由 `scripts/validate_decision_packet.py` 验证，生产任务和测试不再各复制一套规则。当前日期基准是美国市场时区下的工作日保守门，尚未接入交易所节假日日历；节假日前后可能多阻断一天，但不会据此自动下单。

## v6 运行分级

`docs/v6_operating_status.json` 是机器可读的运行范围边界，不替代 `decision_packet.json`，也不改变冻结的前瞻样本门槛：

```text
UNAVAILABLE                 # 关键模型、审查或 IBKR→ChatGPT 证据链不可用
READ_ONLY_SHADOW            # 可读证据并进行只读实时辅助；不是人工试用就绪或交易指令
HUMAN_PILOT_REVIEW_READY    # 公开前瞻 20、脱敏实时审查 20 及只读证据门槛通过，可进入人工试用审核
```

挑战模型的 48 个配对、非重叠前瞻样本是独立的模型晋升证据门槛，不再阻塞现有模型进入人工试用审核；样本达标也不等于挑战模型获准替换或获得真实资金。任何分级下 `automatic_order_allowed` 都必须为 `false`，最终执行层固定为 IBKR 手工操作和明确人工确认。运行状态同时记录数据源、`America/New_York`、市场数据日期、EOD/盘中类型和复权策略；关键字段缺失时构建失败，不会发布模糊状态。

## v4.5 基础修复点

```text
1. Workflow 实际路径是 .github/workflows/main.yml
2. build_decision_report.py 已加入每日链路
3. action_board.json 不再被 build_action_board_v3.py 读入，避免递归套娃
4. active_signals dict 只有内部至少一个信号为 true 才算 active
5. vectorbt validation / evidence 采用下一根 bar 执行假设，避免同日收盘信号同日成交
6. portfolio backtest 使用前一交易日 regime 信号执行，避免同日信号/成交前视偏差
7. latest_summary.json 已退役，避免市场摘要和决策摘要命名冲突
8. Tiingo 刷新改成 tiered-cache-refresh：核心每天、观察池轮流、长尾每周
```

## 摘要文件命名

```text
docs/latest_market_summary.json   # 市场/技术/信号轻量摘要，由 build_latest_summary.py 生成
docs/latest_market_summary.txt    # 同上，纯文本版
docs/latest_decision_summary.json # 决策层摘要，由 build_decision_report.py 生成
docs/decision_packet.json         # GitHub → ChatGPT 的稳定 v5 决策契约（优先读取）
docs/v6_release_status.json       # v6 前瞻样本与人工试用发布闸门
docs/v6_operating_status.json     # v6 当前允许的只读/人工试用运行分级
docs/eason_signal.txt             # 当前阻断/许可的人类可读快照；机器仍优先读取 decision_packet.json
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
2. 如果缓存已经覆盖预期最新交易日，直接跳过 Tiingo 请求
3. 已有缓存的 ticker 只从最新日期的下一天开始增量拉取
4. 新 ticker 才做完整历史下载；上限由 `MAX_NEW_FULL_DOWNLOADS_PER_RUN` 控制（代码默认 8，当前工作流 40）
5. 请求上限由 `MAX_TIINGO_REQUESTS_PER_RUN` 控制（代码默认 35，当前工作流 50）
6. 核心 ticker 每天尝试刷新：SPY, QQQ, SMH, MSFT, SGOV, NVDA 等
7. 观察池 ticker 每 3 天轮流刷新；落后超过 2 个工作日时强制刷新
8. 长尾 ticker 每周轮流刷新；落后超过 5 个工作日时强制刷新
9. 遇到 Tiingo 429 会打开 circuit breaker，后续 ticker 直接用缓存或 defer
10. append-only：已有历史行不覆盖，只追加新日期
```

所以大股票池不会每次全量重拉。API 不够时，系统会优先保留已有缓存，并在 `market_report.json` 的 `update_log` 和 `errors` 里说明哪些是 fresh、cache_fresh_enough_no_request、cache_only_tier_rotation、deferred、cache_after_fetch_error。

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
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/decision_packet.json
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
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/v6_release_status.json
https://raw.githubusercontent.com/eason1232123/eason-quant-cloud-sync/main/docs/v6_operating_status.json
```

如果 `market_report.json` 不是空文件，并且 `action_board.json` 能看到 `final_gate`、`signal_summary`、`vectorbt_evidence`、`portfolio_backtest`，就说明主链路成功。

## 重要限制

```text
GitHub 只负责数据、回测、证据、候选和风险提示。
它不是自动交易系统。
任何买卖前仍然必须做实时价格、新闻、宏观、估值、组合集中度、IBKR bid/ask 和人工确认。
```
