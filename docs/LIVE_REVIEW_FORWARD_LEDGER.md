# v6-T5 脱敏实时审查前瞻账本

这一层把已经通过 v6-T4 契约验证的私有 ChatGPT/Codex 审查，转换成可提交到 GitHub 的脱敏、不可变前瞻事件。GitHub 仍是证据层，ChatGPT 是实时分析层，IBKR 只读快照和人工确认仍是最终执行边界；该流程不创建或发送订单。

## 数据边界

本地录入必须同时提供当前的：

- `private/ibkr/chatgpt_account_context.json`
- `private/ibkr/live_review_request.json`
- `private/ibkr/live_review_response.json`
- `docs/decision_packet.json`
- `docs/model_governance.json`
- `docs/market_report.json`

程序会用当前私有上下文重新构建请求并逐字段比对，再验证响应、量化动作上限、六项检查、时效和公开证据绑定。缺少任一输入、输入过期、上下文变化或响应越权都会显式失败。

公开事件不会复制以下内容：

- 账户号、现金、净值、股数、成本和盈亏
- 私有持仓符号；`REDUCE_REVIEW` 的标的固定脱敏
- 请求 ID、私有上下文哈希、自由文本结论、理由码和来源名称
- 原始私有请求或响应

公开账本只保留受控字段：动作类别、公开 GitHub 候选、审查时间、检查状态计数、冻结模型指纹、数据元信息和隐私转换声明。每个事件都有不可变内容哈希；重复事件幂等，修改历史事件会失败。

## 本地录入

先完成 T3/T4 的私有只读快照和实时审查，再运行：

```text
python -m scripts.build_live_review_forward_ledger record-private-review
```

只有这个本地命令读取 `private/`。GitHub Actions 不生成私有审查事件，也不读取账户数据。

公开输出：

- `docs/live_review_forward_ledger.jsonl`
- `docs/live_review_forward_status.json`

## 前瞻结果

评价规则在事件产生前冻结：

- 观察点：审查所绑定的最新已完成美国市场 EOD 日期
- 假设入场：下一有效交易日收盘价 `close[t+1]`
- 评价周期：20 个交易栏
- 成本：复用共享策略契约中的佣金、滑点和半点差假设
- 价格类型：沿用候选事件冻结的 adjusted/unadjusted 价格基础
- 样本独立性：同一公开符号的评价窗口不得重叠

`BUY_REVIEW` 的结果是“假设买入收益”，不表示真实成交。`WAIT` 和 `NO_TRADE` 只报告被过滤候选的反事实收益，不把现金收益伪造为零。`REDUCE_REVIEW` 因私有标的脱敏，不进入公开逐标的收益评价。

每日公开工作流只运行：

```text
python -m scripts.build_live_review_forward_ledger update-outcomes
```

该命令只读取已脱敏账本和公开 EOD 缓存。未来价格行、缺失历史、价格基础变化、回退数据或不可复现的既有结果都会阻断发布。

## v6 发布审计

运行：

```text
python -m scripts.audit_v6_release
```

生成 `docs/v6_release_status.json`。代码契约通过不等于模型已经完成前瞻验证；审计会分别检查：

- 公开策略 20 日主周期结果至少 20 个（仅作支持证据，不宣称彼此独立）
- 每个治理挑战模型至少 48 个配对、非重叠前瞻样本
- 至少 20 个脱敏实时审查到期结果
- 至少一个经过私有 IBKR 上下文绑定的脱敏实时审查事件

审计把门槛分成两条独立轨道：人工试用审核使用公开策略 20 个结果、脱敏实时审查 20 个结果、模型/账本有效性和 IBKR→ChatGPT 脱敏证据；挑战模型晋升证据使用模型有效性和每个挑战模型 48 个配对样本。后者不再阻塞现有模型的人工试用审核。

人工试用轨道未满足时状态保持 `PROSPECTIVE_VALIDATION_IN_PROGRESS` 并列出具体 blocker；满足后也只允许进入 `READY_FOR_HUMAN_PILOT_REVIEW`。挑战模型样本达标只表示晋升证据可供审核，仍必须通过冻结的收益、胜率、回撤门槛和人工审核。`automatic_order_allowed` 始终为 `false`，仍要求人工确认。
