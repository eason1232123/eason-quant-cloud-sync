# v6-T6 本地只读实时周期

`scripts/run_v6_live_cycle.py` 把 T3、T4 和 T5 的安全步骤收敛为一个本地两阶段入口。它不会启动 TWS、自动登录、绕过 2FA、调用模型 API、提交 Git 或创建订单。

## 前置条件

用户仍需在本机完成一次外部状态准备：

1. 启动并登录 TWS 或 IB Gateway。
2. 完成 IBKR 2FA。
3. 启用 Socket API，并在客户端设置中保持 Read Only。
4. 确认本地端口与 `IBKR_PORT` 一致；Paper TWS 默认通常为 `7497`。

代码只接受 loopback 主机和非零 client ID。TWS 的 Read Only 勾选状态无法通过客户端 API可靠证明，因此仍由用户界面设置负责；代码本身不包含订单调用。

## 第一步：探测与准备

先运行无账户读取的探测：

```text
python -m scripts.run_v6_live_cycle probe
```

端口可达后运行：

```text
python -m scripts.run_v6_live_cycle prepare
```

`prepare` 的固定顺序是：

```text
公开模型产物完整验证
-> 本地 IBKR 端口探测
-> 官方 ibapi 只读快照
-> 私有账户/模型上下文
-> 私有 GPT 实时审查请求
```

任何一步失败都会停止后续步骤。特别是端口离线时不会尝试捕获，也不会拿旧快照继续生成请求。

生成的文件全部位于 Git 已忽略的 `private/ibkr/`：

- `account_snapshot.json`
- `chatgpt_account_context.json`
- `live_review_request.json`

请求默认五分钟过期。命令行只输出时间、状态和私有文件路径，不输出账户号、持仓、股数、余额、请求 ID 或具体结论。

## ChatGPT 实时分析

ChatGPT/Codex 按 `docs/LIVE_REVIEW_CONTRACT.md` 读取私有请求列出的输入，完成实时价格、市场状态、新闻宏观、财报估值、真实账户风险和执行可行性六项检查，并将响应写入：

```text
private/ibkr/live_review_response.json
```

模型只能保持或收紧 GitHub 量化边界，不能凭空创建买入候选。

## 第二步：脱敏入账

响应仍在有效期内时运行：

```text
python -m scripts.run_v6_live_cycle finalize
```

`finalize` 会重新验证当前私有上下文、请求、响应和公开模型证据，随后：

1. 向 `docs/live_review_forward_ledger.jsonl` 追加脱敏不可变事件。
2. 更新 `docs/live_review_forward_status.json`。
3. 重算 `docs/v6_release_status.json`。

它不会自动提交或推送 Git，也不会生成订单。重复执行同一响应是幂等的。

## 运行边界

- 本地周期不在 GitHub Actions 中运行，因为 GitHub 不应读取私有账户材料。
- TWS/IB Gateway 未启动、`ibapi` 缺失、请求过期、证据失配或任何关键检查失败都会显式返回不可用状态。
- `automatic_order_allowed` 始终为 `false`；人工确认始终必需。
- 完成一次周期只证明应用层链路跑通，不代表模型已达到前瞻样本门槛，也不是 IBKR 独立签名证明。
