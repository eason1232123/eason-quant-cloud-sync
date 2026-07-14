# IBKR 本地只读持仓桥接

此桥接只把真实账户数据写入 Git 已忽略的 `private/`，供本机 ChatGPT/Codex 分析。它不会把账户、现金、股数、成本、盈亏或订单上传到 GitHub，也不包含任何下单调用。

## 安全边界

- 仅允许 `127.0.0.1`、`::1` 或 `localhost`。
- 拒绝 `clientId=0`，默认使用独立的 `71`。
- 只调用持仓、账户汇总和账户/组合更新读取接口。
- TWS/IB Gateway 必须由用户手工登录并完成 2FA。
- TWS API 设置必须勾选 **Read Only API**；此设置无法由客户端可靠验证，所以代码不会假装已验证。
- 原始输出固定在 `private/ibkr/account_snapshot.json`，任何其他目录都会被拒绝。

## 官方前置条件

1. 安装并启动当前 Stable/Latest TWS 或 IB Gateway。
2. 在 TWS 的 `Global Configuration → API → Settings` 启用 Socket Client，并保持 Read Only。
3. Paper TWS 默认端口通常为 `7497`，Live TWS 通常为 `7496`；Gateway 常用 `4002/4001`。以软件实际设置为准。
4. 从 IBKR 官方 TWS API 包安装 Python client：进入 `source/pythonclient` 后执行 `python setup.py install`，再用 `python -m pip show ibapi` 核验。

官方参考：

- https://ibkrcampus.com/campus/ibkr-api-page/twsapi-doc/
- https://interactivebrokers.github.io/tws-api/initial_setup.html

## 本地配置

把 `.env.example` 中的非秘密配置设置为本机环境变量。`.env` 只可作为本地记录，脚本不会自动加载它。不要写用户名、密码、2FA、Token 或账户号。

```text
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=71
IBKR_TIMEOUT_SECONDS=15
IBKR_PRIVATE_SNAPSHOT=private/ibkr/account_snapshot.json
```

PowerShell 可临时设置 `$env:IBKR_PORT='7497'`。`run_v6_live_cycle` 在未设置端口或设为 `auto` 时，会只在 loopback 上探测标准 Gateway/TWS 端口；唯一命中才继续，多个命中会要求显式选择。独立的 `capture_ibkr_snapshot` 命令仍使用明确的 `IBKR_PORT`。

## 使用

先只检查本机端口：

```text
python -m scripts.capture_ibkr_snapshot --probe
```

确认 TWS/IB Gateway 已登录、Socket 已启用且 Read Only 后抓取：

```text
python -m scripts.capture_ibkr_snapshot
```

验证五分钟内的已有私有快照：

```text
python -m scripts.capture_ibkr_snapshot --validate-existing --max-age-seconds 300
```

把私有持仓事实与 GitHub 中的公开策略证据组合成仅供本机 GPT/Codex 读取的分析上下文：

```text
python -m scripts.build_local_ibkr_context --max-snapshot-age-seconds 300
```

输出固定为 `private/ibkr/chatgpt_account_context.json`。Git/GitHub 只保存代码、规则和脱敏验证证据；账户号、股数、余额、成本和盈亏只存在于 `private/`。

生成覆盖全部当前持仓、无排名和无数量上限的独立复核请求：

```text
python -m scripts.holding_review_contract build-request
```

TradingAgents、ChatGPT 或 Codex 按 `schemas/holding_review_response.schema.json` 把响应写入 `private/ibkr/holding_review_response.json` 后执行：

```text
python -m scripts.holding_review_contract validate-response
python -m scripts.run_v6_live_cycle finalize-holdings
```

该层只允许 `HOLD`、`REDUCE_REVIEW`、`EXIT_REVIEW` 和 `NO_ACTION`，不能创建买入候选、不能改变量化 `NO_TRADE`，也不会被计入影子证据放行门。公开的 `docs/holding_review_status.json` 不含持仓符号或账户数据。详细边界见 `docs/HOLDING_REVIEW.md`。

若端口离线、官方 `ibapi` 未安装、官方 `accountDownloadEnd` 未返回、Gateway 明确返回 `accountReady=false`/未知值、回调超时、账户为空、关键数字无效或快照过期，命令会以非零状态显式失败。`accountReady` 属于可选的客户端值；完全缺席时只能由官方 `accountDownloadEnd` 证明本轮账户下载完成。TWS 账户更新中的组合价格类型会标记为“未验证为实时”，不得当作实时行情信号。
