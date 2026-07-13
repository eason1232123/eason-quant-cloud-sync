# v6-T4 ChatGPT 私有实时复核契约

这层把 ChatGPT/Codex 的实时判断从自由文本变成可验证的私有 JSON。GitHub 仍然只保存公开量化证据；真实持仓、持仓符号和账户推理留在 Git 已忽略的 `private/`。该契约不调用模型 API，也不创建、预览或发送订单。

## 决策边界

- `DATA_REVIEW_REQUIRED` 或 `NO_TRADE`：GPT 只能返回 `NO_TRADE` 或 `WAIT`。
- `RISK_REVIEW_REQUIRED`：只有真实持仓中的符号可以进入 `REDUCE_REVIEW`。
- `BUY_CANDIDATE_REVIEW_REQUIRED`：只有 GitHub 当前 `top_actionable` 中的符号可以进入 `BUY_REVIEW`。
- 任一必需检查失败或不可用时，只能 `NO_TRADE` 或 `WAIT`。
- `BUY_REVIEW` 必须六项检查全部通过，并至少有一个明确标记为实时的行情源。
- 所有结论仍要求人工确认；`automatic_order_allowed=false`，`order_payload=null`。

## 数据契约

请求 Schema：`schemas/live_review_request.schema.json`

响应 Schema：`schemas/live_review_response.schema.json`

响应的每项证据必须记录：

- 数据源名称与类型
- 来源 URL（私有来源可以为 `null`）
- UTC 观察时间
- 市场时区
- 数据类型，例如实时 bid/ask、延迟盘中、复权 EOD、新闻、估值或私有账户快照

公开网页证据必须提供 URL。公开 EOD 日期还必须等于仓库保守市场时钟计算出的当前已完成美国交易日；节假日仍沿用现有的“可能多阻断、绝不提前放行”策略。

## 本地流程

先安装仓库运行与验证依赖（没有新增依赖）：

```text
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

先按 `docs/IBKR_LOCAL_SETUP.md` 生成五分钟内的新鲜账户上下文，再生成实时复核请求：

```text
python -m scripts.live_review_contract build-request
```

请求输出：`private/ibkr/live_review_request.json`

ChatGPT/Codex 必须同时读取请求列出的三个输入文件，根据响应 Schema 写入：

```text
private/ibkr/live_review_response.json
```

然后验证：

```text
python -m scripts.live_review_contract validate-response
```

请求默认五分钟过期。响应必须绑定请求 ID 和私有上下文 SHA-256，且在同一有效窗口内完成。过期、未来时间、来源缺失、量化越权、候选/持仓不匹配或任何订单载荷都会显式失败。

当前阶段只验证“GPT 输入、证据与判断边界”。后续 v6-T5 才会从已验证的私有响应生成不含账户信息的前瞻审计事件；不会把私有响应直接发布到 GitHub。
