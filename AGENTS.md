# AGENTS.md

This file defines the mandatory working rules for AI coding agents in this repository.

## Repository purpose

`eason-quant-cloud-sync` generates auditable market data, technical indicators, signal backtests, vectorbt evidence, portfolio backtests, walk-forward checks, trade reviews, decision reports, and risk candidates.

It is **not** an automated trading or order-execution system. Any final trading decision still requires current price verification, news and macro review, valuation checks, portfolio concentration checks, broker bid/ask verification, and human confirmation.

## Current architecture

Primary workflow:

```text
.github/workflows/main.yml
scripts/build_report_safe.py
scripts/build_market_universe.py
scripts/build_dynamic_candidates.py
scripts/build_latest_summary.py
scripts/build_vectorbt_validation.py
scripts/build_vectorbt_backtest.py
scripts/build_portfolio_backtest.py
scripts/build_walk_forward_report.py
scripts/build_trade_review.py
scripts/build_decision_report.py
scripts/build_model_candidate_ledger.py
scripts/build_action_board_v3.py
scripts/build_forward_ledger.py
scripts/build_live_review_forward_ledger.py
scripts/audit_v6_release.py
scripts/build_v6_operating_status.py
scripts/run_v6_live_cycle.py
config.py
requirements.txt
docs/
```

Daily processing order:

```text
build_market_universe.py
→ build_report_safe.py
→ build_dynamic_candidates.py
→ build_latest_summary.py
→ build_vectorbt_validation.py
→ build_vectorbt_backtest.py
→ build_portfolio_backtest.py
→ build_walk_forward_report.py
→ build_trade_review.py
→ build_decision_report.py
→ build_action_board_v3.py
```

Additional v6 public-evidence chain:

```text
build_decision_report.py
-> build_model_candidate_ledger.py
-> build_action_board_v3.py
-> validate config/prospective_universe.json
-> build_forward_ledger.py
-> build_live_review_forward_ledger.py update-outcomes
-> validate_model_artifacts.py
-> audit_v6_release.py
-> build_v6_operating_status.py
```

Private `record-private-review` ingestion is local-only and must never run in GitHub Actions.
The `scripts/run_v6_live_cycle.py` orchestrator is also local-only; it may prepare
read-only evidence and finalize sanitized events, but it must not start broker
software, bypass authentication, call an order API, commit, or push.
When `IBKR_PORT` is absent or `auto`, the orchestrator may select exactly one
reachable standard loopback Gateway/TWS port; zero or multiple listeners must fail
closed. Account readiness requires the official matching `accountDownloadEnd`
callback, and any explicit non-true `accountReady` value remains a hard failure.
`docs/v6_operating_status.json` is the public machine-readable scope boundary:
read-only shadow support may be available before the frozen human-pilot release
gates mature, but automatic execution remains permanently prohibited.

## Mandatory workflow

1. Read all relevant files, entry points, configuration, tests, generated schemas, and call relationships before editing.
2. Identify the root cause or exact implementation requirement before proposing a fix.
3. State the intended files and scope before making a non-trivial change.
4. Make the smallest change that correctly solves the task.
5. Preserve existing behavior outside the requested scope.
6. Run the narrowest relevant validation first, followed by broader checks where practical.
7. Never claim completion when validation failed or was not performed.
8. Report exactly what changed, what was tested, and what remains unverified.

## Change boundaries

- Modify only files required for the current task.
- Do not refactor unrelated code.
- Do not rename public functions, scripts, directories, configuration keys, JSON fields, CSV columns, workflow steps, or output files unless explicitly required.
- Preserve backward compatibility unless a breaking change is explicitly requested.
- Do not delete code merely because it appears unused. Confirm references and generated-output dependencies first.
- Do not silently alter the order of the daily processing chain.
- Do not restore retired output names such as `docs/latest_summary.json` or `docs/latest_summary.txt`.
- Do not make `build_action_board_v3.py` consume `action_board.json`; this can create recursive output nesting.
- Do not change the repository from a reporting and decision-support system into an automatic execution system.

## Correctness and failure behavior

- Never fabricate data, prices, timestamps, API responses, logs, calculations, test results, or successful execution.
- Never hard-code an expected result merely to satisfy a check.
- Do not remove tests, weaken assertions, suppress exceptions, or replace failures with misleading defaults.
- Do not use empty exception handlers or broad silent fallbacks.
- A failed critical data source or critical calculation must remain visible in status fields, error output, logs, or workflow failure.
- Do not replace unavailable values with `0` unless zero is mathematically and semantically correct.
- Prefer `null`, a documented status value, or an explicit error object for unavailable results.
- Validate empty input, missing files, malformed JSON/CSV, partial API responses, timeouts, stale caches, duplicate dates, and non-trading days where relevant.

## Quantitative finance rules

### Market data integrity

- Every report that depends on market data must preserve or expose the data source, market timezone, observation timestamp, and freshness status where the schema supports them.
- Distinguish delayed, end-of-day, intraday, adjusted, and unadjusted prices.
- Never present stale, partial, failed, deferred, or cache-only data as fresh current data.
- Preserve the tiered cache-safe refresh behavior and append-only history semantics unless a task explicitly requires a reviewed migration.
- Existing historical price rows must not be overwritten silently.
- Missing market data must not be interpolated or substituted without an explicit documented method and user approval.
- Tiingo rate-limit handling must remain visible through statuses and errors; do not conceal `429` or circuit-breaker behavior.

### Backtest integrity

- Prevent look-ahead bias. Signals calculated from a bar may not execute earlier than the next valid execution point defined by the strategy.
- Preserve the next-bar execution assumption used by vectorbt validation and evidence layers unless a reviewed strategy change explicitly replaces it.
- Preserve the prior-trading-day regime execution assumption in portfolio backtests unless explicitly reviewed.
- Prevent survivorship bias where universe construction can affect results.
- Apply realistic transaction costs, spreads, slippage, and execution assumptions when relevant.
- Separate training, validation, walk-forward, and final test periods.
- Do not optimize parameters on the same period used to present final performance.
- Report sample size, benchmark, data period, strategy parameters, and material assumptions.
- Low-sample or unstable results must be labeled as such.
- Never describe backtest performance as guaranteed future performance.
- Repeated runs with identical inputs and configuration should be reproducible.

### Signal and decision integrity

- An `active_signals` object is active only when at least one contained signal is actually `true`.
- Trading recommendations and action-board outputs must be traceable to input data, strategy rules, gates, and evidence.
- Do not bypass final gates, evidence checks, promotion gates, risk limits, or insufficient-data states just to produce a candidate.
- Do not treat a candidate ranking as an instruction to trade.
- Preserve the distinction between market summary, decision summary, signal evidence, portfolio evidence, and final action-board output.

### Output schema safety

- Preserve existing JSON schemas and CSV columns unless a schema change is explicitly requested.
- When a schema must change, update all producers, consumers, assertions, documentation, and compatibility handling in the same change.
- Validate generated JSON before writing or publishing it.
- Prefer atomic writes for generated files where practical so interrupted jobs do not publish truncated output.
- Do not publish a critical report if required upstream inputs failed validation.

## Security and data safety

- Never commit API keys, passwords, tokens, private URLs, account identifiers, or other secrets.
- Read secrets from environment variables or GitHub Actions secrets.
- Do not log full secrets or sensitive personal information.
- Do not print the value of `TIINGO_API_KEY`.
- Do not execute destructive commands, reset history, force-push, delete caches, or overwrite generated history without explicit authorization.
- Treat changes to workflow permissions, secret usage, scheduled execution, and write access as security-sensitive.

## Dependencies and style

- Use Python 3.11-compatible code.
- Follow the repository's existing architecture, naming, typing, logging, formatting, and error-handling patterns.
- Reuse existing utilities before adding new abstractions.
- Prefer simple, readable, maintainable code over clever abstractions.
- Keep functions focused and avoid unnecessary duplication.
- Comments should explain intent, assumptions, or non-obvious constraints rather than repeat the code.
- Prefer the Python standard library or existing dependencies.
- Do not add a dependency unless it is necessary and justified.
- Do not upgrade a major dependency version unless explicitly requested.
- If dependencies change, update `requirements.txt` and explain compatibility implications.

## Validation commands

Run checks appropriate to the changed scope.

Minimum syntax validation:

```bash
python -m compileall -q scripts config.py
```

Install dependencies when needed:

```bash
python -m pip install -r requirements.txt
```

Repository-cache smoke checks that do not require the Tiingo secret:

```bash
PYTHONPATH=. python -u scripts/build_market_universe.py
PYTHONPATH=. python -u scripts/build_dynamic_candidates.py
```

Then verify required outputs exist and are non-empty:

```bash
test -s docs/market_universe.csv
test -s docs/market_universe_summary.json
test -s docs/dynamic_candidates.csv
test -s docs/dynamic_candidates.json
test -s docs/promotion_queue.csv
test -s docs/promotion_queue.json
```

For generated JSON changed by the task, parse it explicitly:

```bash
python -m json.tool path/to/output.json > /dev/null
```

Do not run or claim a successful live Tiingo refresh unless `TIINGO_API_KEY` is actually available. Do not expose the key in commands or logs.

When practical, run each directly affected builder and verify its expected output schema and cross-file relationships. A full daily build is not required for documentation-only changes.

## Completion report

At the end of every coding task, provide:

1. Root cause or implementation approach.
2. Files changed.
3. Important behavior or schema changes.
4. Commands and tests actually executed.
5. Validation results.
6. Remaining risks, assumptions, or unverified areas.

Do not state that a check passed unless it was actually run and completed successfully.
