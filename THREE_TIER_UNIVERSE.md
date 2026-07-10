# Three-tier market universe

This upgrade separates broad discovery from expensive deep validation.

## Tier 1 — broad listing universe

`scripts/build_market_universe.py` downloads the official Nasdaq Trader symbol directories and builds:

- `docs/market_universe.csv`
- `docs/market_universe_summary.json`

The builder removes ETFs, test issues and obvious warrants, rights, units, preferred shares and debt securities. The result is a broad discovery registry, not a buy list.

## Tier 2 — dynamic candidate ranking

`scripts/build_dynamic_candidates.py` ranks every ticker that already has a usable local `docs/*_daily.csv` cache. The score combines:

- 20-day momentum: 15%
- 60-day momentum: 25%
- 120-day momentum: 25%
- proximity to the 52-week high: 15%
- 20-day average dollar-volume rank: 15%
- lower 60-day volatility: 5%

Outputs:

- `docs/dynamic_candidates.csv`
- `docs/dynamic_candidates.json`

The JSON includes a top-candidate snapshot and a deep-pool list capped at 150 names. The existing configured 94 names remain pinned in the deep pool.

## Tier 3 — deep evidence layer

The existing 94-ticker workflow remains the deep evidence layer for:

- technical and rule evidence
- vectorbt validation
- portfolio backtesting
- walk-forward checks
- trade review
- final action-board gating

## Important limitation

The broad universe becomes rankable only as symbols acquire local price caches. This pull request deliberately does not attempt to download full price history for thousands of symbols in one run, because that would exceed the current Tiingo request budget and weaken workflow reliability.

A later hydration policy can promote a limited number of broad-universe symbols into the cached candidate layer each week after liquidity and fundamental data sources are selected.
