from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from config import TICKERS

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def safe_float(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def pct_change(values: list[float], lookback: int) -> float | None:
    if len(values) <= lookback or values[-lookback - 1] == 0:
        return None
    return (values[-1] / values[-lookback - 1] - 1.0) * 100.0


def annualized_volatility(values: list[float], window: int = 60) -> float | None:
    if len(values) < 3:
        return None
    sample = values[-(window + 1):]
    returns = [sample[index] / sample[index - 1] - 1.0 for index in range(1, len(sample)) if sample[index - 1] != 0]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(252.0) * 100.0


def read_price_file(path: Path) -> dict[str, object] | None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None

    close_keys = ("adjClose", "adj_close", "close", "Close")
    volume_keys = ("adjVolume", "adj_volume", "volume", "Volume")
    closes: list[float] = []
    volumes: list[float] = []
    latest_date = ""

    for row in rows:
        close = next((safe_float(row.get(key)) for key in close_keys if safe_float(row.get(key)) is not None), None)
        if close is None or close <= 0:
            continue
        closes.append(close)
        latest_date = str(row.get("date") or row.get("Date") or latest_date)
        volume = next((safe_float(row.get(key)) for key in volume_keys if safe_float(row.get(key)) is not None), None)
        if volume is not None and volume >= 0:
            volumes.append(volume)

    if len(closes) < 21:
        return None

    high_252 = max(closes[-252:])
    drawdown = (closes[-1] / high_252 - 1.0) * 100.0 if high_252 else None
    avg_dollar_volume_20 = None
    if len(volumes) >= 20:
        recent_volumes = volumes[-20:]
        recent_closes = closes[-20:]
        paired = list(zip(recent_closes[-len(recent_volumes):], recent_volumes))
        if paired:
            avg_dollar_volume_20 = sum(price * volume for price, volume in paired) / len(paired)

    return {
        "latest_date": latest_date,
        "latest_price": closes[-1],
        "history_rows": len(closes),
        "ret_20d_pct": pct_change(closes, 20),
        "ret_60d_pct": pct_change(closes, 60),
        "ret_120d_pct": pct_change(closes, 120),
        "drawdown_52w_pct": drawdown,
        "volatility_60d_pct": annualized_volatility(closes, 60),
        "avg_dollar_volume_20": avg_dollar_volume_20,
    }


def percentile_ranks(values: dict[str, float], higher_is_better: bool = True) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=not higher_is_better)
    count = len(ordered)
    if count <= 1:
        return {ticker: 50.0 for ticker, _ in ordered}
    return {ticker: index / (count - 1) * 100.0 for index, (ticker, _) in enumerate(ordered)}


def main() -> None:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(DOCS.glob("*_daily.csv")):
        ticker = path.name.removesuffix("_daily.csv").upper()
        metrics = read_price_file(path)
        if metrics:
            records[ticker] = {"ticker": ticker, **metrics}

    components = {
        "ret_20d_pct": {ticker: float(record["ret_20d_pct"]) for ticker, record in records.items() if record.get("ret_20d_pct") is not None},
        "ret_60d_pct": {ticker: float(record["ret_60d_pct"]) for ticker, record in records.items() if record.get("ret_60d_pct") is not None},
        "ret_120d_pct": {ticker: float(record["ret_120d_pct"]) for ticker, record in records.items() if record.get("ret_120d_pct") is not None},
        "drawdown_52w_pct": {ticker: float(record["drawdown_52w_pct"]) for ticker, record in records.items() if record.get("drawdown_52w_pct") is not None},
        "avg_dollar_volume_20": {ticker: math.log10(max(float(record["avg_dollar_volume_20"]), 1.0)) for ticker, record in records.items() if record.get("avg_dollar_volume_20") is not None},
        "volatility_60d_pct": {ticker: float(record["volatility_60d_pct"]) for ticker, record in records.items() if record.get("volatility_60d_pct") is not None},
    }

    ranks = {
        "ret20": percentile_ranks(components["ret_20d_pct"]),
        "ret60": percentile_ranks(components["ret_60d_pct"]),
        "ret120": percentile_ranks(components["ret_120d_pct"]),
        "drawdown": percentile_ranks(components["drawdown_52w_pct"]),
        "liquidity": percentile_ranks(components["avg_dollar_volume_20"]),
        "low_vol": percentile_ranks(components["volatility_60d_pct"], higher_is_better=False),
    }

    core_set = {ticker.upper() for ticker in TICKERS}
    ranked: list[dict[str, object]] = []
    for ticker, record in records.items():
        available = [
            ranks["ret20"].get(ticker), ranks["ret60"].get(ticker), ranks["ret120"].get(ticker),
            ranks["drawdown"].get(ticker), ranks["liquidity"].get(ticker), ranks["low_vol"].get(ticker),
        ]
        if any(value is None for value in available):
            continue
        score = (
            0.15 * ranks["ret20"][ticker]
            + 0.25 * ranks["ret60"][ticker]
            + 0.25 * ranks["ret120"][ticker]
            + 0.15 * ranks["drawdown"][ticker]
            + 0.15 * ranks["liquidity"][ticker]
            + 0.05 * ranks["low_vol"][ticker]
        )
        ranked.append({**record, "discovery_score_0_100": round(score, 2), "is_core_94": ticker in core_set})

    ranked.sort(key=lambda item: float(item["discovery_score_0_100"]), reverse=True)
    for index, record in enumerate(ranked, start=1):
        record["rank"] = index

    candidate_limit = 300
    deep_limit = 150
    dynamic_candidates = ranked[:candidate_limit]
    promoted = [record for record in ranked if not bool(record["is_core_94"])][: max(0, deep_limit - len(core_set))]
    deep_pool = sorted(core_set | {str(record["ticker"]) for record in promoted})

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "version": "dynamic-candidates-v1",
        "method": {
            "description": "Cross-sectional ranking over locally cached daily-price histories.",
            "weights": {"20d_momentum": 0.15, "60d_momentum": 0.25, "120d_momentum": 0.25, "52w_proximity": 0.15, "liquidity": 0.15, "low_volatility": 0.05},
            "candidate_limit": candidate_limit,
            "deep_pool_limit": deep_limit,
        },
        "coverage": {
            "cached_tickers_scored": len(ranked),
            "configured_core_count": len(core_set),
            "dynamic_candidate_count": len(dynamic_candidates),
            "deep_pool_count": len(deep_pool),
        },
        "deep_pool": deep_pool,
        "top_candidates": dynamic_candidates[:50],
        "limitations": [
            "Ranking currently covers tickers with local price caches; broad-universe symbols enter ranking only after hydration.",
            "This is a discovery score, not a trade signal or valuation model.",
            "Live news, fundamentals, valuation, portfolio concentration and broker quotes remain mandatory before an order.",
        ],
    }

    (DOCS / "dynamic_candidates.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (DOCS / "dynamic_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank", "ticker", "discovery_score_0_100", "is_core_94", "latest_date", "latest_price",
            "history_rows", "ret_20d_pct", "ret_60d_pct", "ret_120d_pct", "drawdown_52w_pct",
            "volatility_60d_pct", "avg_dollar_volume_20",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dynamic_candidates)
    print(json.dumps(payload["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
