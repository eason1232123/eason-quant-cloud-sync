from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from config import TICKERS

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MIN_HISTORY_ROWS = 252
MIN_DOLLAR_VOLUME = 20_000_000
MAX_VOLATILITY_PCT = 85.0
MAX_CHASE_20D_PCT = 25.0
PROMOTION_LIMIT = 20
DEEP_POOL_CAP = 150


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def period_return(values: list[float], days: int):
    if len(values) <= days or values[-days - 1] == 0:
        return None
    return (values[-1] / values[-days - 1] - 1) * 100


def annualized_volatility(values: list[float], days: int = 60):
    sample = values[-(days + 1) :]
    returns = [sample[index] / sample[index - 1] - 1 for index in range(1, len(sample)) if sample[index - 1] != 0]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(252) * 100


def read_prices(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    closes: list[float] = []
    volumes: list[float | None] = []
    latest_date = ""
    for row in rows:
        close = next(
            (
                number(row.get(key))
                for key in ("adjClose", "adj_close", "close", "Close")
                if number(row.get(key)) is not None
            ),
            None,
        )
        if close is None or close <= 0:
            continue
        closes.append(close)
        latest_date = str(row.get("date") or row.get("Date") or latest_date)
        volume = next(
            (
                number(row.get(key))
                for key in ("adjVolume", "adj_volume", "volume", "Volume")
                if number(row.get(key)) is not None
            ),
            None,
        )
        volumes.append(volume)

    if len(closes) < 21:
        return None

    dollar_volumes = [
        price * volume
        for price, volume in zip(closes[-20:], volumes[-20:])
        if volume is not None and volume >= 0
    ]
    average_dollar_volume = sum(dollar_volumes) / len(dollar_volumes) if dollar_volumes else None
    high_52w = max(closes[-252:])

    return {
        "latest_date": latest_date,
        "latest_price": closes[-1],
        "history_rows": len(closes),
        "ret_20d_pct": period_return(closes, 20),
        "ret_60d_pct": period_return(closes, 60),
        "ret_120d_pct": period_return(closes, 120),
        "drawdown_52w_pct": (closes[-1] / high_52w - 1) * 100,
        "volatility_60d_pct": annualized_volatility(closes),
        "avg_dollar_volume_20": average_dollar_volume,
    }


def percentile_ranks(values: dict[str, float], higher_is_better: bool = True):
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=not higher_is_better)
    if len(ordered) <= 1:
        return {ticker: 50.0 for ticker, _ in ordered}
    return {ticker: index / (len(ordered) - 1) * 100 for index, (ticker, _) in enumerate(ordered)}


def relative_return(record: dict, benchmark: dict | None, key: str):
    if benchmark is None or record.get(key) is None or benchmark.get(key) is None:
        return None
    return record[key] - benchmark[key]


def gate_candidate(record: dict, is_core: bool):
    reasons: list[str] = []
    if record["history_rows"] < MIN_HISTORY_ROWS:
        reasons.append("history_lt_252")
    if record["avg_dollar_volume_20"] is None or record["avg_dollar_volume_20"] < MIN_DOLLAR_VOLUME:
        reasons.append("liquidity_below_20m")
    if record["volatility_60d_pct"] is None or record["volatility_60d_pct"] > MAX_VOLATILITY_PCT:
        reasons.append("volatility_too_high")
    if record["ret_20d_pct"] is not None and record["ret_20d_pct"] > MAX_CHASE_20D_PCT:
        reasons.append("anti_chase_20d")
    if not is_core and record.get("alpha_60d_vs_qqq_pct") is not None and record["alpha_60d_vs_qqq_pct"] <= 0:
        reasons.append("no_60d_alpha_vs_qqq")
    if not is_core and record.get("alpha_120d_vs_spy_pct") is not None and record["alpha_120d_vs_spy_pct"] <= 0:
        reasons.append("no_120d_alpha_vs_spy")
    return reasons


def main():
    records: dict[str, dict] = {}
    for path in sorted(DOCS.glob("*_daily.csv")):
        ticker = path.name.removesuffix("_daily.csv").upper()
        metrics = read_prices(path)
        if metrics:
            records[ticker] = {"ticker": ticker, **metrics}

    qqq = records.get("QQQ")
    spy = records.get("SPY")
    for record in records.values():
        record["alpha_60d_vs_qqq_pct"] = relative_return(record, qqq, "ret_60d_pct")
        record["alpha_120d_vs_spy_pct"] = relative_return(record, spy, "ret_120d_pct")

    fields = {
        "r20": {ticker: row["ret_20d_pct"] for ticker, row in records.items() if row["ret_20d_pct"] is not None},
        "r60": {ticker: row["ret_60d_pct"] for ticker, row in records.items() if row["ret_60d_pct"] is not None},
        "r120": {ticker: row["ret_120d_pct"] for ticker, row in records.items() if row["ret_120d_pct"] is not None},
        "dd": {ticker: row["drawdown_52w_pct"] for ticker, row in records.items() if row["drawdown_52w_pct"] is not None},
        "liq": {
            ticker: math.log10(max(row["avg_dollar_volume_20"], 1))
            for ticker, row in records.items()
            if row["avg_dollar_volume_20"] is not None
        },
        "vol": {ticker: row["volatility_60d_pct"] for ticker, row in records.items() if row["volatility_60d_pct"] is not None},
        "alpha60": {
            ticker: row["alpha_60d_vs_qqq_pct"]
            for ticker, row in records.items()
            if row["alpha_60d_vs_qqq_pct"] is not None
        },
        "alpha120": {
            ticker: row["alpha_120d_vs_spy_pct"]
            for ticker, row in records.items()
            if row["alpha_120d_vs_spy_pct"] is not None
        },
    }
    ranks = {
        "r20": percentile_ranks(fields["r20"]),
        "r60": percentile_ranks(fields["r60"]),
        "r120": percentile_ranks(fields["r120"]),
        "dd": percentile_ranks(fields["dd"]),
        "liq": percentile_ranks(fields["liq"]),
        "vol": percentile_ranks(fields["vol"], higher_is_better=False),
        "alpha60": percentile_ranks(fields["alpha60"]),
        "alpha120": percentile_ranks(fields["alpha120"]),
    }

    core = {ticker.upper() for ticker in TICKERS}
    ranked: list[dict] = []
    required_rank_fields = tuple(ranks)
    for ticker, row in records.items():
        if any(ticker not in ranks[field] for field in required_rank_fields):
            continue
        score = (
            0.10 * ranks["r20"][ticker]
            + 0.20 * ranks["r60"][ticker]
            + 0.20 * ranks["r120"][ticker]
            + 0.10 * ranks["dd"][ticker]
            + 0.15 * ranks["liq"][ticker]
            + 0.05 * ranks["vol"][ticker]
            + 0.10 * ranks["alpha60"][ticker]
            + 0.10 * ranks["alpha120"][ticker]
        )
        is_core = ticker in core
        gate_reasons = gate_candidate(row, is_core)
        ranked.append(
            {
                **row,
                "discovery_score_0_100": round(score, 2),
                "is_core_94": is_core,
                "promotion_eligible": not gate_reasons,
                "gate_reasons": gate_reasons,
            }
        )

    ranked.sort(key=lambda row: row["discovery_score_0_100"], reverse=True)
    for index, row in enumerate(ranked, 1):
        row["rank"] = index

    candidates = ranked[:300]
    promotion_queue = [row for row in ranked if not row["is_core_94"] and row["promotion_eligible"]][:PROMOTION_LIMIT]
    promoted_for_deep_pool = promotion_queue[: max(0, DEEP_POOL_CAP - len(core))]
    deep_pool = sorted(core | {row["ticker"] for row in promoted_for_deep_pool})

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "dynamic-candidates-v2-promotion-gates",
        "purpose": "Discovery ranking with liquidity, history, anti-chase and benchmark-relative promotion gates. Not an order signal.",
        "score_weights": {
            "ret_20d": 0.10,
            "ret_60d": 0.20,
            "ret_120d": 0.20,
            "proximity_52w_high": 0.10,
            "liquidity": 0.15,
            "lower_volatility": 0.05,
            "alpha_60d_vs_qqq": 0.10,
            "alpha_120d_vs_spy": 0.10,
        },
        "promotion_gates": {
            "minimum_history_rows": MIN_HISTORY_ROWS,
            "minimum_avg_dollar_volume_20": MIN_DOLLAR_VOLUME,
            "maximum_volatility_60d_pct": MAX_VOLATILITY_PCT,
            "maximum_ret_20d_pct_before_anti_chase": MAX_CHASE_20D_PCT,
            "non_core_requires_positive_60d_alpha_vs_qqq": True,
            "non_core_requires_positive_120d_alpha_vs_spy": True,
        },
        "coverage": {
            "cached_tickers_read": len(records),
            "cached_tickers_scored": len(ranked),
            "configured_core_count": len(core),
            "dynamic_candidate_count": len(candidates),
            "promotion_queue_count": len(promotion_queue),
            "deep_pool_count": len(deep_pool),
        },
        "deep_pool": deep_pool,
        "promotion_queue": promotion_queue,
        "top_candidates": candidates[:50],
        "warnings": [
            "Promotion eligibility is only a data and price-quality gate; it is not a buy recommendation.",
            "Fundamental quality, valuation, earnings risk, live quotes, news, macro regime and portfolio fit still require GPT review.",
        ],
    }
    (DOCS / "dynamic_candidates.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (DOCS / "promotion_queue.json").write_text(
        json.dumps(
            {
                "generated_at_utc": payload["generated_at_utc"],
                "version": payload["version"],
                "count": len(promotion_queue),
                "candidates": promotion_queue,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    candidate_fields = [
        "rank",
        "ticker",
        "discovery_score_0_100",
        "promotion_eligible",
        "gate_reasons",
        "is_core_94",
        "latest_date",
        "latest_price",
        "history_rows",
        "ret_20d_pct",
        "ret_60d_pct",
        "ret_120d_pct",
        "alpha_60d_vs_qqq_pct",
        "alpha_120d_vs_spy_pct",
        "drawdown_52w_pct",
        "volatility_60d_pct",
        "avg_dollar_volume_20",
    ]
    with (DOCS / "dynamic_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    with (DOCS / "promotion_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(promotion_queue)

    print(json.dumps(payload["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
