from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import TICKERS, START_DATE
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
)
from scripts.market_clock import MARKET_TIMEZONE
from scripts.strategy_contract import (
    EXECUTION_SHIFT_BARS,
    FORWARD_HORIZONS,
    MIN_EFFECTIVE_SAMPLE,
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    benchmark_for_ticker,
    execution_cost_assumptions,
    net_return_after_round_trip_costs,
    next_close_forward_mae,
    next_close_forward_return,
    non_overlapping_signal_mask,
    rule_signals,
)

API_KEY = os.getenv("TIINGO_API_KEY")

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

HORIZONS = list(FORWARD_HORIZONS)
MIN_SAMPLE = MIN_EFFECTIVE_SAMPLE

MAX_TIINGO_REQUESTS_PER_RUN = int(os.getenv("MAX_TIINGO_REQUESTS_PER_RUN", "35"))
MAX_NEW_FULL_DOWNLOADS_PER_RUN = int(os.getenv("MAX_NEW_FULL_DOWNLOADS_PER_RUN", "8"))
REQUEST_SLEEP_SECONDS = float(os.getenv("TIINGO_REQUEST_SLEEP_SECONDS", "0.9"))

CORE_ALWAYS_REFRESH = {
    "SPY", "QQQ", "SMH", "SOXX", "SGOV",
    "MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "META",
    "GLD", "TLT", "IEF",
}


def csv_path(ticker: str) -> Path:
    safe = ticker.replace("/", "-").replace(".", "-")
    return OUT / f"{safe}_daily.csv"


def clean_float(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def redact_sensitive_text(value: Any) -> str:
    text = str(value)
    if API_KEY:
        text = text.replace(API_KEY, "[REDACTED]")
    text = re.sub(r"(?i)(token=)[^&\s]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*:\s*(?:bearer|token)\s+)[^\s]+", r"\1[REDACTED]", text)
    return text


def normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker

    price_col = "adjClose" if "adjClose" in df.columns else "close"
    low_col = "adjLow" if "adjLow" in df.columns else "low"
    high_col = "adjHigh" if "adjHigh" in df.columns else "high"

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["low_price"] = pd.to_numeric(df[low_col], errors="coerce")
    df["high_price"] = pd.to_numeric(df[high_col], errors="coerce")

    return df.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)


def market_data_report_fields(price_map: dict[str, pd.DataFrame], data_source: str) -> dict[str, Any]:
    data_timestamp_by_ticker = {
        ticker: pd.to_datetime(frame["date"]).max().date().isoformat()
        for ticker, frame in sorted(price_map.items())
        if not frame.empty and "date" in frame.columns
    }
    observed_latest_market_date = max(data_timestamp_by_ticker.values()) if data_timestamp_by_ticker else None
    return {
        "data_source": data_source,
        "market_timezone": MARKET_TIMEZONE,
        "data_timestamp": observed_latest_market_date,
        "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
        "data_timestamp_status": "AVAILABLE" if observed_latest_market_date else "MISSING",
        "data_timestamp_by_ticker": data_timestamp_by_ticker,
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "price_basis_by_ticker": {
            ticker: "adjusted" if "adjClose" in frame.columns else "unadjusted"
            for ticker, frame in sorted(price_map.items())
        },
    }


def report_dataframe(rows: list[dict[str, Any]], report: dict[str, Any]) -> pd.DataFrame:
    metadata = {
        "data_source": report.get("data_source"),
        "market_timezone": report.get("market_timezone"),
        "report_generated_at_utc": report.get("generated_at_utc"),
        "data_timestamp": report.get("data_timestamp"),
        "price_frequency": report.get("price_frequency"),
        "price_adjustment_policy": report.get("price_adjustment_policy"),
        "strategy_contract_version": report.get("strategy_contract_version"),
        "rule_fingerprint": report.get("rule_fingerprint"),
        "strategy_fingerprint": report.get("strategy_fingerprint"),
    }
    enriched = [{**metadata, **row} for row in rows]
    return pd.DataFrame(enriched) if enriched else pd.DataFrame(columns=list(metadata))


def load_existing(ticker: str) -> pd.DataFrame:
    path = csv_path(ticker)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        if df.empty:
            return df
        return normalize(df, ticker)
    except Exception:
        return pd.DataFrame()


def cached_latest_date(ticker: str):
    existing = load_existing(ticker)
    if existing.empty:
        return None
    return pd.to_datetime(existing["date"]).max().date()


def get_fetch_start_date(ticker: str) -> str:
    last_date = cached_latest_date(ticker)
    if last_date is None:
        return START_DATE

    start = last_date - timedelta(days=7)
    min_start = pd.to_datetime(START_DATE).date()
    if start < min_start:
        start = min_start
    return start.isoformat()


def fetch_tiingo(ticker: str, start_date: str) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("Missing TIINGO_API_KEY GitHub Secret.")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start_date,
        "resampleFreq": "daily",
        "token": API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=45)
    except requests.RequestException as exc:
        raise RuntimeError(f"{ticker} Tiingo request failed: {type(exc).__name__}") from exc
    if r.status_code != 200:
        raise RuntimeError(f"{ticker} Tiingo HTTP {r.status_code}: {r.reason or 'request failed'}")

    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError(f"{ticker}: empty data from {start_date}")

    return normalize(df, ticker)


def merge_and_save(ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
    old_df = load_existing(ticker)

    if old_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([old_df, new_df], ignore_index=True)

    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    merged = merged.sort_values("date")
    merged = merged.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    preferred = [
        "ticker", "date", "open", "high", "low", "close", "volume",
        "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
        "divCash", "splitFactor", "price", "low_price", "high_price",
    ]
    cols = [c for c in preferred if c in merged.columns] + [
        c for c in merged.columns if c not in preferred
    ]

    merged = merged[cols]
    merged.to_csv(csv_path(ticker), index=False)
    return merged


def request_priority(ticker: str) -> tuple:
    last = cached_latest_date(ticker)
    core_rank = 0 if ticker in CORE_ALWAYS_REFRESH else 1

    if last is None:
        return (core_rank, 0, datetime(1900, 1, 1).date())

    return (core_rank, 1, last)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)

    for n in [1, 5, 10, 20, 60]:
        d[f"ret_{n}d"] = d["price"].pct_change(n, fill_method=None)

    for n in [5, 10, 20, 50, 100, 200]:
        d[f"ma{n}"] = d["price"].rolling(n).mean()

    delta = d["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))

    d["ma20_slope_10d"] = d["ma20"] / d["ma20"].shift(10) - 1
    d["ma50_slope_20d"] = d["ma50"] / d["ma50"].shift(20) - 1
    d["rolling_high_20d"] = d["price"].rolling(20).max()
    d["rolling_high_252d"] = d["price"].rolling(252, min_periods=20).max()
    d["drawdown_from_52w_high"] = d["price"] / d["rolling_high_252d"] - 1
    d["volatility_20d"] = d["ret_1d"].rolling(20).std() * np.sqrt(252)
    d["volatility_60d"] = d["ret_1d"].rolling(60).std() * np.sqrt(252)

    return d


def prepare_benchmarks(price_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    benches = {}
    for ticker in ["QQQ", "SPY", "SMH", "SOXX"]:
        if ticker in price_map:
            benches[ticker] = add_indicators(price_map[ticker])[["date", "price"]].rename(
                columns={"price": f"{ticker}_price"}
            )
    return benches


def choose_relative_benchmark(ticker: str) -> str:
    return benchmark_for_ticker(ticker)


def build_rule_frame(ticker: str, df: pd.DataFrame, price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = add_indicators(df)

    rel_bench = choose_relative_benchmark(ticker)
    if rel_bench in price_map:
        b = add_indicators(price_map[rel_bench])[["date", "ret_20d", "ret_60d"]].rename(
            columns={"ret_20d": "bench_ret_20d", "ret_60d": "bench_ret_60d"}
        )
        d = d.merge(b, on="date", how="left")
        d["relative_20d"] = d["ret_20d"] - d["bench_ret_20d"]
        d["relative_60d"] = d["ret_60d"] - d["bench_ret_60d"]
    else:
        d["relative_20d"] = np.nan
        d["relative_60d"] = np.nan

    for rule_name, signal in rule_signals(d["price"], d["relative_20d"]).items():
        d[f"rule_{rule_name}"] = signal

    return d


def max_adverse_excursion(df: pd.DataFrame, horizon: int) -> pd.Series:
    return next_close_forward_mae(df["price"], df["low_price"], horizon)


def attach_benchmark_forward_returns(df: pd.DataFrame, benchmarks: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    out = df[["date"]].copy()
    for bench_name, bench_df in benchmarks.items():
        b = bench_df.copy()
        price_col = f"{bench_name}_price"
        gross_col = f"{bench_name}_gross_fwd_{horizon}d"
        net_col = f"{bench_name}_fwd_{horizon}d"
        b[gross_col] = next_close_forward_return(b[price_col], horizon)
        b[net_col] = net_return_after_round_trip_costs(b[gross_col])
        out = out.merge(b[["date", gross_col, net_col]], on="date", how="left")
    return out


def prepare_forward_evidence_frame(
    df: pd.DataFrame,
    benchmarks: dict[str, pd.DataFrame],
    horizon: int,
) -> pd.DataFrame:
    d = df.copy()
    d[f"gross_fwd_{horizon}d"] = next_close_forward_return(d["price"], horizon)
    d[f"fwd_{horizon}d"] = net_return_after_round_trip_costs(d[f"gross_fwd_{horizon}d"])
    d[f"mae_{horizon}d"] = max_adverse_excursion(d, horizon)
    return d.merge(attach_benchmark_forward_returns(d, benchmarks, horizon), on="date", how="left")


def summarize_rule(
    ticker: str,
    df: pd.DataFrame,
    rule_col: str,
    benchmarks: dict[str, pd.DataFrame],
    horizon: int,
) -> dict:
    required_columns = {
        f"gross_fwd_{horizon}d",
        f"fwd_{horizon}d",
        f"mae_{horizon}d",
    }
    d = (
        df.copy()
        if required_columns.issubset(df.columns)
        else prepare_forward_evidence_frame(df, benchmarks, horizon)
    )

    raw_signal = d[rule_col].fillna(False).astype(bool)
    raw_signal_count = int(raw_signal.sum())
    required_result_columns = [
        f"gross_fwd_{horizon}d",
        f"fwd_{horizon}d",
        f"mae_{horizon}d",
    ]
    completed_signal = raw_signal & d[required_result_columns].notna().all(axis=1)
    independent_mask = non_overlapping_signal_mask(completed_signal, horizon)
    sig = d[independent_mask]

    result = {
        "ticker": ticker,
        "rule": rule_col.replace("rule_", ""),
        "horizon_days": horizon,
        "raw_signal_count": raw_signal_count,
        "completed_signal_count": int(completed_signal.sum()),
        "effective_sample_count": int(len(sig)),
        "samples": int(len(sig)),
        "valid": bool(len(sig) >= MIN_SAMPLE),
        "min_sample_required": MIN_SAMPLE,
        "sample_independence": "chronological_non_overlapping_signal_windows",
    }

    if len(sig) == 0:
        result["reason"] = "no completed signal samples"
        return result

    fwd = sig[f"fwd_{horizon}d"]
    gross_fwd = sig[f"gross_fwd_{horizon}d"]
    mae = sig[f"mae_{horizon}d"]

    result.update(
        {
            "win_rate": clean_float((fwd > 0).mean()),
            "avg_gross_return": clean_float(gross_fwd.mean()),
            "avg_return": clean_float(fwd.mean()),
            "median_return": clean_float(fwd.median()),
            "worst_return": clean_float(fwd.min()),
            "best_return": clean_float(fwd.max()),
            "avg_mae": clean_float(mae.mean()),
            "worst_mae": clean_float(mae.min()),
            "lookback_start": d["date"].min().date().isoformat(),
            "lookback_end": d["date"].max().date().isoformat(),
            "last_completed_signal_date": sig["date"].max().date().isoformat(),
        }
    )

    for bench_name in ["QQQ", "SPY", "SMH", "SOXX"]:
        col = f"{bench_name}_fwd_{horizon}d"
        if col in sig.columns and sig[col].notna().any():
            paired = sig[[f"fwd_{horizon}d", col]].dropna()
            bench_avg = float(paired[col].mean())
            strategy_avg = float(paired[f"fwd_{horizon}d"].mean())
            result[f"paired_{bench_name}_sample_count"] = int(len(paired))
            result[f"avg_{bench_name}_same_dates"] = clean_float(bench_avg)
            result[f"avg_alpha_vs_{bench_name}"] = clean_float(strategy_avg - bench_avg)
            gross_col = f"{bench_name}_gross_fwd_{horizon}d"
            if gross_col in sig.columns:
                gross_paired = sig[[f"gross_fwd_{horizon}d", gross_col]].dropna()
                if not gross_paired.empty:
                    result[f"avg_{bench_name}_gross_same_dates"] = clean_float(
                        gross_paired[gross_col].mean()
                    )

    return result


def summarize_all_rules(ticker: str, df: pd.DataFrame, price_map: dict[str, pd.DataFrame], benchmarks: dict[str, pd.DataFrame]) -> tuple[dict, list]:
    rule_frame = build_rule_frame(ticker, df, price_map)
    rule_cols = [
        "rule_pullback_reclaim_5dma",
        "rule_rsi_oversold_reclaim_40",
        "rule_ma20_reclaim_bullish",
        "rule_ma50_reclaim_bullish",
        "rule_relative_strength_rebound",
        "rule_momentum_leader",
        "rule_failed_rebound_risk",
    ]

    all_rows = []
    nested = {}
    prepared_by_horizon = {
        horizon: prepare_forward_evidence_frame(rule_frame, benchmarks, horizon)
        for horizon in HORIZONS
    }

    for rule_col in rule_cols:
        nested[rule_col.replace("rule_", "")] = {}
        for horizon in HORIZONS:
            res = summarize_rule(
                ticker,
                prepared_by_horizon[horizon],
                rule_col,
                benchmarks,
                horizon,
            )
            nested[rule_col.replace("rule_", "")][f"{horizon}d"] = res
            all_rows.append(res)

    active = {}
    latest = rule_frame.iloc[-1]
    for rule_col in rule_cols:
        active[rule_col.replace("rule_", "")] = bool(latest.get(rule_col, False))

    nested["active_signals_latest_day"] = active
    nested["latest_date"] = latest["date"].date().isoformat()
    nested["latest_price"] = clean_float(latest["price"])
    nested["trading_days"] = int(len(rule_frame))
    nested["short_history"] = bool(len(rule_frame) < 252)
    nested["very_short_history"] = bool(len(rule_frame) < 90)

    return nested, all_rows


def latest_technical(df: pd.DataFrame, price_map: dict[str, pd.DataFrame], ticker: str) -> dict:
    d = build_rule_frame(ticker, df, price_map)
    row = d.iloc[-1]

    trading_days = int(len(d))
    active = {}
    for col in d.columns:
        if col.startswith("rule_"):
            active[col.replace("rule_", "")] = bool(row.get(col, False))

    return {
        "latest_date": row["date"].date().isoformat(),
        "latest_price": clean_float(row["price"]),
        "trading_days": trading_days,
        "short_history": bool(trading_days < 252),
        "very_short_history": bool(trading_days < 90),
        "new_listing_policy": (
            "Insufficient full-cycle backtest. Can be considered only as watchlist/tiny satellite if momentum, liquidity, "
            "relative strength, news/fundamentals, and market regime all confirm."
            if trading_days < 252 else "Normal backtest evidence available."
        ),
        "ret_1d": clean_float(row.get("ret_1d")),
        "ret_5d": clean_float(row.get("ret_5d")),
        "ret_10d": clean_float(row.get("ret_10d")),
        "ret_20d": clean_float(row.get("ret_20d")),
        "ret_60d": clean_float(row.get("ret_60d")),
        "rsi14": clean_float(row.get("rsi14"), 2),
        "above_ma5": bool(row["price"] > row["ma5"]) if pd.notna(row.get("ma5")) else None,
        "above_ma20": bool(row["price"] > row["ma20"]) if pd.notna(row.get("ma20")) else None,
        "above_ma50": bool(row["price"] > row["ma50"]) if pd.notna(row.get("ma50")) else None,
        "above_ma200": bool(row["price"] > row["ma200"]) if pd.notna(row.get("ma200")) else None,
        "drawdown_from_52w_high": clean_float(row.get("drawdown_from_52w_high")),
        "volatility_20d": clean_float(row.get("volatility_20d")),
        "volatility_60d": clean_float(row.get("volatility_60d")),
        "relative_20d": clean_float(row.get("relative_20d")),
        "relative_60d": clean_float(row.get("relative_60d")),
        "relative_benchmark": choose_relative_benchmark(ticker),
        "active_signals": active,
    }


def corr_90d(price_map: dict[str, pd.DataFrame]) -> dict:
    core = ["QQQ", "SMH", "SPY", "MSFT", "NVDA"]
    core = [x for x in core if x in price_map]
    if len(core) < 3:
        return {}

    merged = None
    for t in core:
        df = price_map[t][["date", "price"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"price": t})
        merged = df if merged is None else merged.merge(df, on="date", how="inner")

    rets = merged.set_index("date").pct_change(fill_method=None).dropna().tail(90)
    return rets.corr().round(4).to_dict()


def evidence_score(row: dict) -> float | None:
    samples = row.get("samples")
    win_rate = row.get("win_rate")
    median_ret = row.get("median_return")
    worst_mae = row.get("worst_mae")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, (int, float))
        or not np.isfinite(float(samples))
        or float(samples) < 0
        or win_rate is None
        or median_ret is None
    ):
        return None

    score = 0.0
    score += min(float(samples), 80) / 80 * 30
    score += max(0, float(win_rate) - 0.5) / 0.3 * 30
    score += max(0, float(median_ret)) / 0.08 * 25
    if worst_mae is not None:
        score += max(0, 0.25 + float(worst_mae)) / 0.25 * 15
    return round(float(min(100, score)), 1)


def rank_rule_evidence(backtests: dict) -> dict:
    ranking = {}
    for ticker, rules in backtests.items():
        rows = []
        for rule_name, horizons in rules.items():
            if rule_name in ["active_signals_latest_day", "latest_date", "latest_price", "trading_days", "short_history", "very_short_history"]:
                continue
            if not isinstance(horizons, dict):
                continue
            res20 = horizons.get("20d", {})
            if not res20:
                continue

            active = rules.get("active_signals_latest_day", {}).get(rule_name, False)
            score = evidence_score(res20)

            rows.append(
                {
                    "ticker": ticker,
                    "rule": rule_name,
                    "samples_20d": res20.get("samples", 0),
                    "samples": res20.get("samples", 0),
                    "valid": res20.get("valid", False),
                    "win_rate_20d": res20.get("win_rate"),
                    "win_rate": res20.get("win_rate"),
                    "median_return_20d": res20.get("median_return"),
                    "median_return": res20.get("median_return"),
                    "worst_mae_20d": res20.get("worst_mae"),
                    "worst_mae": res20.get("worst_mae"),
                    "active_latest_day": active,
                    "evidence_score_0_100": score,
                }
            )
        rows = sorted(
            rows,
            key=lambda x: (
                x["valid"],
                x["active_latest_day"],
                x["evidence_score_0_100"]
                if x["evidence_score_0_100"] is not None
                else float("-inf"),
            ),
            reverse=True,
        )
        ranking[ticker] = rows
    return ranking


def main():
    price_map: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    update_log: dict[str, dict] = {}

    requested = 0
    new_full_downloads = 0

    ordered_tickers = sorted(TICKERS, key=request_priority)

    for ticker in ordered_tickers:
        existing = load_existing(ticker)
        has_cache = not existing.empty
        is_new_full_download = not has_cache

        if requested >= MAX_TIINGO_REQUESTS_PER_RUN:
            if has_cache:
                price_map[ticker] = existing
                update_log[ticker] = {
                    "status": "cache_only_request_cap",
                    "latest_date": pd.to_datetime(existing["date"]).max().date().isoformat(),
                    "total_rows_loaded": int(len(existing)),
                }
                print(f"[CACHE] {ticker}: request cap reached; using cached data")
            else:
                errors[ticker] = "deferred_no_cache_request_cap"
                print(f"[DEFER] {ticker}: no cache and request cap reached")
            continue

        if is_new_full_download and new_full_downloads >= MAX_NEW_FULL_DOWNLOADS_PER_RUN:
            errors[ticker] = "deferred_new_full_download_cap"
            print(f"[DEFER] {ticker}: new full download cap reached")
            continue

        try:
            fetch_start = get_fetch_start_date(ticker)
            new_df = fetch_tiingo(ticker, fetch_start)
            requested += 1
            if is_new_full_download:
                new_full_downloads += 1

            merged = merge_and_save(ticker, new_df)

            price_map[ticker] = merged
            update_log[ticker] = {
                "status": "fresh_from_tiingo",
                "fetch_start": fetch_start,
                "new_rows_downloaded": int(len(new_df)),
                "total_rows_saved": int(len(merged)),
                "latest_date": pd.to_datetime(merged["date"]).max().date().isoformat(),
            }

            print(f"[OK] {ticker}: fetched {len(new_df)} rows from {fetch_start}; saved {len(merged)} total")

        except Exception as e:
            requested += 1
            error_message = redact_sensitive_text(e)
            if has_cache:
                price_map[ticker] = existing
                update_log[ticker] = {
                    "status": "cache_after_fetch_error",
                    "fetch_error": error_message,
                    "latest_date": pd.to_datetime(existing["date"]).max().date().isoformat(),
                    "total_rows_loaded": int(len(existing)),
                }
                errors[ticker] = f"using cached data after fetch error: {error_message}"
                print(f"[CACHE_AFTER_FAIL] {ticker}: {error_message}")
            else:
                errors[ticker] = error_message
                print(f"[FAIL] {ticker}: {error_message}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    benchmarks = prepare_benchmarks(price_map)

    backtests = {}
    all_rows = []
    technicals = {}

    for ticker, df in price_map.items():
        if df.empty or len(df) < 30:
            errors[ticker] = errors.get(ticker, "not enough rows for indicators")
            continue
        try:
            nested, rows = summarize_all_rules(ticker, df, price_map, benchmarks)
            backtests[ticker] = nested
            all_rows.extend(rows)
            technicals[ticker] = latest_technical(df, price_map, ticker)
        except Exception as e:
            errors[ticker] = f"analysis_error: {redact_sensitive_text(e)}"

    market_data_fields = market_data_report_fields(
        price_map,
        "Tiingo Free API with local CSV cache fallback",
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **market_data_fields,
        "update_mode": (
            "large universe, capped Tiingo requests per run, capped new full-history downloads, "
            "cached data used when request cap or Tiingo 429 occurs"
        ),
        "strategy_version": "Eason Master US Market Monitor Cloud Sync v5.1 shared-execution-contract",
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "privacy_mode": "sanitized_public_report_no_cash_no_shares_no_account_value",
        "universe": {
            "configured_ticker_count": len(TICKERS),
            "loaded_ticker_count": len(price_map),
            "fresh_request_count": requested,
            "max_tiingo_requests_per_run": MAX_TIINGO_REQUESTS_PER_RUN,
            "max_new_full_downloads_per_run": MAX_NEW_FULL_DOWNLOADS_PER_RUN,
            "note": "A ticker can remain in the universe even if not refreshed today. ChatGPT should check update_log status and latest_date.",
        },
        "new_listing_policy": {
            "rule": "Short-history stocks are not rejected automatically.",
            "how_to_use": (
                "If a stock has less than 252 trading days, do not treat backtest evidence as reliable. "
                "It can still be recommended only as watchlist or tiny satellite when price action, relative strength, liquidity, "
                "fundamentals/news, and market regime confirm."
            ),
            "position_limit": "Usually 0% to 2% max unless user explicitly accepts high risk and evidence improves.",
        },
        "rule_engine": {
            "rules": [
                "pullback_reclaim_5dma",
                "rsi_oversold_reclaim_40",
                "ma20_reclaim_bullish",
                "ma50_reclaim_bullish",
                "relative_strength_rebound",
                "momentum_leader",
                "failed_rebound_risk",
            ],
            "horizons_days": HORIZONS,
            "minimum_valid_samples": MIN_SAMPLE,
            "sample_independence": "chronological non-overlapping signal windows",
            "execution": {
                "signal_time": "end_of_day_close",
                "entry_time": "next_trading_bar_close",
                "execution_shift_bars": EXECUTION_SHIFT_BARS,
            },
            "cost_assumptions": execution_cost_assumptions(),
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "note": "Samples below minimum are reported but should not be used as primary buy evidence.",
        },
        "backtests": backtests,
        "rule_evidence_ranking": rank_rule_evidence(backtests),
        "technicals": technicals,
        "rolling_90d_correlation": corr_90d(price_map),
        "update_log": update_log,
        "errors": errors,
    }

    report_json = json.dumps(report, indent=2, allow_nan=False)
    (OUT / "market_report.json").write_text(report_json, encoding="utf-8")

    report_dataframe(all_rows, report).to_csv(OUT / "backtest_summary.csv", index=False)

    ranking_rows = []
    for ticker, rows in report["rule_evidence_ranking"].items():
        for row in rows:
            ranking_rows.append({"ticker": ticker, **row})
    report_dataframe(ranking_rows, report).to_csv(OUT / "rule_evidence_ranking.csv", index=False)

    with open(OUT / "index.html", "w", encoding="utf-8") as f:
        f.write(
            "<h1>Eason Quant Cloud Sync</h1>"
            "<p>Sanitized public report. Open market_report.json, latest_summary.json, eason_signal.json, "
            "backtest_summary.csv, or rule_evidence_ranking.csv.</p>"
        )

    print("Saved sanitized docs/market_report.json, docs/backtest_summary.csv, docs/rule_evidence_ranking.csv")


if __name__ == "__main__":
    main()
