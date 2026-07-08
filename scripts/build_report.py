import os
import time
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd

from config import TICKERS, START_DATE

API_KEY = os.getenv("TIINGO_API_KEY")
if not API_KEY:
    raise SystemExit("Missing TIINGO_API_KEY GitHub Secret.")

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

HORIZONS = [5, 10, 20, 60]
MIN_SAMPLE = 20


def csv_path(ticker: str) -> Path:
    return OUT / f"{ticker}_daily.csv"


def normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker

    price_col = "adjClose" if "adjClose" in df.columns else "close"
    low_col = "adjLow" if "adjLow" in df.columns else "low"
    high_col = "adjHigh" if "adjHigh" in df.columns else "high"

    df["price"] = df[price_col]
    df["low_price"] = df[low_col]
    df["high_price"] = df[high_col]

    return df.sort_values("date").reset_index(drop=True)


def get_fetch_start_date(ticker: str) -> str:
    path = csv_path(ticker)
    if not path.exists():
        return START_DATE

    try:
        old = pd.read_csv(path, usecols=["date"])
        if old.empty:
            return START_DATE

        last_date = pd.to_datetime(old["date"]).max().date()
        start = last_date - timedelta(days=7)
        min_start = pd.to_datetime(START_DATE).date()
        if start < min_start:
            start = min_start
        return start.isoformat()
    except Exception:
        return START_DATE


def fetch_tiingo(ticker: str, start_date: str) -> pd.DataFrame:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start_date,
        "resampleFreq": "daily",
        "token": API_KEY,
    }

    r = requests.get(url, params=params, timeout=45)
    if r.status_code != 200:
        raise RuntimeError(f"{ticker} HTTP {r.status_code}: {r.text[:200]}")

    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError(f"{ticker}: empty data from {start_date}")

    return normalize(df, ticker)


def load_existing(ticker: str) -> pd.DataFrame:
    path = csv_path(ticker)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["ret_1d"] = d["price"].pct_change(1)
    d["ret_5d"] = d["price"].pct_change(5)
    d["ret_10d"] = d["price"].pct_change(10)
    d["ret_20d"] = d["price"].pct_change(20)

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

    return d


def prepare_benchmarks(price_map: dict) -> dict:
    benches = {}
    for ticker in ["QQQ", "SPY", "SMH"]:
        if ticker in price_map:
            benches[ticker] = add_indicators(price_map[ticker])[["date", "price"]].rename(
                columns={"price": f"{ticker}_price"}
            )
    return benches


def choose_relative_benchmark(ticker: str) -> str:
    if ticker in ["QQQ", "SPY"]:
        return "SPY" if ticker == "QQQ" else "QQQ"
    if ticker in ["SMH", "SOXX", "SOXQ", "NVDA", "AVGO", "AMD", "ASML", "TSM", "MU"]:
        return "QQQ"
    return "SPY"


def build_rule_frame(ticker: str, df: pd.DataFrame, price_map: dict) -> pd.DataFrame:
    d = add_indicators(df)

    rel_bench = choose_relative_benchmark(ticker)
    if rel_bench in price_map:
        b = add_indicators(price_map[rel_bench])[["date", "ret_20d"]].rename(
            columns={"ret_20d": "bench_ret_20d"}
        )
        d = d.merge(b, on="date", how="left")
        d["relative_20d"] = d["ret_20d"] - d["bench_ret_20d"]
    else:
        d["relative_20d"] = np.nan

    d["rule_pullback_reclaim_5dma"] = (d["ret_5d"] <= -0.04) & (d["price"] > d["ma5"])

    d["rule_rsi_oversold_reclaim_40"] = (
        (d["rsi14"].rolling(5).min().shift(1) < 35)
        & (d["rsi14"] >= 40)
        & (d["price"] > d["ma5"])
    )

    d["rule_ma20_reclaim_bullish"] = (
        (d["price"].shift(1) < d["ma20"].shift(1))
        & (d["price"] > d["ma20"])
        & (d["price"] > d["ma50"])
        & (d["ma20_slope_10d"] > -0.01)
    )

    d["rule_ma50_reclaim_bullish"] = (
        (d["price"].shift(1) < d["ma50"].shift(1))
        & (d["price"] > d["ma50"])
        & (d["price"] > d["ma200"])
    )

    d["rule_relative_strength_rebound"] = (
        (d["ret_5d"] <= -0.035)
        & (d["price"] > d["ma5"])
        & (d["relative_20d"] > 0)
    )

    d["rule_failed_rebound_risk"] = (
        (d["price"] < d["ma5"])
        & (d["price"] < d["price"].rolling(10).min().shift(1))
    )

    return d


def max_adverse_excursion(df: pd.DataFrame, horizon: int) -> pd.Series:
    maes = []
    for i in range(len(df)):
        if i + horizon >= len(df):
            maes.append(np.nan)
        else:
            future_min = df.loc[i + 1 : i + horizon, "low_price"].min()
            maes.append(future_min / df.loc[i, "price"] - 1)
    return pd.Series(maes, index=df.index)


def attach_benchmark_forward_returns(df: pd.DataFrame, benchmarks: dict, horizon: int) -> pd.DataFrame:
    out = df[["date"]].copy()
    for bench_name, bench_df in benchmarks.items():
        b = bench_df.copy()
        price_col = f"{bench_name}_price"
        b[f"{bench_name}_fwd_{horizon}d"] = b[price_col].shift(-horizon) / b[price_col] - 1
        out = out.merge(b[["date", f"{bench_name}_fwd_{horizon}d"]], on="date", how="left")
    return out


def summarize_rule(
    ticker: str,
    df: pd.DataFrame,
    rule_col: str,
    benchmarks: dict,
    horizon: int,
) -> dict:
    d = df.copy()
    d[f"fwd_{horizon}d"] = d["price"].shift(-horizon) / d["price"] - 1
    d[f"mae_{horizon}d"] = max_adverse_excursion(d, horizon)

    b_fwd = attach_benchmark_forward_returns(d, benchmarks, horizon)
    d = d.merge(b_fwd, on="date", how="left")

    sig = d[d[rule_col]].dropna(subset=[f"fwd_{horizon}d", f"mae_{horizon}d"])

    result = {
        "ticker": ticker,
        "rule": rule_col.replace("rule_", ""),
        "horizon_days": horizon,
        "samples": int(len(sig)),
        "valid": bool(len(sig) >= MIN_SAMPLE),
        "min_sample_required": MIN_SAMPLE,
    }

    if len(sig) == 0:
        result["reason"] = "no completed signal samples"
        return result

    fwd = sig[f"fwd_{horizon}d"]
    mae = sig[f"mae_{horizon}d"]

    result.update(
        {
            "win_rate": round(float((fwd > 0).mean()), 4),
            "avg_return": round(float(fwd.mean()), 4),
            "median_return": round(float(fwd.median()), 4),
            "worst_return": round(float(fwd.min()), 4),
            "best_return": round(float(fwd.max()), 4),
            "avg_mae": round(float(mae.mean()), 4),
            "worst_mae": round(float(mae.min()), 4),
            "lookback_start": d["date"].min().date().isoformat(),
            "lookback_end": d["date"].max().date().isoformat(),
        }
    )

    for bench_name in ["QQQ", "SPY", "SMH"]:
        col = f"{bench_name}_fwd_{horizon}d"
        if col in sig.columns and sig[col].notna().any():
            bench_avg = float(sig[col].mean())
            result[f"avg_{bench_name}_same_dates"] = round(bench_avg, 4)
            result[f"avg_alpha_vs_{bench_name}"] = round(float(fwd.mean() - bench_avg), 4)

    return result


def summarize_all_rules(ticker: str, df: pd.DataFrame, price_map: dict, benchmarks: dict) -> tuple[dict, list]:
    rule_frame = build_rule_frame(ticker, df, price_map)
    rule_cols = [
        "rule_pullback_reclaim_5dma",
        "rule_rsi_oversold_reclaim_40",
        "rule_ma20_reclaim_bullish",
        "rule_ma50_reclaim_bullish",
        "rule_relative_strength_rebound",
        "rule_failed_rebound_risk",
    ]

    all_rows = []
    nested = {}

    for rule_col in rule_cols:
        nested[rule_col.replace("rule_", "")] = {}
        for horizon in HORIZONS:
            res = summarize_rule(ticker, rule_frame, rule_col, benchmarks, horizon)
            nested[rule_col.replace("rule_", "")][f"{horizon}d"] = res
            all_rows.append(res)

    active = {}
    latest = rule_frame.iloc[-1]
    for rule_col in rule_cols:
        active[rule_col.replace("rule_", "")] = bool(latest.get(rule_col, False))

    nested["active_signals_latest_day"] = active
    nested["latest_date"] = latest["date"].date().isoformat()
    nested["latest_price"] = round(float(latest["price"]), 4)

    return nested, all_rows


def latest_technical(df: pd.DataFrame, price_map: dict, ticker: str) -> dict:
    d = build_rule_frame(ticker, df, price_map)
    row = d.iloc[-1]

    return {
        "latest_date": row["date"].date().isoformat(),
        "latest_price": round(float(row["price"]), 4),
        "ret_1d": round(float(row["ret_1d"]), 4) if pd.notna(row["ret_1d"]) else None,
        "ret_5d": round(float(row["ret_5d"]), 4) if pd.notna(row["ret_5d"]) else None,
        "ret_20d": round(float(row["ret_20d"]), 4) if pd.notna(row["ret_20d"]) else None,
        "rsi14": round(float(row["rsi14"]), 2) if pd.notna(row["rsi14"]) else None,
        "above_ma5": bool(row["price"] > row["ma5"]) if pd.notna(row["ma5"]) else None,
        "above_ma20": bool(row["price"] > row["ma20"]) if pd.notna(row["ma20"]) else None,
        "above_ma50": bool(row["price"] > row["ma50"]) if pd.notna(row["ma50"]) else None,
        "above_ma200": bool(row["price"] > row["ma200"]) if pd.notna(row["ma200"]) else None,
        "relative_20d": round(float(row["relative_20d"]), 4) if pd.notna(row["relative_20d"]) else None,
        "relative_benchmark": choose_relative_benchmark(ticker),
    }


def corr_90d(price_map: dict) -> dict:
    core = ["QQQ", "SMH", "SPY", "MSFT"]
    if not all(x in price_map for x in core):
        return {}

    merged = None
    for t in core:
        df = price_map[t][["date", "price"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"price": t})
        merged = df if merged is None else merged.merge(df, on="date", how="inner")

    rets = merged.set_index("date").pct_change().dropna().tail(90)
    return rets.corr().round(4).to_dict()


def rank_rule_evidence(backtests: dict) -> dict:
    ranking = {}
    for ticker, rules in backtests.items():
        rows = []
        for rule_name, horizons in rules.items():
            if rule_name in ["active_signals_latest_day", "latest_date", "latest_price"]:
                continue
            res20 = horizons.get("20d", {})
            if not res20:
                continue

            samples = res20.get("samples", 0)
            valid = res20.get("valid", False)
            median_ret = res20.get("median_return", None)
            win_rate = res20.get("win_rate", None)
            worst_mae = res20.get("worst_mae", None)
            active = rules.get("active_signals_latest_day", {}).get(rule_name, False)

            if median_ret is None or win_rate is None:
                evidence_score = 0
            else:
                evidence_score = 0
                evidence_score += min(samples, 80) / 80 * 30
                evidence_score += max(0, win_rate - 0.5) / 0.3 * 30
                evidence_score += max(0, median_ret) / 0.08 * 25
                if worst_mae is not None:
                    evidence_score += max(0, 0.25 + worst_mae) / 0.25 * 15
                evidence_score = round(float(min(100, evidence_score)), 1)

            rows.append(
                {
                    "rule": rule_name,
                    "samples_20d": samples,
                    "valid": valid,
                    "win_rate_20d": win_rate,
                    "median_return_20d": median_ret,
                    "worst_mae_20d": worst_mae,
                    "active_latest_day": active,
                    "evidence_score_0_100": evidence_score,
                }
            )
        rows = sorted(rows, key=lambda x: (x["valid"], x["evidence_score_0_100"]), reverse=True)
        ranking[ticker] = rows
    return ranking


def main():
    price_map = {}
    errors = {}
    update_log = {}

    for ticker in TICKERS:
        try:
            fetch_start = get_fetch_start_date(ticker)
            new_df = fetch_tiingo(ticker, fetch_start)
            merged = merge_and_save(ticker, new_df)

            price_map[ticker] = merged
            update_log[ticker] = {
                "fetch_start": fetch_start,
                "new_rows_downloaded": int(len(new_df)),
                "total_rows_saved": int(len(merged)),
                "latest_date": pd.to_datetime(merged["date"]).max().date().isoformat(),
            }

            print(f"[OK] {ticker}: fetched {len(new_df)} rows from {fetch_start}; saved {len(merged)} total")

        except Exception as e:
            errors[ticker] = str(e)
            print(f"[FAIL] {ticker}: {e}")

        time.sleep(0.8)

    benchmarks = prepare_benchmarks(price_map)

    backtests = {}
    all_rows = []
    technicals = {}

    for ticker, df in price_map.items():
        nested, rows = summarize_all_rules(ticker, df, price_map, benchmarks)
        backtests[ticker] = nested
        all_rows.extend(rows)
        technicals[ticker] = latest_technical(df, price_map, ticker)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Tiingo Free API, adjusted daily prices when available",
        "update_mode": "incremental with 7-day overlap; first run downloads full history",
        "strategy_version": "Eason Master US Market Monitor Cloud Sync v3.0 multi-rule sanitized",
        "privacy_mode": "sanitized_public_report_no_cash_no_shares_no_account_value",
        "rule_engine": {
            "rules": [
                "pullback_reclaim_5dma",
                "rsi_oversold_reclaim_40",
                "ma20_reclaim_bullish",
                "ma50_reclaim_bullish",
                "relative_strength_rebound",
                "failed_rebound_risk",
            ],
            "horizons_days": HORIZONS,
            "minimum_valid_samples": MIN_SAMPLE,
            "note": "Samples below minimum are reported but should not be used as primary buy evidence.",
        },
        "backtests": backtests,
        "rule_evidence_ranking": rank_rule_evidence(backtests),
        "technicals": technicals,
        "rolling_90d_correlation": corr_90d(price_map),
        "update_log": update_log,
        "errors": errors,
    }

    with open(OUT / "market_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(all_rows).to_csv(OUT / "backtest_summary.csv", index=False)

    ranking_rows = []
    for ticker, rows in report["rule_evidence_ranking"].items():
        for row in rows:
            ranking_rows.append({"ticker": ticker, **row})
    pd.DataFrame(ranking_rows).to_csv(OUT / "rule_evidence_ranking.csv", index=False)

    with open(OUT / "index.html", "w", encoding="utf-8") as f:
        f.write(
            "<h1>Eason Quant Cloud Sync</h1>"
            "<p>Sanitized public report. Open market_report.json, backtest_summary.csv, or rule_evidence_ranking.csv.</p>"
        )

    print("Saved sanitized docs/market_report.json, docs/backtest_summary.csv, docs/rule_evidence_ranking.csv")


if __name__ == "__main__":
    main()
