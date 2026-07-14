from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.artifact_io import atomic_write_csv, atomic_write_json
from scripts.market_data_contract import extract_market_data_metadata, read_checked_daily_csv

DATA = Path("data")
OUT = Path("docs")
TRADE_LOG = DATA / "trade_log.csv"
MARKET_REPORT = OUT / "market_report.json"
HORIZONS = [3, 10, 20]
VALID_ACTIONS = {"BUY", "ADD", "COVER", "SELL", "TRIM", "REDUCE"}
MIN_COMPLETED_20D_SAMPLE = 20


def clean_float(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        v = float(value)
        if pd.isna(v) or np.isinf(v):
            return None
        return round(v, digits)
    except Exception:
        return None


def csv_path(ticker: str) -> Path:
    safe = ticker.replace("/", "-").replace(".", "-")
    return OUT / f"{safe}_daily.csv"


def load_price(ticker: str) -> pd.DataFrame:
    path = csv_path(ticker)
    if not path.exists():
        return pd.DataFrame()
    df = read_checked_daily_csv(ticker, OUT)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    if "close" not in df.columns:
        raise ValueError(f"{ticker} price cache lacks unadjusted close required for actual fill review")
    price_col = "close"
    out = df[["date", price_col]].rename(columns={price_col: "price"}).copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out.dropna().sort_values("date").reset_index(drop=True)


def load_trade_log() -> pd.DataFrame:
    if not TRADE_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRADE_LOG)
    if df.empty:
        return df
    if "date" not in df.columns or "ticker" not in df.columns or "action" not in df.columns or "fill_price" not in df.columns:
        raise SystemExit("data/trade_log.csv must include date,ticker,action,fill_price columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["fill_price"] = pd.to_numeric(df["fill_price"], errors="coerce")
    df["shares"] = pd.to_numeric(df.get("shares", 0), errors="coerce")
    df["fees"] = pd.to_numeric(df.get("fees", 0), errors="coerce")
    if df[["date", "ticker", "action", "fill_price", "shares", "fees"]].isna().any().any():
        raise SystemExit("data/trade_log.csv contains invalid or incomplete rows")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    invalid_actions = sorted(set(df["action"]) - VALID_ACTIONS)
    if invalid_actions:
        raise SystemExit(f"data/trade_log.csv contains unsupported actions: {invalid_actions}")
    if (~np.isfinite(df["fill_price"]) | (df["fill_price"] <= 0)).any():
        raise SystemExit("data/trade_log.csv contains non-positive or non-finite fill prices")
    if (~np.isfinite(df["shares"]) | (df["shares"] < 0)).any():
        raise SystemExit("data/trade_log.csv contains invalid share quantities")
    if (~np.isfinite(df["fees"]) | (df["fees"] < 0)).any():
        raise SystemExit("data/trade_log.csv contains invalid fees")
    return df.sort_values("date").reset_index(drop=True)


def future_price_on_or_after(prices: pd.DataFrame, trade_date: pd.Timestamp, horizon: int) -> tuple[str | None, float | None]:
    if prices.empty:
        return None, None
    p = prices[prices["date"] >= trade_date].reset_index(drop=True)
    if p.empty:
        return None, None
    target_idx = horizon
    if target_idx >= len(p):
        return None, None
    row = p.iloc[target_idx]
    return row["date"].date().isoformat(), float(row["price"])


def review_trade(row: pd.Series) -> dict:
    ticker = str(row["ticker"]).upper()
    action = str(row["action"]).upper()
    fill = float(row["fill_price"])
    date = pd.to_datetime(row["date"])
    prices = load_price(ticker)

    out = {
        "date": date.date().isoformat(),
        "ticker": ticker,
        "action": action,
        "shares": None,
        "shares_redacted": True,
        "fees": None,
        "fees_redacted": True,
        "fill_price": clean_float(fill, 4),
        "reason": row.get("reason") if pd.notna(row.get("reason")) else None,
        "signal_source": row.get("signal_source") if pd.notna(row.get("signal_source")) else None,
        "backtest_sample_count": clean_float(row.get("backtest_sample_count"), 0),
        "buy_score": clean_float(row.get("buy_score"), 1),
        "sell_risk_score": clean_float(row.get("sell_risk_score"), 1),
        "expected_thesis": row.get("expected_thesis") if pd.notna(row.get("expected_thesis")) else None,
        "invalidation_level": row.get("invalidation_level") if pd.notna(row.get("invalidation_level")) else None,
        "price_basis": "unadjusted_close_matched_to_actual_fill",
        "return_basis": "gross_mark_to_market_before_unknown_future_exit_costs",
    }

    if prices.empty:
        out["review_status"] = "MISSING_PRICE_HISTORY"
        return out

    for horizon in HORIZONS:
        future_date, future_price = future_price_on_or_after(prices, date, horizon)
        out[f"future_date_{horizon}d"] = future_date
        out[f"future_price_{horizon}d"] = clean_float(future_price, 4)
        if future_price is None:
            out[f"outcome_return_{horizon}d_pct"] = None
            out[f"held_return_after_sell_{horizon}d_pct"] = None
            continue

        raw_ret = future_price / fill - 1
        if action in {"BUY", "ADD", "COVER"}:
            out[f"outcome_return_{horizon}d_pct"] = clean_float(raw_ret * 100, 2)
            out[f"held_return_after_sell_{horizon}d_pct"] = None
        elif action in {"SELL", "TRIM", "REDUCE"}:
            out[f"outcome_return_{horizon}d_pct"] = clean_float((-raw_ret) * 100, 2)
            out[f"held_return_after_sell_{horizon}d_pct"] = clean_float(raw_ret * 100, 2)
        else:
            out[f"outcome_return_{horizon}d_pct"] = clean_float(raw_ret * 100, 2)
            out[f"held_return_after_sell_{horizon}d_pct"] = None

    completed = [out.get(f"outcome_return_{h}d_pct") for h in HORIZONS if out.get(f"outcome_return_{h}d_pct") is not None]
    out["review_status"] = "COMPLETED" if completed else "WAITING_FOR_FUTURE_DATA"
    if completed:
        out["avg_completed_outcome_pct"] = clean_float(np.mean(completed), 2)
    return out


def summarize_reviews(reviews: pd.DataFrame) -> dict:
    if reviews.empty:
        return {
            "trade_count": 0,
            "message": "No real trades logged yet. Fill data/trade_log.csv after actual IBKR executions if you want 3/10/20-day review.",
        }
    out = {"trade_count": int(len(reviews))}
    for horizon in HORIZONS:
        col = f"outcome_return_{horizon}d_pct"
        vals = pd.to_numeric(reviews.get(col), errors="coerce").dropna()
        out[f"completed_{horizon}d_count"] = int(len(vals))
        if len(vals):
            out[f"avg_{horizon}d_outcome_pct"] = clean_float(vals.mean(), 2)
            out[f"median_{horizon}d_outcome_pct"] = clean_float(vals.median(), 2)
            out[f"win_rate_{horizon}d_pct"] = clean_float((vals > 0).mean() * 100, 2)
            out[f"worst_{horizon}d_outcome_pct"] = clean_float(vals.min(), 2)
            out[f"best_{horizon}d_outcome_pct"] = clean_float(vals.max(), 2)
    if "action" in reviews.columns:
        out["actions"] = {str(k): int(v) for k, v in reviews["action"].value_counts().to_dict().items()}
    if "ticker" in reviews.columns:
        out["tickers"] = {str(k): int(v) for k, v in reviews["ticker"].value_counts().to_dict().items()}
    return out


def actual_vs_backtest(reviews: pd.DataFrame) -> dict:
    if reviews.empty:
        return {
            "available": False,
            "reason": "No actual trades logged yet.",
        }
    vals20 = pd.to_numeric(reviews.get("outcome_return_20d_pct"), errors="coerce").dropna()
    if vals20.empty:
        return {
            "available": False,
            "reason": "No completed 20-trading-day outcomes yet.",
            "logged_trade_count": int(len(reviews)),
        }
    if len(vals20) < MIN_COMPLETED_20D_SAMPLE:
        return {
            "available": False,
            "reason": "Insufficient completed 20-trading-day sample for actual-vs-backtest inference.",
            "completed_20d_trade_count": int(len(vals20)),
            "minimum_completed_20d_sample": MIN_COMPLETED_20D_SAMPLE,
        }
    return {
        "available": True,
        "completed_20d_trade_count": int(len(vals20)),
        "minimum_completed_20d_sample": MIN_COMPLETED_20D_SAMPLE,
        "actual_avg_20d_outcome_pct": clean_float(vals20.mean(), 2),
        "actual_median_20d_outcome_pct": clean_float(vals20.median(), 2),
        "actual_20d_win_rate_pct": clean_float((vals20 > 0).mean() * 100, 2),
        "important_limit": "This compares actual logged outcomes only. It does not infer whether unlogged skipped signals would have worked.",
    }


def review_metadata() -> dict[str, Any]:
    if not MARKET_REPORT.exists() or MARKET_REPORT.stat().st_size == 0:
        raise SystemExit("Missing docs/market_report.json; trade review metadata cannot be verified")
    try:
        report = json.loads(MARKET_REPORT.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid docs/market_report.json: {exc}") from exc
    metadata = extract_market_data_metadata(report)
    if metadata.get("metadata_status") != "COMPLETE":
        raise SystemExit("Market report metadata is incomplete; trade review was not published")
    return {
        "data_source": metadata["source"],
        "market_timezone": metadata["market_timezone"],
        "data_timestamp": metadata["data_timestamp"],
        "data_timestamp_granularity": metadata["data_timestamp_granularity"],
        "data_timestamp_status": metadata["data_timestamp_status"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
    }


def main() -> None:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    metadata = review_metadata()
    trades = load_trade_log()
    reviews = pd.DataFrame([review_trade(row) for _, row in trades.iterrows()]) if not trades.empty else pd.DataFrame()
    actual = actual_vs_backtest(reviews)

    summary = {
        "generated_at_utc": generated_at_utc,
        **metadata,
        "version": "trade-review-v3.1-unadjusted-redacted",
        "review_price_basis": "unadjusted_close_matched_to_actual_fill",
        "purpose": "Review actual logged trades at 3/10/20 trading-day horizons and compare real execution with expected thesis.",
        "privacy_warning": "Public outputs always redact exact shares and fees. Broker/account truth belongs only in the local private IBKR layer.",
        "trade_log_path": str(TRADE_LOG),
        "summary": summarize_reviews(reviews),
        "actual_vs_backtest": actual,
    }

    for field, value in {
        "data_source": metadata["data_source"],
        "market_timezone": metadata["market_timezone"],
        "report_generated_at_utc": generated_at_utc,
        "data_timestamp": metadata["data_timestamp"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
    }.items():
        reviews[field] = value
    atomic_write_csv(OUT / "trade_review.csv", reviews)
    atomic_write_json(OUT / "trade_review.json", summary)
    atomic_write_json(
        OUT / "actual_vs_backtest.json",
        {
            "generated_at_utc": generated_at_utc,
            **metadata,
            "review_price_basis": "unadjusted_close_matched_to_actual_fill",
            **actual,
        },
    )

    print("Saved docs/trade_review.json, docs/trade_review.csv, docs/actual_vs_backtest.json")


if __name__ == "__main__":
    main()
