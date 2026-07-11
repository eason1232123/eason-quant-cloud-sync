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

from scripts.market_data_contract import read_checked_daily_csv

DATA = Path("data")
OUT = Path("docs")
TRADE_LOG = DATA / "trade_log.csv"
HORIZONS = [3, 10, 20]


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
    price_col = "price" if "price" in df.columns else "adjClose" if "adjClose" in df.columns else "close"
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
    df["shares"] = pd.to_numeric(df.get("shares", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["date", "ticker", "action", "fill_price"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
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
        "shares": clean_float(row.get("shares"), 6),
        "fill_price": clean_float(fill, 4),
        "reason": row.get("reason") if pd.notna(row.get("reason")) else None,
        "signal_source": row.get("signal_source") if pd.notna(row.get("signal_source")) else None,
        "backtest_sample_count": clean_float(row.get("backtest_sample_count"), 0),
        "buy_score": clean_float(row.get("buy_score"), 1),
        "sell_risk_score": clean_float(row.get("sell_risk_score"), 1),
        "expected_thesis": row.get("expected_thesis") if pd.notna(row.get("expected_thesis")) else None,
        "invalidation_level": row.get("invalidation_level") if pd.notna(row.get("invalidation_level")) else None,
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
    return {
        "available": True,
        "completed_20d_trade_count": int(len(vals20)),
        "actual_avg_20d_outcome_pct": clean_float(vals20.mean(), 2),
        "actual_median_20d_outcome_pct": clean_float(vals20.median(), 2),
        "actual_20d_win_rate_pct": clean_float((vals20 > 0).mean() * 100, 2),
        "important_limit": "This compares actual logged outcomes only. It does not infer whether unlogged skipped signals would have worked.",
    }


def main() -> None:
    trades = load_trade_log()
    reviews = pd.DataFrame([review_trade(row) for _, row in trades.iterrows()]) if not trades.empty else pd.DataFrame()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "trade-review-v3.0",
        "purpose": "Review actual logged trades at 3/10/20 trading-day horizons and compare real execution with expected thesis.",
        "privacy_warning": "This repo may be public. Do not log sensitive account details unless you accept they may be public.",
        "trade_log_path": str(TRADE_LOG),
        "summary": summarize_reviews(reviews),
        "actual_vs_backtest": actual_vs_backtest(reviews),
    }

    reviews.to_csv(OUT / "trade_review.csv", index=False)
    with open(OUT / "trade_review.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(OUT / "actual_vs_backtest.json", "w", encoding="utf-8") as f:
        json.dump(summary["actual_vs_backtest"], f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/trade_review.json, docs/trade_review.csv, docs/actual_vs_backtest.json")


if __name__ == "__main__":
    main()
