from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_data_contract import read_checked_daily_csv

OUT = Path("docs")
INITIAL_CASH = 10000.0
FEES = 0.0005
SLIPPAGE = 0.0005

# Signals are calculated from end-of-day close data. To avoid same-close look-ahead,
# vectorbt execution uses next bar by shifting entries/exits by one trading day.
EXECUTION_SHIFT_DAYS = 1

VALIDATION_TICKERS = [
    "SPY", "QQQ", "SMH", "SOXX", "MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "META", "AVGO",
    "AMD", "ASML", "TSM", "PLTR", "CRWD", "PANW", "VRT", "CEG", "GLD", "TLT", "SGOV",
]


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
    out = df[["date", price_col]].rename(columns={price_col: ticker}).copy()
    out[ticker] = pd.to_numeric(out[ticker], errors="coerce")
    return out.dropna().sort_values("date").drop_duplicates("date", keep="last")


def load_price_matrix() -> pd.DataFrame:
    merged = None
    for ticker in VALIDATION_TICKERS:
        df = load_price(ticker)
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    if merged is None:
        raise SystemExit("No price data found for vectorbt validation")
    merged = merged.sort_values("date").set_index("date")
    merged = merged.ffill().dropna(axis=1, how="all").dropna(axis=0, how="all")
    return merged


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))


def make_rules(close: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)
    rsi14 = rsi(close)

    rules = {}

    entries = (ret5 <= -0.04) & (close > ma5)
    exits = (close < ma20) | (close < ma5)
    rules["pullback_reclaim_5dma"] = (entries.fillna(False), exits.fillna(False))

    entries = (rsi14.rolling(5).min().shift(1) < 35) & (rsi14 >= 40) & (close > ma5)
    exits = (close < ma20) | (rsi14 > 70)
    rules["rsi_oversold_reclaim_40"] = (entries.fillna(False), exits.fillna(False))

    entries = (close.shift(1) < ma20.shift(1)) & (close > ma20) & (close > ma50)
    exits = close < ma20
    rules["ma20_reclaim_bullish"] = (entries.fillna(False), exits.fillna(False))

    entries = (close.shift(1) < ma50.shift(1)) & (close > ma50) & (close > ma200)
    exits = close < ma50
    rules["ma50_reclaim_bullish"] = (entries.fillna(False), exits.fillna(False))

    entries = (close > ma20) & (close > ma50) & (ret20 > 0.05) & (rsi14 < 78)
    exits = (close < ma20) | (rsi14 > 82)
    rules["momentum_leader"] = (entries.fillna(False), exits.fillna(False))

    return rules


def execution_signals(entries: pd.DataFrame, exits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries_exec = entries.shift(EXECUTION_SHIFT_DAYS).fillna(False).astype(bool)
    exits_exec = exits.shift(EXECUTION_SHIFT_DAYS).fillna(False).astype(bool)
    return entries_exec, exits_exec


def to_series_dict(x: Any) -> dict[str, Any]:
    if isinstance(x, pd.Series):
        return {str(k): clean_float(v, 6) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return {str(k): clean_float(v, 6) for k, v in x.iloc[-1].items()}
    return {"ALL": clean_float(x, 6)}


def build_rows(rule_name: str, close: pd.DataFrame, raw_entries: pd.DataFrame, raw_exits: pd.DataFrame) -> list[dict]:
    entries, exits = execution_signals(raw_entries, raw_exits)
    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=INITIAL_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )

    total_return = to_series_dict(pf.total_return())
    max_drawdown = to_series_dict(pf.max_drawdown())
    sharpe = to_series_dict(pf.sharpe_ratio())
    final_value = to_series_dict(pf.final_value())

    try:
        trade_count = to_series_dict(pf.trades.count())
    except Exception:
        trade_count = {ticker: None for ticker in close.columns}

    rows = []
    for ticker in close.columns:
        rows.append(
            {
                "ticker": ticker,
                "rule": rule_name,
                "engine": "vectorbt",
                "execution_assumption": "signals are shifted 1 trading day to avoid same-close look-ahead",
                "vectorbt_total_return_pct": clean_float((total_return.get(ticker) or 0) * 100, 2),
                "vectorbt_max_drawdown_pct": clean_float((max_drawdown.get(ticker) or 0) * 100, 2),
                "vectorbt_sharpe": clean_float(sharpe.get(ticker), 3),
                "vectorbt_final_value": clean_float(final_value.get(ticker), 2),
                "vectorbt_trade_count": clean_float(trade_count.get(ticker), 0),
                "raw_entry_signal_count": int(raw_entries[ticker].sum()) if ticker in raw_entries else 0,
                "executed_entry_count": int(entries[ticker].sum()) if ticker in entries else 0,
                "raw_exit_signal_count": int(raw_exits[ticker].sum()) if ticker in raw_exits else 0,
                "executed_exit_count": int(exits[ticker].sum()) if ticker in exits else 0,
            }
        )
    return rows


def main() -> None:
    close = load_price_matrix()
    rules = make_rules(close)
    all_rows: list[dict] = []
    errors = {}

    for rule_name, (entries, exits) in rules.items():
        try:
            all_rows.extend(build_rows(rule_name, close, entries, exits))
        except Exception as exc:
            errors[rule_name] = str(exc)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["rule", "vectorbt_sharpe", "vectorbt_total_return_pct"], ascending=[True, False, False])

    summary = {
        "available": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "vectorbt-validation-v3.6-next-bar-execution",
        "purpose": "Independent vectorbt validation layer for signal rules. It does not replace pandas portfolio backtest or ChatGPT live review.",
        "vectorbt_version": getattr(vbt, "__version__", None),
        "execution_assumption": "End-of-day signals are executed on the next bar by shifting entries/exits 1 trading day.",
        "data": {
            "loaded_tickers": list(close.columns),
            "start_date": close.index.min().date().isoformat(),
            "end_date": close.index.max().date().isoformat(),
            "rows": int(len(close)),
        },
        "assumptions": {
            "init_cash_per_column": INITIAL_CASH,
            "fees": FEES,
            "slippage": SLIPPAGE,
            "freq": "1D",
            "execution_shift_days": EXECUTION_SHIFT_DAYS,
            "meaning": "Each ticker/rule column is independently backtested, not a real combined portfolio allocation.",
        },
        "top_by_sharpe": df.head(25).to_dict(orient="records") if not df.empty else [],
        "errors": errors,
        "important_limit": "Vectorbt results are a fast validation/audit layer. Final trade decisions still require portfolio-level backtest, walk-forward stability, live market/news checks, real account constraints, and human confirmation.",
    }

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "vectorbt_signal_stats.csv", index=False)
    with open(OUT / "vectorbt_validation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/vectorbt_validation.json and docs/vectorbt_signal_stats.csv")


if __name__ == "__main__":
    main()
