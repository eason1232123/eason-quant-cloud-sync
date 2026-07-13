from __future__ import annotations

import importlib
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

from scripts.market_clock import MARKET_TIMEZONE, latest_completed_us_market_weekday
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
    read_checked_daily_csv,
)
from scripts.strategy_contract import (
    COMMISSION_RATE,
    ENTRY_RULE_SPECS,
    EXECUTION_IMPACT_RATE,
    EXECUTION_SHIFT_BARS,
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    benchmark_for_ticker,
    execution_cost_assumptions,
    exit_signals,
    relative_20d_frame,
    relative_20d_series,
    rule_signals,
    shift_for_execution,
)

OUT = Path("docs")
INITIAL_CASH = 10000.0
vbt: Any | None = None

VALIDATION_TICKERS = [
    "SPY", "QQQ", "SMH", "SOXX", "MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "META", "AVGO",
    "AMD", "ASML", "TSM", "PLTR", "CRWD", "PANW", "VRT", "CEG", "GLD", "TLT", "SGOV",
]


def vectorbt_module() -> Any:
    global vbt
    if vbt is None:
        vbt = importlib.import_module("vectorbt")
    return vbt


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
    if out["date"].duplicated().any():
        raise ValueError(f"{ticker} price cache contains duplicate market dates")
    invalid = out[ticker].isna() | ~np.isfinite(out[ticker]) | (out[ticker] <= 0)
    if invalid.any():
        raise ValueError(f"{ticker} price cache contains invalid close prices")
    out = out.sort_values("date").reset_index(drop=True)
    out.attrs["price_basis"] = "adjusted" if "adjClose" in df.columns else "unadjusted"
    return out


def load_native_price_map() -> tuple[dict[str, pd.Series], dict[str, str], list[str]]:
    prices: dict[str, pd.Series] = {}
    price_basis: dict[str, str] = {}
    missing: list[str] = []
    for ticker in VALIDATION_TICKERS:
        frame = load_price(ticker)
        if frame.empty:
            missing.append(ticker)
            continue
        prices[ticker] = frame.set_index("date")[ticker].astype(float)
        price_basis[ticker] = str(frame.attrs.get("price_basis") or "unknown")
    return prices, price_basis, missing


def load_price_matrix() -> pd.DataFrame:
    merged = None
    for ticker in VALIDATION_TICKERS:
        df = load_price(ticker)
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    if merged is None:
        raise SystemExit("No price data found for vectorbt validation")
    # This compatibility view intentionally preserves gaps. Forward filling would
    # manufacture bars for suspended, missing, or stale tickers.
    return merged.sort_values("date").set_index("date").dropna(axis=1, how="all").dropna(axis=0, how="all")


def make_rules(
    close: pd.Series | pd.DataFrame,
    relative_20d: pd.Series | pd.DataFrame | None = None,
) -> dict[str, tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]]:
    if relative_20d is None and isinstance(close, pd.DataFrame):
        relative_20d = relative_20d_frame(close)
    entries = rule_signals(close, relative_20d)
    exits = exit_signals(close)
    return {
        rule_name: (
            entries[rule_name],
            exits[spec["exit_rule"]],
        )
        for rule_name, spec in ENTRY_RULE_SPECS.items()
    }


def execution_signals(entries: pd.DataFrame, exits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return shift_for_execution(entries), shift_for_execution(exits)


def percent_metric(values: dict[str, Any], ticker: str, digits: int = 2) -> float | None:
    value = values.get(ticker)
    return clean_float(float(value) * 100, digits) if value is not None else None


def to_series_dict(x: Any) -> dict[str, Any]:
    if isinstance(x, pd.Series):
        return {str(k): clean_float(v, 6) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return {str(k): clean_float(v, 6) for k, v in x.iloc[-1].items()}
    return {"ALL": clean_float(x, 6)}


def build_rows(rule_name: str, close: pd.DataFrame, raw_entries: pd.DataFrame, raw_exits: pd.DataFrame) -> list[dict]:
    entries, exits = execution_signals(raw_entries, raw_exits)
    pf = vectorbt_module().Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=INITIAL_CASH,
        fees=COMMISSION_RATE,
        slippage=EXECUTION_IMPACT_RATE,
        upon_long_conflict="exit",
        freq="1D",
    )

    total_return = to_series_dict(pf.total_return())
    max_drawdown = to_series_dict(pf.max_drawdown())
    sharpe = to_series_dict(pf.sharpe_ratio())
    final_value = to_series_dict(pf.final_value())

    trade_count = to_series_dict(pf.trades.count())
    closed_trade_count = to_series_dict(pf.trades.closed.count())
    open_trade_count = to_series_dict(pf.trades.open.count())

    rows = []
    for ticker in close.columns:
        rows.append(
            {
                "ticker": ticker,
                "rule": rule_name,
                "engine": "vectorbt",
                "execution_assumption": "end-of-day signal executed at the next trading bar close",
                "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
                "rule_fingerprint": RULE_FINGERPRINT,
                "strategy_fingerprint": STRATEGY_FINGERPRINT,
                "cost_assumption_source": "configured_not_observed",
                "vectorbt_total_return_pct": percent_metric(total_return, ticker),
                "vectorbt_max_drawdown_pct": percent_metric(max_drawdown, ticker),
                "vectorbt_sharpe": clean_float(sharpe.get(ticker), 3),
                "vectorbt_final_value": clean_float(final_value.get(ticker), 2),
                "vectorbt_trade_count": clean_float(trade_count.get(ticker), 0),
                "vectorbt_closed_trade_count": clean_float(closed_trade_count.get(ticker), 0),
                "vectorbt_open_trade_count": clean_float(open_trade_count.get(ticker), 0),
                "raw_entry_signal_count": int(raw_entries[ticker].sum()) if ticker in raw_entries else 0,
                "shifted_entry_signal_count": int(entries[ticker].sum()) if ticker in entries else 0,
                "executed_entry_count": clean_float(trade_count.get(ticker), 0),
                "raw_exit_signal_count": int(raw_exits[ticker].sum()) if ticker in raw_exits else 0,
                "shifted_exit_signal_count": int(exits[ticker].sum()) if ticker in exits else 0,
                "executed_exit_count": clean_float(closed_trade_count.get(ticker), 0),
            }
        )
    return rows


def main() -> None:
    engine = vectorbt_module()
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    prices, price_basis_by_ticker, missing_tickers = load_native_price_map()
    if not prices:
        raise SystemExit("No price data found for vectorbt validation")
    all_rows: list[dict] = []
    errors: dict[str, str] = {}

    for ticker, native_close in prices.items():
        benchmark = benchmark_for_ticker(ticker)
        benchmark_close = prices.get(benchmark)
        if benchmark_close is None:
            errors[f"{ticker}/benchmark"] = f"required benchmark {benchmark} is missing"
        relative = relative_20d_series(native_close, benchmark_close)
        close = native_close.to_frame(ticker)
        rules = make_rules(close, relative.to_frame(ticker))
        for rule_name, (entries, exits) in rules.items():
            try:
                all_rows.extend(build_rows(rule_name, close, entries, exits))
            except Exception as exc:
                errors[f"{ticker}/{rule_name}"] = f"{type(exc).__name__}: {exc}"

    df = pd.DataFrame(all_rows)
    if not df.empty:
        data_timestamp_by_ticker = {
            ticker: series.index.max().date().isoformat()
            for ticker, series in sorted(prices.items())
        }
        df["data_source"] = "Tiingo daily local CSV cache"
        df["market_timezone"] = MARKET_TIMEZONE
        df["report_generated_at_utc"] = generated_at_utc
        df["data_timestamp"] = df["ticker"].map(data_timestamp_by_ticker)
        df["price_frequency"] = PRICE_FREQUENCY
        df["price_adjustment_policy"] = PRICE_ADJUSTMENT_POLICY
        df["price_basis"] = df["ticker"].map(price_basis_by_ticker)
        df = df.sort_values(["ticker", "rule"], ascending=[True, True])

    data_timestamp_by_ticker = {
        ticker: series.index.max().date().isoformat()
        for ticker, series in sorted(prices.items())
    }
    expected_market_date = latest_completed_us_market_weekday().isoformat()
    stale_tickers = sorted(
        ticker
        for ticker, timestamp in data_timestamp_by_ticker.items()
        if timestamp < expected_market_date
    )
    ranked = (
        df.dropna(subset=["vectorbt_sharpe"])
        .sort_values(
            ["vectorbt_sharpe", "vectorbt_total_return_pct", "ticker", "rule"],
            ascending=[False, False, True, True],
        )
        if not df.empty
        else df
    )
    complete_row_count = len(prices) * len(ENTRY_RULE_SPECS)
    current_data_complete = not missing_tickers and not stale_tickers

    available = bool(
            not errors
            and not missing_tickers
            and not df.empty
            and len(df) == complete_row_count
    )
    summary = {
        "available": available,
        "generated_at_utc": generated_at_utc,
        "version": "vectorbt-validation-v4.1-native-calendars",
        "purpose": "Independent vectorbt validation layer for signal rules. It does not replace pandas portfolio backtest or ChatGPT live review.",
        "vectorbt_version": getattr(engine, "__version__", None),
        "execution_assumption": "End-of-day signals are executed at the next trading bar close.",
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "data_source": "Tiingo daily local CSV cache",
        "market_timezone": MARKET_TIMEZONE,
        "data_timestamp": max(data_timestamp_by_ticker.values()) if data_timestamp_by_ticker else None,
        "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
        "data_timestamp_status": "AVAILABLE" if data_timestamp_by_ticker else "MISSING",
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "bias_controls": {
            "look_ahead_bias": "CONTROLLED_BY_NEXT_BAR_EXECUTION",
            "survivorship_bias": "KNOWN_UNCONTROLLED_CURRENT_VALIDATION_UNIVERSE",
            "selection_bias": "KNOWN_UNCONTROLLED_FIXED_VALIDATION_TICKERS",
        },
        "data": {
            "calendar_policy": "each_ticker_native_valid_dates_no_forward_fill",
            "loaded_tickers": sorted(prices),
            "rows": int(sum(len(series) for series in prices.values())),
            "start_date": min(series.index.min() for series in prices.values()).date().isoformat(),
            "end_date": max(series.index.max() for series in prices.values()).date().isoformat(),
            "missing_tickers": sorted(missing_tickers),
            "stale_tickers": stale_tickers,
            "expected_latest_market_date": expected_market_date,
            "current_data_status": "COMPLETE" if current_data_complete else "INCOMPLETE_OR_STALE",
            "data_timestamp_by_ticker": data_timestamp_by_ticker,
            "price_basis_by_ticker": dict(sorted(price_basis_by_ticker.items())),
            "start_date_by_ticker": {
                ticker: series.index.min().date().isoformat()
                for ticker, series in sorted(prices.items())
            },
            "rows_by_ticker": {
                ticker: int(len(series))
                for ticker, series in sorted(prices.items())
            },
        },
        "assumptions": {
            "init_cash_per_column": INITIAL_CASH,
            **execution_cost_assumptions(),
            "freq": "1D",
            "execution_shift_bars": EXECUTION_SHIFT_BARS,
            "meaning": "Each ticker/rule column is independently backtested, not a real combined portfolio allocation.",
        },
        "top_by_sharpe": ranked.head(25).to_dict(orient="records") if not ranked.empty else [],
        "errors": errors,
        "important_limit": "Vectorbt results are a fast validation/audit layer. Final trade decisions still require portfolio-level backtest, walk-forward stability, live market/news checks, real account constraints, and human confirmation.",
    }

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "vectorbt_signal_stats.csv", index=False)
    with open(OUT / "vectorbt_validation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/vectorbt_validation.json and docs/vectorbt_signal_stats.csv")
    if not available:
        raise SystemExit(
            f"Vectorbt validation incomplete: missing={missing_tickers}, errors={errors}, "
            f"rows={len(df)}/{complete_row_count}"
        )


if __name__ == "__main__":
    main()
