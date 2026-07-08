from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from config import TICKERS

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

INITIAL_CASH = 10000.0
FEES = 0.0005
SLIPPAGE = 0.0005
MIN_SAMPLE = 20
HORIZONS = [5, 10, 20, 60]
EXECUTION_SHIFT_DAYS = 1

ENTRY_RULES = {
    "ma20_reclaim_bullish": {
        "entry_col": "entry_ma20_reclaim_bullish",
        "exit_col": "exit_close_below_ma20",
        "description": "Price reclaims the 20-day moving average while above the 50-day average.",
    },
    "ma50_reclaim_bullish": {
        "entry_col": "entry_ma50_reclaim_bullish",
        "exit_col": "exit_close_below_ma50",
        "description": "Price reclaims the 50-day moving average while above the 200-day average.",
    },
    "momentum_leader": {
        "entry_col": "entry_momentum_leader",
        "exit_col": "exit_close_below_ma20",
        "description": "Above 20/50/200-day averages with positive 20-day momentum and non-overheated RSI.",
    },
    "pullback_reclaim_5dma": {
        "entry_col": "entry_pullback_reclaim_5dma",
        "exit_col": "exit_close_below_ma20",
        "description": "Short pullback followed by a 5-day moving-average reclaim.",
    },
}

RISK_RULES = {
    "risk_break_ma20": {
        "signal_col": "risk_break_ma20",
        "description": "Price breaks below the 20-day moving average after being above it.",
    },
    "risk_break_ma50": {
        "signal_col": "risk_break_ma50",
        "description": "Price breaks below the 50-day moving average after being above it.",
    },
    "risk_failed_rebound": {
        "signal_col": "risk_failed_rebound",
        "description": "Failed rebound: price loses the 5-day average and undercuts the prior 10-day low.",
    },
}


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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def csv_path(ticker: str) -> Path:
    safe = ticker.replace("/", "-").replace(".", "-")
    return OUT / f"{safe}_daily.csv"


def load_price(ticker: str) -> pd.DataFrame:
    path = csv_path(ticker)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    price_col = "price" if "price" in df.columns else "adjClose" if "adjClose" in df.columns else "close"
    low_col = "low_price" if "low_price" in df.columns else "adjLow" if "adjLow" in df.columns else "low"

    out = df[["date", price_col]].rename(columns={price_col: "close"}).copy()
    out["low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col in df.columns else pd.to_numeric(out["close"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()
    d["ma200"] = d["close"].rolling(200).mean()
    d["ret_5d"] = d["close"].pct_change(5)
    d["ret_20d"] = d["close"].pct_change(20)
    d["rsi14"] = rsi(d["close"])
    d["ma20_slope_10d"] = d["ma20"] / d["ma20"].shift(10) - 1

    d["entry_ma20_reclaim_bullish"] = (
        (d["close"].shift(1) <= d["ma20"].shift(1))
        & (d["close"] > d["ma20"])
        & (d["close"] > d["ma50"])
        & (d["ma20_slope_10d"] > -0.01)
    )
    d["entry_ma50_reclaim_bullish"] = (
        (d["close"].shift(1) <= d["ma50"].shift(1))
        & (d["close"] > d["ma50"])
        & (d["close"] > d["ma200"])
    )
    d["entry_momentum_leader"] = (
        (d["close"] > d["ma20"])
        & (d["ma20"] > d["ma50"])
        & (d["close"] > d["ma200"])
        & (d["ret_20d"] > 0.05)
        & (d["rsi14"] < 78)
    )
    d["entry_pullback_reclaim_5dma"] = (d["ret_5d"] <= -0.04) & (d["close"] > d["ma5"])

    d["exit_close_below_ma20"] = d["close"] < d["ma20"]
    d["exit_close_below_ma50"] = d["close"] < d["ma50"]
    d["risk_break_ma20"] = (d["close"].shift(1) >= d["ma20"].shift(1)) & (d["close"] < d["ma20"])
    d["risk_break_ma50"] = (d["close"].shift(1) >= d["ma50"].shift(1)) & (d["close"] < d["ma50"])
    d["risk_failed_rebound"] = (d["close"] < d["ma5"]) & (d["close"] < d["close"].rolling(10).min().shift(1))

    bool_cols = [c for c in d.columns if c.startswith(("entry_", "exit_", "risk_"))]
    for col in bool_cols:
        d[col] = d[col].fillna(False).astype(bool)
    return d


def metric_summary(value: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(value, errors="coerce").dropna()
    if len(v) < 3:
        return {}
    rets = v.pct_change().dropna()
    total_return = v.iloc[-1] / v.iloc[0] - 1
    years = len(rets) / 252
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol and not pd.isna(vol) and vol != 0 else np.nan
    drawdown = v / v.cummax() - 1
    return {
        "strategy_total_return": clean_float(total_return),
        "strategy_cagr": clean_float(cagr),
        "strategy_sharpe": clean_float(sharpe),
        "strategy_max_drawdown": clean_float(drawdown.min()),
        "strategy_worst_day": clean_float(rets.min()),
        "strategy_best_day": clean_float(rets.max()),
    }


def shift_execution(signal: pd.Series) -> pd.Series:
    return signal.shift(EXECUTION_SHIFT_DAYS).fillna(False).astype(bool)


def trade_returns(close: pd.Series, entries: pd.Series, exits: pd.Series) -> list[float]:
    in_pos = False
    entry_price = None
    returns: list[float] = []
    for price, entry, exit_ in zip(close, entries, exits):
        if pd.isna(price):
            continue
        if in_pos and bool(exit_):
            returns.append(float(price) / float(entry_price) - 1)
            in_pos = False
            entry_price = None
        if not in_pos and bool(entry):
            in_pos = True
            entry_price = float(price)
    if in_pos and entry_price:
        returns.append(float(close.iloc[-1]) / float(entry_price) - 1)
    return returns


def latest_position(entries: pd.Series, exits: pd.Series) -> bool:
    in_pos = False
    for entry, exit_ in zip(entries, exits):
        if bool(exit_):
            in_pos = False
        if bool(entry):
            in_pos = True
    return in_pos


def portfolio_value_series(pf: Any, index: pd.Index) -> pd.Series:
    value = pf.value()
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    if isinstance(value, pd.Series):
        return pd.Series(value.to_numpy(), index=index)
    arr = np.asarray(value).reshape(-1)
    return pd.Series(arr, index=index)


def run_vectorbt_strategy(ticker: str, d: pd.DataFrame, strategy_name: str, spec: dict[str, str]) -> dict[str, Any]:
    entry_col = spec["entry_col"]
    exit_col = spec["exit_col"]
    data = d.dropna(subset=["close"]).copy()
    raw_entries = data[entry_col].astype(bool)
    raw_exits = data[exit_col].astype(bool)
    entries = shift_execution(raw_entries)
    exits = shift_execution(raw_exits)

    row: dict[str, Any] = {
        "ticker": ticker,
        "strategy": strategy_name,
        "engine": "vectorbt",
        "execution_assumption": "EOD signal, next-bar execution: entries/exits shifted 1 trading day to avoid same-close look-ahead.",
        "tested_entry_rule": entry_col.replace("entry_", ""),
        "tested_risk_rule": exit_col.replace("exit_", ""),
        "sample_count": int(raw_entries.sum()),
        "executed_entry_count": int(entries.sum()),
        "lookback_start": data["date"].min().date().isoformat() if not data.empty else None,
        "lookback_end": data["date"].max().date().isoformat() if not data.empty else None,
        "forward_horizon": "signal_to_shifted_exit_or_latest_close",
        "valid": bool(int(raw_entries.sum()) >= MIN_SAMPLE and len(data) >= 252),
        "min_sample_required": MIN_SAMPLE,
        "description": spec["description"],
    }
    if raw_entries.sum() == 0 or len(data) < 60:
        row["reason"] = "not enough signals or history"
        return row

    close = data["close"]
    pf = vbt.Portfolio.from_signals(close=close, entries=entries, exits=exits, init_cash=INITIAL_CASH, fees=FEES, slippage=SLIPPAGE, freq="1D")
    value = portfolio_value_series(pf, data.index)
    row.update(metric_summary(value))

    buy_hold_return = close.iloc[-1] / close.iloc[0] - 1
    returns = trade_returns(close, entries, exits)
    row.update(
        {
            "buy_hold_return_same_window": clean_float(buy_hold_return),
            "alpha_vs_buy_hold": clean_float(row.get("strategy_total_return", 0) - buy_hold_return if row.get("strategy_total_return") is not None else None),
            "trade_count": int(len(returns)),
            "trade_win_rate": clean_float(np.mean([r > 0 for r in returns]) if returns else None),
            "avg_trade_return": clean_float(np.mean(returns) if returns else None),
            "median_trade_return": clean_float(np.median(returns) if returns else None),
            "worst_trade_return": clean_float(np.min(returns) if returns else None),
            "best_trade_return": clean_float(np.max(returns) if returns else None),
            "latest_entry_signal_active": bool(raw_entries.iloc[-1]),
            "latest_exit_signal_active": bool(raw_exits.iloc[-1]),
            "latest_position_active": bool(latest_position(entries, exits)),
        }
    )
    return row


def attach_benchmark(data: pd.DataFrame, benchmark: pd.DataFrame | None, horizon: int) -> pd.Series:
    if benchmark is None or benchmark.empty:
        return pd.Series([np.nan] * len(data), index=data.index)
    b = benchmark[["date", "close"]].copy().rename(columns={"close": "bench_close"})
    b[f"bench_fwd_{horizon}d"] = b["bench_close"].shift(-horizon) / b["bench_close"] - 1
    merged = data[["date"]].merge(b[["date", f"bench_fwd_{horizon}d"]], on="date", how="left")
    return pd.Series(merged[f"bench_fwd_{horizon}d"].to_numpy(), index=data.index)


def future_mae(data: pd.DataFrame, horizon: int) -> pd.Series:
    # For row i, this gives min low over rows i+1 ... i+horizon.
    return data["low"].rolling(horizon).min().shift(-horizon) / data["close"] - 1


def forward_evidence_row(ticker: str, d: pd.DataFrame, rule_name: str, rule_type: str, signal_col: str, horizon: int, benchmark: pd.DataFrame | None, description: str) -> dict[str, Any]:
    data = d.copy()
    data[f"fwd_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1
    data[f"mae_{horizon}d"] = future_mae(data, horizon)
    data[f"bench_fwd_{horizon}d"] = attach_benchmark(data, benchmark, horizon)
    sig = data[data[signal_col]].dropna(subset=[f"fwd_{horizon}d"])

    row: dict[str, Any] = {
        "ticker": ticker,
        "rule": rule_name,
        "rule_type": rule_type,
        "tested_rule": signal_col.replace("entry_", "").replace("risk_", ""),
        "horizon_days": horizon,
        "sample_count": int(len(sig)),
        "valid": bool(len(sig) >= MIN_SAMPLE),
        "min_sample_required": MIN_SAMPLE,
        "lookback_start": data["date"].min().date().isoformat() if not data.empty else None,
        "lookback_end": data["date"].max().date().isoformat() if not data.empty else None,
        "measurement_assumption": "Forward evidence measures close-to-close outcome after the signal day; executable vectorbt strategy uses next-bar shifted signals.",
        "description": description,
    }
    if sig.empty:
        row["reason"] = "no completed samples"
        return row

    fwd = sig[f"fwd_{horizon}d"]
    mae = sig[f"mae_{horizon}d"]
    bench = sig[f"bench_fwd_{horizon}d"]
    row.update(
        {
            "win_rate": clean_float((fwd > 0).mean()),
            "avoidance_rate": clean_float((fwd < 0).mean()) if rule_type == "risk" else None,
            "avg_forward_return": clean_float(fwd.mean()),
            "median_forward_return": clean_float(fwd.median()),
            "worst_forward_return": clean_float(fwd.min()),
            "best_forward_return": clean_float(fwd.max()),
            "avg_mae": clean_float(mae.mean()),
            "worst_mae": clean_float(mae.min()),
            "benchmark": "QQQ",
            "avg_benchmark_same_dates": clean_float(bench.mean()) if bench.notna().any() else None,
            "alpha_vs_benchmark_same_dates": clean_float(fwd.mean() - bench.mean()) if bench.notna().any() else None,
        }
    )
    return row


def evidence_score(row: dict[str, Any]) -> float:
    sample = row.get("sample_count") or 0
    valid_bonus = 20 if row.get("valid") else 0
    if row.get("rule_type") == "risk":
        avoidance = row.get("avoidance_rate") or 0
        worst = abs(row.get("worst_forward_return") or 0)
        return round(min(100, valid_bonus + min(sample, 80) / 80 * 30 + float(avoidance) * 35 + min(worst, 0.25) / 0.25 * 15), 1)
    win = row.get("win_rate") or 0
    median = row.get("median_forward_return") or 0
    alpha = row.get("alpha_vs_benchmark_same_dates") or 0
    return round(min(100, valid_bonus + min(sample, 80) / 80 * 30 + max(0, float(win) - 0.5) / 0.3 * 25 + max(0, float(median)) / 0.08 * 15 + max(0, float(alpha)) / 0.08 * 10), 1)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def main() -> None:
    price_map: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        df = load_price(ticker)
        if not df.empty and len(df) >= 60:
            price_map[ticker] = add_signals(df)

    benchmark = price_map.get("QQQ")
    strategy_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    latest_active: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    for ticker, d in price_map.items():
        try:
            for strategy_name, spec in ENTRY_RULES.items():
                strategy_rows.append(run_vectorbt_strategy(ticker, d, strategy_name, spec))
                for horizon in HORIZONS:
                    evidence_rows.append(forward_evidence_row(ticker, d, strategy_name, "entry", spec["entry_col"], horizon, benchmark, spec["description"]))
            for risk_name, spec in RISK_RULES.items():
                for horizon in HORIZONS:
                    evidence_rows.append(forward_evidence_row(ticker, d, risk_name, "risk", spec["signal_col"], horizon, benchmark, spec["description"]))

            latest = d.iloc[-1]
            latest_active[ticker] = {
                "latest_date": latest["date"].date().isoformat(),
                "latest_price": clean_float(latest["close"]),
                "entries": {name: bool(latest[spec["entry_col"]]) for name, spec in ENTRY_RULES.items()},
                "risks": {name: bool(latest[spec["signal_col"]]) for name, spec in RISK_RULES.items()},
            }
        except Exception as exc:
            errors[ticker] = str(exc)

    strategy_df = pd.DataFrame(strategy_rows)
    evidence_df = pd.DataFrame(evidence_rows)

    if not strategy_df.empty:
        strategy_df = ensure_columns(strategy_df, ["valid", "alpha_vs_buy_hold", "strategy_sharpe", "sample_count"])
        strategy_df = strategy_df.sort_values(["valid", "alpha_vs_buy_hold", "strategy_sharpe", "sample_count"], ascending=[False, False, False, False], na_position="last")
        strategy_df.to_csv(OUT / "vectorbt_strategy_summary.csv", index=False)

    if not evidence_df.empty:
        evidence_df["evidence_score_0_100"] = evidence_df.apply(lambda r: evidence_score(r.to_dict()), axis=1)
        evidence_df = ensure_columns(evidence_df, ["valid", "evidence_score_0_100", "sample_count", "rule_type", "horizon_days"])
        evidence_df = evidence_df.sort_values(["valid", "evidence_score_0_100", "sample_count"], ascending=[False, False, False], na_position="last")
        evidence_df.to_csv(OUT / "vectorbt_forward_evidence.csv", index=False)

    if evidence_df.empty:
        top_entry_evidence = []
        top_risk_evidence = []
    else:
        top_entry_evidence = evidence_df[(evidence_df["rule_type"] == "entry") & (evidence_df["horizon_days"] == 20)].head(20).to_dict("records")
        top_risk_evidence = evidence_df[(evidence_df["rule_type"] == "risk") & (evidence_df["horizon_days"] == 20)].head(20).to_dict("records")

    report = {
        "available": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "vectorbt-validation-layer-v1.2-next-bar-execution",
        "engine": "vectorbt",
        "vectorbt_version": getattr(vbt, "__version__", None),
        "execution_assumption": "Executable strategy backtests shift entry/exit signals by 1 trading day to avoid same-close look-ahead. Forward evidence remains signal-close outcome measurement.",
        "data_source": "Local Tiingo CSV cache in docs generated by scripts/build_report_safe.py",
        "loaded_ticker_count": len(price_map),
        "configured_ticker_count": len(TICKERS),
        "initial_cash": INITIAL_CASH,
        "fees": FEES,
        "slippage": SLIPPAGE,
        "minimum_valid_samples": MIN_SAMPLE,
        "horizons_days": HORIZONS,
        "required_evidence_fields": {
            "tested_risk_rule": True,
            "sample_count": True,
            "lookback_window": True,
            "forward_horizon": True,
            "win_rate_or_avoidance_rate": True,
            "average_and_median_forward_return": True,
            "worst_case_or_max_drawdown": True,
            "compare_vs_holding_or_benchmark": True,
        },
        "entry_rules": ENTRY_RULES,
        "risk_rules": RISK_RULES,
        "top_strategy_results": strategy_df.head(20).to_dict("records") if not strategy_df.empty else [],
        "top_entry_forward_evidence_20d": top_entry_evidence,
        "top_risk_forward_evidence_20d": top_risk_evidence,
        "latest_active_signals": latest_active,
        "output_files": [
            "docs/vectorbt_report.json",
            "docs/vectorbt_strategy_summary.csv",
            "docs/vectorbt_forward_evidence.csv",
        ],
        "important_limits": [
            "This is a validation layer, not an automatic trading system.",
            "It uses end-of-day adjusted/cached prices and cannot replace live IBKR bid/ask confirmation.",
            "Short-history tickers can show weak or invalid sample counts; do not use them as primary evidence.",
            "A risk rule with high avoidance rate means it often preceded weakness historically; it is not a guaranteed sell signal.",
        ],
        "errors": errors,
    }

    with open(OUT / "vectorbt_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/vectorbt_report.json, docs/vectorbt_strategy_summary.csv, docs/vectorbt_forward_evidence.csv")


if __name__ == "__main__":
    main()
