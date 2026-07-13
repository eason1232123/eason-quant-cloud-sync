from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import TICKERS
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
    FORWARD_HORIZONS,
    MIN_EFFECTIVE_SAMPLE,
    RISK_RULE_SPECS,
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    benchmark_for_ticker,
    execution_cost_assumptions,
    exit_signals,
    net_return_after_round_trip_costs,
    next_close_forward_mae,
    next_close_forward_return,
    non_overlapping_signal_mask,
    risk_signals,
    rsi as strategy_rsi,
    rule_signals,
    shift_for_execution,
)

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

vbt: Any | None = None

INITIAL_CASH = 10000.0
MIN_SAMPLE = MIN_EFFECTIVE_SAMPLE
HORIZONS = list(FORWARD_HORIZONS)

ENTRY_RULES = {
    name: {
        "entry_col": f"entry_{name}",
        "exit_col": f"exit_{spec['exit_rule']}",
        "description": spec["description"],
    }
    for name, spec in ENTRY_RULE_SPECS.items()
}

RISK_RULES = {
    name: {
        "signal_col": f"risk_{name}",
        "description": spec["description"],
    }
    for name, spec in RISK_RULE_SPECS.items()
}


def vectorbt_module() -> Any:
    global vbt
    if vbt is None:
        vbt = importlib.import_module("vectorbt")
    return vbt


def configured_cost_assumptions() -> dict[str, Any]:
    return {
        **execution_cost_assumptions(),
        "assumption_source": "configured_not_observed",
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
    df = read_checked_daily_csv(ticker, OUT)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    price_col = "price" if "price" in df.columns else "adjClose" if "adjClose" in df.columns else "close"
    low_col = "low_price" if "low_price" in df.columns else "adjLow" if "adjLow" in df.columns else "low"

    out = df[["date", price_col]].rename(columns={price_col: "close"}).copy()
    if low_col not in df.columns:
        raise ValueError(f"{ticker} price cache is missing a low-price field required for MAE")
    out["low"] = pd.to_numeric(df[low_col], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if out["date"].duplicated().any():
        raise ValueError(f"{ticker} price cache contains duplicate market dates")
    invalid_close = out["close"].isna() | ~np.isfinite(out["close"]) | (out["close"] <= 0)
    invalid_low = out["low"].isna() | ~np.isfinite(out["low"]) | (out["low"] <= 0)
    if invalid_close.any() or invalid_low.any():
        raise ValueError(f"{ticker} price cache contains invalid close or low prices")
    out = out.sort_values("date").reset_index(drop=True)
    out.attrs["price_basis"] = "adjusted" if "adjClose" in df.columns else "unadjusted"
    return out


def relative_20d_series(data: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.Series:
    relative = pd.Series(np.nan, index=data.index, dtype=float)
    if benchmark is None or benchmark.empty:
        return relative

    aligned = data[["date", "close"]].merge(
        benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    ticker_return = aligned["close"].pct_change(20, fill_method=None)
    benchmark_return = aligned["benchmark_close"].pct_change(20, fill_method=None)
    return pd.Series((ticker_return - benchmark_return).to_numpy(), index=data.index)


def add_signals(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()
    d["ma200"] = d["close"].rolling(200).mean()
    d["ret_5d"] = d["close"].pct_change(5, fill_method=None)
    d["ret_20d"] = d["close"].pct_change(20, fill_method=None)
    d["rsi14"] = strategy_rsi(d["close"])
    d["ma20_slope_10d"] = d["ma20"] / d["ma20"].shift(10) - 1
    d["relative_20d"] = relative_20d_series(d, benchmark)

    for name, signal in rule_signals(d["close"], d["relative_20d"]).items():
        if name in ENTRY_RULE_SPECS:
            d[f"entry_{name}"] = signal
    for name, signal in exit_signals(d["close"]).items():
        d[f"exit_{name}"] = signal
    for name, signal in risk_signals(d["close"]).items():
        d[f"risk_{name}"] = signal
    return d


def metric_summary(value: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(value, errors="coerce").dropna()
    if len(v) < 3:
        return {}
    rets = v.pct_change(fill_method=None).dropna()
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
    return shift_for_execution(signal)


def engine_trade_statistics(portfolio: Any) -> tuple[list[float], int, int]:
    closed = portfolio.trades.closed
    opened = portfolio.trades.open
    closed_count = int(np.asarray(closed.count()).reshape(-1)[0])
    open_count = int(np.asarray(opened.count()).reshape(-1)[0])
    returns = np.asarray(closed.returns.values, dtype=float).reshape(-1)
    if len(returns) != closed_count:
        raise ValueError("vectorbt closed-trade count does not match closed return records")
    if not np.isfinite(returns).all():
        raise ValueError("vectorbt returned non-finite closed-trade returns")
    return returns.tolist(), closed_count, open_count


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
    raw_signal_count = int(raw_entries.sum())

    row: dict[str, Any] = {
        "ticker": ticker,
        "strategy": strategy_name,
        "engine": "vectorbt",
        "strategy_contract": STRATEGY_CONTRACT_VERSION,
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "cost_assumption_source": "configured_not_observed",
        "execution_assumption": "Signal on trading bar t; entry/exit can execute only at trading bar t+1 close.",
        "tested_entry_rule": entry_col.replace("entry_", ""),
        "tested_risk_rule": exit_col.replace("exit_", ""),
        "raw_signal_count": raw_signal_count,
        "effective_samples": 0,
        "closed_trade_count": 0,
        "sample_count": 0,
        "trade_count": 0,
        "executed_entry_count": 0,
        "open_position_count": 0,
        "lookback_start": data["date"].min().date().isoformat() if not data.empty else None,
        "lookback_end": data["date"].max().date().isoformat() if not data.empty else None,
        "forward_horizon": "next_close_to_next_close_after_exit_signal",
        "valid": False,
        "min_sample_required": MIN_SAMPLE,
        "description": spec["description"],
    }
    if raw_signal_count == 0 or len(data) < 60:
        row["reason"] = "not enough signals or history"
        return row

    close = data["close"]
    pf = vectorbt_module().Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=INITIAL_CASH,
        fees=COMMISSION_RATE,
        slippage=EXECUTION_IMPACT_RATE,
        upon_long_conflict="exit",
        freq="1D",
    )
    value = portfolio_value_series(pf, data.index)
    row.update(metric_summary(value))

    buy_hold_return = close.iloc[-1] / close.iloc[0] - 1
    returns, closed_trade_count, open_position_count = engine_trade_statistics(pf)
    executed_entry_count = closed_trade_count + open_position_count
    strategy_total_return = row.get("strategy_total_return")
    row.update(
        {
            "sample_count": int(closed_trade_count),
            "effective_samples": int(closed_trade_count),
            "closed_trade_count": int(closed_trade_count),
            "open_position_count": int(open_position_count),
            "executed_entry_count": int(executed_entry_count),
            "valid": bool(closed_trade_count >= MIN_SAMPLE and len(data) >= 252),
            "buy_hold_return_same_window": clean_float(buy_hold_return),
            "alpha_vs_buy_hold": clean_float(strategy_total_return - buy_hold_return) if strategy_total_return is not None else None,
            "trade_count": int(closed_trade_count),
            "trade_win_rate": clean_float(np.mean([r > 0 for r in returns]) if returns else None),
            "avg_trade_return": clean_float(np.mean(returns) if returns else None),
            "median_trade_return": clean_float(np.median(returns) if returns else None),
            "worst_trade_return": clean_float(np.min(returns) if returns else None),
            "best_trade_return": clean_float(np.max(returns) if returns else None),
            "latest_entry_signal_active": bool(raw_entries.iloc[-1]),
            "latest_exit_signal_active": bool(raw_exits.iloc[-1]),
            "latest_position_active": bool(open_position_count > 0),
        }
    )
    return row


def attach_benchmark(data: pd.DataFrame, benchmark: pd.DataFrame | None, horizon: int) -> pd.Series:
    if benchmark is None or benchmark.empty:
        return pd.Series([np.nan] * len(data), index=data.index)
    b = benchmark[["date", "close"]].copy().rename(columns={"close": "bench_close"})
    b[f"bench_fwd_{horizon}d"] = next_close_forward_return(b["bench_close"], horizon)
    merged = data[["date"]].merge(b[["date", f"bench_fwd_{horizon}d"]], on="date", how="left")
    return pd.Series(merged[f"bench_fwd_{horizon}d"].to_numpy(), index=data.index)


def future_mae(data: pd.DataFrame, horizon: int) -> pd.Series:
    return next_close_forward_mae(data["close"], data["low"], horizon)


def forward_evidence_row(
    ticker: str,
    d: pd.DataFrame,
    rule_name: str,
    rule_type: str,
    signal_col: str,
    horizon: int,
    benchmark: pd.DataFrame | None,
    description: str,
    benchmark_name: str = "QQQ",
) -> dict[str, Any]:
    data = d.copy()
    raw_signal = data[signal_col].fillna(False).astype(bool)
    gross_col = f"gross_fwd_{horizon}d"
    net_col = f"fwd_{horizon}d"
    bench_gross_col = f"bench_gross_fwd_{horizon}d"
    bench_net_col = f"bench_fwd_{horizon}d"
    data[gross_col] = next_close_forward_return(data["close"], horizon)
    data[net_col] = net_return_after_round_trip_costs(data[gross_col])
    data[f"mae_{horizon}d"] = future_mae(data, horizon)
    data[bench_gross_col] = attach_benchmark(data, benchmark, horizon)
    data[bench_net_col] = net_return_after_round_trip_costs(data[bench_gross_col])
    required_results = [gross_col, net_col, f"mae_{horizon}d", bench_gross_col, bench_net_col]
    completed_signal = raw_signal & data[required_results].notna().all(axis=1)
    completed_signal_count = int(completed_signal.sum())
    effective_mask = non_overlapping_signal_mask(completed_signal, horizon)
    sig = data[effective_mask]
    effective_samples = int(len(sig))

    row: dict[str, Any] = {
        "ticker": ticker,
        "rule": rule_name,
        "rule_type": rule_type,
        "strategy_contract": STRATEGY_CONTRACT_VERSION,
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "cost_assumption_source": "configured_not_observed",
        "tested_rule": signal_col.replace("entry_", "").replace("risk_", ""),
        "horizon_days": horizon,
        "raw_signal_count": int(raw_signal.sum()),
        "completed_signal_count": completed_signal_count,
        "effective_samples": effective_samples,
        "closed_trade_count": 0,
        "sample_count": effective_samples,
        "valid": bool(effective_samples >= MIN_SAMPLE),
        "min_sample_required": MIN_SAMPLE,
        "lookback_start": data["date"].min().date().isoformat() if not data.empty else None,
        "lookback_end": data["date"].max().date().isoformat() if not data.empty else None,
        "measurement_assumption": "Signal on bar t enters at t+1 close and exits horizon trading bars after that entry close; samples are chronological and non-overlapping.",
        "benchmark_data_required_for_completed_sample": True,
        "description": description,
    }
    if sig.empty:
        row["reason"] = "no completed samples"
        return row

    gross_fwd = sig[gross_col]
    fwd = sig[net_col]
    mae = sig[f"mae_{horizon}d"]
    gross_bench = sig[bench_gross_col]
    bench = sig[bench_net_col]
    row.update(
        {
            "win_rate": clean_float((fwd > 0).mean()),
            "avoidance_rate": clean_float((fwd < 0).mean()) if rule_type == "risk" else None,
            "avg_gross_forward_return": clean_float(gross_fwd.mean()),
            "avg_forward_return": clean_float(fwd.mean()),
            "median_forward_return": clean_float(fwd.median()),
            "worst_forward_return": clean_float(fwd.min()),
            "best_forward_return": clean_float(fwd.max()),
            "avg_mae": clean_float(mae.mean()),
            "worst_mae": clean_float(mae.min()),
            "benchmark": benchmark_name,
            "paired_benchmark_sample_count": int(len(sig)),
            "avg_gross_benchmark_same_dates": clean_float(gross_bench.mean()) if gross_bench.notna().any() else None,
            "avg_benchmark_same_dates": clean_float(bench.mean()) if bench.notna().any() else None,
            "alpha_vs_benchmark_same_dates": clean_float(fwd.mean() - bench.mean()) if bench.notna().any() else None,
        }
    )
    return row


def numeric_metric(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def evidence_score(row: dict[str, Any]) -> float | None:
    sample = numeric_metric(row, "sample_count")
    if sample is None:
        return None
    valid_bonus = 20 if row.get("valid") else 0
    if row.get("rule_type") == "risk":
        avoidance = numeric_metric(row, "avoidance_rate")
        worst_return = numeric_metric(row, "worst_forward_return")
        if avoidance is None or worst_return is None:
            return None
        worst = abs(worst_return)
        return round(min(100, valid_bonus + min(sample, 80) / 80 * 30 + float(avoidance) * 35 + min(worst, 0.25) / 0.25 * 15), 1)
    win = numeric_metric(row, "win_rate")
    median = numeric_metric(row, "median_forward_return")
    alpha = numeric_metric(row, "alpha_vs_benchmark_same_dates")
    if win is None or median is None or alpha is None:
        return None
    return round(min(100, valid_bonus + min(sample, 80) / 80 * 30 + max(0, float(win) - 0.5) / 0.3 * 25 + max(0, float(median)) / 0.08 * 15 + max(0, float(alpha)) / 0.08 * 10), 1)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def main() -> None:
    engine = vectorbt_module()
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    expected_market_date = latest_completed_us_market_weekday().isoformat()
    raw_price_map: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        df = load_price(ticker)
        if not df.empty and len(df) >= 60:
            raw_price_map[ticker] = df

    missing_tickers = sorted(set(TICKERS) - set(raw_price_map))
    data_timestamp_by_ticker = {
        ticker: frame["date"].max().date().isoformat()
        for ticker, frame in sorted(raw_price_map.items())
    }
    price_basis_by_ticker = {
        ticker: str(frame.attrs.get("price_basis") or "unknown")
        for ticker, frame in sorted(raw_price_map.items())
    }
    stale_tickers = sorted(
        ticker
        for ticker, timestamp in data_timestamp_by_ticker.items()
        if timestamp < expected_market_date
    )

    price_map: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for ticker, df in raw_price_map.items():
        benchmark_name = benchmark_for_ticker(ticker)
        if benchmark_name not in raw_price_map:
            errors[f"{ticker}/benchmark"] = f"required benchmark {benchmark_name} is missing"
        price_map[ticker] = add_signals(df, raw_price_map.get(benchmark_name))

    strategy_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    latest_active: dict[str, dict[str, Any]] = {}
    for ticker, d in price_map.items():
        try:
            benchmark_name = benchmark_for_ticker(ticker)
            benchmark = raw_price_map.get(benchmark_name)
            for strategy_name, spec in ENTRY_RULES.items():
                strategy_rows.append(run_vectorbt_strategy(ticker, d, strategy_name, spec))
                for horizon in HORIZONS:
                    evidence_rows.append(
                        forward_evidence_row(
                            ticker,
                            d,
                            strategy_name,
                            "entry",
                            spec["entry_col"],
                            horizon,
                            benchmark,
                            spec["description"],
                            benchmark_name,
                        )
                    )
            for risk_name, spec in RISK_RULES.items():
                for horizon in HORIZONS:
                    evidence_rows.append(
                        forward_evidence_row(
                            ticker,
                            d,
                            risk_name,
                            "risk",
                            spec["signal_col"],
                            horizon,
                            benchmark,
                            spec["description"],
                            benchmark_name,
                        )
                    )

            latest = d.iloc[-1]
            latest_active[ticker] = {
                "latest_date": latest["date"].date().isoformat(),
                "latest_price": clean_float(latest["close"]),
                "data_fresh": latest["date"].date().isoformat() == expected_market_date,
                "decision_eligible": latest["date"].date().isoformat() == expected_market_date,
                "entries": {name: bool(latest[spec["entry_col"]]) for name, spec in ENTRY_RULES.items()},
                "risks": {name: bool(latest[spec["signal_col"]]) for name, spec in RISK_RULES.items()},
            }
        except Exception as exc:
            errors[ticker] = str(exc)

    strategy_df = pd.DataFrame(strategy_rows)
    evidence_df = pd.DataFrame(evidence_rows)

    row_metadata = {
        "data_source": "Tiingo daily local CSV cache",
        "market_timezone": MARKET_TIMEZONE,
        "report_generated_at_utc": generated_at_utc,
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
    }
    for frame in (strategy_df, evidence_df):
        if not frame.empty:
            for field, value in row_metadata.items():
                frame[field] = value
            frame["data_timestamp"] = frame["ticker"].map(data_timestamp_by_ticker)
            frame["price_basis"] = frame["ticker"].map(price_basis_by_ticker)

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

    expected_strategy_rows = len(price_map) * len(ENTRY_RULES)
    expected_evidence_rows = len(price_map) * (len(ENTRY_RULES) + len(RISK_RULES)) * len(HORIZONS)
    available = bool(
        not missing_tickers
        and not errors
        and len(strategy_df) == expected_strategy_rows
        and len(evidence_df) == expected_evidence_rows
    )
    report = {
        "available": available,
        "generated_at_utc": generated_at_utc,
        "version": "vectorbt-validation-layer-v2.0-shared-contract",
        "engine": "vectorbt",
        "vectorbt_version": getattr(engine, "__version__", None),
        "strategy_contract": STRATEGY_CONTRACT_VERSION,
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "execution_assumption": "Signals on trading bar t execute only at trading bar t+1 close. Forward evidence uses the same next-close basis and chronological non-overlapping samples.",
        "data_source": "Tiingo daily local CSV cache",
        "market_timezone": MARKET_TIMEZONE,
        "data_timestamp": max(data_timestamp_by_ticker.values()) if data_timestamp_by_ticker else None,
        "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
        "data_timestamp_status": "AVAILABLE" if data_timestamp_by_ticker else "MISSING",
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "data_quality": {
            "expected_latest_market_date": expected_market_date,
            "data_timestamp_by_ticker": data_timestamp_by_ticker,
            "price_basis_by_ticker": price_basis_by_ticker,
            "missing_tickers": missing_tickers,
            "stale_tickers": stale_tickers,
            "current_signal_status": "COMPLETE" if not stale_tickers and not missing_tickers else "PARTIAL_STALE_OR_MISSING",
            "calendar_policy": "each_ticker_native_valid_dates_no_forward_fill",
        },
        "bias_controls": {
            "look_ahead_bias": "CONTROLLED_BY_NEXT_BAR_EXECUTION",
            "overlapping_forward_samples": "CONTROLLED_BY_CHRONOLOGICAL_NON_OVERLAP",
            "survivorship_bias": "KNOWN_UNCONTROLLED_CURRENT_CONFIGURED_UNIVERSE",
            "selection_bias": "KNOWN_UNCONTROLLED_MANUALLY_CONFIGURED_UNIVERSE",
        },
        "loaded_ticker_count": len(price_map),
        "configured_ticker_count": len(TICKERS),
        "initial_cash": INITIAL_CASH,
        "fees": COMMISSION_RATE,
        "slippage": EXECUTION_IMPACT_RATE,
        "cost_assumptions": configured_cost_assumptions(),
        "minimum_valid_samples": MIN_SAMPLE,
        "horizons_days": HORIZONS,
        "required_evidence_fields": {
            "tested_risk_rule": True,
            "sample_count": True,
            "raw_signal_count": True,
            "effective_samples": True,
            "closed_trade_count": True,
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
    if not available:
        raise SystemExit(
            f"Vectorbt evidence incomplete: missing={missing_tickers}, errors={errors}, "
            f"strategy_rows={len(strategy_df)}/{expected_strategy_rows}, "
            f"evidence_rows={len(evidence_df)}/{expected_evidence_rows}"
        )


if __name__ == "__main__":
    main()
