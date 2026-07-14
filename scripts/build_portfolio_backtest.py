from __future__ import annotations

import ast
import inspect
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.artifact_io import atomic_write_csv, atomic_write_json
from scripts.market_clock import MARKET_TIMEZONE
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
    read_checked_daily_csv,
)
from scripts.model_governance import (
    choose_regime as governance_choose_regime,
    governance_fingerprint,
    incumbent_model,
    load_governance_config,
    model_fingerprint,
)
from scripts.strategy_contract import (
    COMMISSION_BPS_PER_SIDE,
    COMMISSION_RATE,
    EXECUTION_IMPACT_RATE,
    EXECUTION_SHIFT_BARS,
    FULL_SPREAD_BPS,
    HALF_SPREAD_BPS_PER_SIDE,
    RULE_FINGERPRINT,
    SLIPPAGE_BPS_PER_SIDE,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    execution_cost_assumptions,
)
from scripts.validate_validation_split import (
    DEFAULT_MANIFEST,
    canonical_fingerprint,
    full_model_fingerprint,
    load_strict_json,
    split_manifest_fingerprint,
)

OUT = Path("docs")

INITIAL_CAPITAL = 20000.0
PORTFOLIO_CONTRACT_VERSION = "v6-rc2-governed-incumbent-pending-rebalance"
SLIPPAGE_BPS = SLIPPAGE_BPS_PER_SIDE
MIN_TRADE_DOLLARS = 25.0
EXECUTION_SHIFT_DAYS = EXECUTION_SHIFT_BARS

CORE_TICKERS = ["QQQ", "SMH", "MSFT", "SPY"]
BENCHMARK_TICKERS = ["SPY", "QQQ", "SMH"]
CASH_TICKER = "SGOV"

MODEL_GOVERNANCE = load_governance_config()
INCUMBENT_MODEL = incumbent_model(MODEL_GOVERNANCE)
INCUMBENT_MODEL_ID = str(INCUMBENT_MODEL["model_id"])
MODEL_GOVERNANCE_FINGERPRINT = governance_fingerprint(MODEL_GOVERNANCE)
INCUMBENT_MODEL_FINGERPRINT = model_fingerprint(INCUMBENT_MODEL)
BASE_WEIGHTS = dict(INCUMBENT_MODEL["regime_weights"]["base"])
DEFENSIVE_WEIGHTS = dict(INCUMBENT_MODEL["regime_weights"]["defensive"])
SEVERE_DEFENSIVE_WEIGHTS = dict(INCUMBENT_MODEL["regime_weights"]["severe_defensive"])


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
    invalid_price = out[ticker].isna() | ~np.isfinite(out[ticker]) | (out[ticker] <= 0)
    if invalid_price.any():
        dates = out.loc[invalid_price, "date"].dt.date.astype(str).tolist()
        raise ValueError(f"{ticker} price cache has missing or invalid prices on: {dates[:5]}")
    if out["date"].duplicated().any():
        duplicates = out.loc[out["date"].duplicated(keep=False), "date"].dt.date.astype(str).unique().tolist()
        raise ValueError(f"{ticker} price cache has duplicate dates: {duplicates[:5]}")
    return out.sort_values("date").reset_index(drop=True)


def load_prices(tickers: list[str]) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = load_price(ticker)
        if df.empty:
            continue
        frames[ticker] = df
    if len(frames) < len(tickers):
        missing = [t for t in tickers if t not in frames]
        raise SystemExit(f"Missing required price history for: {missing}")

    # Leading rows before every asset exists are outside the tradeable universe.
    # Once all assets have started, a missing asset bar is a critical data gap;
    # an inner join would silently conceal it and calculate a multi-day return as
    # though it were one ordinary execution bar.
    common_start = max(frame["date"].min() for frame in frames.values())
    merged = None
    for ticker in tickers:
        frame = frames[ticker]
        merged = frame if merged is None else merged.merge(
            frame,
            on="date",
            how="outer",
            validate="one_to_one",
        )
    if merged is None:
        raise SystemExit("No required price history was loaded")
    merged = merged[merged["date"] >= common_start].sort_values("date").reset_index(drop=True)
    missing_mask = merged[tickers].isna()
    if missing_mask.any().any():
        details = {
            ticker: {
                "missing_rows": int(missing_mask[ticker].sum()),
                "first_missing_date": merged.loc[missing_mask[ticker], "date"].iloc[0].date().isoformat(),
            }
            for ticker in tickers
            if missing_mask[ticker].any()
        }
        raise SystemExit(f"Missing required asset bars after common start: {details}")
    return merged


def add_trend_columns(prices: pd.DataFrame) -> pd.DataFrame:
    d = prices.copy()
    for ticker in CORE_TICKERS:
        d[f"{ticker}_ma50"] = d[ticker].rolling(50).mean()
        d[f"{ticker}_ma200"] = d[ticker].rolling(200).mean()
        d[f"{ticker}_ret20"] = d[ticker].pct_change(20, fill_method=None)
    return d


def choose_regime(row: pd.Series) -> str:
    return governance_choose_regime(row)


def weights_for_regime(regime: str) -> dict[str, float]:
    if regime == "severe_defensive":
        return SEVERE_DEFENSIVE_WEIGHTS.copy()
    if regime == "defensive":
        return DEFENSIVE_WEIGHTS.copy()
    return BASE_WEIGHTS.copy()


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    if set(w) != {*CORE_TICKERS, "CASH"}:
        raise ValueError("Portfolio weights must contain every core ticker and CASH exactly once")
    if any(not np.isfinite(value) or value < 0 for value in w.values()):
        raise ValueError("Portfolio weights must be finite and non-negative")
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Portfolio weights must have a positive total")
    return {k: float(v) / total for k, v in w.items()}


def is_month_change(prev_date: pd.Timestamp | None, date: pd.Timestamp) -> bool:
    if prev_date is None:
        return True
    return (prev_date.year, prev_date.month) != (date.year, date.month)


def should_rebalance(current_weights: dict[str, float], target: dict[str, float], regime_changed: bool, month_changed: bool) -> bool:
    if regime_changed or month_changed:
        return True
    for asset, target_w in target.items():
        if abs(current_weights.get(asset, 0.0) - target_w) >= 0.05:
            return True
    return False


def build_cash_returns() -> pd.Series | None:
    sgov = load_price(CASH_TICKER)
    if sgov.empty:
        return None
    sgov["date"] = pd.to_datetime(sgov["date"])
    sgov["cash_ret"] = sgov[CASH_TICKER].pct_change(fill_method=None)
    return sgov.set_index("date")["cash_ret"]


def validate_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Core asset price history is empty")
    required = ["date", *CORE_TICKERS]
    missing_columns = [column for column in required if column not in prices.columns]
    if missing_columns:
        raise ValueError(f"Core asset price history is missing columns: {missing_columns}")

    d = prices[required].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if d["date"].isna().any():
        raise ValueError("Core asset price history contains invalid dates")
    d["date"] = d["date"].dt.tz_localize(None)
    if d["date"].duplicated().any():
        duplicates = d.loc[d["date"].duplicated(keep=False), "date"].dt.date.astype(str).unique().tolist()
        raise ValueError(f"Core asset price history contains duplicate dates: {duplicates[:5]}")

    d = d.sort_values("date").reset_index(drop=True)
    for ticker in CORE_TICKERS:
        d[ticker] = pd.to_numeric(d[ticker], errors="coerce")
        invalid = d[ticker].isna() | ~np.isfinite(d[ticker]) | (d[ticker] <= 0)
        if invalid.any():
            dates = d.loc[invalid, "date"].dt.date.astype(str).tolist()
            raise ValueError(f"{ticker} has missing or invalid price bars on: {dates[:5]}")
    return d


def attach_cash_proxy_returns(
    prices: pd.DataFrame,
    cash_returns: pd.Series | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cash_returns is None or cash_returns.empty:
        raise ValueError(
            f"{CASH_TICKER} cash-proxy returns are unavailable; portfolio backtest is not tradeable"
        )

    cash = cash_returns.rename("CASH_RET").reset_index()
    date_column = cash.columns[0]
    cash = cash.rename(columns={date_column: "date"})
    cash["date"] = pd.to_datetime(cash["date"], errors="coerce")
    if cash["date"].isna().any():
        raise ValueError(f"{CASH_TICKER} cash-proxy returns contain invalid dates")
    cash["date"] = cash["date"].dt.tz_localize(None)
    if cash["date"].duplicated().any():
        raise ValueError(f"{CASH_TICKER} cash-proxy returns contain duplicate dates")
    cash["CASH_RET"] = pd.to_numeric(cash["CASH_RET"], errors="coerce")

    d = prices.merge(cash, on="date", how="left", validate="one_to_one")
    observed = d["CASH_RET"].notna() & np.isfinite(d["CASH_RET"])
    if not observed.any():
        raise ValueError(
            f"{CASH_TICKER} has no observed cash-proxy return overlapping core asset history"
        )

    first_observed_position = int(np.flatnonzero(observed.to_numpy())[0])
    excluded_pre_proxy_rows = first_observed_position
    d = d.iloc[first_observed_position:].reset_index(drop=True)
    invalid = d["CASH_RET"].isna() | ~np.isfinite(d["CASH_RET"]) | (d["CASH_RET"] <= -1)
    if invalid.any():
        dates = d.loc[invalid, "date"].dt.date.astype(str).tolist()
        raise ValueError(
            f"{CASH_TICKER} cash-proxy return is missing or invalid after proxy start on: {dates[:5]}"
        )
    if len(d) < 2:
        raise ValueError("Insufficient common core-asset and cash-proxy bars for next-bar execution")

    return d, {
        "cash_proxy_ticker": CASH_TICKER,
        "cash_proxy_status": "OBSERVED_COMPLETE_AFTER_START",
        "cash_proxy_first_observed_return_date": d.iloc[0]["date"].date().isoformat(),
        "excluded_pre_cash_proxy_rows": excluded_pre_proxy_rows,
        "missing_cash_proxy_policy": "pre-proxy rows are untradeable; any later missing return fails the backtest",
    }


def rebalance_signal_reasons(
    current_weights: dict[str, float],
    target: dict[str, float],
    regime_changed: bool,
    month_changed: bool,
    initial_allocation: bool,
) -> list[str]:
    if initial_allocation:
        return ["initial_allocation"]
    reasons = []
    if regime_changed:
        reasons.append("regime_change")
    if month_changed:
        reasons.append("monthly_rebalance")
    if any(abs(current_weights.get(asset, 0.0) - target_w) >= 0.05 for asset, target_w in target.items()):
        reasons.append("drift_rebalance")
    return reasons


def simulate_strategy(prices: pd.DataFrame, cash_returns: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = add_trend_columns(validate_price_frame(prices))
    d["raw_regime_signal"] = d.apply(choose_regime, axis=1)
    d, cash_proxy_metadata = attach_cash_proxy_returns(d, cash_returns)

    shares = {ticker: 0.0 for ticker in CORE_TICKERS}
    cash = INITIAL_CAPITAL
    records = []
    trades = []
    prev_date: pd.Timestamp | None = None
    prev_raw_regime: str | None = None
    active_regime = "warmup_base"
    pending_rebalance: dict[str, Any] | None = None
    last_execution_signal_date: str | None = None

    for i, row in d.iterrows():
        date = pd.to_datetime(row["date"])
        if i > 0:
            cash *= 1 + float(row["CASH_RET"])

        prices_now = {ticker: float(row[ticker]) for ticker in CORE_TICKERS}
        holdings_value = {ticker: shares[ticker] * prices_now[ticker] for ticker in CORE_TICKERS}
        total_value = cash + sum(holdings_value.values())

        current_weights = {ticker: holdings_value[ticker] / total_value if total_value else 0.0 for ticker in CORE_TICKERS}
        current_weights["CASH"] = cash / total_value if total_value else 1.0

        raw_regime_signal = str(row["raw_regime_signal"])
        executed_signal: dict[str, Any] | None = None
        if pending_rebalance is not None:
            executed_signal = pending_rebalance
            target = pending_rebalance["target_weights"]
            active_regime = str(pending_rebalance["regime"])
            pre_trade_value = total_value
            planned_deltas = {
                ticker: (
                    pre_trade_value * target[ticker]
                    - shares[ticker] * prices_now[ticker]
                )
                for ticker in CORE_TICKERS
            }
            # Sell first so purchases can use the same rebalance's proceeds.
            execution_order = sorted(CORE_TICKERS, key=lambda ticker: planned_deltas[ticker] >= 0)
            for ticker in execution_order:
                delta = planned_deltas[ticker]
                if abs(delta) < MIN_TRADE_DOLLARS:
                    continue

                if delta > 0:
                    execution_price = prices_now[ticker] * (1 + EXECUTION_IMPACT_RATE)
                    maximum_notional = cash / (1 + COMMISSION_RATE)
                    buy_shares = min(delta, maximum_notional) / execution_price
                    if buy_shares <= 0:
                        continue
                    dollars = buy_shares * execution_price
                    commission = dollars * COMMISSION_RATE
                    shares[ticker] += buy_shares
                    cash -= dollars + commission
                    action = "BUY"
                    traded_shares = buy_shares
                else:
                    execution_price = prices_now[ticker] * (1 - EXECUTION_IMPACT_RATE)
                    sell_shares = min(abs(delta) / execution_price, shares[ticker])
                    if sell_shares <= 0:
                        continue
                    dollars = sell_shares * execution_price
                    commission = dollars * COMMISSION_RATE
                    shares[ticker] -= sell_shares
                    cash += dollars - commission
                    action = "SELL"
                    traded_shares = sell_shares

                execution_impact_dollars = abs(execution_price - prices_now[ticker]) * traded_shares
                trigger_reasons = list(pending_rebalance["trigger_reasons"])
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "signal_date": pending_rebalance["signal_date"],
                        "execution_date": date.date().isoformat(),
                        "ticker": ticker,
                        "action": action,
                        "shares": clean_float(traded_shares, 6),
                        "reference_close": clean_float(prices_now[ticker], 4),
                        "execution_price": clean_float(execution_price, 4),
                        "execution_price_kind": "model_close_with_configured_impact_not_historical_fill",
                        "dollars": clean_float(dollars, 2),
                        "commission_dollars": clean_float(commission, 4),
                        "execution_impact_dollars": clean_float(execution_impact_dollars, 4),
                        "cost_calibration_status": "configured_not_observed",
                        "regime_used_for_execution": active_regime,
                        "raw_regime_signal_same_day": raw_regime_signal,
                        "trigger_reasons": "|".join(trigger_reasons),
                        "reason": (
                            "regime_change"
                            if "regime_change" in trigger_reasons
                            else "monthly_or_drift_rebalance"
                        ),
                    }
                )

            holdings_value = {ticker: shares[ticker] * prices_now[ticker] for ticker in CORE_TICKERS}
            total_value = cash + sum(holdings_value.values())
            current_weights = {ticker: holdings_value[ticker] / total_value if total_value else 0.0 for ticker in CORE_TICKERS}
            current_weights["CASH"] = cash / total_value if total_value else 1.0
            last_execution_signal_date = str(pending_rebalance["signal_date"])
            pending_rebalance = None

        target_for_signal = normalize_weights(weights_for_regime(raw_regime_signal))
        month_changed = is_month_change(prev_date, date)
        regime_changed = prev_raw_regime is not None and raw_regime_signal != prev_raw_regime
        signal_reasons = rebalance_signal_reasons(
            current_weights,
            target_for_signal,
            regime_changed,
            month_changed,
            initial_allocation=prev_date is None,
        )
        if signal_reasons:
            pending_rebalance = {
                "signal_date": date.date().isoformat(),
                "regime": raw_regime_signal,
                "target_weights": target_for_signal,
                "trigger_reasons": signal_reasons,
            }

        records.append(
            {
                "date": date.date().isoformat(),
                "strategy_value": clean_float(total_value, 4),
                "regime": active_regime,
                "raw_regime_signal_same_day": raw_regime_signal,
                "execution_assumption": "all rebalance signals execute at the next valid common trading-bar close",
                "rebalance_signal_generated": bool(signal_reasons),
                "rebalance_signal_reasons": "|".join(signal_reasons) if signal_reasons else None,
                "executed_signal_date": executed_signal["signal_date"] if executed_signal else None,
                "cash": clean_float(cash, 4),
                "cash_weight": clean_float(current_weights.get("CASH"), 6),
                "QQQ_weight": clean_float(current_weights.get("QQQ"), 6),
                "SMH_weight": clean_float(current_weights.get("SMH"), 6),
                "MSFT_weight": clean_float(current_weights.get("MSFT"), 6),
                "SPY_weight": clean_float(current_weights.get("SPY"), 6),
                "tech_ai_concentration": clean_float(current_weights.get("QQQ", 0) + current_weights.get("SMH", 0) + current_weights.get("MSFT", 0), 6),
                "semiconductor_exposure": clean_float(current_weights.get("SMH", 0), 6),
                "MSFT_exposure": clean_float(current_weights.get("MSFT", 0), 6),
            }
        )

        prev_date = date
        prev_raw_regime = raw_regime_signal

    equity = pd.DataFrame(records)
    trade_log = pd.DataFrame(trades)
    equity["date"] = pd.to_datetime(equity["date"])
    configured_costs = execution_cost_assumptions()
    configured_costs["calibration_status"] = "configured_not_observed"
    return equity, trade_log, {
        "initial_capital": INITIAL_CAPITAL,
        "slippage_bps": SLIPPAGE_BPS,
        "execution_shift_days": EXECUTION_SHIFT_DAYS,
        "execution_assumption": "Regime, monthly, drift, and initial-allocation signals are calculated after a close and execute only at the next valid common trading-bar close.",
        "execution_price_kind": "model_close_with_configured_impact_not_historical_fill",
        "execution_costs": configured_costs,
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "model_governance_fingerprint": MODEL_GOVERNANCE_FINGERPRINT,
        "incumbent_model_id": INCUMBENT_MODEL_ID,
        "incumbent_model_fingerprint": INCUMBENT_MODEL_FINGERPRINT,
        "base_weights": BASE_WEIGHTS,
        "defensive_weights": DEFENSIVE_WEIGHTS,
        "severe_defensive_weights": SEVERE_DEFENSIVE_WEIGHTS,
        "rebalance_policy": "monthly, regime-change, or drift >= 5 percentage points",
        "regime_policy": "base unless QQQ/SMH/SPY trend weakness triggers defensive or severe defensive weights",
        "model_governance_policy": "Only the frozen incumbent is used in this retrospective portfolio; challengers require paired non-overlapping prospective evidence before any capped blend.",
        "cash_proxy": cash_proxy_metadata,
        "pending_rebalance_at_end": pending_rebalance,
        "last_execution_signal_date": last_execution_signal_date,
    }


def metric_summary(values: pd.Series, dates: pd.Series | None = None) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 3:
        return {}
    rets = v.pct_change(fill_method=None).dropna()
    total_return = v.iloc[-1] / v.iloc[0] - 1
    years = len(rets) / 252
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol and not pd.isna(vol) and vol != 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = (rets.mean() * 252) / downside if downside and not pd.isna(downside) and downside != 0 else np.nan
    running_max = v.cummax()
    dd = v / running_max - 1
    max_dd = dd.min()
    win_rate = (rets > 0).mean()
    avg_daily = rets.mean()
    best_day = rets.max()
    worst_day = rets.min()
    calmar = cagr / abs(max_dd) if max_dd and not pd.isna(max_dd) and max_dd != 0 else np.nan

    return {
        "start_date": pd.to_datetime(dates.iloc[0]).date().isoformat() if dates is not None and len(dates) else None,
        "end_date": pd.to_datetime(dates.iloc[-1]).date().isoformat() if dates is not None and len(dates) else None,
        "trading_days": int(len(v)),
        "years": clean_float(years, 2),
        "total_return_pct": clean_float(total_return * 100, 2),
        "cagr_pct": clean_float(cagr * 100, 2),
        "volatility_pct": clean_float(vol * 100, 2),
        "sharpe": clean_float(sharpe, 3),
        "sortino": clean_float(sortino, 3),
        "calmar": clean_float(calmar, 3),
        "max_drawdown_pct": clean_float(max_dd * 100, 2),
        "daily_win_rate_pct": clean_float(win_rate * 100, 2),
        "avg_daily_return_pct": clean_float(avg_daily * 100, 4),
        "best_day_pct": clean_float(best_day * 100, 2),
        "worst_day_pct": clean_float(worst_day * 100, 2),
        "final_value": clean_float(v.iloc[-1], 2),
    }


def benchmark_curves(prices: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    out = equity[["date", "strategy_value"]].copy()
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    out = out.merge(px[["date"] + BENCHMARK_TICKERS], on="date", how="left")
    for ticker in BENCHMARK_TICKERS:
        first_series = out[ticker].dropna()
        if first_series.empty:
            out[f"buy_hold_{ticker}"] = np.nan
        else:
            first = first_series.iloc[0]
            out[f"buy_hold_{ticker}"] = INITIAL_CAPITAL * out[ticker] / first
    out = out.drop(columns=BENCHMARK_TICKERS)
    return out


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trade_count": 0}
    by_action = trades["action"].value_counts().to_dict()
    by_ticker = trades["ticker"].value_counts().to_dict()
    return {
        "trade_count": int(len(trades)),
        "buy_count": int(by_action.get("BUY", 0)),
        "sell_count": int(by_action.get("SELL", 0)),
        "trades_by_ticker": {k: int(v) for k, v in by_ticker.items()},
        "avg_trade_dollars": clean_float(trades["dollars"].mean(), 2),
        "max_trade_dollars": clean_float(trades["dollars"].max(), 2),
        "total_commission_dollars": clean_float(trades["commission_dollars"].sum(), 2),
        "total_execution_impact_dollars": clean_float(trades["execution_impact_dollars"].sum(), 2),
        "cost_calibration_status": "configured_not_observed",
    }


def exposure_summary(equity: pd.DataFrame) -> dict:
    if equity.empty:
        return {}
    latest = equity.iloc[-1]
    return {
        "latest_cash_weight_pct": clean_float(latest["cash_weight"] * 100, 2),
        "latest_QQQ_weight_pct": clean_float(latest["QQQ_weight"] * 100, 2),
        "latest_SMH_weight_pct": clean_float(latest["SMH_weight"] * 100, 2),
        "latest_MSFT_weight_pct": clean_float(latest["MSFT_weight"] * 100, 2),
        "latest_SPY_weight_pct": clean_float(latest["SPY_weight"] * 100, 2),
        "latest_tech_ai_concentration_pct": clean_float(latest["tech_ai_concentration"] * 100, 2),
        "latest_semiconductor_exposure_pct": clean_float(latest["semiconductor_exposure"] * 100, 2),
        "latest_MSFT_exposure_pct": clean_float(latest["MSFT_exposure"] * 100, 2),
        "avg_cash_weight_pct": clean_float(equity["cash_weight"].mean() * 100, 2),
        "max_tech_ai_concentration_pct": clean_float(equity["tech_ai_concentration"].max() * 100, 2),
        "max_semiconductor_exposure_pct": clean_float(equity["semiconductor_exposure"].max() * 100, 2),
        "max_MSFT_exposure_pct": clean_float(equity["MSFT_exposure"].max() * 100, 2),
    }


def attach_output_metadata(
    frame: pd.DataFrame,
    *,
    generated_at_utc: str,
    data_timestamp: str | None,
    portfolio_fingerprint: str,
    split_fingerprint: str,
    model_fingerprint: str,
) -> pd.DataFrame:
    out = frame.copy()
    metadata = {
        "data_source": "Tiingo daily local CSV cache",
        "market_timezone": MARKET_TIMEZONE,
        "report_generated_at_utc": generated_at_utc,
        "data_timestamp": data_timestamp,
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "portfolio_contract_fingerprint": portfolio_fingerprint,
        "split_manifest_fingerprint": split_fingerprint,
        "full_model_fingerprint": model_fingerprint,
        "model_governance_fingerprint": MODEL_GOVERNANCE_FINGERPRINT,
        "incumbent_model_id": INCUMBENT_MODEL_ID,
        "incumbent_model_fingerprint": INCUMBENT_MODEL_FINGERPRINT,
    }
    for field, value in metadata.items():
        out[field] = value
    return out


def _normalized_function_ast(function: Any) -> str:
    return ast.dump(
        ast.parse(textwrap.dedent(inspect.getsource(function))),
        annotate_fields=True,
        include_attributes=False,
    )


def portfolio_contract_payload() -> dict[str, Any]:
    return {
        "version": PORTFOLIO_CONTRACT_VERSION,
        "initial_capital": INITIAL_CAPITAL,
        "minimum_trade_dollars": MIN_TRADE_DOLLARS,
        "core_tickers": CORE_TICKERS,
        "benchmark_tickers": BENCHMARK_TICKERS,
        "cash_proxy_ticker": CASH_TICKER,
        "model_governance_fingerprint": MODEL_GOVERNANCE_FINGERPRINT,
        "incumbent_model_id": INCUMBENT_MODEL_ID,
        "incumbent_model_fingerprint": INCUMBENT_MODEL_FINGERPRINT,
        "weights": {
            "base": BASE_WEIGHTS,
            "defensive": DEFENSIVE_WEIGHTS,
            "severe_defensive": SEVERE_DEFENSIVE_WEIGHTS,
        },
        "execution_shift_bars": EXECUTION_SHIFT_BARS,
        "rebalance_policy": {
            "monthly": True,
            "regime_change": True,
            "absolute_drift_threshold": 0.05,
            "pending_to_next_common_bar": True,
            "sell_before_buy": True,
        },
        "cash_proxy_policy": "exclude_pre_proxy_rows_and_fail_on_any_later_gap",
        "implementation_ast": {
            function.__name__: _normalized_function_ast(function)
            for function in (
                choose_regime,
                weights_for_regime,
                normalize_weights,
                should_rebalance,
                attach_cash_proxy_returns,
                rebalance_signal_reasons,
                simulate_strategy,
                metric_summary,
                benchmark_curves,
            )
        },
    }


def main() -> None:
    manifest = load_strict_json(DEFAULT_MANIFEST)
    portfolio_payload = portfolio_contract_payload()
    portfolio_fingerprint = canonical_fingerprint(portfolio_payload)
    split_fingerprint = split_manifest_fingerprint(manifest)
    model_fingerprint = full_model_fingerprint(
        STRATEGY_FINGERPRINT,
        portfolio_payload,
        manifest,
    )
    prices = load_prices(CORE_TICKERS)
    cash_returns = build_cash_returns()
    equity, trades, assumptions = simulate_strategy(prices, cash_returns)
    assumptions.update(
        {
            "portfolio_contract_version": PORTFOLIO_CONTRACT_VERSION,
            "portfolio_contract_fingerprint": portfolio_fingerprint,
            "split_manifest_fingerprint": split_fingerprint,
            "full_model_fingerprint": model_fingerprint,
        }
    )
    vs = benchmark_curves(prices, equity)

    strategy_metrics = metric_summary(equity["strategy_value"], equity["date"])
    benchmark_metrics = {}
    for col in [c for c in vs.columns if c.startswith("buy_hold_")]:
        benchmark_metrics[col.replace("buy_hold_", "")] = metric_summary(vs[col], vs["date"])

    comparison = []
    for ticker, metrics in benchmark_metrics.items():
        strategy_cagr = strategy_metrics.get("cagr_pct")
        benchmark_cagr = metrics.get("cagr_pct")
        strategy_drawdown = strategy_metrics.get("max_drawdown_pct")
        benchmark_drawdown = metrics.get("max_drawdown_pct")
        strategy_final = strategy_metrics.get("final_value")
        benchmark_final = metrics.get("final_value")
        comparison.append(
            {
                "benchmark": ticker,
                "strategy_cagr_minus_benchmark_pct": clean_float(strategy_cagr - benchmark_cagr, 2)
                if strategy_cagr is not None and benchmark_cagr is not None
                else None,
                "strategy_maxdd_minus_benchmark_pct": clean_float(strategy_drawdown - benchmark_drawdown, 2)
                if strategy_drawdown is not None and benchmark_drawdown is not None
                else None,
                "strategy_final_minus_benchmark_dollars": clean_float(strategy_final - benchmark_final, 2)
                if strategy_final is not None and benchmark_final is not None
                else None,
            }
        )

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    data_timestamp = equity["date"].max().date().isoformat() if not equity.empty else None
    report = {
        "available": True,
        "generated_at_utc": generated_at_utc,
        "data_source": "Tiingo daily local CSV cache",
        "market_timezone": MARKET_TIMEZONE,
        "data_timestamp": data_timestamp,
        "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
        "data_timestamp_status": "AVAILABLE" if data_timestamp else "MISSING",
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "execution_price_kind": "model_close_with_configured_impact_not_historical_fill",
        "market_data_status": "CACHE_VALIDATED_NOT_LIVE",
        "version": "portfolio-backtest-v3.1-governed-incumbent-next-bar",
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "portfolio_contract_version": PORTFOLIO_CONTRACT_VERSION,
        "portfolio_contract_fingerprint": portfolio_fingerprint,
        "split_manifest_fingerprint": split_fingerprint,
        "full_model_fingerprint": model_fingerprint,
        "model_governance_fingerprint": MODEL_GOVERNANCE_FINGERPRINT,
        "incumbent_model_id": INCUMBENT_MODEL_ID,
        "incumbent_model_fingerprint": INCUMBENT_MODEL_FINGERPRINT,
        "cost_calibration_status": "configured_not_observed",
        "purpose": "Model-level portfolio backtest for Eason strategy. This is not live execution and not personalized account truth.",
        "bias_controls": {
            "look_ahead_bias": "CONTROLLED_BY_PENDING_NEXT_BAR_REBALANCE",
            "survivorship_bias": "KNOWN_UNCONTROLLED_CURRENT_FIXED_ASSET_SET",
            "selection_bias": "KNOWN_UNCONTROLLED_MANUALLY_SELECTED_MODEL_ASSETS",
        },
        "important_limits": [
            "Uses end-of-day prices from checked cached docs CSV files under the reported adjustment policy; not intraday bid/ask.",
            "Uses model weights, not Eason's private real account shares/cash.",
            "Regime, monthly, drift, and initial-allocation signals execute only on the next valid common trading bar.",
            "Rows before observed SGOV returns are untradeable; a missing SGOV return after proxy start fails the backtest instead of being replaced by zero.",
            f"Commission ({COMMISSION_BPS_PER_SIDE} bps/side), slippage ({SLIPPAGE_BPS_PER_SIDE} bps/side), and spread ({FULL_SPREAD_BPS} bps full, {HALF_SPREAD_BPS_PER_SIDE} bps/side) are configured assumptions, not observed fills.",
            "Portfolio rebalances are model assumptions and may not match actual fills.",
            "Final trading decisions still require ChatGPT live quote/news/macro/portfolio review and human confirmation.",
        ],
        "assumptions": assumptions,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "strategy_vs_benchmarks": comparison,
        "trade_metrics": trade_metrics(trades),
        "exposure_summary": exposure_summary(equity),
        "latest_regime": equity.iloc[-1]["regime"] if not equity.empty else None,
        "latest_raw_regime_signal_same_day": equity.iloc[-1]["raw_regime_signal_same_day"] if not equity.empty else None,
        "latest_strategy_value": clean_float(equity.iloc[-1]["strategy_value"], 2) if not equity.empty else None,
    }

    OUT.mkdir(exist_ok=True)
    atomic_write_csv(
        OUT / "portfolio_equity_curve.csv",
        attach_output_metadata(
            equity,
            generated_at_utc=generated_at_utc,
            data_timestamp=data_timestamp,
            portfolio_fingerprint=portfolio_fingerprint,
            split_fingerprint=split_fingerprint,
            model_fingerprint=model_fingerprint,
        ),
    )
    atomic_write_csv(
        OUT / "portfolio_trades.csv",
        attach_output_metadata(
            trades,
            generated_at_utc=generated_at_utc,
            data_timestamp=data_timestamp,
            portfolio_fingerprint=portfolio_fingerprint,
            split_fingerprint=split_fingerprint,
            model_fingerprint=model_fingerprint,
        ),
    )
    atomic_write_csv(
        OUT / "portfolio_vs_benchmark.csv",
        attach_output_metadata(
            vs,
            generated_at_utc=generated_at_utc,
            data_timestamp=data_timestamp,
            portfolio_fingerprint=portfolio_fingerprint,
            split_fingerprint=split_fingerprint,
            model_fingerprint=model_fingerprint,
        ),
    )
    atomic_write_json(OUT / "portfolio_backtest.json", report)

    print("Saved docs/portfolio_backtest.json, docs/portfolio_equity_curve.csv, docs/portfolio_trades.csv, docs/portfolio_vs_benchmark.csv")


if __name__ == "__main__":
    main()
