from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import textwrap
from typing import Any

import pandas as pd


STRATEGY_CONTRACT_VERSION = "v6-rc2-shared-rules-next-close"
RULE_IMPLEMENTATION_REVISION = "shared-rule-engine-2026-07-12.1"
EXECUTION_SHIFT_BARS = 1
COMMISSION_BPS_PER_SIDE = 5.0
SLIPPAGE_BPS_PER_SIDE = 5.0
FULL_SPREAD_BPS = 2.0
HALF_SPREAD_BPS_PER_SIDE = FULL_SPREAD_BPS / 2
COMMISSION_RATE = COMMISSION_BPS_PER_SIDE / 10_000
EXECUTION_IMPACT_RATE = (SLIPPAGE_BPS_PER_SIDE + HALF_SPREAD_BPS_PER_SIDE) / 10_000

FORWARD_HORIZONS = (5, 10, 20, 60)
PRIMARY_EVALUATION_HORIZON = 20
MIN_EFFECTIVE_SAMPLE = 20

RULE_PARAMETERS: dict[str, dict[str, float]] = {
    "pullback_reclaim_5dma": {"ret_5d_max": -0.04},
    "rsi_oversold_reclaim_40": {
        "oversold_level": 35.0,
        "reclaim_level": 40.0,
        "oversold_lookback": 5.0,
    },
    "ma20_reclaim_bullish": {"ma20_slope_10d_min": -0.01},
    "ma50_reclaim_bullish": {},
    "relative_strength_rebound": {"ret_5d_max": -0.035},
    "momentum_leader": {"ret_20d_min": 0.05, "rsi_max": 78.0},
}

EXIT_PARAMETERS: dict[str, dict[str, float]] = {
    "close_below_ma20": {},
    "close_below_ma50": {},
    "close_below_ma20_or_rsi_above_70": {"rsi_max": 70.0},
    "close_below_ma20_or_rsi_above_82": {"rsi_max": 82.0},
}

ENTRY_RULE_SPECS: dict[str, dict[str, str]] = {
    "pullback_reclaim_5dma": {
        "formula": "ret_5d<=-0.04 and close>ma5",
        "exit_rule": "close_below_ma20",
        "description": "Five-day pullback reclaims the 5-day average.",
    },
    "rsi_oversold_reclaim_40": {
        "formula": "prior_5d_rsi_min<35 and rsi14>=40 and close>ma5",
        "exit_rule": "close_below_ma20_or_rsi_above_70",
        "description": "RSI recovers through 40 after an oversold reading while price is above MA5.",
    },
    "ma20_reclaim_bullish": {
        "formula": "prior_close<prior_ma20 and close>ma20 and close>ma50 and ma20_slope_10d>-0.01",
        "exit_rule": "close_below_ma20",
        "description": "Price reclaims MA20 in a bullish intermediate trend.",
    },
    "ma50_reclaim_bullish": {
        "formula": "prior_close<prior_ma50 and close>ma50 and close>ma200",
        "exit_rule": "close_below_ma50",
        "description": "Price reclaims MA50 while remaining above MA200.",
    },
    "relative_strength_rebound": {
        "formula": "ret_5d<=-0.035 and close>ma5 and relative_20d>0",
        "exit_rule": "close_below_ma20",
        "description": "A short pullback rebounds while 20-day relative strength stays positive.",
    },
    "momentum_leader": {
        "formula": "close>ma20 and close>ma50 and ret_20d>0.05 and relative_20d>0 and rsi14<78",
        "exit_rule": "close_below_ma20_or_rsi_above_82",
        "description": "Positive 20-day momentum and relative strength without extreme RSI.",
    },
}

RISK_RULE_SPECS: dict[str, dict[str, str]] = {
    "failed_rebound_risk": {
        "formula": "close<ma5 and close<prior_10d_low",
        "description": "Price breaks below MA5 and the prior 10-day close low.",
    },
    "break_ma20": {
        "formula": "prior_close>=prior_ma20 and close<ma20",
        "description": "Price breaks below MA20.",
    },
    "break_ma50": {
        "formula": "prior_close>=prior_ma50 and close<ma50",
        "description": "Price breaks below MA50.",
    },
}


SEMICONDUCTOR_TICKERS = {
    "SMH", "SOXX", "NVDA", "AVGO", "AMD", "ASML", "TSM", "MU", "LRCX", "AMAT",
    "KLAC", "ARM", "INTC", "MRVL", "QCOM", "ON", "ADI", "TXN", "NXPI", "MCHP", "MPWR",
}
SOFTWARE_TICKERS = {
    "IGV", "CRWD", "PLTR", "SNOW", "DDOG", "NET", "NOW", "PANW", "MDB", "ORCL",
    "CRM", "ADBE", "ZS", "OKTA", "TEAM", "SHOP", "CFLT", "ESTC", "GTLB", "PATH",
}
DEFENSIVE_TICKERS = {"GLD", "TLT", "IEF", "SHY", "XLV", "XLP", "XLU", "USMV", "VYM", "SGOV"}


def benchmark_for_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker == "QQQ":
        return "SPY"
    if ticker == "SPY":
        return "QQQ"
    if ticker == "SMH":
        return "QQQ"
    if ticker in SEMICONDUCTOR_TICKERS:
        return "SMH"
    if ticker in SOFTWARE_TICKERS:
        return "QQQ"
    if ticker in DEFENSIVE_TICKERS:
        return "SPY"
    return "SPY"


def rsi(close: pd.Series | pd.DataFrame, period: int = 14) -> pd.Series | pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss.replace(0, float("nan"))))


def relative_20d_frame(close: pd.DataFrame) -> pd.DataFrame:
    returns = close.pct_change(20, fill_method=None)
    relative = close * float("nan")
    for ticker in close.columns:
        benchmark = benchmark_for_ticker(str(ticker))
        if benchmark in returns.columns:
            relative[ticker] = returns[ticker] - returns[benchmark]
    return relative


def relative_20d_series(close: pd.Series, benchmark_close: pd.Series | None) -> pd.Series:
    if benchmark_close is None:
        return pd.Series(float("nan"), index=close.index, dtype=float)
    aligned_benchmark = benchmark_close.reindex(close.index)
    return close.pct_change(20, fill_method=None) - aligned_benchmark.pct_change(20, fill_method=None)


def rule_signals(
    close: pd.Series | pd.DataFrame,
    relative_20d: pd.Series | pd.DataFrame | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ret5 = close.pct_change(5, fill_method=None)
    ret20 = close.pct_change(20, fill_method=None)
    rsi14 = rsi(close)
    ma20_slope_10d = ma20 / ma20.shift(10) - 1
    relative = relative_20d if relative_20d is not None else close * float("nan")

    pullback = RULE_PARAMETERS["pullback_reclaim_5dma"]
    rsi_reclaim = RULE_PARAMETERS["rsi_oversold_reclaim_40"]
    ma20_reclaim = RULE_PARAMETERS["ma20_reclaim_bullish"]
    relative_rebound = RULE_PARAMETERS["relative_strength_rebound"]
    momentum = RULE_PARAMETERS["momentum_leader"]

    signals = {
        "pullback_reclaim_5dma": (ret5 <= pullback["ret_5d_max"]) & (close > ma5),
        "rsi_oversold_reclaim_40": (
            (
                rsi14.rolling(int(rsi_reclaim["oversold_lookback"])).min().shift(1)
                < rsi_reclaim["oversold_level"]
            )
            & (rsi14 >= rsi_reclaim["reclaim_level"])
            & (close > ma5)
        ),
        "ma20_reclaim_bullish": (
            (close.shift(1) < ma20.shift(1))
            & (close > ma20)
            & (close > ma50)
            & (ma20_slope_10d > ma20_reclaim["ma20_slope_10d_min"])
        ),
        "ma50_reclaim_bullish": (
            (close.shift(1) < ma50.shift(1))
            & (close > ma50)
            & (close > ma200)
        ),
        "relative_strength_rebound": (
            (ret5 <= relative_rebound["ret_5d_max"])
            & (close > ma5)
            & (relative > 0)
        ),
        "momentum_leader": (
            (close > ma20)
            & (close > ma50)
            & (ret20 > momentum["ret_20d_min"])
            & (relative > 0)
            & (rsi14 < momentum["rsi_max"])
        ),
        "failed_rebound_risk": (
            (close < ma5)
            & (close < close.rolling(10).min().shift(1))
        ),
    }
    return {name: signal.fillna(False).astype(bool) for name, signal in signals.items()}


def risk_signals(close: pd.Series | pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    signals = {
        "failed_rebound_risk": (close < ma5) & (close < close.rolling(10).min().shift(1)),
        "break_ma20": (close.shift(1) >= ma20.shift(1)) & (close < ma20),
        "break_ma50": (close.shift(1) >= ma50.shift(1)) & (close < ma50),
    }
    return {name: signal.fillna(False).astype(bool) for name, signal in signals.items()}


def exit_signals(close: pd.Series | pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    rsi14 = rsi(close)
    exits = {
        "close_below_ma20": close < ma20,
        "close_below_ma50": close < ma50,
        "close_below_ma20_or_rsi_above_70": (
            (close < ma20)
            | (rsi14 > EXIT_PARAMETERS["close_below_ma20_or_rsi_above_70"]["rsi_max"])
        ),
        "close_below_ma20_or_rsi_above_82": (
            (close < ma20)
            | (rsi14 > EXIT_PARAMETERS["close_below_ma20_or_rsi_above_82"]["rsi_max"])
        ),
    }
    return {name: signal.fillna(False).astype(bool) for name, signal in exits.items()}


def shift_for_execution(signal: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return signal.shift(EXECUTION_SHIFT_BARS).fillna(False).astype(bool)


def next_close_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    entry = close.shift(-EXECUTION_SHIFT_BARS)
    exit_ = close.shift(-(EXECUTION_SHIFT_BARS + horizon))
    return exit_ / entry - 1


def next_close_forward_mae(close: pd.Series, low: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    numeric_close = pd.to_numeric(close, errors="coerce")
    numeric_low = pd.to_numeric(low, errors="coerce")
    entry_price = numeric_close.shift(-EXECUTION_SHIFT_BARS)
    forward_low = (
        numeric_low.iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .min()
        .iloc[::-1]
        .shift(-(EXECUTION_SHIFT_BARS + 1))
    )
    valid = (
        entry_price.notna()
        & forward_low.notna()
        & entry_price.map(lambda value: math.isfinite(float(value)))
        & forward_low.map(lambda value: math.isfinite(float(value)))
        & (entry_price > 0)
        & (forward_low > 0)
    )
    return (forward_low / entry_price - 1).where(valid)


def non_overlapping_signal_mask(signal: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    selected = pd.Series(False, index=signal.index, dtype=bool)
    next_allowed_position = 0
    for position, active in enumerate(signal.fillna(False).astype(bool).to_numpy()):
        if active and position >= next_allowed_position:
            selected.iloc[position] = True
            # The next signal may occur on the prior window's exit bar because
            # its own entry is one bar later and therefore does not overlap.
            next_allowed_position = position + EXECUTION_SHIFT_BARS + horizon
    return selected


def net_return_after_round_trip_costs(gross_return: pd.Series | float) -> pd.Series | float:
    entry_multiplier = (1 + EXECUTION_IMPACT_RATE) * (1 + COMMISSION_RATE)
    exit_multiplier = (1 - EXECUTION_IMPACT_RATE) * (1 - COMMISSION_RATE)
    return (1 + gross_return) * exit_multiplier / entry_multiplier - 1


def execution_cost_assumptions() -> dict[str, Any]:
    return {
        "assumption_source": "configured_not_observed",
        "commission_bps_per_side": COMMISSION_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "full_spread_bps": FULL_SPREAD_BPS,
        "half_spread_bps_per_side": HALF_SPREAD_BPS_PER_SIDE,
        "spread_observation_status": "NOT_OBSERVED",
        "applicability": "uniform_model_assumption_for_all_tickers_and_historical_periods",
        "limitation": "Actual commissions, market impact, and bid-ask spreads vary by ticker, order size, venue, and time.",
        "vectorbt_fees_rate": COMMISSION_RATE,
        "vectorbt_slippage_plus_half_spread_rate": EXECUTION_IMPACT_RATE,
        "execution_shift_bars": EXECUTION_SHIFT_BARS,
        "execution_price_basis": "next_trading_bar_close",
    }


def _benchmark_mapping_payload() -> dict[str, list[str]]:
    return {
        "semiconductor_to_SMH": sorted(SEMICONDUCTOR_TICKERS - {"SMH"}),
        "software_to_QQQ": sorted(SOFTWARE_TICKERS),
        "defensive_to_SPY": sorted(DEFENSIVE_TICKERS),
        "QQQ_to_SPY": ["QQQ"],
        "SPY_to_QQQ": ["SPY"],
        "SMH_to_QQQ": ["SMH"],
        "default_to_SPY": ["*"],
    }


def _normalized_function_ast(function: Any) -> str:
    source = textwrap.dedent(inspect.getsource(function))
    return ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)


def rule_contract_payload(*, include_implementation: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "implementation_revision": RULE_IMPLEMENTATION_REVISION,
        "entry_rules": ENTRY_RULE_SPECS,
        "risk_rules": RISK_RULE_SPECS,
        "rule_parameters": RULE_PARAMETERS,
        "exit_parameters": EXIT_PARAMETERS,
        "benchmark_mapping": _benchmark_mapping_payload(),
    }
    if include_implementation:
        payload["implementation_ast"] = {
            function.__name__: _normalized_function_ast(function)
            for function in (
                benchmark_for_ticker,
                rsi,
                relative_20d_frame,
                relative_20d_series,
                rule_signals,
                risk_signals,
                exit_signals,
            )
        }
    return payload


def rule_fingerprint() -> str:
    encoded = json.dumps(
        rule_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strategy_contract_payload() -> dict[str, Any]:
    return {
        "version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": rule_fingerprint(),
        "rule_contract": rule_contract_payload(include_implementation=False),
        "evaluation": {
            "forward_horizons": list(FORWARD_HORIZONS),
            "primary_horizon": PRIMARY_EVALUATION_HORIZON,
            "minimum_effective_sample": MIN_EFFECTIVE_SAMPLE,
            "sample_independence": "chronological_non_overlapping_next_close_windows",
        },
        "costs": execution_cost_assumptions(),
        "execution_implementation_ast": {
            function.__name__: _normalized_function_ast(function)
            for function in (
                shift_for_execution,
                next_close_forward_return,
                next_close_forward_mae,
                non_overlapping_signal_mask,
                net_return_after_round_trip_costs,
            )
        },
        "fingerprint_scope": (
            "rules, benchmark mapping, evaluation horizons, next-close execution, and configured costs; "
            "portfolio weights and train/validation/test split are intentionally excluded until frozen"
        ),
    }


def strategy_fingerprint() -> str:
    encoded = json.dumps(
        strategy_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


RULE_FINGERPRINT = rule_fingerprint()
STRATEGY_FINGERPRINT = strategy_fingerprint()
