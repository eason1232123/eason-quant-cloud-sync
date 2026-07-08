from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT = Path("docs")

INITIAL_CAPITAL = 20000.0
SLIPPAGE_BPS = 5.0
MIN_TRADE_DOLLARS = 25.0
EXECUTION_SHIFT_DAYS = 1

CORE_TICKERS = ["QQQ", "SMH", "MSFT", "SPY"]
BENCHMARK_TICKERS = ["SPY", "QQQ", "SMH"]
CASH_TICKER = "SGOV"

BASE_WEIGHTS = {
    "QQQ": 0.30,
    "SMH": 0.25,
    "MSFT": 0.20,
    "SPY": 0.10,
    "CASH": 0.15,
}

DEFENSIVE_WEIGHTS = {
    "QQQ": 0.18,
    "SMH": 0.12,
    "MSFT": 0.16,
    "SPY": 0.14,
    "CASH": 0.40,
}

SEVERE_DEFENSIVE_WEIGHTS = {
    "QQQ": 0.10,
    "SMH": 0.08,
    "MSFT": 0.12,
    "SPY": 0.10,
    "CASH": 0.60,
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


def csv_path(ticker: str) -> Path:
    safe = ticker.replace("/", "-").replace(".", "-")
    return OUT / f"{safe}_daily.csv"


def load_price(ticker: str) -> pd.DataFrame:
    path = csv_path(ticker)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    price_col = "price" if "price" in df.columns else "adjClose" if "adjClose" in df.columns else "close"
    out = df[["date", price_col]].rename(columns={price_col: ticker}).copy()
    out[ticker] = pd.to_numeric(out[ticker], errors="coerce")
    out = out.dropna().sort_values("date").drop_duplicates("date", keep="last")
    return out


def load_prices(tickers: list[str]) -> pd.DataFrame:
    merged = None
    loaded = []
    for ticker in tickers:
        df = load_price(ticker)
        if df.empty:
            continue
        loaded.append(ticker)
        merged = df if merged is None else merged.merge(df, on="date", how="inner")
    if merged is None or len(loaded) < len(tickers):
        missing = [t for t in tickers if t not in loaded]
        raise SystemExit(f"Missing required price history for: {missing}")
    return merged.sort_values("date").reset_index(drop=True)


def add_trend_columns(prices: pd.DataFrame) -> pd.DataFrame:
    d = prices.copy()
    for ticker in CORE_TICKERS:
        d[f"{ticker}_ma50"] = d[ticker].rolling(50).mean()
        d[f"{ticker}_ma200"] = d[ticker].rolling(200).mean()
        d[f"{ticker}_ret20"] = d[ticker].pct_change(20)
    return d


def choose_regime(row: pd.Series) -> str:
    qqq_below_200 = row["QQQ"] < row.get("QQQ_ma200", np.nan)
    smh_below_200 = row["SMH"] < row.get("SMH_ma200", np.nan)
    spy_below_200 = row["SPY"] < row.get("SPY_ma200", np.nan)
    qqq_weak = row.get("QQQ_ret20", 0) <= -0.08
    smh_weak = row.get("SMH_ret20", 0) <= -0.12

    if pd.isna(row.get("QQQ_ma200")) or pd.isna(row.get("SMH_ma200")) or pd.isna(row.get("SPY_ma200")):
        return "warmup_base"
    if spy_below_200 and qqq_below_200 and smh_below_200:
        return "severe_defensive"
    if (qqq_below_200 and smh_below_200) or (qqq_weak and smh_weak):
        return "defensive"
    return "base"


def weights_for_regime(regime: str) -> dict[str, float]:
    if regime == "severe_defensive":
        return SEVERE_DEFENSIVE_WEIGHTS.copy()
    if regime == "defensive":
        return DEFENSIVE_WEIGHTS.copy()
    return BASE_WEIGHTS.copy()


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    total = sum(w.values())
    if total == 0:
        return BASE_WEIGHTS.copy()
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
    sgov["cash_ret"] = sgov[CASH_TICKER].pct_change().fillna(0.0)
    return sgov.set_index("date")["cash_ret"]


def simulate_strategy(prices: pd.DataFrame, cash_returns: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = add_trend_columns(prices)
    d = d.dropna(subset=CORE_TICKERS).reset_index(drop=True)

    # Regime is known only after the signal day closes. Execution therefore uses
    # the previous row's regime signal at today's close. This avoids same-close
    # signal + same-close fill look-ahead.
    d["raw_regime_signal"] = d.apply(choose_regime, axis=1)
    d["exec_regime"] = d["raw_regime_signal"].shift(EXECUTION_SHIFT_DAYS).fillna("warmup_base")

    if cash_returns is None:
        d["CASH_RET"] = 0.0
    else:
        d = d.merge(cash_returns.rename("CASH_RET").reset_index(), on="date", how="left")
        d["CASH_RET"] = d["CASH_RET"].fillna(0.0)

    shares = {ticker: 0.0 for ticker in CORE_TICKERS}
    cash = INITIAL_CAPITAL
    records = []
    trades = []
    prev_date = None
    prev_regime = None

    for i, row in d.iterrows():
        date = pd.to_datetime(row["date"])
        if i > 0:
            cash *= 1 + float(row.get("CASH_RET", 0.0) or 0.0)

        prices_now = {ticker: float(row[ticker]) for ticker in CORE_TICKERS}
        holdings_value = {ticker: shares[ticker] * prices_now[ticker] for ticker in CORE_TICKERS}
        total_value = cash + sum(holdings_value.values())

        current_weights = {ticker: holdings_value[ticker] / total_value if total_value else 0.0 for ticker in CORE_TICKERS}
        current_weights["CASH"] = cash / total_value if total_value else 1.0

        regime = str(row["exec_regime"])
        raw_regime_signal = str(row["raw_regime_signal"])
        target = normalize_weights(weights_for_regime(regime))
        month_changed = is_month_change(prev_date, date)
        regime_changed = prev_regime is not None and regime != prev_regime

        if should_rebalance(current_weights, target, regime_changed, month_changed):
            pre_trade_value = total_value
            for ticker in CORE_TICKERS:
                target_dollars = pre_trade_value * target[ticker]
                current_dollars = shares[ticker] * prices_now[ticker]
                delta = target_dollars - current_dollars
                if abs(delta) < MIN_TRADE_DOLLARS:
                    continue

                slip = SLIPPAGE_BPS / 10000.0
                if delta > 0:
                    execution_price = prices_now[ticker] * (1 + slip)
                    buy_shares = min(delta, cash) / execution_price
                    if buy_shares <= 0:
                        continue
                    dollars = buy_shares * execution_price
                    shares[ticker] += buy_shares
                    cash -= dollars
                    action = "BUY"
                    traded_shares = buy_shares
                else:
                    execution_price = prices_now[ticker] * (1 - slip)
                    sell_shares = min(abs(delta) / execution_price, shares[ticker])
                    if sell_shares <= 0:
                        continue
                    dollars = sell_shares * execution_price
                    shares[ticker] -= sell_shares
                    cash += dollars
                    action = "SELL"
                    traded_shares = sell_shares

                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": ticker,
                        "action": action,
                        "shares": clean_float(traded_shares, 6),
                        "execution_price": clean_float(execution_price, 4),
                        "dollars": clean_float(dollars, 2),
                        "regime_used_for_execution": regime,
                        "raw_regime_signal_same_day": raw_regime_signal,
                        "reason": "regime_change" if regime_changed else "monthly_or_drift_rebalance",
                    }
                )

            holdings_value = {ticker: shares[ticker] * prices_now[ticker] for ticker in CORE_TICKERS}
            total_value = cash + sum(holdings_value.values())
            current_weights = {ticker: holdings_value[ticker] / total_value if total_value else 0.0 for ticker in CORE_TICKERS}
            current_weights["CASH"] = cash / total_value if total_value else 1.0

        records.append(
            {
                "date": date.date().isoformat(),
                "strategy_value": clean_float(total_value, 4),
                "regime": regime,
                "raw_regime_signal_same_day": raw_regime_signal,
                "execution_assumption": "previous-day regime signal, current-day close execution",
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
        prev_regime = regime

    equity = pd.DataFrame(records)
    trade_log = pd.DataFrame(trades)
    equity["date"] = pd.to_datetime(equity["date"])
    return equity, trade_log, {
        "initial_capital": INITIAL_CAPITAL,
        "slippage_bps": SLIPPAGE_BPS,
        "execution_shift_days": EXECUTION_SHIFT_DAYS,
        "execution_assumption": "Regime signal is calculated after the prior close and executed on the next bar/current row. This avoids same-close look-ahead.",
        "base_weights": BASE_WEIGHTS,
        "defensive_weights": DEFENSIVE_WEIGHTS,
        "severe_defensive_weights": SEVERE_DEFENSIVE_WEIGHTS,
        "rebalance_policy": "monthly, regime-change, or drift >= 5 percentage points",
        "regime_policy": "base unless QQQ/SMH/SPY trend weakness triggers defensive or severe defensive weights",
    }


def metric_summary(values: pd.Series, dates: pd.Series | None = None) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 3:
        return {}
    rets = v.pct_change().dropna()
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


def main() -> None:
    prices = load_prices(CORE_TICKERS)
    cash_returns = build_cash_returns()
    equity, trades, assumptions = simulate_strategy(prices, cash_returns)
    vs = benchmark_curves(prices, equity)

    strategy_metrics = metric_summary(equity["strategy_value"], equity["date"])
    benchmark_metrics = {}
    for col in [c for c in vs.columns if c.startswith("buy_hold_")]:
        benchmark_metrics[col.replace("buy_hold_", "")] = metric_summary(vs[col], vs["date"])

    comparison = []
    for ticker, metrics in benchmark_metrics.items():
        comparison.append(
            {
                "benchmark": ticker,
                "strategy_cagr_minus_benchmark_pct": clean_float(strategy_metrics.get("cagr_pct", 0) - metrics.get("cagr_pct", 0), 2),
                "strategy_maxdd_minus_benchmark_pct": clean_float(strategy_metrics.get("max_drawdown_pct", 0) - metrics.get("max_drawdown_pct", 0), 2),
                "strategy_final_minus_benchmark_dollars": clean_float(strategy_metrics.get("final_value", 0) - metrics.get("final_value", 0), 2),
            }
        )

    report = {
        "available": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "portfolio-backtest-v2.1-next-bar-regime-execution",
        "purpose": "Model-level portfolio backtest for Eason strategy. This is not live execution and not personalized account truth.",
        "important_limits": [
            "Uses daily adjusted prices from cached docs CSV files; not intraday bid/ask.",
            "Uses model weights, not Eason's private real account shares/cash.",
            "Regime signal is shifted one trading day to avoid same-close signal/fill look-ahead.",
            "Cash return uses SGOV when available; otherwise cash earns 0% before SGOV history.",
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
    equity.to_csv(OUT / "portfolio_equity_curve.csv", index=False)
    trades.to_csv(OUT / "portfolio_trades.csv", index=False)
    vs.to_csv(OUT / "portfolio_vs_benchmark.csv", index=False)
    with open(OUT / "portfolio_backtest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/portfolio_backtest.json, docs/portfolio_equity_curve.csv, docs/portfolio_trades.csv, docs/portfolio_vs_benchmark.csv")


if __name__ == "__main__":
    main()
