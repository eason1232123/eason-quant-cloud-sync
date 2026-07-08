from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT = Path("docs")
PORTFOLIO_VS_PATH = OUT / "portfolio_vs_benchmark.csv"
EQUITY_PATH = OUT / "portfolio_equity_curve.csv"

PERIODS = [
    {"name": "train_2005_2016", "start": "2005-01-01", "end": "2016-12-31", "purpose": "training / long-cycle sanity check"},
    {"name": "validation_2017_2021", "start": "2017-01-01", "end": "2021-12-31", "purpose": "validation / bull-cycle and covid shock check"},
    {"name": "test_2022_latest", "start": "2022-01-01", "end": None, "purpose": "out-of-sample style test / inflation-rate-hike-AI cycle"},
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


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}. Run scripts/build_portfolio_backtest.py first.")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty required file: {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def metrics(values: pd.Series, dates: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 3:
        return {"available": False, "reason": "insufficient rows"}
    rets = v.pct_change().dropna()
    years = len(rets) / 252
    total_return = v.iloc[-1] / v.iloc[0] - 1
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol and not pd.isna(vol) and vol != 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = (rets.mean() * 252) / downside if downside and not pd.isna(downside) and downside != 0 else np.nan
    dd = v / v.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd and not pd.isna(max_dd) and max_dd != 0 else np.nan
    return {
        "available": True,
        "start_date": pd.to_datetime(dates.iloc[0]).date().isoformat(),
        "end_date": pd.to_datetime(dates.iloc[-1]).date().isoformat(),
        "trading_days": int(len(v)),
        "years": clean_float(years, 2),
        "total_return_pct": clean_float(total_return * 100, 2),
        "cagr_pct": clean_float(cagr * 100, 2),
        "volatility_pct": clean_float(vol * 100, 2),
        "sharpe": clean_float(sharpe, 3),
        "sortino": clean_float(sortino, 3),
        "calmar": clean_float(calmar, 3),
        "max_drawdown_pct": clean_float(max_dd * 100, 2),
        "daily_win_rate_pct": clean_float((rets > 0).mean() * 100, 2),
        "final_value": clean_float(v.iloc[-1], 2),
    }


def period_slice(df: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    d = df[df["date"] >= pd.to_datetime(start)].copy()
    if end:
        d = d[d["date"] <= pd.to_datetime(end)]
    return d.reset_index(drop=True)


def build_walk_forward(vs: pd.DataFrame) -> dict:
    rows = []
    benchmark_cols = [c for c in vs.columns if c.startswith("buy_hold_")]
    for p in PERIODS:
        d = period_slice(vs, p["start"], p["end"])
        if d.empty or len(d) < 30:
            rows.append({"period": p, "available": False, "reason": "insufficient data"})
            continue

        strategy = metrics(d["strategy_value"], d["date"])
        benchmarks = {c.replace("buy_hold_", ""): metrics(d[c], d["date"]) for c in benchmark_cols}
        comparisons = []
        for bench, m in benchmarks.items():
            if not strategy.get("available") or not m.get("available"):
                continue
            comparisons.append(
                {
                    "benchmark": bench,
                    "strategy_cagr_minus_benchmark_pct": clean_float(strategy.get("cagr_pct", 0) - m.get("cagr_pct", 0), 2),
                    "strategy_maxdd_minus_benchmark_pct": clean_float(strategy.get("max_drawdown_pct", 0) - m.get("max_drawdown_pct", 0), 2),
                    "strategy_sharpe_minus_benchmark": clean_float(strategy.get("sharpe", 0) - m.get("sharpe", 0), 3),
                    "strategy_final_minus_benchmark_dollars": clean_float(strategy.get("final_value", 0) - m.get("final_value", 0), 2),
                }
            )
        rows.append(
            {
                "period": p,
                "available": True,
                "strategy_metrics": strategy,
                "benchmark_metrics": benchmarks,
                "strategy_vs_benchmarks": comparisons,
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "walk-forward-v2.5",
        "purpose": "Check whether the portfolio model works across train/validation/test-style periods instead of only one lucky historical window.",
        "important_limit": "This is not true machine-learning walk-forward optimization. It is a fixed-rule period stability check.",
        "periods": rows,
    }


def regime_report(equity: pd.DataFrame) -> dict:
    if "regime" not in equity.columns:
        return {"available": False, "reason": "regime column missing"}
    d = equity.copy()
    d["ret"] = d["strategy_value"].pct_change()
    rows = []
    for regime, g in d.groupby("regime"):
        v = g["strategy_value"]
        rets = g["ret"].dropna()
        dd = v / v.cummax() - 1 if len(v) else pd.Series(dtype=float)
        rows.append(
            {
                "regime": regime,
                "trading_days": int(len(g)),
                "share_of_history_pct": clean_float(len(g) / len(d) * 100, 2),
                "avg_daily_return_pct": clean_float(rets.mean() * 100, 4) if len(rets) else None,
                "daily_win_rate_pct": clean_float((rets > 0).mean() * 100, 2) if len(rets) else None,
                "max_regime_drawdown_pct": clean_float(dd.min() * 100, 2) if len(dd) else None,
                "avg_cash_weight_pct": clean_float(g.get("cash_weight", pd.Series(dtype=float)).mean() * 100, 2),
                "avg_tech_ai_concentration_pct": clean_float(g.get("tech_ai_concentration", pd.Series(dtype=float)).mean() * 100, 2),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "market-regime-v2.5",
        "purpose": "Show how often the model is in base/defensive/severe defensive mode and how strategy returns behaved in each regime.",
        "regimes": sorted(rows, key=lambda x: x["trading_days"], reverse=True),
    }


def overfitting_check(walk: dict) -> dict:
    warnings = []
    scores = []
    available_periods = [p for p in walk.get("periods", []) if p.get("available")]

    for p in available_periods:
        name = p.get("period", {}).get("name")
        sm = p.get("strategy_metrics", {})
        cagr = sm.get("cagr_pct")
        maxdd = sm.get("max_drawdown_pct")
        sharpe = sm.get("sharpe")
        if cagr is not None and cagr > 0:
            scores.append(1)
        else:
            scores.append(0)
            warnings.append(f"{name}: strategy CAGR <= 0")
        if sharpe is not None and sharpe > 0.3:
            scores.append(1)
        else:
            scores.append(0)
            warnings.append(f"{name}: Sharpe <= 0.3")
        if maxdd is not None and maxdd > -45:
            scores.append(1)
        else:
            scores.append(0)
            warnings.append(f"{name}: drawdown worse than -45% or unavailable")

        comps = p.get("strategy_vs_benchmarks", [])
        if comps:
            better_cagr_count = sum(1 for c in comps if (c.get("strategy_cagr_minus_benchmark_pct") or -999) > -3)
            if better_cagr_count >= 1:
                scores.append(1)
            else:
                scores.append(0)
                warnings.append(f"{name}: strategy CAGR lags all checked benchmarks by more than 3pct")

    score_pct = round(sum(scores) / len(scores) * 100, 1) if scores else 0.0
    if score_pct >= 75:
        verdict = "PASS_STABILITY_CHECK"
    elif score_pct >= 50:
        verdict = "MIXED_NEEDS_CAUTION"
    else:
        verdict = "FAIL_OR_OVERFIT_RISK"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "overfitting-check-v2.5",
        "purpose": "Flag whether the model only works in one historical window or remains acceptable across multiple periods.",
        "verdict": verdict,
        "score_0_100": score_pct,
        "warnings": warnings,
        "important_limit": "This is a heuristic stability check, not a formal statistical proof.",
    }


def main() -> None:
    vs = load_csv(PORTFOLIO_VS_PATH)
    equity = load_csv(EQUITY_PATH)
    walk = build_walk_forward(vs)
    regime = regime_report(equity)
    overfit = overfitting_check(walk)

    with open(OUT / "walk_forward_report.json", "w", encoding="utf-8") as f:
        json.dump(walk, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(OUT / "market_regime_report.json", "w", encoding="utf-8") as f:
        json.dump(regime, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(OUT / "overfitting_check.json", "w", encoding="utf-8") as f:
        json.dump(overfit, f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/walk_forward_report.json, docs/market_regime_report.json, docs/overfitting_check.json")


if __name__ == "__main__":
    main()
