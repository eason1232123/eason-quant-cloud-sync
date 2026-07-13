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

from scripts.model_governance import (
    governance_fingerprint,
    incumbent_model,
    load_governance_config,
    model_fingerprint,
)
from scripts.market_clock import MARKET_TIMEZONE
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
)
from scripts.strategy_contract import RULE_FINGERPRINT, STRATEGY_CONTRACT_VERSION, STRATEGY_FINGERPRINT
from scripts.validate_validation_split import (
    DEFAULT_MANIFEST,
    load_strict_json,
    split_manifest_fingerprint,
    validate_split_manifest,
)

OUT = Path("docs")
PORTFOLIO_VS_PATH = OUT / "portfolio_vs_benchmark.csv"
EQUITY_PATH = OUT / "portfolio_equity_curve.csv"
PORTFOLIO_REPORT_PATH = OUT / "portfolio_backtest.json"
REBASED_INITIAL_VALUE = 10000.0


def validation_periods(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": item["name"],
            "start": item["signal_start_market_date"],
            "end": item["signal_end_market_date"],
            "classification": item["classification"],
            "label_maturation_end_market_date": item["label_maturation_end_market_date"],
            "purpose": "retrospective fixed-rule stability diagnostic; not pristine out-of-sample evidence",
        }
        for item in manifest["historical_partitions"]
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


def load_portfolio_report(path: Path = PORTFOLIO_REPORT_PATH) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Missing required file: {path}. Run scripts/build_portfolio_backtest.py first.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid portfolio report {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("available") is not True:
        raise SystemExit("Portfolio report is unavailable; retrospective diagnostics cannot be published")
    expected_contract = {
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    actual_contract = {key: value.get(key) for key in expected_contract}
    if actual_contract != expected_contract:
        raise SystemExit("Portfolio report strategy fingerprints do not match the running contract")
    return value


def metrics(values: pd.Series, dates: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce")
    if v.isna().any() or not np.isfinite(v).all() or (v <= 0).any():
        return {"available": False, "reason": "missing, non-finite, or non-positive values"}
    if len(v) < 3:
        return {"available": False, "reason": "insufficient rows"}
    rebased = v / v.iloc[0] * REBASED_INITIAL_VALUE
    rets = v.pct_change(fill_method=None).dropna()
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
        "source_series_start_value": clean_float(v.iloc[0], 2),
        "source_series_end_value": clean_float(v.iloc[-1], 2),
        "rebased_initial_value": REBASED_INITIAL_VALUE,
        "final_value": clean_float(rebased.iloc[-1], 2),
        "final_value_basis": "period_rebased_to_10000",
    }


def period_slice(df: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    d = df[df["date"] >= pd.to_datetime(start)].copy()
    if end:
        d = d[d["date"] <= pd.to_datetime(end)]
    return d.reset_index(drop=True)


def build_walk_forward(
    vs: pd.DataFrame,
    periods: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict:
    rows = []
    benchmark_cols = [c for c in vs.columns if c.startswith("buy_hold_")]
    for p in periods:
        d = period_slice(vs, p["start"], p["end"])
        if d.empty or len(d) < 30:
            rows.append(
                {
                    "period": p,
                    "available": False,
                    "coverage_status": "INSUFFICIENT_OR_MISSING",
                    "reason": "insufficient data within the frozen signal partition",
                }
            )
            continue

        actual_start = d["date"].min().date().isoformat()
        actual_end = d["date"].max().date().isoformat()
        coverage_complete = actual_start <= p["start"] and actual_end >= p["end"]

        strategy = metrics(d["strategy_value"], d["date"])
        benchmarks = {c.replace("buy_hold_", ""): metrics(d[c], d["date"]) for c in benchmark_cols}
        comparisons = []
        for bench, m in benchmarks.items():
            if not strategy.get("available") or not m.get("available"):
                continue
            strategy_cagr = strategy.get("cagr_pct")
            benchmark_cagr = m.get("cagr_pct")
            strategy_drawdown = strategy.get("max_drawdown_pct")
            benchmark_drawdown = m.get("max_drawdown_pct")
            strategy_sharpe = strategy.get("sharpe")
            benchmark_sharpe = m.get("sharpe")
            strategy_final = strategy.get("final_value")
            benchmark_final = m.get("final_value")
            comparisons.append(
                {
                    "benchmark": bench,
                    "strategy_cagr_minus_benchmark_pct": clean_float(strategy_cagr - benchmark_cagr, 2)
                    if strategy_cagr is not None and benchmark_cagr is not None
                    else None,
                    "strategy_maxdd_minus_benchmark_pct": clean_float(strategy_drawdown - benchmark_drawdown, 2)
                    if strategy_drawdown is not None and benchmark_drawdown is not None
                    else None,
                    "strategy_sharpe_minus_benchmark": clean_float(strategy_sharpe - benchmark_sharpe, 3)
                    if strategy_sharpe is not None and benchmark_sharpe is not None
                    else None,
                    "strategy_final_minus_benchmark_dollars": clean_float(strategy_final - benchmark_final, 2)
                    if strategy_final is not None and benchmark_final is not None
                    else None,
                }
            )
        rows.append(
            {
                "period": p,
                "available": True,
                "coverage_status": "COMPLETE" if coverage_complete else "PARTIAL",
                "eligible_for_validation_claim": False,
                "actual_coverage_start": actual_start,
                "actual_coverage_end": actual_end,
                "strategy_metrics": strategy,
                "benchmark_metrics": benchmarks,
                "strategy_vs_benchmarks": comparisons,
            }
        )
    return {
        **metadata,
        "version": "retrospective-period-stability-v3.0",
        "purpose": "Describe fixed-rule historical stability inside frozen, purged partitions without claiming pristine out-of-sample validation.",
        "evidence_classification": "RETROSPECTIVE_CONTAMINATED",
        "eligible_as_pristine_out_of_sample": False,
        "important_limit": "This is a retrospective fixed-rule period diagnostic, not true walk-forward optimization and not pristine out-of-sample evidence.",
        "periods": rows,
    }


def regime_report(equity: pd.DataFrame, metadata: dict[str, Any]) -> dict:
    if "regime" not in equity.columns:
        return {**metadata, "available": False, "reason": "regime column missing"}
    d = equity.copy()
    d["ret"] = d["strategy_value"].pct_change(fill_method=None)
    d["regime_episode"] = d["regime"].ne(d["regime"].shift()).cumsum()
    rows = []
    for regime, g in d.groupby("regime"):
        rets = g["ret"].dropna()
        episode_drawdowns = []
        for _, episode in g.groupby("regime_episode"):
            values = episode["strategy_value"]
            if not values.empty:
                episode_drawdowns.append(float((values / values.cummax() - 1).min()))
        rows.append(
            {
                "regime": regime,
                "trading_days": int(len(g)),
                "share_of_history_pct": clean_float(len(g) / len(d) * 100, 2),
                "avg_daily_return_pct": clean_float(rets.mean() * 100, 4) if len(rets) else None,
                "daily_win_rate_pct": clean_float((rets > 0).mean() * 100, 2) if len(rets) else None,
                "max_regime_drawdown_pct": clean_float(min(episode_drawdowns) * 100, 2)
                if episode_drawdowns
                else None,
                "drawdown_definition": "minimum drawdown within contiguous regime episodes",
                "avg_cash_weight_pct": clean_float(g.get("cash_weight", pd.Series(dtype=float)).mean() * 100, 2),
                "avg_tech_ai_concentration_pct": clean_float(g.get("tech_ai_concentration", pd.Series(dtype=float)).mean() * 100, 2),
            }
        )
    return {
        **metadata,
        "version": "market-regime-v3.0-contiguous-episodes",
        "purpose": "Show how often the model is in base/defensive/severe defensive mode and how strategy returns behaved in each regime.",
        "regimes": sorted(rows, key=lambda x: x["trading_days"], reverse=True),
    }


def overfitting_check(walk: dict, metadata: dict[str, Any]) -> dict:
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
            better_cagr_count = sum(
                1
                for c in comps
                if c.get("strategy_cagr_minus_benchmark_pct") is not None
                and float(c["strategy_cagr_minus_benchmark_pct"]) > -3
            )
            if better_cagr_count >= 1:
                scores.append(1)
            else:
                scores.append(0)
                warnings.append(f"{name}: strategy CAGR lags all checked benchmarks by more than 3pct")

    score_pct = round(sum(scores) / len(scores) * 100, 1) if scores else 0.0
    if score_pct >= 75:
        heuristic_verdict = "PASS_STABILITY_CHECK"
    elif score_pct >= 50:
        heuristic_verdict = "MIXED_NEEDS_CAUTION"
    else:
        heuristic_verdict = "FAIL_OR_OVERFIT_RISK"

    warnings.append(
        "All scored periods were visible during model development; heuristic stability cannot validate V6."
    )

    return {
        **metadata,
        "version": "overfitting-check-v3.0-retrospective-only",
        "purpose": "Report retrospective stability heuristics while preventing them from being mistaken for model validation.",
        "verdict": "RETROSPECTIVE_ONLY_NOT_VALIDATED",
        "raw_heuristic_verdict": heuristic_verdict,
        "score_0_100": score_pct,
        "evidence_classification": "RETROSPECTIVE_CONTAMINATED",
        "prospective_validation_status": "NOT_YET_SUFFICIENT",
        "warnings": warnings,
        "important_limit": "This heuristic is descriptive only; a PASS-like raw score is not V6 validation and cannot authorize a trade.",
    }


def main() -> None:
    manifest = load_strict_json(DEFAULT_MANIFEST)
    validate_split_manifest(manifest)
    governance = load_governance_config()
    governance_fp = governance_fingerprint(governance)
    incumbent = incumbent_model(governance)
    incumbent_fp = model_fingerprint(incumbent)
    portfolio_report = load_portfolio_report()
    vs = load_csv(PORTFOLIO_VS_PATH)
    equity = load_csv(EQUITY_PATH)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "generated_at_utc": generated_at_utc,
        "data_source": portfolio_report.get("data_source"),
        "market_timezone": portfolio_report.get("market_timezone"),
        "data_timestamp": portfolio_report.get("data_timestamp"),
        "data_timestamp_granularity": portfolio_report.get("data_timestamp_granularity"),
        "data_timestamp_status": portfolio_report.get("data_timestamp_status"),
        "price_frequency": portfolio_report.get("price_frequency"),
        "price_adjustment_policy": portfolio_report.get("price_adjustment_policy"),
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "split_manifest_fingerprint": split_manifest_fingerprint(manifest),
        "portfolio_contract_version": portfolio_report.get("portfolio_contract_version"),
        "portfolio_contract_fingerprint": portfolio_report.get("portfolio_contract_fingerprint"),
        "full_model_fingerprint": portfolio_report.get("full_model_fingerprint"),
        "model_governance_fingerprint": portfolio_report.get("model_governance_fingerprint"),
        "incumbent_model_id": portfolio_report.get("incumbent_model_id"),
        "incumbent_model_fingerprint": portfolio_report.get("incumbent_model_fingerprint"),
        "survivorship_bias_status": "KNOWN_UNCONTROLLED_CURRENT_FIXED_ASSET_SET",
    }
    required_metadata = {
        "data_source": str,
        "market_timezone": str,
        "data_timestamp": str,
        "data_timestamp_granularity": str,
        "data_timestamp_status": str,
        "price_frequency": str,
        "price_adjustment_policy": str,
    }
    if any(not isinstance(metadata[key], expected) or not metadata[key] for key, expected in required_metadata.items()):
        raise SystemExit("Portfolio report metadata is incomplete; retrospective reports were not published")
    if metadata["market_timezone"] != MARKET_TIMEZONE:
        raise SystemExit("Portfolio report market timezone is inconsistent")
    if metadata["data_timestamp_granularity"] != DATA_TIMESTAMP_GRANULARITY:
        raise SystemExit("Portfolio report timestamp granularity is inconsistent")
    if metadata["price_frequency"] != PRICE_FREQUENCY or metadata["price_adjustment_policy"] != PRICE_ADJUSTMENT_POLICY:
        raise SystemExit("Portfolio report price metadata is inconsistent")
    if portfolio_report.get("split_manifest_fingerprint") != metadata["split_manifest_fingerprint"]:
        raise SystemExit("Portfolio report belongs to a different validation split")
    for field in ("portfolio_contract_fingerprint", "full_model_fingerprint"):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise SystemExit(f"Portfolio report {field} is missing or invalid")
    expected_governance = {
        "model_governance_fingerprint": governance_fp,
        "incumbent_model_id": incumbent["model_id"],
        "incumbent_model_fingerprint": incumbent_fp,
    }
    actual_governance = {field: metadata.get(field) for field in expected_governance}
    if actual_governance != expected_governance:
        raise SystemExit("Portfolio report model-governance identity is missing or mismatched")

    periods = validation_periods(manifest)
    walk = build_walk_forward(vs, periods, metadata)
    regime = regime_report(equity, metadata)
    overfit = overfitting_check(walk, metadata)

    with open(OUT / "walk_forward_report.json", "w", encoding="utf-8") as f:
        json.dump(walk, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(OUT / "market_regime_report.json", "w", encoding="utf-8") as f:
        json.dump(regime, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(OUT / "overfitting_check.json", "w", encoding="utf-8") as f:
        json.dump(overfit, f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/walk_forward_report.json, docs/market_regime_report.json, docs/overfitting_check.json")


if __name__ == "__main__":
    main()
