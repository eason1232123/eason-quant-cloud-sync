from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

OUT = Path("docs")
REPORT_PATH = OUT / "market_report.json"
MIN_SAMPLE = 20
PRIMARY_HORIZON = "20d"

BUY_RULES = {
    "pullback_reclaim_5dma",
    "rsi_oversold_reclaim_40",
    "ma20_reclaim_bullish",
    "ma50_reclaim_bullish",
    "relative_strength_rebound",
    "momentum_leader",
}
RISK_RULES = {"failed_rebound_risk"}


def clean_float(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        v = float(value)
        if pd.isna(v):
            return None
        return round(v, digits)
    except Exception:
        return None


def as_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def pct(value: Any) -> Any:
    v = clean_float(value, 4)
    if v is None:
        return None
    return clean_float(v * 100, 2)


def get_alpha(row: dict, ticker: str, technical: dict) -> tuple[str | None, Any]:
    preferred = technical.get("relative_benchmark")
    keys = []
    if preferred:
        keys.append(f"avg_alpha_vs_{preferred}")
    keys += ["avg_alpha_vs_QQQ", "avg_alpha_vs_SPY", "avg_alpha_vs_SMH", "avg_alpha_vs_SOXX"]
    for key in keys:
        if key in row and row.get(key) is not None:
            return key.replace("avg_alpha_vs_", ""), clean_float(row.get(key))
    return None, None


def evidence_passes(row: dict, alpha: Any) -> tuple[bool, list[str]]:
    reasons = []
    samples = row.get("samples", 0) or 0
    win_rate = row.get("win_rate")
    median_return = row.get("median_return")
    avg_return = row.get("avg_return")
    worst_mae = row.get("worst_mae")

    if samples < MIN_SAMPLE:
        reasons.append(f"sample<{MIN_SAMPLE}")
    if win_rate is None or float(win_rate) < 0.55:
        reasons.append("win_rate<55%")
    if median_return is None or float(median_return) <= 0:
        reasons.append("median_return<=0")
    if avg_return is None or float(avg_return) <= 0:
        reasons.append("avg_return<=0")
    if alpha is None or float(alpha) <= 0:
        reasons.append("alpha_vs_benchmark<=0")
    if worst_mae is not None and float(worst_mae) < -0.15:
        reasons.append("worst_MAE<-15%")

    return len(reasons) == 0, reasons


def build_buy_candidates(report: dict) -> list[dict]:
    candidates = []
    backtests = report.get("backtests", {})
    technicals = report.get("technicals", {})
    rankings = report.get("rule_evidence_ranking", {})

    score_lookup: dict[tuple[str, str], Any] = {}
    for ticker, rows in rankings.items():
        for r in rows or []:
            score_lookup[(ticker, r.get("rule"))] = r.get("evidence_score_0_100")

    for ticker, rules in backtests.items():
        if not isinstance(rules, dict):
            continue
        tech = technicals.get(ticker, {})
        active = rules.get("active_signals_latest_day", {}) or {}
        for rule in BUY_RULES:
            if not as_bool(active.get(rule)):
                continue
            row = (rules.get(rule, {}) or {}).get(PRIMARY_HORIZON, {}) or {}
            bench, alpha = get_alpha(row, ticker, tech)
            passes, fail_reasons = evidence_passes(row, alpha)
            candidates.append(
                {
                    "ticker": ticker,
                    "rule": rule,
                    "status": "ACTIONABLE_CANDIDATE" if passes else "WATCH_ONLY_INSUFFICIENT_EVIDENCE",
                    "latest_date": rules.get("latest_date") or tech.get("latest_date"),
                    "latest_price": tech.get("latest_price") or rules.get("latest_price"),
                    "horizon_days": 20,
                    "samples": row.get("samples", 0),
                    "valid": row.get("valid", False),
                    "win_rate_pct": pct(row.get("win_rate")),
                    "avg_return_pct": pct(row.get("avg_return")),
                    "median_return_pct": pct(row.get("median_return")),
                    "worst_return_pct": pct(row.get("worst_return")),
                    "avg_mae_pct": pct(row.get("avg_mae")),
                    "worst_mae_pct": pct(row.get("worst_mae")),
                    "benchmark": bench,
                    "avg_alpha_vs_benchmark_pct": pct(alpha),
                    "evidence_score_0_100": score_lookup.get((ticker, rule), 0),
                    "fail_reasons": fail_reasons,
                }
            )

    return sorted(
        candidates,
        key=lambda x: (
            x["status"] == "ACTIONABLE_CANDIDATE",
            x.get("evidence_score_0_100") or 0,
            x.get("samples") or 0,
        ),
        reverse=True,
    )


def build_risk_candidates(report: dict) -> list[dict]:
    risks = []
    backtests = report.get("backtests", {})
    technicals = report.get("technicals", {})

    for ticker, tech in technicals.items():
        active = tech.get("active_signals", {}) or {}
        reasons = []
        severity = 0

        if as_bool(active.get("failed_rebound_risk")):
            severity += 35
            reasons.append("failed_rebound_risk active")
        if tech.get("above_ma5") is False and (tech.get("ret_5d") is not None and float(tech.get("ret_5d")) <= -0.04):
            severity += 20
            reasons.append("5d pullback not reclaimed")
        if tech.get("above_ma20") is False:
            severity += 10
            reasons.append("below MA20")
        if tech.get("above_ma50") is False:
            severity += 15
            reasons.append("below MA50")
        if tech.get("above_ma200") is False:
            severity += 20
            reasons.append("below MA200")
        dd = tech.get("drawdown_from_52w_high")
        if dd is not None and float(dd) <= -0.12:
            severity += 15
            reasons.append("drawdown from 52w high <= -12%")

        if severity > 0:
            row = ((backtests.get(ticker, {}) or {}).get("failed_rebound_risk", {}) or {}).get(PRIMARY_HORIZON, {}) or {}
            risks.append(
                {
                    "ticker": ticker,
                    "severity_0_100": min(100, severity),
                    "latest_date": tech.get("latest_date"),
                    "latest_price": tech.get("latest_price"),
                    "reasons": reasons,
                    "failed_rebound_samples_20d": row.get("samples", 0),
                    "failed_rebound_valid_20d": row.get("valid", False),
                    "failed_rebound_avg_return_pct": pct(row.get("avg_return")),
                    "failed_rebound_median_return_pct": pct(row.get("median_return")),
                    "failed_rebound_worst_mae_pct": pct(row.get("worst_mae")),
                    "ret_5d_pct": pct(tech.get("ret_5d")),
                    "ret_20d_pct": pct(tech.get("ret_20d")),
                    "drawdown_from_52w_high_pct": pct(dd),
                }
            )

    return sorted(risks, key=lambda x: x["severity_0_100"], reverse=True)


def top_watchlist(report: dict, limit: int = 25) -> list[dict]:
    rows = []
    for ticker, ranking in (report.get("rule_evidence_ranking", {}) or {}).items():
        for r in ranking or []:
            if r.get("valid"):
                rows.append(
                    {
                        "ticker": ticker,
                        "rule": r.get("rule"),
                        "active_latest_day": r.get("active_latest_day", False),
                        "samples_20d": r.get("samples_20d"),
                        "win_rate_20d_pct": pct(r.get("win_rate_20d")),
                        "median_return_20d_pct": pct(r.get("median_return_20d")),
                        "worst_mae_20d_pct": pct(r.get("worst_mae_20d")),
                        "evidence_score_0_100": r.get("evidence_score_0_100"),
                    }
                )
    return sorted(rows, key=lambda x: (x.get("active_latest_day"), x.get("evidence_score_0_100") or 0), reverse=True)[:limit]


def freshness(report: dict) -> dict:
    dates = []
    for item in (report.get("technicals", {}) or {}).values():
        d = item.get("latest_date")
        if d:
            dates.append(d)
    return {
        "source_report_generated_at_utc": report.get("generated_at_utc"),
        "latest_price_date_max": max(dates) if dates else None,
        "latest_price_date_min": min(dates) if dates else None,
        "loaded_ticker_count": (report.get("universe", {}) or {}).get("loaded_ticker_count"),
        "configured_ticker_count": (report.get("universe", {}) or {}).get("configured_ticker_count"),
        "fresh_request_count": (report.get("universe", {}) or {}).get("fresh_request_count"),
        "errors_count": len(report.get("errors", {}) or {}),
    }


def main() -> None:
    if not REPORT_PATH.exists():
        raise SystemExit("docs/market_report.json not found. Run scripts/build_report.py first.")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    buy_candidates = build_buy_candidates(report)
    risk_candidates = build_risk_candidates(report)
    watch = top_watchlist(report)

    actionable = [x for x in buy_candidates if x["status"] == "ACTIONABLE_CANDIDATE"]
    high_risk = [x for x in risk_candidates if x["severity_0_100"] >= 65]

    if high_risk:
        final_action = "RISK_REVIEW_REQUIRED"
        reason = "At least one ticker has high technical/rule risk. Review before adding risk."
    elif actionable:
        final_action = "BUY_CANDIDATE_REVIEW_REQUIRED"
        reason = "At least one active signal has valid 20d backtest evidence. Still require price/news/portfolio guardrail confirmation."
    else:
        final_action = "NO_TRADE"
        reason = "No active signal passed the minimum sample, win-rate, median-return, alpha, and MAE filters."

    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Derived from docs/market_report.json and docs/backtest_summary.csv produced by scripts/build_report.py",
        "decision_policy": {
            "buy_requires": [
                "active latest-day signal",
                "20d sample count >= 20",
                "win rate >= 55%",
                "average and median forward return > 0",
                "average alpha versus selected benchmark > 0",
                "worst 20d MAE not worse than -15%",
                "separate real-time price/news/portfolio guardrail confirmation before order",
            ],
            "risk_review_requires": [
                "failed rebound risk",
                "pullback not reclaimed",
                "break of key moving averages",
                "large drawdown from 52-week high",
            ],
            "important_limit": "This is an evidence report, not an automatic trading bot. Final orders require live quote and account confirmation.",
        },
        "freshness": freshness(report),
        "final_action": final_action,
        "reason": reason,
        "actionable_buy_candidates": actionable[:10],
        "all_active_buy_candidates": buy_candidates[:50],
        "risk_candidates": risk_candidates[:50],
        "top_valid_watchlist_evidence": watch,
        "errors": report.get("errors", {}),
    }

    with open(OUT / "eason_signal.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False, allow_nan=False)

    latest_summary = {
        "generated_at_utc": decision["generated_at_utc"],
        "final_action": final_action,
        "reason": reason,
        "actionable_buy_count": len(actionable),
        "risk_candidate_count": len(risk_candidates),
        "high_risk_count": len(high_risk),
        "top_actionable_buy": actionable[0] if actionable else None,
        "top_risk": risk_candidates[0] if risk_candidates else None,
        "freshness": decision["freshness"],
    }
    with open(OUT / "latest_summary.json", "w", encoding="utf-8") as f:
        json.dump(latest_summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    pd.DataFrame(buy_candidates).to_csv(OUT / "signal_candidates.csv", index=False)
    pd.DataFrame(risk_candidates).to_csv(OUT / "risk_candidates.csv", index=False)

    print("Saved docs/eason_signal.json, docs/latest_summary.json, docs/signal_candidates.csv, docs/risk_candidates.csv")


if __name__ == "__main__":
    main()
