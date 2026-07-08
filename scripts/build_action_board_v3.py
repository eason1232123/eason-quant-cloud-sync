from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT = Path("docs")

# Important: do NOT read docs/action_board.json here.
# This script writes action_board.json, so reading it as an input creates a recursive
# self-nesting file that grows on every run.
FILES = {
    "eason_signal": OUT / "eason_signal.json",
    "portfolio_backtest": OUT / "portfolio_backtest.json",
    "walk_forward_report": OUT / "walk_forward_report.json",
    "market_regime_report": OUT / "market_regime_report.json",
    "overfitting_check": OUT / "overfitting_check.json",
    "trade_review": OUT / "trade_review.json",
    "actual_vs_backtest": OUT / "actual_vs_backtest.json",
    "vectorbt_validation": OUT / "vectorbt_validation.json",
    "vectorbt_report": OUT / "vectorbt_report.json",
    "market_report": OUT / "market_report.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"available": False, "reason": f"{path} not found or empty"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("available", True)
            return data
        return {"available": False, "reason": f"{path} did not contain a JSON object"}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def first_item(*values: Any) -> Any:
    for value in values:
        if isinstance(value, list) and value:
            return value[0]
        if value not in (None, [], {}, ""):
            return value
    return None


def get_signal_action(signal: dict[str, Any]) -> str:
    # build_decision_report.py writes final_action.
    # legacy build_eason_signal.py wrote today_action.
    return str(signal.get("final_action") or signal.get("today_action") or "UNKNOWN")


def compact_portfolio(portfolio: dict[str, Any]) -> dict[str, Any]:
    if not portfolio.get("available"):
        return portfolio
    return {
        "available": True,
        "version": portfolio.get("version"),
        "latest_regime": portfolio.get("latest_regime"),
        "strategy_metrics": portfolio.get("strategy_metrics", {}),
        "strategy_vs_benchmarks": portfolio.get("strategy_vs_benchmarks", []),
        "exposure_summary": portfolio.get("exposure_summary", {}),
        "trade_metrics": portfolio.get("trade_metrics", {}),
        "important_limits": portfolio.get("important_limits", []),
    }


def compact_walk_forward(walk: dict[str, Any]) -> dict[str, Any]:
    if not walk.get("available"):
        return walk
    periods = []
    for p in walk.get("periods", []):
        periods.append(
            {
                "period": p.get("period", {}).get("name"),
                "purpose": p.get("period", {}).get("purpose"),
                "available": p.get("available"),
                "strategy_metrics": p.get("strategy_metrics", {}),
                "strategy_vs_benchmarks": p.get("strategy_vs_benchmarks", []),
                "reason": p.get("reason"),
            }
        )
    return {
        "available": True,
        "version": walk.get("version"),
        "important_limit": walk.get("important_limit"),
        "periods": periods,
    }


def compact_vectorbt_validation(vbt_validation: dict[str, Any]) -> dict[str, Any]:
    if not vbt_validation.get("available"):
        return vbt_validation
    return {
        "available": True,
        "version": vbt_validation.get("version"),
        "vectorbt_version": vbt_validation.get("vectorbt_version"),
        "purpose": vbt_validation.get("purpose"),
        "data": vbt_validation.get("data", {}),
        "assumptions": vbt_validation.get("assumptions", {}),
        "top_by_sharpe": vbt_validation.get("top_by_sharpe", [])[:15],
        "important_limit": vbt_validation.get("important_limit"),
        "errors": vbt_validation.get("errors", {}),
    }


def compact_vectorbt_report(vbt_report: dict[str, Any]) -> dict[str, Any]:
    if not vbt_report.get("available"):
        return vbt_report
    return {
        "available": True,
        "version": vbt_report.get("version"),
        "engine": vbt_report.get("engine"),
        "execution_assumption": vbt_report.get("execution_assumption"),
        "loaded_ticker_count": vbt_report.get("loaded_ticker_count"),
        "configured_ticker_count": vbt_report.get("configured_ticker_count"),
        "minimum_valid_samples": vbt_report.get("minimum_valid_samples"),
        "horizons_days": vbt_report.get("horizons_days", []),
        "required_evidence_fields": vbt_report.get("required_evidence_fields", {}),
        "top_strategy_results": vbt_report.get("top_strategy_results", [])[:15],
        "top_entry_forward_evidence_20d": vbt_report.get("top_entry_forward_evidence_20d", [])[:15],
        "top_risk_forward_evidence_20d": vbt_report.get("top_risk_forward_evidence_20d", [])[:15],
        "latest_active_signals": vbt_report.get("latest_active_signals", {}),
        "output_files": vbt_report.get("output_files", []),
        "important_limits": vbt_report.get("important_limits", []),
        "errors": vbt_report.get("errors", {}),
    }


def final_gate(signal: dict[str, Any], overfit: dict[str, Any], trade_review: dict[str, Any], vectorbt_validation: dict[str, Any], vectorbt_report: dict[str, Any]) -> dict[str, Any]:
    base = get_signal_action(signal)
    warnings: list[str] = []

    if not signal.get("available"):
        warnings.append("eason_signal.json unavailable; decision layer did not complete")

    overfit_verdict = overfit.get("verdict")
    if overfit_verdict in {"FAIL_OR_OVERFIT_RISK", "MIXED_NEEDS_CAUTION"}:
        warnings.append(f"overfitting_check={overfit_verdict}")

    if vectorbt_validation.get("available") and vectorbt_validation.get("errors"):
        warnings.append("vectorbt validation has rule-level errors; inspect vectorbt_validation.json")

    if not vectorbt_report.get("available"):
        warnings.append("vectorbt evidence layer unavailable; do not treat GitHub signal as fully vectorbt-validated")
    else:
        required = vectorbt_report.get("required_evidence_fields", {})
        if required and not all(required.values()):
            warnings.append("vectorbt required evidence fields incomplete")
        if vectorbt_report.get("errors"):
            warnings.append("vectorbt evidence layer has ticker-level errors; inspect vectorbt_report.json")

    actual = trade_review.get("actual_vs_backtest", {}) if isinstance(trade_review, dict) else {}
    if actual.get("available") and actual.get("actual_20d_win_rate_pct") is not None:
        if float(actual.get("actual_20d_win_rate_pct", 0)) < 45:
            warnings.append("actual 20d trade win rate <45%; reduce confidence until reviewed")

    if base == "NO_TRADE":
        recommended = "NO_TRADE_UNLESS_LIVE_RISK_OVERRIDE"
    elif base in {"RISK_REVIEW_REQUIRED", "WAIT_OR_REDUCE"}:
        recommended = "REVIEW_DEFENSE_FIRST"
    elif base in {"BUY_CANDIDATE_REVIEW_REQUIRED", "BUY_ALLOWED"}:
        recommended = "WAIT_FOR_CHATGPT_LIVE_REVIEW"
    elif base in {"WAIT", "SMALL_BUY_OR_WAIT", "CAUTION"}:
        recommended = "WAIT_OR_SMALL_ONLY_AFTER_CHATGPT_REVIEW"
    else:
        recommended = "WAIT_FOR_CHATGPT_LIVE_REVIEW"

    if warnings and recommended.startswith("WAIT"):
        recommended = "WAIT_OR_REDUCE_SIZE_AFTER_CHATGPT_REVIEW"

    return {
        "quant_signal": base,
        "automatic_order_allowed": False,
        "chatgpt_review_required": True,
        "recommended_default_action": recommended,
        "warnings": warnings,
        "hard_rule": "GitHub evidence can create candidates only. Vectorbt/backtest evidence + ChatGPT live review + IBKR quote + human confirmation are required before any order.",
    }


def signal_summary(signal: dict[str, Any]) -> dict[str, Any]:
    buys = signal.get("actionable_buy_candidates") or signal.get("buy_candidates") or []
    risks = signal.get("risk_candidates") or []
    return {
        "final_action": get_signal_action(signal),
        "reason": signal.get("reason"),
        "market_regime": signal.get("market_regime"),
        "buy_permission": signal.get("buy_permission"),
        "data_status": signal.get("data_status"),
        "top_actionable_buy": first_item(buys),
        "top_risk": first_item(risks),
        "freshness": signal.get("freshness"),
        "universe": signal.get("universe"),
    }


def main() -> None:
    loaded = {name: load_json(path) for name, path in FILES.items()}
    signal = loaded.get("eason_signal", {})
    portfolio = loaded.get("portfolio_backtest", {})
    walk = loaded.get("walk_forward_report", {})
    regime = loaded.get("market_regime_report", {})
    overfit = loaded.get("overfitting_check", {})
    trade = loaded.get("trade_review", {})
    actual = loaded.get("actual_vs_backtest", {})
    vectorbt_validation = loaded.get("vectorbt_validation", {})
    vectorbt_report = loaded.get("vectorbt_report", {})
    market_report = loaded.get("market_report", {})

    master = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "eason-master-action-board-v3.7-no-self-nesting",
        "purpose": "One compact file for ChatGPT to review GitHub quant evidence, vectorbt validation/evidence, portfolio backtest, walk-forward stability, regime behavior, and actual trade review before live-market judgment.",
        "roles": {
            "github": "data, backtest, vectorbt validation/evidence, stability, risk, and trade-review evidence layer",
            "chatgpt": "live quote/news/macro/valuation/account-risk reviewer and execution planner",
            "human": "final broker confirmation and order execution",
        },
        "final_gate": final_gate(signal, overfit, trade, vectorbt_validation, vectorbt_report),
        "signal_summary": signal_summary(signal),
        "market_report_status": {
            "available": market_report.get("available", False),
            "generated_at_utc": market_report.get("generated_at_utc"),
            "strategy_version": market_report.get("strategy_version"),
            "universe": market_report.get("universe", {}),
            "errors_count": len(market_report.get("errors", {}) or {}) if isinstance(market_report.get("errors", {}), dict) else None,
        },
        "vectorbt_validation": compact_vectorbt_validation(vectorbt_validation),
        "vectorbt_evidence": compact_vectorbt_report(vectorbt_report),
        "portfolio_backtest": compact_portfolio(portfolio),
        "walk_forward_report": compact_walk_forward(walk),
        "market_regime_report": regime,
        "overfitting_check": overfit,
        "trade_review": trade,
        "actual_vs_backtest": actual,
        "required_live_checks_before_order": [
            "IBKR bid/ask/last or two public quote sources",
            "current price vs tested signal area",
            "same-day SPY/QQQ/VIX/10Y/breadth/regime",
            "fresh company/sector/macro news",
            "earnings/guidance/valuation check",
            "real account cash, position size, QQQ+SMH+MSFT concentration, semiconductor exposure, MSFT exposure",
            "exact limit price, invalidation level, and cancel condition",
        ],
    }

    OUT.mkdir(exist_ok=True)
    with open(OUT / "eason_master_status.json", "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, allow_nan=False)

    with open(OUT / "action_board.json", "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, allow_nan=False)

    print("Saved docs/eason_master_status.json and docs/action_board.json without recursive self-nesting")


if __name__ == "__main__":
    main()
