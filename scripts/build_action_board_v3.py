from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_decision_packet import validate_invariants, validate_schema

OUT = Path("docs")
DECISION_PACKET_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "decision_packet.schema.json"

# Important: do NOT read docs/action_board.json here.
# This script writes action_board.json, so reading it as an input creates a recursive
# self-nesting file that grows on every run.
FILES = {
    "decision_packet": OUT / "decision_packet.json",
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


CORE_TICKERS = ["SPY", "QQQ", "SMH", "SOXX", "MSFT", "SGOV", "NVDA", "AVGO", "ASML", "TSM", "AMD", "MU"]


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


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return default
        parsed = int(value)
        if float(value) != parsed or parsed < 0:
            return default
        return parsed
    except (TypeError, ValueError, OverflowError):
        return default


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
    validation_data = vectorbt_validation.get("data", {})
    if not isinstance(validation_data, dict):
        validation_data = {}
    validation_rows = as_int(validation_data.get("rows"))
    validation_tickers = validation_data.get("loaded_tickers")
    vectorbt_validation_usable = bool(
        vectorbt_validation.get("available")
        and not vectorbt_validation.get("errors")
        and validation_rows is not None
        and validation_rows > 0
        and isinstance(validation_tickers, list)
        and validation_tickers
        and vectorbt_validation.get("top_by_sharpe")
    )
    vectorbt_loaded = as_int(vectorbt_report.get("loaded_ticker_count"))
    required = vectorbt_report.get("required_evidence_fields", {})
    required_complete = bool(required) and all(value is True for value in required.values())
    vectorbt_evidence_usable = bool(
        vectorbt_report.get("available")
        and vectorbt_loaded is not None
        and vectorbt_loaded > 0
        and required_complete
        and vectorbt_report.get("top_strategy_results")
        and vectorbt_report.get("top_entry_forward_evidence_20d")
    )

    if not signal.get("available"):
        warnings.append("eason_signal.json unavailable; decision layer did not complete")
    if signal.get("data_status") == "STALE_MODEL_PORTFOLIO_DATA":
        warnings.append("model-portfolio data is stale or missing; refresh before any decision")

    overfit_verdict = overfit.get("verdict")
    if overfit_verdict in {"FAIL_OR_OVERFIT_RISK", "MIXED_NEEDS_CAUTION"}:
        warnings.append(f"overfitting_check={overfit_verdict}")

    if not vectorbt_validation_usable:
        warnings.append("independent vectorbt validation unavailable or incomplete; refresh before using the decision")

    if not vectorbt_evidence_usable:
        warnings.append("vectorbt evidence layer unavailable or incomplete; refresh before treating a buy candidate as validated")
    else:
        if vectorbt_report.get("errors"):
            warnings.append("vectorbt evidence layer has ticker-level errors; inspect vectorbt_report.json")

    actual = trade_review.get("actual_vs_backtest", {}) if isinstance(trade_review, dict) else {}
    if actual.get("available") and actual.get("actual_20d_win_rate_pct") is not None:
        if float(actual["actual_20d_win_rate_pct"]) < 45:
            warnings.append("actual 20d trade win rate <45%; reduce confidence until reviewed")

    critical_evidence_usable = vectorbt_validation_usable and vectorbt_evidence_usable
    if base == "DATA_REVIEW_REQUIRED" or not critical_evidence_usable:
        recommended = "REFRESH_DATA_BEFORE_DECISION"
    elif base == "NO_TRADE":
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
        "portfolio_scope": signal.get("portfolio_scope"),
        "top_actionable_buy": first_item(buys),
        "top_risk": first_item(risks),
        "top_model_portfolio_risk": first_item(signal.get("model_portfolio_high_risks")),
        "top_watchlist_risk_advisory": first_item(signal.get("watchlist_high_risks_advisory")),
        "freshness": signal.get("freshness"),
        "universe": signal.get("universe"),
    }


def market_report_status(market_report: dict[str, Any]) -> dict[str, Any]:
    universe = market_report.get("universe", {}) if isinstance(market_report.get("universe"), dict) else {}
    errors = market_report.get("errors")
    return {
        "available": market_report.get("available", False),
        "generated_at_utc": market_report.get("generated_at_utc"),
        "data_source": market_report.get("data_source"),
        "market_timezone": market_report.get("market_timezone"),
        "data_timestamp": market_report.get("data_timestamp"),
        "data_timestamp_status": market_report.get("data_timestamp_status"),
        "price_frequency": market_report.get("price_frequency"),
        "price_adjustment_policy": market_report.get("price_adjustment_policy"),
        "strategy_version": market_report.get("strategy_version"),
        "universe": universe,
        "errors_count": len(errors) if isinstance(errors, dict) else None,
    }


def score_github_data_quality(
    market_report: dict[str, Any],
    vectorbt_report: dict[str, Any],
    decision_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    universe = market_report.get("universe", {}) if isinstance(market_report.get("universe"), dict) else {}
    configured = as_int(universe.get("configured_ticker_count"))
    if configured is None:
        configured = as_int(vectorbt_report.get("configured_ticker_count"))
    loaded = as_int(universe.get("loaded_ticker_count"))
    if loaded is None:
        loaded = as_int(vectorbt_report.get("loaded_ticker_count"))
    fresh = as_int(universe.get("fresh_request_count"))
    errors = as_int(universe.get("errors_count"))
    coverage_gaps = as_int(universe.get("coverage_gaps_count"))
    tiingo_circuit_open = bool(universe.get("tiingo_circuit_open"))
    vbt_available = bool(vectorbt_report.get("available"))
    coverage_ratio = round(loaded / configured, 4) if loaded is not None and configured else None
    packet_quality = (decision_packet or {}).get("data_quality", {}) if isinstance(decision_packet, dict) else {}
    stale_ticker_count = as_int(packet_quality.get("stale_ticker_count"))
    packet_data_status = packet_quality.get("data_status")
    latest_price_date_max = packet_quality.get("latest_price_date_max")
    if latest_price_date_max is None:
        latest_price_date_max = market_report.get("data_timestamp")
    reasons: list[str] = []
    count_metadata_missing = any(
        value is None for value in (configured, loaded, fresh, errors, coverage_gaps, stale_ticker_count)
    )

    if not market_report.get("available"):
        reasons.append("market_report unavailable")
    if not vbt_available:
        reasons.append("vectorbt_report unavailable")
    if tiingo_circuit_open:
        reasons.append("Tiingo circuit open")
    if errors:
        reasons.append(f"true_errors={errors}")
    if coverage_gaps:
        reasons.append(f"coverage_gaps={coverage_gaps}")
    if coverage_ratio is not None and coverage_ratio < 0.70:
        reasons.append(f"coverage_ratio={coverage_ratio:.0%} below 70%")
    if stale_ticker_count:
        reasons.append(f"stale_tickers={stale_ticker_count}")
    if packet_data_status == "STALE_MODEL_PORTFOLIO_DATA":
        reasons.append("model-portfolio data stale")
    if count_metadata_missing:
        reasons.append("required data-quality count metadata missing or invalid")

    if not market_report.get("available") or not vbt_available or count_metadata_missing:
        grade = "Unusable"
    elif tiingo_circuit_open or (errors is not None and errors >= 10) or packet_data_status == "STALE_MODEL_PORTFOLIO_DATA":
        grade = "Low"
    elif coverage_ratio is not None and coverage_ratio < 0.70:
        grade = "Medium-Low"
    elif coverage_gaps or fresh == 0 or stale_ticker_count:
        grade = "Medium"
    else:
        grade = "High"

    return {
        "grade": grade,
        "configured_ticker_count": configured,
        "loaded_ticker_count": loaded,
        "coverage_ratio": coverage_ratio,
        "fresh_request_count": fresh,
        "coverage_gaps_count": coverage_gaps,
        "errors_count": errors,
        "tiingo_circuit_open": tiingo_circuit_open,
        "latest_price_date_max": latest_price_date_max,
        "expected_latest_market_date": universe.get("expected_latest_market_date"),
        "stale_ticker_count": stale_ticker_count,
        "decision_data_status": packet_data_status,
        "reasons": reasons,
        "usage_rule": "Use High/Medium as evidence; Medium-Low only as auxiliary evidence; Low/Unusable cannot justify trades by itself.",
    }


def compact_active_signals(vbt_report: dict[str, Any], limit: int = 30) -> dict[str, Any]:
    signals = vbt_report.get("latest_active_signals", {}) if isinstance(vbt_report.get("latest_active_signals"), dict) else {}
    compact: dict[str, Any] = {}

    for ticker in CORE_TICKERS:
        if ticker in signals:
            compact[ticker] = signals[ticker]

    for ticker, data in signals.items():
        if ticker in compact or not isinstance(data, dict):
            continue
        entries = data.get("entries", {}) if isinstance(data.get("entries"), dict) else {}
        risks = data.get("risks", {}) if isinstance(data.get("risks"), dict) else {}
        if any(bool(v) for v in entries.values()) or any(bool(v) for v in risks.values()):
            compact[ticker] = data
        if len(compact) >= limit:
            break

    return compact


def finalize_decision_packet(packet: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(packet, dict) or not packet.get("schema_version"):
        return {
            "schema_version": "decision-packet-v5.0",
            "available": False,
            "reason": "decision packet was not produced by build_decision_report.py",
        }

    finalized = dict(packet)
    finalized.pop("available", None)
    finalized["evidence_gate"] = {
        "quant_signal": gate.get("quant_signal"),
        "recommended_default_action": gate.get("recommended_default_action"),
        "warnings": gate.get("warnings", []),
        "automatic_order_allowed": False,
    }
    decision = dict(finalized.get("decision", {}))
    decision["chatgpt_review_required"] = bool(gate.get("chatgpt_review_required", True))
    decision["default_human_action"] = str(gate.get("recommended_default_action") or "WAIT_FOR_CHATGPT_LIVE_REVIEW")
    decision["automatic_order_allowed"] = False
    finalized["decision"] = decision
    return finalized


def build_chatgpt_snapshot(
    loaded: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    sig_summary: dict[str, Any],
    decision_packet: dict[str, Any],
) -> dict[str, Any]:
    market_report = loaded.get("market_report", {})
    vectorbt_report = loaded.get("vectorbt_report", {})
    signal = loaded.get("eason_signal", {})
    portfolio = loaded.get("portfolio_backtest", {})
    regime = loaded.get("market_regime_report", {})
    overfit = loaded.get("overfitting_check", {})
    trade = loaded.get("trade_review", {})

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "chatgpt-compact-snapshot-v2.0-decision-contract",
        "purpose": "Small connector-friendly evidence file for ChatGPT. Read this before large market_report.json/action_board.json to avoid large JSON connector truncation/empty-content issues.",
        "read_priority_for_chatgpt": [
            "docs/decision_packet.json",
            "docs/chatgpt_snapshot.json",
            "docs/eason_master_status.json",
            "docs/action_board.json",
            "docs/market_report.json",
        ],
        "decision_contract": decision_packet,
        "market_data": decision_packet.get("market_data"),
        "github_data_quality": score_github_data_quality(market_report, vectorbt_report, decision_packet),
        "final_gate": gate,
        "signal_summary": sig_summary,
        "market_report_status": market_report_status(market_report),
        "vectorbt_evidence_summary": {
            "available": vectorbt_report.get("available", False),
            "engine": vectorbt_report.get("engine"),
            "execution_assumption": vectorbt_report.get("execution_assumption"),
            "loaded_ticker_count": vectorbt_report.get("loaded_ticker_count"),
            "configured_ticker_count": vectorbt_report.get("configured_ticker_count"),
            "minimum_valid_samples": vectorbt_report.get("minimum_valid_samples"),
            "horizons_days": vectorbt_report.get("horizons_days", []),
            "top_entry_forward_evidence_20d": vectorbt_report.get("top_entry_forward_evidence_20d", [])[:8],
            "top_risk_forward_evidence_20d": vectorbt_report.get("top_risk_forward_evidence_20d", [])[:8],
            "latest_active_signals_compact": compact_active_signals(vectorbt_report),
        },
        "portfolio_backtest_summary": compact_portfolio(portfolio),
        "market_regime_summary": {
            "available": regime.get("available", False),
            "generated_at_utc": regime.get("generated_at_utc"),
            "version": regime.get("version"),
            "latest_regime": first_item(regime.get("latest_regime"), regime.get("current_regime")),
        },
        "stability_summary": {
            "overfitting_verdict": overfit.get("verdict"),
            "overfitting_available": overfit.get("available", False),
            "trade_review_available": trade.get("available", False),
        },
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


def write_snapshot_txt(snapshot: dict[str, Any]) -> str:
    quality = snapshot.get("github_data_quality", {})
    signal = snapshot.get("signal_summary", {})
    gate = snapshot.get("final_gate", {})
    market = snapshot.get("market_report_status", {})
    vbt = snapshot.get("vectorbt_evidence_summary", {})
    lines = [
        "Eason ChatGPT Compact Snapshot",
        f"generated_at_utc: {snapshot.get('generated_at_utc')}",
        f"data_quality: {quality.get('grade')}",
        f"coverage: {quality.get('loaded_ticker_count')}/{quality.get('configured_ticker_count')} ratio={quality.get('coverage_ratio')}",
        f"latest_price_date: {quality.get('latest_price_date_max')}",
        f"tiingo_circuit_open: {quality.get('tiingo_circuit_open')}",
        f"coverage_gaps_count: {quality.get('coverage_gaps_count')}",
        f"errors_count: {quality.get('errors_count')}",
        f"market_report_generated_at_utc: {market.get('generated_at_utc')}",
        f"data_source: {market.get('data_source')}",
        f"market_timezone: {market.get('market_timezone')}",
        f"data_timestamp: {market.get('data_timestamp')}",
        f"price_frequency: {market.get('price_frequency')}",
        f"price_adjustment_policy: {market.get('price_adjustment_policy')}",
        f"quant_signal: {gate.get('quant_signal')}",
        f"recommended_default_action: {gate.get('recommended_default_action')}",
        f"signal_final_action: {signal.get('final_action')}",
        f"signal_market_regime: {signal.get('market_regime')}",
        f"vectorbt_available: {vbt.get('available')}",
        "Use JSON for details. This TXT is only a quick health check.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    loaded = {name: load_json(path) for name, path in FILES.items()}
    decision_packet_input = loaded.get("decision_packet", {})
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

    gate = final_gate(signal, overfit, trade, vectorbt_validation, vectorbt_report)
    sig_summary = signal_summary(signal)
    decision_packet = finalize_decision_packet(decision_packet_input, gate)

    master = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "eason-master-action-board-v5.0-decision-contract",
        "purpose": "One compact file for ChatGPT to review GitHub quant evidence, vectorbt validation/evidence, portfolio backtest, walk-forward stability, regime behavior, and actual trade review before live-market judgment.",
        "roles": {
            "github": "data, backtest, vectorbt validation/evidence, stability, risk, and trade-review evidence layer",
            "chatgpt": "live quote/news/macro/valuation/account-risk reviewer and execution planner",
            "human": "final broker confirmation and order execution",
        },
        "final_gate": gate,
        "decision_contract": decision_packet,
        "market_data": decision_packet.get("market_data"),
        "signal_summary": sig_summary,
        "market_report_status": market_report_status(market_report),
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

    snapshot = build_chatgpt_snapshot(loaded, gate, sig_summary, decision_packet)

    schema = json.loads(DECISION_PACKET_SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_schema(decision_packet, schema)
    validate_invariants(decision_packet)
    json.dumps(master, allow_nan=False)
    json.dumps(snapshot, allow_nan=False)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "eason_master_status.json", "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, allow_nan=False)

    with open(OUT / "action_board.json", "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, allow_nan=False)

    with open(OUT / "chatgpt_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, allow_nan=False)

    with open(OUT / "chatgpt_snapshot.txt", "w", encoding="utf-8") as f:
        f.write(write_snapshot_txt(snapshot))

    with open(OUT / "decision_packet.json", "w", encoding="utf-8") as f:
        json.dump(decision_packet, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    print("Saved docs/decision_packet.json, docs/eason_master_status.json, docs/action_board.json, docs/chatgpt_snapshot.json, and docs/chatgpt_snapshot.txt without recursive self-nesting")


if __name__ == "__main__":
    main()
