from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_clock import reference_market_context, weekday_lag
from scripts.market_data_contract import extract_market_data_metadata
from scripts.shadow_evidence_policy import (
    load_shadow_evidence_policy,
    prospective_start_market_date,
)
from scripts.validate_decision_packet import validate_invariants, validate_schema

OUT = Path("docs")
REPORT_PATH = OUT / "market_report.json"
PORTFOLIO_PATH = OUT / "portfolio_backtest.json"
DECISION_SUMMARY_PATH = OUT / "latest_decision_summary.json"
DECISION_PACKET_PATH = OUT / "decision_packet.json"
DECISION_PACKET_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "decision_packet.schema.json"
MIN_SAMPLE = 20
PRIMARY_HORIZON = "20d"
MAX_STALENESS_BUSINESS_DAYS = 0
MAX_EXPECTED_MARKET_DATE_AGE_DAYS = 0
HIGH_RISK_THRESHOLD = 65
DEFAULT_MODEL_RISK_SCOPE = ("QQQ", "SMH", "MSFT", "SPY")

BUY_RULES = {
    "pullback_reclaim_5dma",
    "rsi_oversold_reclaim_40",
    "ma20_reclaim_bullish",
    "ma50_reclaim_bullish",
    "relative_strength_rebound",
    "momentum_leader",
}

SHADOW_POLICY = load_shadow_evidence_policy()

LIVE_REVIEW_CHECKLIST = [
    "Read action_board.json, eason_signal.json, portfolio_backtest.json, and market_report.json freshness first.",
    "Check current tradable bid/ask/last from IBKR if available; otherwise cross-check at least two public quote sources.",
    "Check whether current price is still close to the tested signal entry area; do not chase if price already ran far away.",
    "Check same-day regime: SPY/QQQ trend, VIX, 10Y yield, breadth, and semiconductor leadership.",
    "Check fresh news, earnings, guidance, analyst changes, regulation, and macro events for the candidate and key leaders.",
    "Check portfolio constraints: cash floor, offense cash, QQQ/SMH/MSFT concentration, semis max, MSFT max, and same-day repeat-buy rule.",
    "Decide final action: buy/add, wait, pause, trim, sell, or no trade. GitHub evidence alone is never an automatic order.",
]


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


def pct(value: Any) -> Any:
    v = clean_float(value, 4)
    if v is None:
        return None
    return clean_float(v * 100, 2)


def load_optional_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"load_error": str(exc)}


def business_day_age(latest: Any, reference: Any) -> int | None:
    return weekday_lag(latest, reference)


def optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        if float(value) != parsed or parsed < 0:
            return None
        return parsed
    except (TypeError, ValueError, OverflowError):
        return None


def market_data_metadata(report: dict) -> dict[str, Any]:
    return extract_market_data_metadata(report)


def market_date_context(report: dict, as_of: Any = None) -> dict[str, str | None]:
    universe = report.get("universe", {}) if isinstance(report.get("universe"), dict) else {}
    return reference_market_context(universe.get("expected_latest_market_date"), as_of_utc=as_of)


def expected_market_date(report: dict, as_of: Any = None) -> str | None:
    context = market_date_context(report, as_of)
    if context["reference_market_date_status"] != "ALIGNED":
        return None
    return context["reference_market_date"]


def technical_analysis_available(technical: Any) -> bool:
    if not isinstance(technical, dict):
        return False
    if not technical.get("latest_date"):
        return False
    latest_price = clean_float(technical.get("latest_price"))
    trading_days = technical.get("trading_days")
    if latest_price is None or latest_price <= 0:
        return False
    if not isinstance(trading_days, int) or isinstance(trading_days, bool) or trading_days <= 0:
        return False
    if any(
        not isinstance(technical.get(key), bool)
        for key in ("above_ma5", "above_ma20", "above_ma50", "above_ma200")
    ):
        return False
    if any(
        clean_float(technical.get(key)) is None
        for key in ("drawdown_from_52w_high", "ret_5d", "ret_20d")
    ):
        return False
    active = technical.get("active_signals")
    required_signals = BUY_RULES | {"failed_rebound_risk"}
    return bool(
        isinstance(active, dict)
        and all(isinstance(active.get(rule), bool) for rule in required_signals)
    )


def ticker_freshness(report: dict, as_of: Any = None) -> dict[str, dict[str, Any]]:
    reference_date = market_date_context(report, as_of)["reference_market_date"]
    statuses: dict[str, dict[str, Any]] = {}
    technicals = report.get("technicals", {}) or {}
    if not isinstance(technicals, dict):
        technicals = {}
    update_log = report.get("update_log", {}) or {}
    if not isinstance(update_log, dict):
        update_log = {}

    loaded_tickers = set(technicals)
    loaded_tickers.update(
        ticker
        for ticker, item in update_log.items()
        if isinstance(item, dict) and item.get("latest_date") and not item.get("quarantined", False)
    )

    for ticker in sorted(loaded_tickers):
        technical = technicals.get(ticker, {})
        update = update_log.get(ticker, {})
        latest_date = technical.get("latest_date") if isinstance(technical, dict) else None
        technical_available = technical_analysis_available(technical)
        if latest_date is None and isinstance(update, dict):
            latest_date = update.get("latest_date")
        age = business_day_age(latest_date, reference_date)
        market_date_fresh = age is not None and age <= MAX_STALENESS_BUSINESS_DAYS
        decision_eligible = market_date_fresh and technical_available
        statuses[ticker] = {
            "latest_date": latest_date,
            "reference_date": reference_date,
            "staleness_business_days": age,
            "market_date_fresh": market_date_fresh,
            "data_fresh": decision_eligible,
            "technical_analysis_available": technical_available,
            "decision_eligible": decision_eligible,
            "exclusion_reason": (
                None
                if decision_eligible
                else "technical_analysis_missing_or_failed"
                if market_date_fresh
                else "stale_or_missing_market_data"
            ),
        }
    return statuses


def model_risk_scope(portfolio: dict) -> dict[str, Any]:
    portfolio_is_object = isinstance(portfolio, dict)
    portfolio_declared_available = portfolio.get("available") is True if portfolio_is_object else False
    portfolio_has_load_error = bool(portfolio.get("load_error")) if portfolio_is_object else True
    raw_assumptions = portfolio.get("assumptions", {}) if portfolio_is_object else {}
    assumptions = raw_assumptions if isinstance(raw_assumptions, dict) else {}
    tickers_set: set[str] = set()
    valid_weight_maps: set[str] = set()
    for key in ("base_weights", "defensive_weights", "severe_defensive_weights"):
        weights = assumptions.get(key)
        if not isinstance(weights, dict) or not weights:
            continue
        parsed_weights: list[float] = []
        map_tickers: set[str] = set()
        map_valid = True
        for ticker, weight in weights.items():
            normalized = str(ticker).strip().upper()
            clean_weight = clean_float(weight)
            if not normalized or clean_weight is None or clean_weight < 0:
                map_valid = False
                break
            parsed_weights.append(clean_weight)
            if normalized and normalized != "CASH" and clean_weight is not None and clean_weight > 0:
                map_tickers.add(normalized)
        if map_valid and sum(parsed_weights) > 0:
            valid_weight_maps.add(key)
            tickers_set.update(map_tickers)
    tickers = sorted(tickers_set)
    source = "portfolio_backtest.assumptions.*_weights"
    if not tickers:
        tickers = list(DEFAULT_MODEL_RISK_SCOPE)
        source = "decision_contract.default_model_scope"

    latest_regime_available = bool(
        portfolio_is_object
        and portfolio.get("latest_regime") in {"base", "defensive", "severe_defensive"}
    )
    context_available = bool(
        portfolio_is_object
        and portfolio
        and portfolio_declared_available
        and not portfolio_has_load_error
        and isinstance(raw_assumptions, dict)
        and len(valid_weight_maps) == 3
        and tickers_set
        and latest_regime_available
    )
    if not portfolio_is_object or not portfolio:
        context_reason = "portfolio_backtest_missing"
    elif portfolio_has_load_error:
        context_reason = "portfolio_backtest_load_error"
    elif not portfolio_declared_available:
        context_reason = "portfolio_backtest_unavailable"
    elif not isinstance(raw_assumptions, dict) or len(valid_weight_maps) != 3 or not tickers_set:
        context_reason = "portfolio_weights_missing_or_invalid"
    elif not latest_regime_available:
        context_reason = "portfolio_regime_missing_or_invalid"
    else:
        context_reason = None
    return {
        "scope_type": "public_model_portfolio_symbols_only",
        "source": source,
        "tickers": tickers,
        "portfolio_context_available": context_available,
        "portfolio_context_reason": context_reason,
        "contains_private_shares_or_cash": False,
        "actual_holdings_check_required_by_chatgpt": True,
    }


def get_alpha(row: dict, technical: dict) -> tuple[str | None, Any]:
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
    samples = optional_nonnegative_int(row.get("samples"))
    valid = row.get("valid") if isinstance(row.get("valid"), bool) else None
    win_rate = clean_float(row.get("win_rate"))
    median_return = clean_float(row.get("median_return"))
    avg_return = clean_float(row.get("avg_return"))
    worst_mae = clean_float(row.get("worst_mae"))

    if samples is None:
        reasons.append("samples_missing_or_invalid")
    elif samples < MIN_SAMPLE:
        reasons.append(f"sample<{MIN_SAMPLE}")
    if valid is not True:
        reasons.append("valid_backtest_flag_missing_or_false")
    if win_rate is None or win_rate < 0.55:
        reasons.append("win_rate<55%")
    if median_return is None or median_return <= 0:
        reasons.append("median_return<=0")
    if avg_return is None or avg_return <= 0:
        reasons.append("avg_return<=0")
    if alpha is None or alpha <= 0:
        reasons.append("alpha_vs_benchmark<=0")
    if worst_mae is None:
        reasons.append("worst_MAE_missing_or_invalid")
    elif worst_mae < -0.15:
        reasons.append("worst_MAE<-15%")

    return len(reasons) == 0, reasons


def build_buy_candidates(report: dict, freshness_by_ticker: dict[str, dict[str, Any]] | None = None) -> list[dict]:
    candidates = []
    backtests = report.get("backtests", {})
    technicals = report.get("technicals", {})
    rankings = report.get("rule_evidence_ranking", {})
    freshness_by_ticker = freshness_by_ticker or ticker_freshness(report)

    score_lookup: dict[tuple[str, str], Any] = {}
    if isinstance(rankings, dict):
        for ticker, rows in rankings.items():
            for r in rows or []:
                if isinstance(r, dict):
                    score_lookup[(ticker, r.get("rule"))] = r.get("evidence_score_0_100")

    for ticker, rules in backtests.items():
        if not isinstance(rules, dict):
            continue
        tech = technicals.get(ticker, {}) if isinstance(technicals, dict) else {}
        active = rules.get("active_signals_latest_day", {}) or {}
        for rule in BUY_RULES:
            if active.get(rule) is not True:
                continue
            row = (rules.get(rule, {}) or {}).get(PRIMARY_HORIZON, {}) or {}
            bench, alpha = get_alpha(row, tech)
            passes, fail_reasons = evidence_passes(row, alpha)
            evidence_score = clean_float(score_lookup.get((ticker, rule)), 2)
            if evidence_score is None:
                passes = False
                fail_reasons = [*fail_reasons, "evidence_score_missing_or_invalid"]
            technical_status = freshness_by_ticker.get(
                ticker,
                {
                    "latest_date": tech.get("latest_date"),
                    "reference_date": None,
                    "staleness_business_days": None,
                    "data_fresh": False,
                    "decision_eligible": False,
                    "exclusion_reason": "stale_or_missing_market_data",
                },
            )
            signal_latest_date = rules.get("latest_date")
            reference_date = technical_status.get("reference_date")
            signal_age = business_day_age(signal_latest_date, reference_date)
            signal_fresh = signal_age == 0
            technical_fresh = bool(technical_status.get("decision_eligible", False))
            decision_eligible = technical_fresh and signal_fresh
            ages = [
                age
                for age in (technical_status.get("staleness_business_days"), signal_age)
                if age is not None
            ]
            combined_age = max(ages) if ages else None
            if not decision_eligible:
                fail_reasons = [*fail_reasons, "stale_or_missing_market_data"]
                status = "STALE_DATA_EXCLUDED"
            else:
                status = "QUANT_PASS_NEEDS_CHATGPT_REVIEW" if passes else "WATCH_ONLY_INSUFFICIENT_QUANT_EVIDENCE"
            candidates.append(
                {
                    "ticker": ticker,
                    "rule": rule,
                    "status": status,
                    "latest_date": signal_latest_date,
                    "signal_latest_date": signal_latest_date,
                    "technical_latest_date": tech.get("latest_date"),
                    "latest_price": tech.get("latest_price") or rules.get("latest_price"),
                    "data_fresh": decision_eligible,
                    "technical_data_fresh": technical_fresh,
                    "signal_data_fresh": signal_fresh,
                    "staleness_business_days": combined_age,
                    "technical_staleness_business_days": technical_status.get("staleness_business_days"),
                    "signal_staleness_business_days": signal_age,
                    "decision_eligible": decision_eligible,
                    "horizon_days": 20,
                    "samples": optional_nonnegative_int(row.get("samples")),
                    "valid": row.get("valid") if isinstance(row.get("valid"), bool) else None,
                    "win_rate_pct": pct(row.get("win_rate")),
                    "avg_return_pct": pct(row.get("avg_return")),
                    "median_return_pct": pct(row.get("median_return")),
                    "worst_return_pct": pct(row.get("worst_return")),
                    "avg_mae_pct": pct(row.get("avg_mae")),
                    "worst_mae_pct": pct(row.get("worst_mae")),
                    "benchmark": bench,
                    "avg_alpha_vs_benchmark_pct": pct(alpha),
                    "evidence_score_0_100": evidence_score,
                    "fail_reasons": fail_reasons,
                    "final_order_permission": "NOT_ALLOWED_UNTIL_CHATGPT_LIVE_REVIEW",
                }
            )

    return sorted(
        candidates,
        key=lambda x: (
            0 if x["status"] == "QUANT_PASS_NEEDS_CHATGPT_REVIEW" else 1,
            0 if x.get("data_fresh", False) else 1,
            -(x.get("evidence_score_0_100") or 0),
            -(x.get("samples") or 0),
            x["ticker"],
            x["rule"],
        ),
    )


def select_shadow_candidates(
    buy_candidates: list[dict[str, Any]],
    *,
    reference_market_date: str,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the deterministic public research cohort without changing execution eligibility."""
    frozen_policy = policy or SHADOW_POLICY
    maximum = frozen_policy["selection"]["maximum_candidates_per_market_date"]
    start = prospective_start_market_date(frozen_policy)
    observation = date.fromisoformat(reference_market_date)
    rule_valid = [
        row
        for row in buy_candidates
        if row.get("status")
        in {
            "QUANT_PASS_NEEDS_CHATGPT_REVIEW",
            "WATCH_ONLY_INSUFFICIENT_QUANT_EVIDENCE",
        }
        and row.get("rule") in BUY_RULES
        and row.get("data_fresh") is True
        and row.get("decision_eligible") is True
        and row.get("latest_date") == reference_market_date
        and isinstance(row.get("latest_price"), (int, float))
        and not isinstance(row.get("latest_price"), bool)
        and float(row["latest_price"]) > 0
        and isinstance(row.get("benchmark"), str)
        and bool(row["benchmark"])
        and isinstance(row.get("evidence_score_0_100"), (int, float))
        and not isinstance(row.get("evidence_score_0_100"), bool)
    ]
    selected = rule_valid[:maximum]
    evidence_eligible = observation >= start
    if not evidence_eligible:
        status = "BEFORE_PROSPECTIVE_START"
    elif selected:
        status = "READY_FOR_SAME_DAY_SHADOW_REVIEW"
    else:
        status = "NO_FRESH_RULE_VALID_ACTIVE_SIGNAL"
    fields = (
        "ticker",
        "rule",
        "status",
        "latest_date",
        "latest_price",
        "data_fresh",
        "staleness_business_days",
        "decision_eligible",
        "horizon_days",
        "samples",
        "valid",
        "win_rate_pct",
        "avg_return_pct",
        "median_return_pct",
        "worst_return_pct",
        "avg_mae_pct",
        "worst_mae_pct",
        "benchmark",
        "avg_alpha_vs_benchmark_pct",
        "evidence_score_0_100",
        "fail_reasons",
    )
    compact = []
    for rank, row in enumerate(selected, start=1):
        compact.append(
            {
                "candidate_type": "SHADOW_CANDIDATE",
                "selection_rank": rank,
                **{key: row.get(key) for key in fields},
                "prospective_evidence_eligible": evidence_eligible,
                "counterfactual_only": True,
                "execution_eligible": False,
                "automatic_order_allowed": False,
            }
        )
    return {
        "candidate_type": "SHADOW_CANDIDATE",
        "collection_status": status,
        "prospective_start_market_date": start.isoformat(),
        "allow_historical_backfill": False,
        "selection_policy": frozen_policy["selection"]["ranking_policy"],
        "maximum_candidates_per_market_date": maximum,
        "candidate_count": len(compact),
        "top": compact,
        "counterfactual_only": True,
        "execution_eligible": False,
        "automatic_order_allowed": False,
    }


def build_risk_candidates(report: dict, freshness_by_ticker: dict[str, dict[str, Any]] | None = None) -> list[dict]:
    risks = []
    backtests = report.get("backtests", {})
    technicals = report.get("technicals", {})
    freshness_by_ticker = freshness_by_ticker or ticker_freshness(report)
    if not isinstance(technicals, dict):
        return risks

    for ticker, tech in technicals.items():
        active = tech.get("active_signals", {}) or {}
        reasons = []
        severity = 0

        if active.get("failed_rebound_risk") is True:
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
            data_status = freshness_by_ticker.get(
                ticker,
                {
                    "staleness_business_days": None,
                    "data_fresh": False,
                    "decision_eligible": False,
                    "exclusion_reason": "stale_or_missing_market_data",
                },
            )
            risks.append(
                {
                    "ticker": ticker,
                    "severity_0_100": min(100, severity),
                    "latest_date": tech.get("latest_date"),
                    "latest_price": tech.get("latest_price"),
                    "data_fresh": data_status.get("data_fresh", False),
                    "staleness_business_days": data_status.get("staleness_business_days"),
                    "decision_eligible": data_status.get("decision_eligible", False),
                    "data_exclusion_reason": data_status.get("exclusion_reason"),
                    "reasons": reasons,
                    "failed_rebound_samples_20d": optional_nonnegative_int(row.get("samples")),
                    "failed_rebound_valid_20d": row.get("valid") if isinstance(row.get("valid"), bool) else None,
                    "failed_rebound_avg_return_pct": pct(row.get("avg_return")),
                    "failed_rebound_median_return_pct": pct(row.get("median_return")),
                    "failed_rebound_worst_mae_pct": pct(row.get("worst_mae")),
                    "ret_5d_pct": pct(tech.get("ret_5d")),
                    "ret_20d_pct": pct(tech.get("ret_20d")),
                    "drawdown_from_52w_high_pct": pct(dd),
                    "final_order_permission": "NOT_ALLOWED_UNTIL_CHATGPT_LIVE_REVIEW",
                }
            )

    return sorted(risks, key=lambda x: (x.get("decision_eligible", False), x["severity_0_100"]), reverse=True)


def top_watchlist(
    report: dict,
    freshness_by_ticker: dict[str, dict[str, Any]] | None = None,
    limit: int = 25,
) -> list[dict]:
    rows = []
    ranking = report.get("rule_evidence_ranking", {}) or {}
    freshness_by_ticker = freshness_by_ticker or ticker_freshness(report)
    if not isinstance(ranking, dict):
        return rows
    for ticker, items in ranking.items():
        data_status = freshness_by_ticker.get(ticker, {})
        if not data_status.get("decision_eligible", False):
            continue
        for r in items or []:
            if isinstance(r, dict) and r.get("valid"):
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
                        "data_fresh": True,
                        "staleness_business_days": data_status.get("staleness_business_days"),
                    }
                )
    return sorted(rows, key=lambda x: (x.get("active_latest_day"), x.get("evidence_score_0_100") or 0), reverse=True)[:limit]


def freshness(
    report: dict,
    freshness_by_ticker: dict[str, dict[str, Any]] | None = None,
    as_of: Any = None,
) -> dict:
    statuses = freshness_by_ticker if freshness_by_ticker is not None else ticker_freshness(report, as_of)
    date_context = market_date_context(report, as_of)
    market_data = market_data_metadata(report)
    reference_market_date = date_context["reference_market_date"]
    reported_expected_market_date = date_context["reported_expected_market_date"]
    universe = report.get("universe", {}) if isinstance(report.get("universe"), dict) else {}
    loaded_ticker_count = optional_nonnegative_int(universe.get("loaded_ticker_count"))
    configured_ticker_count = optional_nonnegative_int(universe.get("configured_ticker_count"))
    fresh_request_count = optional_nonnegative_int(universe.get("fresh_request_count"))
    errors = report.get("errors")
    errors_count = len(errors) if isinstance(errors, dict) else None
    count_metadata_complete = all(
        value is not None
        for value in (loaded_ticker_count, configured_ticker_count, fresh_request_count, errors_count)
    )
    data_quality_metadata_status = (
        "COMPLETE"
        if market_data["metadata_status"] == "COMPLETE" and count_metadata_complete
        else "MISSING_OR_INVALID"
    )
    dates = [item.get("latest_date") for item in statuses.values() if item.get("latest_date")]
    stale = sorted(ticker for ticker, item in statuses.items() if not item.get("decision_eligible"))
    fresh = sorted(ticker for ticker, item in statuses.items() if item.get("decision_eligible"))
    stale_details = [
        {
            "ticker": ticker,
            "latest_date": statuses[ticker].get("latest_date"),
            "staleness_business_days": statuses[ticker].get("staleness_business_days"),
            "reason": statuses[ticker].get("exclusion_reason"),
        }
        for ticker in stale
    ]
    return {
        "source_report_generated_at_utc": report.get("generated_at_utc"),
        "market_data": market_data,
        "data_quality_metadata_status": data_quality_metadata_status,
        "reported_expected_market_date": reported_expected_market_date,
        "reference_market_date": reference_market_date,
        "reference_market_date_status": date_context["reference_market_date_status"],
        "max_staleness_business_days": MAX_STALENESS_BUSINESS_DAYS,
        "max_expected_market_date_age_days": MAX_EXPECTED_MARKET_DATE_AGE_DAYS,
        "latest_price_date_max": max(dates) if dates else None,
        "latest_price_date_min": min(dates) if dates else None,
        "data_status": "FRESH" if not stale else "PARTIAL_STALE_WATCHLIST",
        "fresh_ticker_count": len(fresh),
        "stale_ticker_count": len(stale),
        "stale_tickers": stale_details,
        "loaded_ticker_count": loaded_ticker_count,
        "configured_ticker_count": configured_ticker_count,
        "fresh_request_count": fresh_request_count,
        "errors_count": errors_count,
    }


def portfolio_digest(portfolio: dict) -> dict:
    if not portfolio:
        return {"available": False, "reason": "portfolio_backtest.json not generated yet"}
    if "load_error" in portfolio:
        return {"available": False, "reason": portfolio["load_error"]}
    scope = model_risk_scope(portfolio)
    if not scope["portfolio_context_available"]:
        return {
            "available": False,
            "reason": scope["portfolio_context_reason"],
            "model_risk_scope": scope,
        }
    return {
        "available": True,
        "version": portfolio.get("version"),
        "latest_regime": portfolio.get("latest_regime"),
        "model_risk_scope": scope,
        "assumptions": portfolio.get("assumptions", {}),
        "strategy_metrics": portfolio.get("strategy_metrics", {}),
        "strategy_vs_benchmarks": portfolio.get("strategy_vs_benchmarks", []),
        "exposure_summary": portfolio.get("exposure_summary", {}),
        "important_limits": portfolio.get("important_limits", []),
    }


def build_action_board(
    decision: dict,
    actionable: list[dict],
    buy_candidates: list[dict],
    risk_candidates: list[dict],
    model_high_risk: list[dict],
    advisory_high_risk: list[dict],
    stale_model_tickers: list[str],
    portfolio: dict,
) -> dict:
    final_action = decision["final_action"]
    shadow_research = select_shadow_candidates(
        buy_candidates,
        reference_market_date=decision["freshness"]["reference_market_date"],
    )
    if final_action == "DATA_REVIEW_REQUIRED":
        chatgpt_task = "Model-portfolio market data is stale or missing. Refresh data before using quant signals or risk flags."
        default_human_action = "NO_TRADE_UNTIL_DATA_REFRESH"
    elif final_action == "RISK_REVIEW_REQUIRED":
        chatgpt_task = "Prioritize risk review for model-portfolio symbols, then reconcile against actual IBKR holdings before deciding pause/hold/trim/sell."
        default_human_action = "DO_NOT_ADD_RISK_BEFORE_REVIEW"
    elif final_action == "BUY_CANDIDATE_REVIEW_REQUIRED":
        chatgpt_task = "Review top quant-pass candidate with live price, news, macro, valuation, portfolio backtest, and portfolio constraints before deciding buy/wait/no trade."
        default_human_action = "WAIT_FOR_CHATGPT_FINAL_REVIEW"
    else:
        chatgpt_task = "Usually no trade. Only override if fresh live news or market risk appears after the daily report."
        default_human_action = "NO_TRADE"

    return {
        "generated_at_utc": decision["generated_at_utc"],
        "purpose": "Bridge GitHub quantitative evidence and portfolio backtest to ChatGPT final market judgment.",
        "github_is_final_decision": False,
        "automatic_order_allowed": False,
        "default_human_action_before_chatgpt_review": default_human_action,
        "quant_layer_final_action": final_action,
        "quant_layer_reason": decision["reason"],
        "market_data": decision["market_data"],
        "portfolio_backtest": portfolio_digest(portfolio),
        "chatgpt_final_review_required": final_action != "NO_TRADE",
        "chatgpt_task": chatgpt_task,
        "shadow_research_review": {
            "review_due": shadow_research["collection_status"]
            == "READY_FOR_SAME_DAY_SHADOW_REVIEW",
            "collection_status": shadow_research["collection_status"],
            "candidate_count": shadow_research["candidate_count"],
            "candidates": shadow_research["top"],
            "allowed_assessments": ["BUY_REVIEW", "WAIT", "REJECT", "NO_TRADE"],
            "task": (
                "Assess every frozen shadow candidate for prospective research even when the "
                "execution decision remains NO_TRADE. The assessment cannot create an order."
            ),
            "counterfactual_only": True,
            "execution_eligible": False,
            "automatic_order_allowed": False,
        },
        "top_quant_buy_candidate": actionable[0] if actionable else None,
        "top_active_buy_candidate_even_if_insufficient": buy_candidates[0] if buy_candidates else None,
        "top_risk_candidate": model_high_risk[0] if model_high_risk else (risk_candidates[0] if risk_candidates else None),
        "top_model_portfolio_risk": model_high_risk[0] if model_high_risk else None,
        "top_watchlist_risk_advisory": advisory_high_risk[0] if advisory_high_risk else None,
        "model_portfolio_high_risk_count": len(model_high_risk),
        "watchlist_high_risk_advisory_count": len(advisory_high_risk),
        "stale_model_tickers": stale_model_tickers,
        "actionable_quant_buy_count": len(actionable),
        "required_live_checks_before_any_order": LIVE_REVIEW_CHECKLIST,
        "live_override_rules": {
            "block_buy_if": [
                "candidate gaps far above the tested entry area or latest report price",
                "fresh negative earnings/guidance/regulatory/news changes the thesis",
                "VIX/rates/macro regime deteriorates materially",
                "QQQ/SMH/MSFT concentration, semis exposure, or cash floor would be violated",
                "current bid/ask spread or liquidity makes limit execution poor",
                "public quote sources conflict materially and IBKR quote is unavailable",
                "portfolio-level backtest is clearly worse than QQQ/SPY without a risk-reduction benefit",
            ],
            "allow_buy_only_if": [
                "quant evidence passes",
                "portfolio-level backtest is acceptable versus QQQ/SPY/SMH",
                "current price still offers acceptable risk/reward",
                "news/fundamentals do not invalidate the signal",
                "portfolio cash and concentration guardrails pass",
                "explicit limit price and invalidation level are set",
            ],
            "risk_review_priority": "Only fresh model-portfolio risks can block the quant layer. Watchlist risks remain advisory until ChatGPT confirms an actual holding.",
        },
        "recommended_user_prompt_to_chatgpt": "Read my GitHub action_board/eason_signal/portfolio_backtest/market_report, then check live price/news/macro and tell me whether to buy, wait, pause, trim, or do nothing with exact limit price and invalidation level.",
        "freshness": decision["freshness"],
    }


def build_decision_packet(
    decision: dict,
    action_board: dict,
    model_high_risk: list[dict],
    advisory_high_risk: list[dict],
    stale_risks: list[dict],
) -> dict:
    freshness_summary = decision["freshness"]
    packet_data_status = freshness_summary.get("data_status")
    if packet_data_status not in {
        "FRESH",
        "PARTIAL_STALE_WATCHLIST",
        "STALE_MODEL_PORTFOLIO_DATA",
        "EXPECTED_MARKET_DATE_MISSING_OR_INVALID",
        "MARKET_DATA_METADATA_MISSING",
        "PORTFOLIO_CONTEXT_MISSING",
    }:
        raise ValueError(f"unsupported decision data_status: {packet_data_status!r}")
    packet_data_quality = {
        "source_report_generated_at_utc": freshness_summary.get("source_report_generated_at_utc"),
        "data_quality_metadata_status": freshness_summary.get("data_quality_metadata_status"),
        "reported_expected_market_date": freshness_summary.get("reported_expected_market_date"),
        "reference_market_date": freshness_summary.get("reference_market_date"),
        "reference_market_date_status": freshness_summary.get("reference_market_date_status"),
        "max_staleness_business_days": freshness_summary.get("max_staleness_business_days", 0),
        "max_expected_market_date_age_days": freshness_summary.get("max_expected_market_date_age_days", 0),
        "latest_price_date_max": freshness_summary.get("latest_price_date_max"),
        "latest_price_date_min": freshness_summary.get("latest_price_date_min"),
        "data_status": packet_data_status,
        "fresh_ticker_count": freshness_summary.get("fresh_ticker_count"),
        "stale_ticker_count": freshness_summary.get("stale_ticker_count"),
        "stale_tickers_in_packet_count": min(len(freshness_summary.get("stale_tickers", [])), 20),
        "stale_tickers": freshness_summary.get("stale_tickers", [])[:20],
        "loaded_ticker_count": freshness_summary.get("loaded_ticker_count"),
        "configured_ticker_count": freshness_summary.get("configured_ticker_count"),
        "fresh_request_count": freshness_summary.get("fresh_request_count"),
        "errors_count": freshness_summary.get("errors_count"),
    }
    scope = decision["portfolio_scope"]
    packet_portfolio_scope = {
        "scope_type": scope["scope_type"],
        "source": scope["source"],
        "tickers": scope["tickers"],
        "portfolio_context_available": scope["portfolio_context_available"],
        "portfolio_context_reason": scope["portfolio_context_reason"],
        "contains_private_shares_or_cash": scope["contains_private_shares_or_cash"],
        "actual_holdings_check_required_by_chatgpt": scope["actual_holdings_check_required_by_chatgpt"],
    }
    candidate_fields = (
        "ticker",
        "rule",
        "status",
        "latest_date",
        "latest_price",
        "data_fresh",
        "staleness_business_days",
        "decision_eligible",
        "horizon_days",
        "samples",
        "valid",
        "win_rate_pct",
        "avg_return_pct",
        "median_return_pct",
        "worst_return_pct",
        "avg_mae_pct",
        "worst_mae_pct",
        "benchmark",
        "avg_alpha_vs_benchmark_pct",
        "evidence_score_0_100",
        "fail_reasons",
        "final_order_permission",
    )
    compact_actionable = [
        {
            "candidate_type": "EXECUTION_CANDIDATE",
            **{key: row.get(key) for key in candidate_fields},
            "counterfactual_only": False,
            "execution_eligible": True,
            "automatic_order_allowed": False,
        }
        for row in decision["actionable_buy_candidates"][:5]
    ]
    shadow_candidates = select_shadow_candidates(
        decision["all_active_buy_candidates"],
        reference_market_date=freshness_summary["reference_market_date"],
    )
    stale_risk_fields = (
        "ticker",
        "latest_date",
        "staleness_business_days",
        "data_fresh",
        "decision_eligible",
        "data_exclusion_reason",
        "risk_scope",
    )
    compact_stale_risks = [
        {key: row.get(key) for key in stale_risk_fields}
        for row in stale_risks[:20]
    ]
    model_regime = decision.get("market_regime")
    if model_regime not in {"base", "defensive", "severe_defensive"}:
        model_regime = None
    return {
        "schema_version": "decision-packet-v5.1",
        "generated_at_utc": decision["generated_at_utc"],
        "purpose": "Small, stable handoff contract from GitHub evidence to ChatGPT live-market judgment.",
        "roles": {
            "github": "data quality, reproducible quant evidence, portfolio-model risk scope, and candidate generation",
            "chatgpt": "live quote/news/macro/valuation and actual-account risk reconciliation",
            "human": "final broker confirmation and execution",
        },
        "market_data": freshness_summary["market_data"],
        "data_quality": packet_data_quality,
        "portfolio_scope": packet_portfolio_scope,
        "market_context": {
            "model_regime": model_regime,
            "live_regime_check_required": True,
        },
        "evidence_gate": {
            "quant_signal": decision["final_action"],
            "recommended_default_action": action_board["default_human_action_before_chatgpt_review"],
            "warnings": [],
            "automatic_order_allowed": False,
        },
        "decision": {
            "final_action": decision["final_action"],
            "reason": decision["reason"],
            "buy_permission": decision["buy_permission"],
            "automatic_order_allowed": False,
            "chatgpt_review_required": action_board["chatgpt_final_review_required"],
            "default_human_action": action_board["default_human_action_before_chatgpt_review"],
        },
        "candidates": {
            "execution": {
                "candidate_type": "EXECUTION_CANDIDATE",
                "candidate_count": decision.get(
                    "actionable_buy_count_total",
                    len(decision["actionable_buy_candidates"]),
                ),
                "top": compact_actionable,
                "counterfactual_only": False,
                "automatic_order_allowed": False,
            },
            "shadow": shadow_candidates,
        },
        "risks": {
            "model_portfolio_high": model_high_risk[:10],
            "watchlist_high_advisory": advisory_high_risk[:10],
            "stale_excluded": compact_stale_risks,
        },
        "execution_contract": {
            "automatic_order_allowed": False,
            "required_live_checks": LIVE_REVIEW_CHECKLIST,
            "required_final_inputs": [
                "IBKR tradable quote",
                "actual holdings and cash",
                "fresh news and earnings context",
                "same-day macro and market regime",
                "explicit limit price, invalidation level, and cancel condition",
            ],
        },
    }


def compile_decision_outputs(
    report: dict,
    portfolio: dict,
    generated_at_utc: str | None = None,
    as_of: Any = None,
) -> dict[str, Any]:
    generated_at_utc = generated_at_utc or datetime.now(timezone.utc).isoformat()
    decision_as_of = as_of if as_of is not None else generated_at_utc
    freshness_by_ticker = ticker_freshness(report, decision_as_of)
    freshness_summary = freshness(report, freshness_by_ticker, decision_as_of)
    scope = model_risk_scope(portfolio)
    scope_tickers = set(scope["tickers"])

    buy_candidates = build_buy_candidates(report, freshness_by_ticker)
    risk_candidates = build_risk_candidates(report, freshness_by_ticker)
    for row in risk_candidates:
        row["risk_scope"] = "model_portfolio" if row["ticker"] in scope_tickers else "watchlist_advisory"

    watch = top_watchlist(report, freshness_by_ticker)
    actionable = [
        row
        for row in buy_candidates
        if row["status"] == "QUANT_PASS_NEEDS_CHATGPT_REVIEW" and row.get("decision_eligible", False)
    ]
    model_high_risk = [
        row
        for row in risk_candidates
        if row["ticker"] in scope_tickers
        and row.get("decision_eligible", False)
        and row["severity_0_100"] >= HIGH_RISK_THRESHOLD
    ]
    advisory_high_risk = [
        row
        for row in risk_candidates
        if row["ticker"] not in scope_tickers
        and row.get("decision_eligible", False)
        and row["severity_0_100"] >= HIGH_RISK_THRESHOLD
    ]
    stale_risks = [row for row in risk_candidates if not row.get("decision_eligible", False)]
    stale_model_tickers = sorted(
        ticker
        for ticker in scope_tickers
        if not freshness_by_ticker.get(ticker, {}).get("decision_eligible", False)
    )

    if freshness_summary.get("reference_market_date_status") != "ALIGNED":
        final_action = "DATA_REVIEW_REQUIRED"
        reason = "Reported expected market date is missing or disagrees with the runtime market clock; refresh the market report before using quant signals."
        buy_permission = "BLOCKED_BY_MARKET_DATE_CONTEXT"
        freshness_summary["data_status"] = "EXPECTED_MARKET_DATE_MISSING_OR_INVALID"
    elif freshness_summary.get("data_quality_metadata_status") != "COMPLETE":
        final_action = "DATA_REVIEW_REQUIRED"
        reason = "Market-data source, timezone, timestamp, price basis, or coverage metadata is missing or invalid."
        buy_permission = "BLOCKED_BY_SOURCE_METADATA"
        freshness_summary["data_status"] = "MARKET_DATA_METADATA_MISSING"
    elif not scope.get("portfolio_context_available", False):
        final_action = "DATA_REVIEW_REQUIRED"
        reason = "Portfolio backtest context is missing or invalid; diagnostics remain available but buy review is blocked."
        buy_permission = "BLOCKED_BY_PORTFOLIO_CONTEXT"
        freshness_summary["data_status"] = "PORTFOLIO_CONTEXT_MISSING"
    elif stale_model_tickers:
        final_action = "DATA_REVIEW_REQUIRED"
        reason = "Model-portfolio data is stale or missing; refresh it before using quant signals."
        buy_permission = "BLOCKED_BY_MODEL_PORTFOLIO_DATA"
        freshness_summary["data_status"] = "STALE_MODEL_PORTFOLIO_DATA"
    elif model_high_risk:
        final_action = "RISK_REVIEW_REQUIRED"
        reason = "At least one fresh model-portfolio ticker has high technical/rule risk. Review defense before adding risk."
        buy_permission = "BLOCKED_BY_MODEL_PORTFOLIO_RISK"
    elif actionable:
        final_action = "BUY_CANDIDATE_REVIEW_REQUIRED"
        reason = "At least one fresh active signal has valid 20d evidence. ChatGPT live review is still required."
        buy_permission = "CHATGPT_LIVE_REVIEW_REQUIRED"
    else:
        final_action = "NO_TRADE"
        reason = "No fresh active signal passed the sample, return, alpha, and MAE filters."
        buy_permission = "NO_QUANT_CANDIDATE"

    decision = {
        "schema_version": "eason-signal-v5.1",
        "generated_at_utc": generated_at_utc,
        "source": "Derived from market_report.json, portfolio_backtest.json, and cached quant evidence.",
        "decision_policy": {
            "github_role": "quantitative evidence filter and public portfolio-model risk scope only",
            "chatgpt_role": "final live-market reviewer, actual-account risk reviewer, and execution planner",
            "human_role": "final order confirmation in broker",
            "automatic_order_allowed": False,
            "max_staleness_business_days": MAX_STALENESS_BUSINESS_DAYS,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "watchlist_risk_behavior": "advisory_only_until_actual_holding_is_confirmed",
            "buy_requires": [
                "fresh latest-day signal",
                "20d sample count >= 20",
                "win rate >= 55%",
                "average and median forward return > 0",
                "average alpha versus selected benchmark > 0",
                "worst 20d MAE not worse than -15%",
                "portfolio-level backtest acceptable versus QQQ/SPY/SMH",
                "separate ChatGPT live price/news/macro/actual-account confirmation before order",
            ],
            "important_limit": "This is an evidence report, not an automatic trading bot.",
        },
        "freshness": freshness_summary,
        "market_data": freshness_summary["market_data"],
        "data_status": freshness_summary["data_status"],
        "portfolio_scope": scope,
        "portfolio_backtest": portfolio_digest(portfolio),
        "market_regime": portfolio.get("latest_regime") if isinstance(portfolio, dict) else None,
        "buy_permission": buy_permission,
        "universe": report.get("universe", {}),
        "final_action": final_action,
        "reason": reason,
        "actionable_buy_candidates": actionable[:10],
        "actionable_buy_count_total": len(actionable),
        "all_active_buy_candidates": buy_candidates[:50],
        "risk_candidates": risk_candidates[:50],
        "model_portfolio_high_risks": model_high_risk[:20],
        "watchlist_high_risks_advisory": advisory_high_risk[:20],
        "stale_risks_excluded": stale_risks[:50],
        "top_valid_watchlist_evidence": watch,
        "errors": report.get("errors", {}),
    }

    action_board = build_action_board(
        decision,
        actionable,
        buy_candidates,
        risk_candidates,
        model_high_risk,
        advisory_high_risk,
        stale_model_tickers,
        portfolio,
    )
    decision_packet = build_decision_packet(
        decision,
        action_board,
        model_high_risk,
        advisory_high_risk,
        stale_risks,
    )
    latest_decision_summary = {
        "generated_at_utc": generated_at_utc,
        "summary_type": "decision",
        "source_file": "eason_signal.json",
        "decision_packet_file": "decision_packet.json",
        "summary_file_version": "v5.1-shadow-evidence-without-execution-upgrade",
        "final_action": final_action,
        "reason": reason,
        "buy_permission": buy_permission,
        "market_regime": decision.get("market_regime"),
        "market_data": decision["market_data"],
        "actionable_buy_count": len(actionable),
        "shadow_candidate_count": action_board["shadow_research_review"]["candidate_count"],
        "shadow_collection_status": action_board["shadow_research_review"][
            "collection_status"
        ],
        "risk_candidate_count": len(risk_candidates),
        "high_risk_count": len(model_high_risk),
        "model_portfolio_high_risk_count": len(model_high_risk),
        "watchlist_high_risk_advisory_count": len(advisory_high_risk),
        "stale_risk_excluded_count": len(stale_risks),
        "top_actionable_buy": actionable[0] if actionable else None,
        "top_risk": risk_candidates[0] if risk_candidates else None,
        "top_model_portfolio_risk": model_high_risk[0] if model_high_risk else None,
        "top_watchlist_risk_advisory": advisory_high_risk[0] if advisory_high_risk else None,
        "chatgpt_final_review_required": action_board["chatgpt_final_review_required"],
        "default_human_action_before_chatgpt_review": action_board["default_human_action_before_chatgpt_review"],
        "portfolio_backtest_available": action_board["portfolio_backtest"].get("available", False),
        "freshness": freshness_summary,
    }
    return {
        "decision": decision,
        "action_board": action_board,
        "decision_packet": decision_packet,
        "latest_decision_summary": latest_decision_summary,
        "buy_candidates": buy_candidates,
        "risk_candidates": risk_candidates,
    }


def main() -> None:
    if not REPORT_PATH.exists() or REPORT_PATH.stat().st_size == 0:
        raise SystemExit("docs/market_report.json not found or empty. Run scripts/build_report_safe.py first.")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    portfolio = load_optional_json(PORTFOLIO_PATH)
    outputs = compile_decision_outputs(report, portfolio)

    schema = json.loads(DECISION_PACKET_SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_schema(outputs["decision_packet"], schema)
    validate_invariants(outputs["decision_packet"])
    for payload in (
        outputs["decision"],
        outputs["action_board"],
        outputs["latest_decision_summary"],
        outputs["decision_packet"],
    ):
        json.dumps(payload, allow_nan=False)

    OUT.mkdir(parents=True, exist_ok=True)
    for path, payload in [
        (OUT / "eason_signal.json", outputs["decision"]),
        (OUT / "action_board.json", outputs["action_board"]),
        (DECISION_SUMMARY_PATH, outputs["latest_decision_summary"]),
        (DECISION_PACKET_PATH, outputs["decision_packet"]),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")

    packet = outputs["decision_packet"]
    signal_text = "\n".join(
        [
            "Eason Quant v5.1 Decision Status",
            f"generated_at_utc: {packet['generated_at_utc']}",
            f"data_source: {packet['market_data']['source']}",
            f"market_timezone: {packet['market_data']['market_timezone']}",
            f"data_timestamp: {packet['market_data']['data_timestamp']}",
            f"price_frequency: {packet['market_data']['price_frequency']}",
            f"price_adjustment_policy: {packet['market_data']['price_adjustment_policy']}",
            f"final_action: {packet['decision']['final_action']}",
            f"buy_permission: {packet['decision']['buy_permission']}",
            f"reason: {packet['decision']['reason']}",
            "automatic_order_allowed: false",
            "Primary machine-readable file: docs/decision_packet.json",
        ]
    )
    (OUT / "eason_signal.txt").write_text(signal_text + "\n", encoding="utf-8")

    csv_metadata = {
        "data_source": outputs["decision_packet"]["market_data"]["source"],
        "market_timezone": outputs["decision_packet"]["market_data"]["market_timezone"],
        "data_timestamp": outputs["decision_packet"]["market_data"]["data_timestamp"],
        "price_frequency": outputs["decision_packet"]["market_data"]["price_frequency"],
        "price_adjustment_policy": outputs["decision_packet"]["market_data"]["price_adjustment_policy"],
    }
    buy_rows = [{**csv_metadata, **row} for row in outputs["buy_candidates"]]
    risk_rows = [{**csv_metadata, **row} for row in outputs["risk_candidates"]]
    pd.DataFrame(buy_rows).to_csv(OUT / "signal_candidates.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(OUT / "risk_candidates.csv", index=False)

    print(
        "Saved docs/eason_signal.json, docs/action_board.json, docs/latest_decision_summary.json, "
        "docs/decision_packet.json, docs/eason_signal.txt, docs/signal_candidates.csv, and docs/risk_candidates.csv"
    )


if __name__ == "__main__":
    main()
