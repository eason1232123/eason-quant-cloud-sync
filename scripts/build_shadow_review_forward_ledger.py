from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_forward_ledger import (  # noqa: E402
    ForwardLedgerError,
    _event_hash,
    _load_prices,
    _parse_required_date,
    _parse_required_timestamp,
    _seal_event,
    _unique_object,
    _write_json_atomic,
    _write_jsonl_atomic,
    load_public_json,
)
from scripts.market_clock import (  # noqa: E402
    MARKET_TIMEZONE,
    latest_completed_us_market_weekday,
)
from scripts.market_data_contract import (  # noqa: E402
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
    extract_market_data_metadata,
)
from scripts.shadow_evidence_policy import (  # noqa: E402
    ALLOWED_ASSESSMENTS,
    load_shadow_evidence_policy,
    prospective_start_market_date,
)
from scripts.shadow_review_contract import (  # noqa: E402
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    ShadowReviewContractError,
    validate_shadow_review_response,
)
from scripts.strategy_contract import (  # noqa: E402
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    execution_cost_assumptions,
    net_return_after_round_trip_costs,
)
from scripts.validate_decision_packet import (  # noqa: E402
    DEFAULT_SCHEMA as DEFAULT_DECISION_SCHEMA,
    validate_invariants,
    validate_schema as validate_decision_schema,
)
from scripts.validate_validation_split import (  # noqa: E402
    assert_finite_json,
    canonical_fingerprint,
)


DEFAULT_PACKET = ROOT / "docs" / "decision_packet.json"
DEFAULT_REPORT = ROOT / "docs" / "market_report.json"
DEFAULT_LEDGER = ROOT / "docs" / "shadow_review_forward_ledger.jsonl"
DEFAULT_SUMMARY = ROOT / "docs" / "shadow_review_forward_status.json"
DEFAULT_PRICES_DIR = ROOT / "docs"

LEDGER_SCHEMA_VERSION = "shadow-review-forward-event-v1"
SUMMARY_SCHEMA_VERSION = "shadow-review-forward-status-v1"
PREDICTION_NAMESPACE = "eason-shadow-review-prediction-v1"
OUTCOME_NAMESPACE = "eason-shadow-review-outcome-v1"
PREDICTION_EVENT_TYPE = "SHADOW_REVIEW_PREDICTION"
OUTCOME_EVENT_TYPE = "SHADOW_REVIEW_OUTCOME"
EVALUATION_STATUS = "PENDING"


class ShadowReviewForwardLedgerError(ForwardLedgerError):
    """Raised when shadow review evidence would be incomplete, mutable, or retrospective."""


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ShadowReviewForwardLedgerError(
            f"{context} fields changed: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _prediction_event_id(
    *,
    request_id: str,
    candidate_id: str,
    reviewed_at_utc: str,
    reviewer_fingerprint: str,
    assessment: str,
) -> str:
    return "srp_" + canonical_fingerprint(
        {
            "namespace": PREDICTION_NAMESPACE,
            "request_id": request_id,
            "candidate_id": candidate_id,
            "reviewed_at_utc": reviewed_at_utc,
            "reviewer_fingerprint": reviewer_fingerprint,
            "assessment": assessment,
        }
    )


def _outcome_event_id(prediction_event_id: str) -> str:
    return "sro_" + canonical_fingerprint(
        {
            "namespace": OUTCOME_NAMESPACE,
            "prediction_event_id": prediction_event_id,
            "horizon_bars": 20,
        }
    )


def _reviewer_fingerprint(response: dict[str, Any]) -> str:
    return canonical_fingerprint(response["reviewer"])


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ShadowReviewForwardLedgerError(
                f"invalid shadow ledger JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise ShadowReviewForwardLedgerError("shadow ledger rows must be objects")
        _validate_event(event)
        events.append(event)
    ids = [event["event_id"] for event in events]
    if len(ids) != len(set(ids)):
        raise ShadowReviewForwardLedgerError("duplicate shadow review event id")
    _validate_lineage(events)
    return events


def _validate_hash(event: dict[str, Any]) -> None:
    if event.get("immutable_payload_hash") != _event_hash(event):
        raise ShadowReviewForwardLedgerError("shadow review immutable payload hash mismatch")


PREDICTION_KEYS = {
    "schema_version",
    "event_type",
    "event_id",
    "recorded_at_utc",
    "evidence_classification",
    "shadow_review_request_contract_version",
    "shadow_review_response_contract_version",
    "shadow_evidence_policy_fingerprint",
    "strategy_contract_version",
    "rule_fingerprint",
    "strategy_fingerprint",
    "public_decision_fingerprint",
    "reviewer",
    "sanitized_check_summary",
    "prediction",
    "market_data",
    "contains_private_account_data",
    "counterfactual_only",
    "execution_eligible",
    "automatic_order_allowed",
    "human_confirmation_required",
    "immutable_payload_hash",
}
OUTCOME_KEYS = {
    "schema_version",
    "event_type",
    "event_id",
    "prediction_event_id",
    "recorded_at_utc",
    "evidence_classification",
    "shadow_evidence_policy_fingerprint",
    "strategy_contract_version",
    "rule_fingerprint",
    "strategy_fingerprint",
    "public_decision_fingerprint",
    "cost_contract",
    "outcome",
    "market_data",
    "contains_private_account_data",
    "counterfactual_only",
    "execution_eligible",
    "automatic_order_allowed",
    "human_confirmation_required",
    "immutable_payload_hash",
}


def _validate_event(event: dict[str, Any]) -> None:
    if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ShadowReviewForwardLedgerError("unsupported shadow review ledger version")
    event_type = event.get("event_type")
    if event_type == PREDICTION_EVENT_TYPE:
        _exact_keys(event, PREDICTION_KEYS, "shadow prediction event")
        _validate_prediction(event)
    elif event_type == OUTCOME_EVENT_TYPE:
        _exact_keys(event, OUTCOME_KEYS, "shadow outcome event")
        _validate_outcome(event)
    else:
        raise ShadowReviewForwardLedgerError("unsupported shadow review event type")
    if (
        event["evidence_classification"] != "PROSPECTIVE"
        or event["contains_private_account_data"] is not False
        or event["counterfactual_only"] is not True
        or event["execution_eligible"] is not False
        or event["automatic_order_allowed"] is not False
        or event["human_confirmation_required"] is not True
    ):
        raise ShadowReviewForwardLedgerError("shadow event safety boundary changed")
    if event["strategy_contract_version"] != STRATEGY_CONTRACT_VERSION:
        raise ShadowReviewForwardLedgerError("shadow event strategy contract changed")
    if event["rule_fingerprint"] != RULE_FINGERPRINT:
        raise ShadowReviewForwardLedgerError("shadow event rule fingerprint changed")
    if event["strategy_fingerprint"] != STRATEGY_FINGERPRINT:
        raise ShadowReviewForwardLedgerError("shadow event strategy fingerprint changed")
    if event["shadow_evidence_policy_fingerprint"] != canonical_fingerprint(
        load_shadow_evidence_policy()
    ):
        raise ShadowReviewForwardLedgerError("shadow event belongs to a different evidence policy")
    _parse_required_timestamp(event["recorded_at_utc"], "shadow event recorded_at_utc")
    _validate_hash(event)


def _validate_prediction(event: dict[str, Any]) -> None:
    prediction = event.get("prediction")
    reviewer = event.get("reviewer")
    checks = event.get("sanitized_check_summary")
    market = event.get("market_data")
    if not all(isinstance(item, dict) for item in (prediction, reviewer, checks, market)):
        raise ShadowReviewForwardLedgerError("shadow prediction objects are missing")
    _exact_keys(reviewer, {"system", "reviewer_fingerprint"}, "shadow reviewer")
    _exact_keys(
        checks,
        {"required_check_count", "pass_count", "fail_count", "unavailable_count"},
        "shadow check summary",
    )
    _exact_keys(
        prediction,
        {
            "request_id",
            "cohort_candidate_count",
            "candidate_id",
            "candidate_type",
            "selection_rank",
            "reviewed_at_utc",
            "observation_market_date",
            "public_symbol",
            "rule",
            "signal_status",
            "benchmark_symbol",
            "assessment",
            "confidence",
            "evaluation_status",
            "entry_close_offset_bars",
            "evaluation_horizon_bars",
            "candidate_price_basis",
            "benchmark_price_basis",
        },
        "shadow prediction",
    )
    _exact_keys(
        market,
        {
            "source",
            "market_timezone",
            "data_timestamp",
            "source_report_generated_at_utc",
            "price_frequency",
            "price_adjustment_policy",
        },
        "shadow prediction market data",
    )
    if reviewer["system"] not in {"CHATGPT", "CODEX"}:
        raise ShadowReviewForwardLedgerError("invalid shadow reviewer system")
    fingerprint = reviewer["reviewer_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ShadowReviewForwardLedgerError("invalid shadow reviewer fingerprint")
    if (
        prediction["candidate_type"] != "SHADOW_CANDIDATE"
        or prediction["assessment"] not in ALLOWED_ASSESSMENTS
        or prediction["confidence"] not in {"LOW", "MEDIUM", "HIGH"}
        or prediction["evaluation_status"] != EVALUATION_STATUS
        or prediction["entry_close_offset_bars"] != 1
        or prediction["evaluation_horizon_bars"] != 20
        or prediction["candidate_price_basis"] not in {"adjusted", "unadjusted"}
        or prediction["benchmark_price_basis"] not in {"adjusted", "unadjusted"}
    ):
        raise ShadowReviewForwardLedgerError("invalid shadow prediction contract")
    if (
        isinstance(prediction["cohort_candidate_count"], bool)
        or not isinstance(prediction["cohort_candidate_count"], int)
        or not 1 <= prediction["cohort_candidate_count"] <= 3
        or prediction["selection_rank"] not in range(1, 4)
    ):
        raise ShadowReviewForwardLedgerError("invalid shadow cohort size or rank")
    observation = _parse_required_date(
        prediction["observation_market_date"], "shadow observation date"
    )
    reviewed = _parse_required_timestamp(prediction["reviewed_at_utc"], "shadow review timestamp")
    if reviewed != _parse_required_timestamp(event["recorded_at_utc"], "recorded timestamp"):
        raise ShadowReviewForwardLedgerError("shadow review and record timestamps differ")
    if observation < prospective_start_market_date(load_shadow_evidence_policy()):
        raise ShadowReviewForwardLedgerError("shadow prediction predates prospective start")
    if latest_completed_us_market_weekday(reviewed) != observation:
        raise ShadowReviewForwardLedgerError("shadow prediction is retrospective")
    if (
        market["market_timezone"] != MARKET_TIMEZONE
        or market["data_timestamp"] != observation.isoformat()
        or market["price_frequency"] != PRICE_FREQUENCY
        or market["price_adjustment_policy"] != PRICE_ADJUSTMENT_POLICY
    ):
        raise ShadowReviewForwardLedgerError("shadow prediction market metadata is invalid")
    counts = [
        checks["required_check_count"],
        checks["pass_count"],
        checks["fail_count"],
        checks["unavailable_count"],
    ]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ShadowReviewForwardLedgerError("shadow check counts must be non-negative integers")
    if counts[0] != 4 or sum(counts[1:]) != 4:
        raise ShadowReviewForwardLedgerError("shadow check counts do not reconcile")
    if prediction["assessment"] == "BUY_REVIEW" and checks["pass_count"] != 4:
        raise ShadowReviewForwardLedgerError("shadow BUY_REVIEW must preserve four passing checks")
    expected_id = _prediction_event_id(
        request_id=prediction["request_id"],
        candidate_id=prediction["candidate_id"],
        reviewed_at_utc=prediction["reviewed_at_utc"],
        reviewer_fingerprint=reviewer["reviewer_fingerprint"],
        assessment=prediction["assessment"],
    )
    if event["event_id"] != expected_id:
        raise ShadowReviewForwardLedgerError("shadow prediction event identity is invalid")


def _validate_outcome(event: dict[str, Any]) -> None:
    outcome = event.get("outcome")
    market = event.get("market_data")
    if not isinstance(outcome, dict) or not isinstance(market, dict):
        raise ShadowReviewForwardLedgerError("shadow outcome objects are missing")
    _exact_keys(
        outcome,
        {
            "assessment",
            "public_symbol",
            "benchmark_symbol",
            "observation_market_date",
            "entry_market_date",
            "exit_market_date",
            "entry_close",
            "exit_close",
            "benchmark_entry_close",
            "benchmark_exit_close",
            "horizon_bars",
            "status",
            "gross_counterfactual_return",
            "net_counterfactual_return_after_frozen_costs",
            "close_path_max_adverse_excursion",
            "benchmark_gross_return_same_dates",
            "gross_return_minus_benchmark",
            "net_return_after_costs_minus_benchmark",
            "return_definition",
            "interpretation",
            "candidate_price_basis",
            "benchmark_price_basis",
        },
        "shadow outcome",
    )
    _exact_keys(
        market,
        {
            "source",
            "market_timezone",
            "data_timestamp",
            "source_report_generated_at_utc",
            "report_available_through_market_date",
            "price_frequency",
            "price_adjustment_policy",
        },
        "shadow outcome market data",
    )
    if event["event_id"] != _outcome_event_id(event.get("prediction_event_id", "")):
        raise ShadowReviewForwardLedgerError("shadow outcome identity is invalid")
    if outcome["assessment"] not in ALLOWED_ASSESSMENTS:
        raise ShadowReviewForwardLedgerError("shadow outcome assessment is invalid")
    if outcome["status"] != "MATURED" or outcome["horizon_bars"] != 20:
        raise ShadowReviewForwardLedgerError("shadow outcome maturity contract is invalid")
    numeric = (
        "entry_close",
        "exit_close",
        "benchmark_entry_close",
        "benchmark_exit_close",
        "gross_counterfactual_return",
        "net_counterfactual_return_after_frozen_costs",
        "close_path_max_adverse_excursion",
        "benchmark_gross_return_same_dates",
        "gross_return_minus_benchmark",
        "net_return_after_costs_minus_benchmark",
    )
    for field in numeric:
        value = outcome[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ShadowReviewForwardLedgerError(f"shadow outcome {field} must be finite")
    if any(outcome[field] <= 0 for field in numeric[:4]):
        raise ShadowReviewForwardLedgerError("shadow outcome prices must be positive")
    gross = outcome["exit_close"] / outcome["entry_close"] - 1.0
    net = float(net_return_after_round_trip_costs(gross))
    benchmark = outcome["benchmark_exit_close"] / outcome["benchmark_entry_close"] - 1.0
    expected = {
        "gross_counterfactual_return": gross,
        "net_counterfactual_return_after_frozen_costs": net,
        "benchmark_gross_return_same_dates": benchmark,
        "gross_return_minus_benchmark": gross - benchmark,
        "net_return_after_costs_minus_benchmark": net - benchmark,
    }
    if any(
        not math.isclose(float(outcome[field]), value, rel_tol=1e-10, abs_tol=1e-10)
        for field, value in expected.items()
    ):
        raise ShadowReviewForwardLedgerError("shadow outcome return calculation is invalid")
    if outcome["close_path_max_adverse_excursion"] > 1e-12:
        raise ShadowReviewForwardLedgerError("shadow outcome adverse excursion cannot be positive")


def _validate_lineage(events: list[dict[str, Any]]) -> None:
    predictions = {
        event["event_id"]: event
        for event in events
        if event["event_type"] == PREDICTION_EVENT_TYPE
    }
    outcomes: set[str] = set()
    cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    date_requests: dict[str, str] = {}
    for prediction in predictions.values():
        item = prediction["prediction"]
        observation = item["observation_market_date"]
        request_id = item["request_id"]
        prior = date_requests.setdefault(observation, request_id)
        if prior != request_id:
            raise ShadowReviewForwardLedgerError("multiple shadow cohorts exist for one market date")
        cohorts.setdefault((observation, request_id), []).append(prediction)
    for cohort in cohorts.values():
        expected_count = cohort[0]["prediction"]["cohort_candidate_count"]
        ranks = sorted(event["prediction"]["selection_rank"] for event in cohort)
        if len(cohort) != expected_count or ranks != list(range(1, expected_count + 1)):
            raise ShadowReviewForwardLedgerError("shadow cohort is incomplete; cherry-picking is prohibited")
        if any(event["prediction"]["cohort_candidate_count"] != expected_count for event in cohort):
            raise ShadowReviewForwardLedgerError("shadow cohort candidate counts disagree")
    for event in events:
        if event["event_type"] != OUTCOME_EVENT_TYPE:
            continue
        prediction_id = event["prediction_event_id"]
        if prediction_id not in predictions or prediction_id in outcomes:
            raise ShadowReviewForwardLedgerError("shadow outcome lineage is missing or duplicated")
        prediction = predictions[prediction_id]["prediction"]
        outcome = event["outcome"]
        bindings = (
            ("assessment", "assessment"),
            ("public_symbol", "public_symbol"),
            ("benchmark_symbol", "benchmark_symbol"),
            ("observation_market_date", "observation_market_date"),
            ("candidate_price_basis", "candidate_price_basis"),
            ("benchmark_price_basis", "benchmark_price_basis"),
        )
        if any(outcome[left] != prediction[right] for left, right in bindings):
            raise ShadowReviewForwardLedgerError("shadow outcome does not match its prediction")
        outcomes.add(prediction_id)


def _market_metadata(packet: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    decision_schema = load_public_json(DEFAULT_DECISION_SCHEMA, "decision packet schema")
    validate_decision_schema(packet, decision_schema)
    validate_invariants(packet)
    metadata = extract_market_data_metadata(report)
    if metadata["metadata_status"] != "COMPLETE":
        raise ShadowReviewForwardLedgerError("market report metadata is incomplete")
    packet_market = packet["market_data"]
    expected = {
        "source": metadata["source"],
        "market_timezone": metadata["market_timezone"],
        "report_generated_at_utc": metadata["report_generated_at_utc"],
        "data_timestamp": metadata["data_timestamp"],
        "data_timestamp_granularity": metadata["data_timestamp_granularity"],
        "data_timestamp_status": metadata["data_timestamp_status"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
        "metadata_status": metadata["metadata_status"],
    }
    if packet_market != expected:
        raise ShadowReviewForwardLedgerError("decision packet and market report metadata differ")
    return metadata


def _prediction_events(
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    packet: dict[str, Any],
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if request["decision_packet_fingerprint"] != canonical_fingerprint(packet):
        raise ShadowReviewForwardLedgerError("shadow request is not bound to current decision packet")
    if request["observation_market_date"] != metadata["data_timestamp"]:
        raise ShadowReviewForwardLedgerError("shadow request market date differs from current evidence")
    policy_fingerprint = canonical_fingerprint(load_shadow_evidence_policy())
    reviewer_fingerprint = _reviewer_fingerprint(response)
    price_basis = report.get("price_basis_by_ticker")
    if not isinstance(price_basis, dict):
        raise ShadowReviewForwardLedgerError("market report price basis map is missing")
    request_by_id = {row["candidate_id"]: row for row in request["candidates"]}
    events = []
    for assessment in response["assessments"]:
        candidate = request_by_id[assessment["candidate_id"]]
        candidate_basis = price_basis.get(candidate["symbol"])
        benchmark_basis = price_basis.get(candidate["benchmark"])
        if candidate_basis not in {"adjusted", "unadjusted"} or benchmark_basis not in {
            "adjusted",
            "unadjusted",
        }:
            raise ShadowReviewForwardLedgerError("candidate or benchmark has no frozen price basis")
        counts = Counter(check["status"] for check in assessment["checks"])
        event_id = _prediction_event_id(
            request_id=request["request_id"],
            candidate_id=candidate["candidate_id"],
            reviewed_at_utc=response["reviewed_at_utc"],
            reviewer_fingerprint=reviewer_fingerprint,
            assessment=assessment["assessment"],
        )
        events.append(
            _seal_event(
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "event_type": PREDICTION_EVENT_TYPE,
                    "event_id": event_id,
                    "recorded_at_utc": response["reviewed_at_utc"],
                    "evidence_classification": "PROSPECTIVE",
                    "shadow_review_request_contract_version": REQUEST_SCHEMA_VERSION,
                    "shadow_review_response_contract_version": RESPONSE_SCHEMA_VERSION,
                    "shadow_evidence_policy_fingerprint": policy_fingerprint,
                    "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
                    "rule_fingerprint": RULE_FINGERPRINT,
                    "strategy_fingerprint": STRATEGY_FINGERPRINT,
                    "public_decision_fingerprint": request["decision_packet_fingerprint"],
                    "reviewer": {
                        "system": response["reviewer"]["system"],
                        "reviewer_fingerprint": reviewer_fingerprint,
                    },
                    "sanitized_check_summary": {
                        "required_check_count": 4,
                        "pass_count": counts.get("PASS", 0),
                        "fail_count": counts.get("FAIL", 0),
                        "unavailable_count": counts.get("UNAVAILABLE", 0),
                    },
                    "prediction": {
                        "request_id": request["request_id"],
                        "cohort_candidate_count": len(request["candidates"]),
                        "candidate_id": candidate["candidate_id"],
                        "candidate_type": "SHADOW_CANDIDATE",
                        "selection_rank": candidate["selection_rank"],
                        "reviewed_at_utc": response["reviewed_at_utc"],
                        "observation_market_date": request["observation_market_date"],
                        "public_symbol": candidate["symbol"],
                        "rule": candidate["rule"],
                        "signal_status": candidate["signal_status"],
                        "benchmark_symbol": candidate["benchmark"],
                        "assessment": assessment["assessment"],
                        "confidence": assessment["confidence"],
                        "evaluation_status": EVALUATION_STATUS,
                        "entry_close_offset_bars": 1,
                        "evaluation_horizon_bars": 20,
                        "candidate_price_basis": candidate_basis,
                        "benchmark_price_basis": benchmark_basis,
                    },
                    "market_data": {
                        "source": metadata["source"],
                        "market_timezone": metadata["market_timezone"],
                        "data_timestamp": metadata["data_timestamp"],
                        "source_report_generated_at_utc": metadata["report_generated_at_utc"],
                        "price_frequency": metadata["price_frequency"],
                        "price_adjustment_policy": metadata["price_adjustment_policy"],
                    },
                    "contains_private_account_data": False,
                    "counterfactual_only": True,
                    "execution_eligible": False,
                    "automatic_order_allowed": False,
                    "human_confirmation_required": True,
                }
            )
        )
    return events


def _calculate_outcome(
    prediction_event: dict[str, Any],
    *,
    report: dict[str, Any],
    metadata: dict[str, Any],
    prices_dir: Path,
    as_of_market_date: date,
) -> dict[str, Any] | None:
    prediction = prediction_event["prediction"]
    observation = _parse_required_date(
        prediction["observation_market_date"], "shadow observation date"
    )
    report_date = _parse_required_date(metadata["data_timestamp"], "market report date")
    symbol = prediction["public_symbol"]
    benchmark = prediction["benchmark_symbol"]
    try:
        dates, prices = _load_prices(
            symbol,
            prices_dir,
            report_data_timestamp=report_date,
            as_of_market_date=as_of_market_date,
        )
        benchmark_dates, benchmark_prices = _load_prices(
            benchmark,
            prices_dir,
            report_data_timestamp=report_date,
            as_of_market_date=as_of_market_date,
        )
    except ForwardLedgerError as exc:
        raise ShadowReviewForwardLedgerError(str(exc)) from exc
    try:
        observation_index = dates.index(observation)
    except ValueError as exc:
        raise ShadowReviewForwardLedgerError(
            f"shadow observation date is absent from {symbol} price history"
        ) from exc
    entry_index = observation_index + 1
    exit_index = entry_index + 20
    if exit_index >= len(dates):
        return None
    entry_date = dates[entry_index]
    exit_date = dates[exit_index]
    benchmark_by_date = dict(zip(benchmark_dates, benchmark_prices))
    if entry_date not in benchmark_by_date or exit_date not in benchmark_by_date:
        if not benchmark_dates or benchmark_dates[-1] < exit_date:
            return None
        raise ShadowReviewForwardLedgerError(
            f"{benchmark} benchmark lacks exact candidate entry or exit market dates"
        )
    price_basis = report.get("price_basis_by_ticker")
    if not isinstance(price_basis, dict):
        raise ShadowReviewForwardLedgerError("current market report price basis map is missing")
    if (
        price_basis.get(symbol) != prediction["candidate_price_basis"]
        or price_basis.get(benchmark) != prediction["benchmark_price_basis"]
    ):
        raise ShadowReviewForwardLedgerError("candidate or benchmark price basis changed")
    entry = prices[entry_index]
    exit_price = prices[exit_index]
    benchmark_entry = benchmark_by_date[entry_date]
    benchmark_exit = benchmark_by_date[exit_date]
    gross = exit_price / entry - 1.0
    net = float(net_return_after_round_trip_costs(gross))
    benchmark_gross = benchmark_exit / benchmark_entry - 1.0
    mae = min(price / entry - 1.0 for price in prices[entry_index : exit_index + 1])
    values = (gross, net, benchmark_gross, mae, gross - benchmark_gross, net - benchmark_gross)
    if any(not math.isfinite(value) for value in values):
        raise ShadowReviewForwardLedgerError("non-finite matured shadow outcome")
    interpretation = (
        "HYPOTHETICAL_BUY_REVIEW_RETURN_NO_EXECUTION_CLAIM"
        if prediction["assessment"] == "BUY_REVIEW"
        else "COUNTERFACTUAL_FILTERED_CANDIDATE_RETURN_NO_EXECUTION_CLAIM"
    )
    return {
        "assessment": prediction["assessment"],
        "public_symbol": symbol,
        "benchmark_symbol": benchmark,
        "observation_market_date": observation.isoformat(),
        "entry_market_date": entry_date.isoformat(),
        "exit_market_date": exit_date.isoformat(),
        "entry_close": round(entry, 10),
        "exit_close": round(exit_price, 10),
        "benchmark_entry_close": round(benchmark_entry, 10),
        "benchmark_exit_close": round(benchmark_exit, 10),
        "horizon_bars": 20,
        "status": "MATURED",
        "gross_counterfactual_return": round(gross, 12),
        "net_counterfactual_return_after_frozen_costs": round(net, 12),
        "close_path_max_adverse_excursion": round(mae, 12),
        "benchmark_gross_return_same_dates": round(benchmark_gross, 12),
        "gross_return_minus_benchmark": round(gross - benchmark_gross, 12),
        "net_return_after_costs_minus_benchmark": round(net - benchmark_gross, 12),
        "return_definition": (
            "candidate close[t+21] / close[t+1] - 1; benchmark uses the same market dates; "
            "close-path MAE uses candidate closes from t+1 through t+21"
        ),
        "interpretation": interpretation,
        "candidate_price_basis": prediction["candidate_price_basis"],
        "benchmark_price_basis": prediction["benchmark_price_basis"],
    }


def _outcome_event(
    prediction: dict[str, Any],
    outcome: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return _seal_event(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": OUTCOME_EVENT_TYPE,
            "event_id": _outcome_event_id(prediction["event_id"]),
            "prediction_event_id": prediction["event_id"],
            "recorded_at_utc": metadata["report_generated_at_utc"],
            "evidence_classification": "PROSPECTIVE",
            "shadow_evidence_policy_fingerprint": prediction[
                "shadow_evidence_policy_fingerprint"
            ],
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "public_decision_fingerprint": prediction["public_decision_fingerprint"],
            "cost_contract": execution_cost_assumptions(),
            "outcome": outcome,
            "market_data": {
                "source": metadata["source"],
                "market_timezone": metadata["market_timezone"],
                "data_timestamp": outcome["exit_market_date"],
                "source_report_generated_at_utc": metadata["report_generated_at_utc"],
                "report_available_through_market_date": metadata["data_timestamp"],
                "price_frequency": metadata["price_frequency"],
                "price_adjustment_policy": metadata["price_adjustment_policy"],
            },
            "contains_private_account_data": False,
            "counterfactual_only": True,
            "execution_eligible": False,
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
        }
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 12) if values else None


def _positive_rate(values: list[float]) -> float | None:
    return round(sum(value > 0 for value in values) / len(values), 12) if values else None


def _metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(event["outcome"]["net_counterfactual_return_after_frozen_costs"]) for event in outcomes]
    mae = [float(event["outcome"]["close_path_max_adverse_excursion"]) for event in outcomes]
    relative = [float(event["outcome"]["net_return_after_costs_minus_benchmark"]) for event in outcomes]
    return {
        "sample_count": len(outcomes),
        "mean_hypothetical_net_return_after_frozen_costs": _mean(returns),
        "positive_hypothetical_return_rate": _positive_rate(returns),
        "mean_close_path_max_adverse_excursion": _mean(mae),
        "mean_net_return_after_costs_minus_benchmark": _mean(relative),
        "positive_relative_return_rate": _positive_rate(relative),
    }


def _build_summary(events: list[dict[str, Any]], *, metadata: dict[str, Any]) -> dict[str, Any]:
    predictions = [event for event in events if event["event_type"] == PREDICTION_EVENT_TYPE]
    outcomes = [event for event in events if event["event_type"] == OUTCOME_EVENT_TYPE]
    labels = Counter(event["prediction"]["assessment"] for event in predictions)
    buy = [event for event in outcomes if event["outcome"]["assessment"] == "BUY_REVIEW"]
    filtered = [event for event in outcomes if event["outcome"]["assessment"] != "BUY_REVIEW"]
    policy = load_shadow_evidence_policy()
    minimum = policy["evaluation"]["minimum_matured_samples_for_human_pilot"]
    if metadata["data_timestamp"] < policy["prospective_start_market_date"]:
        status = "BEFORE_PROSPECTIVE_START"
    elif not predictions:
        status = "WAITING_FOR_SAME_DAY_CHATGPT_SHADOW_REVIEWS"
    elif len(outcomes) < minimum:
        status = "PROSPECTIVE_SHADOW_EVIDENCE_ACCUMULATING"
    else:
        status = "MINIMUM_PROSPECTIVE_SHADOW_SAMPLE_REACHED"
    buy_returns = [
        float(event["outcome"]["net_counterfactual_return_after_frozen_costs"])
        for event in buy
    ]
    filtered_returns = [
        float(event["outcome"]["net_counterfactual_return_after_frozen_costs"])
        for event in filtered
    ]
    discrimination = None
    if buy_returns and filtered_returns:
        discrimination = round(
            sum(buy_returns) / len(buy_returns) - sum(filtered_returns) / len(filtered_returns),
            12,
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": metadata["report_generated_at_utc"],
        "status": status,
        "evidence_classification": "PROSPECTIVE",
        "prospective_start_market_date": policy["prospective_start_market_date"],
        "historical_backfill_allowed": False,
        "data_source": metadata["source"],
        "market_timezone": metadata["market_timezone"],
        "data_timestamp": metadata["data_timestamp"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
        "shadow_evidence_policy_fingerprint": canonical_fingerprint(policy),
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "evaluation_policy": {
            "entry_close_offset_bars": 1,
            "evaluation_horizon_bars": 20,
            "minimum_matured_samples_for_human_pilot": minimum,
            "every_selected_candidate_must_be_assessed": True,
            "overlapping_windows_may_be_correlated": True,
        },
        "ledger_counts": {
            "prediction_events": len(predictions),
            "matured_outcome_events": len(outcomes),
            "pending_outcomes": len(predictions) - len(outcomes),
        },
        "assessment_counts": {
            label.lower(): labels.get(label, 0) for label in ALLOWED_ASSESSMENTS
        },
        "matured_metrics": {
            "buy_review": _metrics(buy),
            "wait_reject_or_no_trade_filter": _metrics(filtered),
            "buy_review_mean_net_return_minus_filtered_mean": discrimination,
            "sample_confidence": (
                "MINIMUM_SAMPLE_REACHED" if len(outcomes) >= minimum else "INSUFFICIENT_SAMPLE"
            ),
        },
        "release_gate_eligible_matured_sample_count": len(outcomes),
        "real_trade_count_used_as_release_prerequisite": False,
        "contains_private_account_data": False,
        "counterfactual_only": True,
        "execution_eligible": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "important_limit": (
            "Shadow outcomes are counterfactual and may contain correlated overlapping windows. "
            "They do not prove an order was placed and do not guarantee future performance."
        ),
    }
    assert_finite_json(summary)
    return summary


def build_shadow_review_forward_ledger(
    *,
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    packet_path: Path = DEFAULT_PACKET,
    report_path: Path = DEFAULT_REPORT,
    ledger_path: Path = DEFAULT_LEDGER,
    summary_path: Path | None = DEFAULT_SUMMARY,
    prices_dir: Path = DEFAULT_PRICES_DIR,
    as_of_market_date: date | None = None,
    validation_now: datetime | None = None,
) -> dict[str, Any]:
    if (request is None) is not (response is None):
        raise ShadowReviewForwardLedgerError("shadow request and response must be supplied together")
    packet = load_public_json(packet_path, "decision packet")
    report = load_public_json(report_path, "market report")
    metadata = _market_metadata(packet, report)
    as_of = as_of_market_date or latest_completed_us_market_weekday(validation_now)
    if _parse_required_date(metadata["data_timestamp"], "market report date") > as_of:
        raise ShadowReviewForwardLedgerError("market report date is in the future")
    events = _load_ledger(ledger_path)
    event_by_id = {event["event_id"]: event for event in events}
    for prediction in [event for event in events if event["event_type"] == PREDICTION_EVENT_TYPE]:
        outcome_id = _outcome_event_id(prediction["event_id"])
        calculated = _calculate_outcome(
            prediction,
            report=report,
            metadata=metadata,
            prices_dir=prices_dir,
            as_of_market_date=as_of,
        )
        existing = event_by_id.get(outcome_id)
        if calculated is None:
            if existing is not None:
                raise ShadowReviewForwardLedgerError("matured shadow outcome is no longer reproducible")
            continue
        candidate = _outcome_event(prediction, calculated, metadata=metadata)
        if existing is None:
            events.append(candidate)
            event_by_id[candidate["event_id"]] = candidate
        else:
            stable = copy.deepcopy(existing)
            proposed = copy.deepcopy(candidate)
            for item in (stable, proposed):
                item.pop("recorded_at_utc", None)
                item.pop("immutable_payload_hash", None)
                item["market_data"].pop("source_report_generated_at_utc", None)
                item["market_data"].pop("report_available_through_market_date", None)
            if stable != proposed:
                raise ShadowReviewForwardLedgerError("immutable shadow outcome changed")

    if request is not None and response is not None:
        try:
            validate_shadow_review_response(request, response, now=validation_now)
        except ShadowReviewContractError as exc:
            raise ShadowReviewForwardLedgerError(str(exc)) from exc
        candidates = _prediction_events(
            request,
            response,
            packet=packet,
            report=report,
            metadata=metadata,
        )
        existing_requests = {
            event["prediction"]["request_id"]
            for event in events
            if event["event_type"] == PREDICTION_EVENT_TYPE
            and event["prediction"]["observation_market_date"]
            == request["observation_market_date"]
        }
        if existing_requests and existing_requests != {request["request_id"]}:
            raise ShadowReviewForwardLedgerError("a different shadow cohort already exists for this date")
        for candidate in candidates:
            existing = event_by_id.get(candidate["event_id"])
            if existing is None:
                events.append(candidate)
                event_by_id[candidate["event_id"]] = candidate
            elif existing != candidate:
                raise ShadowReviewForwardLedgerError("immutable shadow prediction changed")

    events.sort(
        key=lambda event: (
            event["recorded_at_utc"],
            0 if event["event_type"] == PREDICTION_EVENT_TYPE else 1,
            event["event_id"],
        )
    )
    _validate_lineage(events)
    summary = _build_summary(events, metadata=metadata)
    _write_jsonl_atomic(ledger_path, events)
    if summary_path is not None:
        _write_json_atomic(summary_path, summary)
    return summary


def validate_shadow_review_forward_artifacts(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    summary_path: Path = DEFAULT_SUMMARY,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    events = _load_ledger(ledger_path)
    report = load_public_json(report_path, "market report")
    metadata = extract_market_data_metadata(report)
    if metadata["metadata_status"] != "COMPLETE":
        raise ShadowReviewForwardLedgerError("market report metadata is incomplete")
    committed = load_public_json(summary_path, "shadow review forward status")
    expected = _build_summary(events, metadata=metadata)
    if committed != expected:
        raise ShadowReviewForwardLedgerError("committed shadow review status is stale")
    return {
        "status": "VALID",
        "prediction_events": expected["ledger_counts"]["prediction_events"],
        "matured_outcome_events": expected["ledger_counts"]["matured_outcome_events"],
        "pending_outcomes": expected["ledger_counts"]["pending_outcomes"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record and mature counterfactual shadow reviews.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    update = subparsers.add_parser("update-outcomes")
    update.add_argument("--as-of-market-date", type=date.fromisoformat)
    record = subparsers.add_parser("record-review")
    record.add_argument("--request", type=Path, required=True)
    record.add_argument("--response", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    args = parser.parse_args(argv)
    try:
        if args.command == "update-outcomes":
            result = build_shadow_review_forward_ledger(as_of_market_date=args.as_of_market_date)
        elif args.command == "record-review":
            result = build_shadow_review_forward_ledger(
                request=load_public_json(args.request, "shadow request"),
                response=load_public_json(args.response, "shadow response"),
            )
        else:
            result = validate_shadow_review_forward_artifacts()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    except (ShadowReviewForwardLedgerError, AssertionError, ValueError) as exc:
        print(f"shadow review forward ledger FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
