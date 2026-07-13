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
    _reject_constant,
    _seal_event,
    _unique_object,
    _validate_market_inputs,
    _write_json_atomic,
    _write_jsonl_atomic,
    TICKER_PATTERN,
    load_public_json,
)
from scripts.live_review_contract import (  # noqa: E402
    DEFAULT_DECISION_PACKET,
    DEFAULT_MODEL_GOVERNANCE,
    DEFAULT_PRIVATE_CONTEXT,
    DEFAULT_REQUEST,
    DEFAULT_RESPONSE,
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    ACTION_CEILINGS,
    MAXIMUM_LIVE_ACTION,
    LiveReviewError,
    _load_object,
    _parse_timestamp as _parse_live_review_timestamp,
    build_live_review_request,
    validate_live_review_response,
    validate_response_from_files,
)
from scripts.market_clock import (  # noqa: E402
    MARKET_TIMEZONE,
    latest_completed_us_market_weekday,
    parse_market_date,
)
from scripts.market_data_contract import (  # noqa: E402
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
    extract_market_data_metadata,
)
from scripts.model_governance import governance_fingerprint, load_governance_config  # noqa: E402
from scripts.strategy_contract import (  # noqa: E402
    EXECUTION_SHIFT_BARS,
    MIN_EFFECTIVE_SAMPLE,
    PRIMARY_EVALUATION_HORIZON,
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    execution_cost_assumptions,
    net_return_after_round_trip_costs,
)
from scripts.validate_validation_split import (  # noqa: E402
    DEFAULT_MANIFEST,
    ValidationSplitError,
    assert_finite_json,
    canonical_fingerprint,
    load_anchor_market_dates,
    load_strict_json,
    validate_split_manifest,
)


DEFAULT_REPORT = ROOT / "docs" / "market_report.json"
DEFAULT_LEDGER = ROOT / "docs" / "live_review_forward_ledger.jsonl"
DEFAULT_SUMMARY = ROOT / "docs" / "live_review_forward_status.json"
DEFAULT_PRICES_DIR = ROOT / "docs"

LEDGER_SCHEMA_VERSION = "v6-live-review-forward-event-v1"
SUMMARY_SCHEMA_VERSION = "v6-live-review-forward-status-v1"
PREDICTION_NAMESPACE = "eason-v6-sanitized-live-review-prediction-v1"
OUTCOME_NAMESPACE = "eason-v6-sanitized-live-review-outcome-v1"
EVALUATION_HORIZON_BARS = PRIMARY_EVALUATION_HORIZON
ENTRY_CLOSE_OFFSET_BARS = EXECUTION_SHIFT_BARS

PUBLIC_ACTIONS = {"NO_TRADE", "WAIT", "REDUCE_REVIEW", "BUY_REVIEW"}
EVALUABLE_ACTIONS = {"NO_TRADE", "WAIT", "BUY_REVIEW"}
EVALUATION_STATUSES = {
    "PENDING",
    "NOT_EVALUATED_NO_PUBLIC_CANDIDATE",
    "NOT_EVALUATED_PRIVATE_REDUCTION_REDACTED",
    "NOT_EVALUATED_OVERLAPPING_WINDOW",
}


class LiveReviewForwardLedgerError(ForwardLedgerError):
    """Raised when a private review cannot become safe prospective public evidence."""


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LiveReviewForwardLedgerError(
            f"{context} fields differ from the public contract: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _canonical_id(prefix: str, namespace: str, identity: dict[str, Any]) -> str:
    return f"{prefix}_{canonical_fingerprint({'namespace': namespace, **identity})}"


def _prediction_event_id(
    *,
    reviewed_at_utc: str,
    reviewer_fingerprint: str,
    governance_fingerprint_value: str,
    public_decision_fingerprint: str,
    quant_final_action: str,
    public_action: str,
    public_symbol: str | None,
) -> str:
    return _canonical_id(
        "lrp",
        PREDICTION_NAMESPACE,
        {
            "reviewed_at_utc": reviewed_at_utc,
            "reviewer_fingerprint": reviewer_fingerprint,
            "model_governance_fingerprint": governance_fingerprint_value,
            "public_decision_fingerprint": public_decision_fingerprint,
            "quant_final_action": quant_final_action,
            "public_action": public_action,
            "public_symbol": public_symbol,
        },
    )


def _outcome_event_id(prediction_event_id: str) -> str:
    return _canonical_id(
        "lro",
        OUTCOME_NAMESPACE,
        {
            "prediction_event_id": prediction_event_id,
            "evaluation_horizon_bars": EVALUATION_HORIZON_BARS,
        },
    )


def _reviewer_fingerprint(response: dict[str, Any]) -> str:
    reviewer = response["reviewer"]
    return canonical_fingerprint(
        {
            "system": reviewer["system"],
            "model_id": reviewer["model_id"],
        }
    )


def _prediction_for_market_date(
    events: list[dict[str, Any]],
    observation_market_date: str,
) -> dict[str, Any] | None:
    matches = [
        event
        for event in events
        if event.get("event_type") == "LIVE_REVIEW_PREDICTION"
        and event.get("prediction", {}).get("observation_market_date")
        == observation_market_date
    ]
    if len(matches) > 1:
        raise LiveReviewForwardLedgerError(
            f"duplicate live-review prediction cohort at {observation_market_date}"
        )
    return matches[0] if matches else None


def live_review_due_status(
    *,
    packet_path: Path = DEFAULT_DECISION_PACKET,
    ledger_path: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    packet = load_public_json(packet_path, "decision_packet")
    market_data = packet.get("market_data")
    if not isinstance(market_data, dict):
        raise LiveReviewForwardLedgerError("decision_packet.market_data is required")
    observation = _parse_required_date(
        market_data.get("data_timestamp"),
        "decision_packet.market_data.data_timestamp",
    ).isoformat()
    decision_fp = canonical_fingerprint(packet)
    events = load_live_review_ledger(ledger_path)
    prediction_dates = [
        event["prediction"]["observation_market_date"]
        for event in events
        if event["event_type"] == "LIVE_REVIEW_PREDICTION"
    ]
    if prediction_dates and observation < max(prediction_dates):
        raise LiveReviewForwardLedgerError(
            "current public decision predates the latest live-review prediction"
        )
    existing = _prediction_for_market_date(events, observation)
    if existing is not None and existing["public_decision_fingerprint"] != decision_fp:
        raise LiveReviewForwardLedgerError(
            "current market date already has a live review bound to different public decision evidence"
        )
    due = existing is None
    return {
        "status": (
            "LIVE_REVIEW_DUE"
            if due
            else "LIVE_REVIEW_ALREADY_RECORDED_FOR_MARKET_DATE"
        ),
        "review_due": due,
        "data_timestamp": observation,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }


def _public_symbol_and_status(
    request: dict[str, Any],
    response: dict[str, Any],
) -> tuple[str | None, str, str]:
    action = response["conclusion"]["action"]
    quant_action = request["quant_boundary"]["quant_final_action"]
    candidates = request["candidate_symbols"]
    if action == "REDUCE_REVIEW":
        return None, "PRIVATE_HELD_SYMBOL_REDACTED", "NOT_EVALUATED_PRIVATE_REDUCTION_REDACTED"
    if action == "BUY_REVIEW":
        return response["conclusion"]["symbol"], "PUBLIC_GITHUB_CANDIDATE", "PENDING"
    if quant_action == "BUY_CANDIDATE_REVIEW_REQUIRED" and candidates:
        return candidates[0], "FIRST_FROZEN_GITHUB_CANDIDATE", "PENDING"
    return None, "NO_PUBLIC_EVALUATION_SYMBOL", "NOT_EVALUATED_NO_PUBLIC_CANDIDATE"


def _last_evaluable_prediction(
    events: list[dict[str, Any]],
    symbol: str,
    *,
    exclude_event_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    outcomes = {
        event["prediction_event_id"]: event
        for event in events
        if event["event_type"] == "LIVE_REVIEW_OUTCOME"
    }
    predictions = [
        event
        for event in events
        if event["event_type"] == "LIVE_REVIEW_PREDICTION"
        and event["event_id"] != exclude_event_id
        and event["prediction"]["public_symbol"] == symbol
        and event["prediction"]["evaluation_status"] == "PENDING"
    ]
    if not predictions:
        return None
    predictions.sort(
        key=lambda event: (
            event["prediction"]["observation_market_date"],
            event["prediction"]["reviewed_at_utc"],
            event["event_id"],
        )
    )
    last = predictions[-1]
    return last, outcomes.get(last["event_id"])


def _apply_non_overlap_policy(
    events: list[dict[str, Any]],
    *,
    symbol: str | None,
    observation_market_date: str,
    proposed_status: str,
    exclude_event_id: str | None = None,
) -> str:
    if proposed_status != "PENDING" or symbol is None:
        return proposed_status
    previous = _last_evaluable_prediction(
        events,
        symbol,
        exclude_event_id=exclude_event_id,
    )
    if previous is None:
        return proposed_status
    _, previous_outcome = previous
    if previous_outcome is None:
        return "NOT_EVALUATED_OVERLAPPING_WINDOW"
    prior_exit = _parse_required_date(
        previous_outcome["outcome"]["exit_market_date"],
        "previous live-review outcome exit_market_date",
    )
    observation = _parse_required_date(
        observation_market_date,
        "new live-review observation_market_date",
    )
    if observation < prior_exit:
        return "NOT_EVALUATED_OVERLAPPING_WINDOW"
    return proposed_status


def _validate_public_source_binding(
    request: dict[str, Any],
    packet: dict[str, Any],
    governance: dict[str, Any],
) -> None:
    github = request["source_context"]["github"]
    packet_market = packet.get("market_data")
    if not isinstance(packet_market, dict):
        raise LiveReviewForwardLedgerError("decision_packet.market_data is required")
    expected_github = {
        "source": packet_market.get("source"),
        "market_timezone": packet_market.get("market_timezone"),
        "report_generated_at_utc": packet_market.get("report_generated_at_utc"),
        "data_timestamp": packet_market.get("data_timestamp"),
        "price_frequency": packet_market.get("price_frequency"),
        "price_adjustment_policy": packet_market.get("price_adjustment_policy"),
        "decision_schema_version": packet.get("schema_version"),
        "model_governance_fingerprint": governance.get("model_governance_fingerprint"),
    }
    if github != expected_github:
        raise LiveReviewForwardLedgerError(
            "private review request is not bound to the supplied current public artifacts"
        )
    decision = packet.get("decision")
    if not isinstance(decision, dict):
        raise LiveReviewForwardLedgerError("decision_packet.decision is required")
    if request["quant_boundary"]["quant_final_action"] != decision.get("final_action"):
        raise LiveReviewForwardLedgerError("private review quant action differs from decision_packet")
    if request["quant_boundary"]["quant_buy_permission"] != decision.get("buy_permission"):
        raise LiveReviewForwardLedgerError("private review buy permission differs from decision_packet")


def _prediction_event(
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    packet: dict[str, Any],
    report: dict[str, Any],
    metadata: dict[str, Any],
    governance: dict[str, Any],
    split_result: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    conclusion = response["conclusion"]
    action = conclusion["action"]
    symbol, disclosure, proposed_status = _public_symbol_and_status(request, response)
    observation_market_date = metadata["data_timestamp"]
    reviewer_fp = _reviewer_fingerprint(response)
    governance_fp = governance_fingerprint(governance)
    decision_fp = canonical_fingerprint(packet)
    reviewed_at = response["reviewed_at_utc"]
    event_id = _prediction_event_id(
        reviewed_at_utc=reviewed_at,
        reviewer_fingerprint=reviewer_fp,
        governance_fingerprint_value=governance_fp,
        public_decision_fingerprint=decision_fp,
        quant_final_action=request["quant_boundary"]["quant_final_action"],
        public_action=action,
        public_symbol=symbol,
    )
    existing_cohort = _prediction_for_market_date(events, observation_market_date)
    if existing_cohort is not None and existing_cohort["event_id"] != event_id:
        raise LiveReviewForwardLedgerError(
            f"live-review prediction cohort already exists at {observation_market_date}"
        )
    evaluation_status = _apply_non_overlap_policy(
        events,
        symbol=symbol,
        observation_market_date=observation_market_date,
        proposed_status=proposed_status,
        exclude_event_id=event_id,
    )
    price_basis = None
    if symbol is not None:
        price_basis_by_ticker = report.get("price_basis_by_ticker")
        if not isinstance(price_basis_by_ticker, dict):
            raise LiveReviewForwardLedgerError("market_report.price_basis_by_ticker is required")
        price_basis = price_basis_by_ticker.get(symbol)
        if price_basis not in {"adjusted", "unadjusted"}:
            raise LiveReviewForwardLedgerError(
                f"public evaluation symbol {symbol} has no valid frozen price basis"
            )
    check_counts = Counter(check["status"] for check in response["checks"])
    return _seal_event(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": "LIVE_REVIEW_PREDICTION",
            "event_id": event_id,
            "recorded_at_utc": reviewed_at,
            "evidence_classification": "PROSPECTIVE",
            "live_review_request_contract_version": REQUEST_SCHEMA_VERSION,
            "live_review_response_contract_version": RESPONSE_SCHEMA_VERSION,
            "split_manifest_fingerprint": split_result["split_manifest_fingerprint"],
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "model_governance_fingerprint": governance_fp,
            "public_decision_fingerprint": decision_fp,
            "reviewer": {
                "system": response["reviewer"]["system"],
                "reviewer_fingerprint": reviewer_fp,
            },
            "quant_boundary": {
                "quant_final_action": request["quant_boundary"]["quant_final_action"],
                "maximum_live_action": request["quant_boundary"]["maximum_live_action"],
            },
            "sanitized_check_summary": {
                "required_check_count": len(response["checks"]),
                "pass_count": check_counts.get("PASS", 0),
                "fail_count": check_counts.get("FAIL", 0),
                "unavailable_count": check_counts.get("UNAVAILABLE", 0),
            },
            "prediction": {
                "reviewed_at_utc": reviewed_at,
                "observation_market_date": observation_market_date,
                "public_action": action,
                "public_symbol": symbol,
                "symbol_disclosure": disclosure,
                "confidence": conclusion["confidence"],
                "evaluation_status": evaluation_status,
                "entry_close_offset_bars": ENTRY_CLOSE_OFFSET_BARS,
                "evaluation_horizon_bars": EVALUATION_HORIZON_BARS,
                "non_overlapping_by_symbol": evaluation_status == "PENDING",
                "price_basis": price_basis,
            },
            "market_data": {
                "source": metadata["source"],
                "market_timezone": metadata["market_timezone"],
                "data_timestamp": metadata["data_timestamp"],
                "source_report_generated_at_utc": metadata["report_generated_at_utc"],
                "price_frequency": metadata["price_frequency"],
                "price_adjustment_policy": metadata["price_adjustment_policy"],
            },
            "privacy_transformation": {
                "classification": "SANITIZED_PUBLIC_DERIVED_EVIDENCE",
                "private_request_identifiers_removed": True,
                "private_position_symbols_removed": True,
                "selected_private_reduction_symbol_removed": action == "REDUCE_REVIEW",
                "free_text_removed": True,
                "exact_account_values_removed": True,
            },
            "contains_private_account_data": False,
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
        }
    )


def _validate_prediction(event: dict[str, Any]) -> None:
    prediction = event.get("prediction")
    reviewer = event.get("reviewer")
    quant_boundary = event.get("quant_boundary")
    checks = event.get("sanitized_check_summary")
    market = event.get("market_data")
    privacy = event.get("privacy_transformation")
    if not all(isinstance(value, dict) for value in (prediction, reviewer, quant_boundary, checks, market, privacy)):
        raise LiveReviewForwardLedgerError("live-review prediction nested objects are required")
    _exact_keys(
        prediction,
        {
            "reviewed_at_utc",
            "observation_market_date",
            "public_action",
            "public_symbol",
            "symbol_disclosure",
            "confidence",
            "evaluation_status",
            "entry_close_offset_bars",
            "evaluation_horizon_bars",
            "non_overlapping_by_symbol",
            "price_basis",
        },
        "prediction",
    )
    _exact_keys(reviewer, {"system", "reviewer_fingerprint"}, "reviewer")
    _exact_keys(quant_boundary, {"quant_final_action", "maximum_live_action"}, "quant_boundary")
    _exact_keys(
        checks,
        {"required_check_count", "pass_count", "fail_count", "unavailable_count"},
        "sanitized_check_summary",
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
        "market_data",
    )
    _exact_keys(
        privacy,
        {
            "classification",
            "private_request_identifiers_removed",
            "private_position_symbols_removed",
            "selected_private_reduction_symbol_removed",
            "free_text_removed",
            "exact_account_values_removed",
        },
        "privacy_transformation",
    )
    action = prediction["public_action"]
    symbol = prediction["public_symbol"]
    evaluation_status = prediction["evaluation_status"]
    if action not in PUBLIC_ACTIONS or evaluation_status not in EVALUATION_STATUSES:
        raise LiveReviewForwardLedgerError("unsupported public live-review action or evaluation status")
    if prediction["confidence"] not in {"LOW", "MEDIUM", "HIGH"}:
        raise LiveReviewForwardLedgerError("invalid public live-review confidence")
    quant_action = quant_boundary["quant_final_action"]
    if (
        quant_action not in ACTION_CEILINGS
        or quant_boundary["maximum_live_action"] != MAXIMUM_LIVE_ACTION[quant_action]
        or action not in ACTION_CEILINGS[quant_action]
    ):
        raise LiveReviewForwardLedgerError("public live-review action exceeds its quant boundary")
    if prediction["entry_close_offset_bars"] != ENTRY_CLOSE_OFFSET_BARS:
        raise LiveReviewForwardLedgerError("live-review entry offset differs from frozen strategy")
    if prediction["evaluation_horizon_bars"] != EVALUATION_HORIZON_BARS:
        raise LiveReviewForwardLedgerError("live-review horizon differs from frozen strategy")
    if prediction["non_overlapping_by_symbol"] is not (evaluation_status == "PENDING"):
        raise LiveReviewForwardLedgerError("live-review non-overlap flag is inconsistent")
    symbol_evaluation_statuses = {"PENDING", "NOT_EVALUATED_OVERLAPPING_WINDOW"}
    if evaluation_status in symbol_evaluation_statuses:
        if (
            action not in EVALUABLE_ACTIONS
            or not isinstance(symbol, str)
            or not TICKER_PATTERN.fullmatch(symbol)
        ):
            raise LiveReviewForwardLedgerError("symbol-level live-review evaluation requires a public symbol")
        if prediction["price_basis"] not in {"adjusted", "unadjusted"}:
            raise LiveReviewForwardLedgerError("symbol-level live-review evaluation has invalid price basis")
        expected_disclosure = (
            "PUBLIC_GITHUB_CANDIDATE"
            if action == "BUY_REVIEW"
            else "FIRST_FROZEN_GITHUB_CANDIDATE"
        )
        if prediction["symbol_disclosure"] != expected_disclosure:
            raise LiveReviewForwardLedgerError("public symbol disclosure method is invalid")
    elif action == "REDUCE_REVIEW":
        if (
            evaluation_status != "NOT_EVALUATED_PRIVATE_REDUCTION_REDACTED"
            or symbol is not None
            or prediction["price_basis"] is not None
            or prediction["symbol_disclosure"] != "PRIVATE_HELD_SYMBOL_REDACTED"
        ):
            raise LiveReviewForwardLedgerError("private reduction symbol was not fully redacted")
    elif (
        action not in {"WAIT", "NO_TRADE"}
        or evaluation_status != "NOT_EVALUATED_NO_PUBLIC_CANDIDATE"
        or symbol is not None
        or prediction["price_basis"] is not None
        or prediction["symbol_disclosure"] != "NO_PUBLIC_EVALUATION_SYMBOL"
    ):
        raise LiveReviewForwardLedgerError("non-evaluable live-review prediction is inconsistent")
    if (
        privacy["classification"] != "SANITIZED_PUBLIC_DERIVED_EVIDENCE"
        or privacy["private_request_identifiers_removed"] is not True
        or privacy["private_position_symbols_removed"] is not True
        or privacy["selected_private_reduction_symbol_removed"]
        is not (action == "REDUCE_REVIEW")
        or privacy["free_text_removed"] is not True
        or privacy["exact_account_values_removed"] is not True
    ):
        raise LiveReviewForwardLedgerError("privacy transformation is incomplete")
    counts = [checks[key] for key in ("required_check_count", "pass_count", "fail_count", "unavailable_count")]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise LiveReviewForwardLedgerError("live-review check counts must be non-negative integers")
    if checks["required_check_count"] != 6 or sum(counts[1:]) != counts[0]:
        raise LiveReviewForwardLedgerError("live-review check counts do not reconcile")
    reviewed = _parse_required_timestamp(prediction["reviewed_at_utc"], "prediction.reviewed_at_utc")
    recorded = _parse_required_timestamp(event["recorded_at_utc"], "event.recorded_at_utc")
    if reviewed != recorded:
        raise LiveReviewForwardLedgerError("live-review recorded timestamp differs from review timestamp")
    observation = _parse_required_date(
        prediction["observation_market_date"],
        "prediction.observation_market_date",
    )
    report_generated = _parse_required_timestamp(
        market["source_report_generated_at_utc"],
        "market_data.source_report_generated_at_utc",
    )
    if reviewed < report_generated or report_generated.date() < observation:
        raise LiveReviewForwardLedgerError("live-review timestamps predate their public evidence")
    if (
        not isinstance(market["source"], str)
        or not market["source"].strip()
        or market["market_timezone"] != MARKET_TIMEZONE
        or market["data_timestamp"] != observation.isoformat()
        or market["price_frequency"] != PRICE_FREQUENCY
        or market["price_adjustment_policy"] != PRICE_ADJUSTMENT_POLICY
    ):
        raise LiveReviewForwardLedgerError("live-review prediction market metadata is invalid")
    for field in ("reviewer_fingerprint",):
        value = reviewer[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LiveReviewForwardLedgerError("live-review reviewer fingerprint is invalid")
    if reviewer["system"] not in {"CHATGPT", "CODEX"}:
        raise LiveReviewForwardLedgerError("live-review reviewer system is invalid")
    expected_id = _prediction_event_id(
        reviewed_at_utc=prediction["reviewed_at_utc"],
        reviewer_fingerprint=reviewer["reviewer_fingerprint"],
        governance_fingerprint_value=event["model_governance_fingerprint"],
        public_decision_fingerprint=event["public_decision_fingerprint"],
        quant_final_action=quant_boundary["quant_final_action"],
        public_action=action,
        public_symbol=symbol,
    )
    if event["event_id"] != expected_id:
        raise LiveReviewForwardLedgerError("live-review prediction identity is invalid")


def _validate_outcome(event: dict[str, Any]) -> None:
    outcome = event.get("outcome")
    market = event.get("market_data")
    if not isinstance(outcome, dict) or not isinstance(market, dict):
        raise LiveReviewForwardLedgerError("live-review outcome nested objects are required")
    _exact_keys(
        outcome,
        {
            "public_action",
            "public_symbol",
            "observation_market_date",
            "entry_market_date",
            "exit_market_date",
            "entry_close",
            "exit_close",
            "horizon_bars",
            "status",
            "gross_counterfactual_return",
            "net_counterfactual_return_after_frozen_costs",
            "close_path_max_adverse_excursion",
            "return_definition",
            "interpretation",
            "price_basis",
        },
        "outcome",
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
        "market_data",
    )
    prediction_id = event.get("prediction_event_id")
    if not isinstance(prediction_id, str) or event["event_id"] != _outcome_event_id(prediction_id):
        raise LiveReviewForwardLedgerError("live-review outcome identity is invalid")
    if outcome["status"] != "MATURED" or outcome["horizon_bars"] != EVALUATION_HORIZON_BARS:
        raise LiveReviewForwardLedgerError("live-review outcome maturity contract is invalid")
    if outcome["public_action"] not in EVALUABLE_ACTIONS:
        raise LiveReviewForwardLedgerError("live-review outcome action is not evaluable")
    if not isinstance(outcome["public_symbol"], str) or not outcome["public_symbol"]:
        raise LiveReviewForwardLedgerError("live-review outcome public symbol is missing")
    for field in (
        "entry_close",
        "exit_close",
        "gross_counterfactual_return",
        "net_counterfactual_return_after_frozen_costs",
        "close_path_max_adverse_excursion",
    ):
        value = outcome[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise LiveReviewForwardLedgerError(f"live-review outcome {field} must be finite")
    if outcome["entry_close"] <= 0 or outcome["exit_close"] <= 0:
        raise LiveReviewForwardLedgerError("live-review outcome prices must be positive")
    observation = _parse_required_date(outcome["observation_market_date"], "outcome observation date")
    entry = _parse_required_date(outcome["entry_market_date"], "outcome entry date")
    exit_date = _parse_required_date(outcome["exit_market_date"], "outcome exit date")
    if not observation < entry < exit_date:
        raise LiveReviewForwardLedgerError("live-review outcome dates are not strictly ordered")
    if market["data_timestamp"] != exit_date.isoformat():
        raise LiveReviewForwardLedgerError("live-review outcome market timestamp differs from exit date")
    available_through = _parse_required_date(
        market["report_available_through_market_date"],
        "outcome report available-through date",
    )
    report_generated = _parse_required_timestamp(
        market["source_report_generated_at_utc"],
        "outcome source report timestamp",
    )
    recorded = _parse_required_timestamp(event["recorded_at_utc"], "outcome recorded_at_utc")
    if available_through < exit_date or report_generated.date() < available_through:
        raise LiveReviewForwardLedgerError("live-review outcome report timing is invalid")
    if recorded != report_generated:
        raise LiveReviewForwardLedgerError("live-review outcome recorded time differs from its report")
    if (
        not isinstance(market["source"], str)
        or not market["source"].strip()
        or market["market_timezone"] != MARKET_TIMEZONE
        or market["price_frequency"] != PRICE_FREQUENCY
        or market["price_adjustment_policy"] != PRICE_ADJUSTMENT_POLICY
        or outcome["price_basis"] not in {"adjusted", "unadjusted"}
    ):
        raise LiveReviewForwardLedgerError("live-review outcome market metadata is invalid")
    expected_gross = float(outcome["exit_close"]) / float(outcome["entry_close"]) - 1.0
    expected_net = float(net_return_after_round_trip_costs(expected_gross))
    if not math.isclose(
        float(outcome["gross_counterfactual_return"]),
        expected_gross,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ) or not math.isclose(
        float(outcome["net_counterfactual_return_after_frozen_costs"]),
        expected_net,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise LiveReviewForwardLedgerError("live-review outcome return calculation is invalid")
    if float(outcome["close_path_max_adverse_excursion"]) > 1e-12:
        raise LiveReviewForwardLedgerError("live-review close-path adverse excursion cannot be positive")
    expected_interpretation = (
        "HYPOTHETICAL_BUY_REVIEW_RETURN_NO_EXECUTION_CLAIM"
        if outcome["public_action"] == "BUY_REVIEW"
        else "COUNTERFACTUAL_FILTERED_CANDIDATE_RETURN_NO_EXECUTION_CLAIM"
    )
    if outcome["interpretation"] != expected_interpretation:
        raise LiveReviewForwardLedgerError("live-review outcome interpretation is invalid")


PREDICTION_TOP_LEVEL_KEYS = {
    "schema_version",
    "event_type",
    "event_id",
    "recorded_at_utc",
    "evidence_classification",
    "live_review_request_contract_version",
    "live_review_response_contract_version",
    "split_manifest_fingerprint",
    "strategy_contract_version",
    "rule_fingerprint",
    "strategy_fingerprint",
    "model_governance_fingerprint",
    "public_decision_fingerprint",
    "reviewer",
    "quant_boundary",
    "sanitized_check_summary",
    "prediction",
    "market_data",
    "privacy_transformation",
    "contains_private_account_data",
    "automatic_order_allowed",
    "human_confirmation_required",
    "immutable_payload_hash",
}

OUTCOME_TOP_LEVEL_KEYS = {
    "schema_version",
    "event_type",
    "event_id",
    "prediction_event_id",
    "recorded_at_utc",
    "evidence_classification",
    "split_manifest_fingerprint",
    "strategy_contract_version",
    "rule_fingerprint",
    "strategy_fingerprint",
    "model_governance_fingerprint",
    "public_decision_fingerprint",
    "cost_contract",
    "outcome",
    "market_data",
    "contains_private_account_data",
    "automatic_order_allowed",
    "human_confirmation_required",
    "immutable_payload_hash",
}


def _validate_event(event: dict[str, Any]) -> None:
    try:
        assert_finite_json(event)
    except ValidationSplitError as exc:
        raise LiveReviewForwardLedgerError(str(exc)) from exc
    if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise LiveReviewForwardLedgerError("unsupported live-review forward ledger schema")
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise LiveReviewForwardLedgerError("live-review forward event_id is missing")
    if event.get("immutable_payload_hash") != _event_hash(event):
        raise LiveReviewForwardLedgerError(f"live-review forward event hash mismatch: {event_id}")
    expected_contract = {
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    if any(event.get(key) != value for key, value in expected_contract.items()):
        raise LiveReviewForwardLedgerError(f"live-review forward strategy contract mismatch: {event_id}")
    if event.get("evidence_classification") != "PROSPECTIVE":
        raise LiveReviewForwardLedgerError("live-review forward evidence must be prospective")
    for field in (
        "split_manifest_fingerprint",
        "model_governance_fingerprint",
        "public_decision_fingerprint",
    ):
        value = event.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LiveReviewForwardLedgerError(f"live-review forward {field} is invalid")
    if event.get("contains_private_account_data") is not False:
        raise LiveReviewForwardLedgerError("live-review forward ledger contains private account data")
    if event.get("automatic_order_allowed") is not False or event.get("human_confirmation_required") is not True:
        raise LiveReviewForwardLedgerError("live-review forward event violates execution boundary")
    event_type = event.get("event_type")
    if event_type == "LIVE_REVIEW_PREDICTION":
        _exact_keys(event, PREDICTION_TOP_LEVEL_KEYS, "live-review prediction event")
        if (
            event.get("live_review_request_contract_version") != REQUEST_SCHEMA_VERSION
            or event.get("live_review_response_contract_version") != RESPONSE_SCHEMA_VERSION
        ):
            raise LiveReviewForwardLedgerError("live-review private contract version mismatch")
        _validate_prediction(event)
    elif event_type == "LIVE_REVIEW_OUTCOME":
        _exact_keys(event, OUTCOME_TOP_LEVEL_KEYS, "live-review outcome event")
        if event.get("cost_contract") != execution_cost_assumptions():
            raise LiveReviewForwardLedgerError("live-review outcome cost contract mismatch")
        _validate_outcome(event)
    else:
        raise LiveReviewForwardLedgerError(f"unsupported live-review forward event type: {event_type}")


def load_live_review_ledger(path: Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LiveReviewForwardLedgerError(f"could not read live-review ledger {path}: {exc}") from exc
    events: list[dict[str, Any]] = []
    ids: set[str] = set()
    prediction_dates: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise LiveReviewForwardLedgerError(f"blank live-review JSONL record at line {line_number}")
        try:
            event = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except json.JSONDecodeError as exc:
            raise LiveReviewForwardLedgerError(
                f"invalid live-review JSONL record at line {line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise LiveReviewForwardLedgerError(
                f"live-review JSONL record at line {line_number} must be an object"
            )
        _validate_event(event)
        if event["event_id"] in ids:
            raise LiveReviewForwardLedgerError(f"duplicate live-review event_id: {event['event_id']}")
        ids.add(event["event_id"])
        if event["event_type"] == "LIVE_REVIEW_PREDICTION":
            observation = event["prediction"]["observation_market_date"]
            if observation in prediction_dates:
                raise LiveReviewForwardLedgerError(
                    f"duplicate live-review prediction cohort at {observation}"
                )
            prediction_dates.add(observation)
        events.append(event)
    predictions = {
        event["event_id"]: event
        for event in events
        if event["event_type"] == "LIVE_REVIEW_PREDICTION"
    }
    outcomes: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] != "LIVE_REVIEW_OUTCOME":
            continue
        prediction_id = event["prediction_event_id"]
        prediction = predictions.get(prediction_id)
        if prediction is None:
            raise LiveReviewForwardLedgerError("live-review outcome references an unknown prediction")
        if prediction_id in outcomes:
            raise LiveReviewForwardLedgerError("duplicate live-review outcome for one prediction")
        if prediction["prediction"]["evaluation_status"] != "PENDING":
            raise LiveReviewForwardLedgerError("outcome references a non-evaluable live-review prediction")
        outcome = event["outcome"]
        if (
            outcome["public_action"] != prediction["prediction"]["public_action"]
            or outcome["public_symbol"] != prediction["prediction"]["public_symbol"]
            or outcome["observation_market_date"]
            != prediction["prediction"]["observation_market_date"]
            or outcome["price_basis"] != prediction["prediction"]["price_basis"]
            or event["model_governance_fingerprint"]
            != prediction["model_governance_fingerprint"]
            or event["public_decision_fingerprint"]
            != prediction["public_decision_fingerprint"]
        ):
            raise LiveReviewForwardLedgerError("live-review outcome does not match its prediction")
        outcomes[prediction_id] = event
    _validate_non_overlapping_samples(predictions.values(), outcomes)
    return events


def _validate_non_overlapping_samples(
    predictions: Any,
    outcomes: dict[str, dict[str, Any]],
) -> None:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for event in predictions:
        prediction = event["prediction"]
        if prediction["evaluation_status"] != "PENDING":
            continue
        by_symbol.setdefault(prediction["public_symbol"], []).append(event)
    for symbol, symbol_predictions in by_symbol.items():
        symbol_predictions.sort(
            key=lambda event: (
                event["prediction"]["observation_market_date"],
                event["prediction"]["reviewed_at_utc"],
                event["event_id"],
            )
        )
        previous: dict[str, Any] | None = None
        for current in symbol_predictions:
            if previous is not None:
                previous_outcome = outcomes.get(previous["event_id"])
                if previous_outcome is None:
                    raise LiveReviewForwardLedgerError(
                        f"evaluable {symbol} live-review samples overlap while an earlier outcome is pending"
                    )
                prior_exit = _parse_required_date(
                    previous_outcome["outcome"]["exit_market_date"],
                    "prior live-review exit date",
                )
                current_observation = _parse_required_date(
                    current["prediction"]["observation_market_date"],
                    "current live-review observation date",
                )
                if current_observation < prior_exit:
                    raise LiveReviewForwardLedgerError(
                        f"evaluable {symbol} live-review sample windows overlap"
                    )
            previous = current


def _calculate_outcome(
    prediction_event: dict[str, Any],
    *,
    report: dict[str, Any],
    metadata: dict[str, Any],
    prices_dir: Path,
    as_of_market_date: date,
) -> dict[str, Any] | None:
    prediction = prediction_event["prediction"]
    if prediction["evaluation_status"] != "PENDING":
        return None
    symbol = prediction["public_symbol"]
    report_date = _parse_required_date(metadata["data_timestamp"], "market_report.data_timestamp")
    try:
        dates, prices = _load_prices(
            symbol,
            prices_dir,
            report_data_timestamp=report_date,
            as_of_market_date=as_of_market_date,
        )
    except ForwardLedgerError as exc:
        raise LiveReviewForwardLedgerError(str(exc)) from exc
    observation = _parse_required_date(
        prediction["observation_market_date"],
        "live-review prediction observation date",
    )
    try:
        observation_index = dates.index(observation)
    except ValueError as exc:
        raise LiveReviewForwardLedgerError(
            f"live-review observation date is absent from {symbol} price history"
        ) from exc
    entry_index = observation_index + ENTRY_CLOSE_OFFSET_BARS
    exit_index = entry_index + EVALUATION_HORIZON_BARS
    if exit_index >= len(dates):
        return None
    price_basis_by_ticker = report.get("price_basis_by_ticker")
    current_basis = price_basis_by_ticker.get(symbol) if isinstance(price_basis_by_ticker, dict) else None
    if current_basis != prediction["price_basis"]:
        raise LiveReviewForwardLedgerError(
            f"{symbol} price basis changed since the immutable live-review prediction"
        )
    entry_price = prices[entry_index]
    exit_price = prices[exit_index]
    gross = exit_price / entry_price - 1.0
    net = net_return_after_round_trip_costs(gross)
    path_mae = min(price / entry_price - 1.0 for price in prices[entry_index : exit_index + 1])
    if not all(math.isfinite(float(value)) for value in (gross, net, path_mae)):
        raise LiveReviewForwardLedgerError("non-finite matured live-review outcome")
    interpretation = (
        "HYPOTHETICAL_BUY_REVIEW_RETURN_NO_EXECUTION_CLAIM"
        if prediction["public_action"] == "BUY_REVIEW"
        else "COUNTERFACTUAL_FILTERED_CANDIDATE_RETURN_NO_EXECUTION_CLAIM"
    )
    return {
        "public_action": prediction["public_action"],
        "public_symbol": symbol,
        "observation_market_date": observation.isoformat(),
        "entry_market_date": dates[entry_index].isoformat(),
        "exit_market_date": dates[exit_index].isoformat(),
        "entry_close": round(entry_price, 10),
        "exit_close": round(exit_price, 10),
        "horizon_bars": EVALUATION_HORIZON_BARS,
        "status": "MATURED",
        "gross_counterfactual_return": round(gross, 12),
        "net_counterfactual_return_after_frozen_costs": round(float(net), 12),
        "close_path_max_adverse_excursion": round(path_mae, 12),
        "return_definition": "close[t+1+h] / close[t+1] - 1; close-path MAE uses closes from t+1 through t+1+h",
        "interpretation": interpretation,
        "price_basis": prediction["price_basis"],
    }


def _outcome_event(
    prediction_event: dict[str, Any],
    outcome: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return _seal_event(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": "LIVE_REVIEW_OUTCOME",
            "event_id": _outcome_event_id(prediction_event["event_id"]),
            "prediction_event_id": prediction_event["event_id"],
            "recorded_at_utc": metadata["report_generated_at_utc"],
            "evidence_classification": "PROSPECTIVE",
            "split_manifest_fingerprint": prediction_event["split_manifest_fingerprint"],
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "model_governance_fingerprint": prediction_event[
                "model_governance_fingerprint"
            ],
            "public_decision_fingerprint": prediction_event[
                "public_decision_fingerprint"
            ],
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
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
        }
    )


def _equivalent_prediction(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return existing == candidate


def _equivalent_outcome(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    stable = copy.deepcopy(existing)
    proposed = copy.deepcopy(candidate)
    for event in (stable, proposed):
        event.pop("recorded_at_utc", None)
        event.pop("immutable_payload_hash", None)
        event["market_data"].pop("source_report_generated_at_utc", None)
        event["market_data"].pop("report_available_through_market_date", None)
    return stable == proposed


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 12)


def _rate_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(value > 0 for value in values) / len(values), 12)


def _build_summary(
    events: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    split_result: dict[str, Any],
    governance_fp: str,
    new_prediction_events: int,
    new_outcome_events: int,
) -> dict[str, Any]:
    predictions = [event for event in events if event["event_type"] == "LIVE_REVIEW_PREDICTION"]
    outcomes = [event for event in events if event["event_type"] == "LIVE_REVIEW_OUTCOME"]
    evaluable = [event for event in predictions if event["prediction"]["evaluation_status"] == "PENDING"]
    action_counts = Counter(event["prediction"]["public_action"] for event in predictions)
    evaluation_counts = Counter(event["prediction"]["evaluation_status"] for event in predictions)
    buy_returns = [
        float(event["outcome"]["net_counterfactual_return_after_frozen_costs"])
        for event in outcomes
        if event["outcome"]["public_action"] == "BUY_REVIEW"
    ]
    filtered_returns = [
        float(event["outcome"]["net_counterfactual_return_after_frozen_costs"])
        for event in outcomes
        if event["outcome"]["public_action"] in {"WAIT", "NO_TRADE"}
    ]
    buy_mae = [
        float(event["outcome"]["close_path_max_adverse_excursion"])
        for event in outcomes
        if event["outcome"]["public_action"] == "BUY_REVIEW"
    ]
    if not predictions:
        status = "WAITING_FOR_VALIDATED_PRIVATE_LIVE_REVIEW"
    elif not outcomes:
        status = "PROSPECTIVE_LIVE_REVIEWS_RECORDED_OUTCOMES_PENDING"
    elif len(outcomes) < MIN_EFFECTIVE_SAMPLE:
        status = "PROSPECTIVE_LIVE_REVIEW_EVIDENCE_ACCUMULATING_LOW_SAMPLE"
    else:
        status = "MINIMUM_PROSPECTIVE_LIVE_REVIEW_SAMPLE_REACHED"
    pending = len(evaluable) - len(outcomes)
    if pending < 0:
        raise LiveReviewForwardLedgerError("more live-review outcomes exist than evaluable predictions")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": metadata["report_generated_at_utc"],
        "status": status,
        "evidence_classification": "PROSPECTIVE",
        "data_source": metadata["source"],
        "market_timezone": metadata["market_timezone"],
        "data_timestamp": metadata["data_timestamp"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
        "split_manifest_fingerprint": split_result["split_manifest_fingerprint"],
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "model_governance_fingerprint": governance_fp,
        "live_review_request_contract_version": REQUEST_SCHEMA_VERSION,
        "live_review_response_contract_version": RESPONSE_SCHEMA_VERSION,
        "evaluation_policy": {
            "entry_close_offset_bars": ENTRY_CLOSE_OFFSET_BARS,
            "evaluation_horizon_bars": EVALUATION_HORIZON_BARS,
            "non_overlapping_by_symbol_required": True,
            "minimum_effective_sample": MIN_EFFECTIVE_SAMPLE,
            "buy_result_interpretation": "hypothetical candidate return; no execution claim",
            "wait_result_interpretation": "counterfactual filtered-candidate return; not a cash return",
        },
        "ledger_counts": {
            "prediction_events": len(predictions),
            "evaluable_prediction_events": len(evaluable),
            "non_evaluable_prediction_events": len(predictions) - len(evaluable),
            "matured_outcome_events": len(outcomes),
            "pending_outcomes": pending,
            "new_prediction_events": new_prediction_events,
            "new_outcome_events": new_outcome_events,
        },
        "public_action_counts": {
            "buy_review": action_counts.get("BUY_REVIEW", 0),
            "wait": action_counts.get("WAIT", 0),
            "no_trade": action_counts.get("NO_TRADE", 0),
            "private_reduction_review_redacted": action_counts.get("REDUCE_REVIEW", 0),
        },
        "evaluation_status_counts": {
            status_name.lower(): evaluation_counts.get(status_name, 0)
            for status_name in sorted(EVALUATION_STATUSES)
        },
        "matured_metrics": {
            "buy_review": {
                "sample_count": len(buy_returns),
                "mean_hypothetical_net_return_after_frozen_costs": _mean_or_none(buy_returns),
                "positive_hypothetical_return_rate": _rate_or_none(buy_returns),
                "mean_close_path_max_adverse_excursion": _mean_or_none(buy_mae),
            },
            "wait_or_no_trade_filter": {
                "sample_count": len(filtered_returns),
                "mean_counterfactual_candidate_net_return_after_frozen_costs": _mean_or_none(
                    filtered_returns
                ),
                "positive_counterfactual_candidate_return_rate": _rate_or_none(filtered_returns),
            },
            "sample_confidence": (
                "MINIMUM_SAMPLE_REACHED"
                if len(outcomes) >= MIN_EFFECTIVE_SAMPLE
                else "INSUFFICIENT_SAMPLE"
            ),
        },
        "privacy": {
            "classification": "SANITIZED_PUBLIC_DERIVED_EVIDENCE",
            "private_request_or_response_published": False,
            "private_reduction_symbols_published": False,
            "contains_private_account_data": False,
        },
        "contains_private_account_data": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "important_limit": (
            "This ledger measures sanitized prospective review outcomes. It does not prove an order "
            "was placed, does not expose private holdings, and does not guarantee future performance."
        ),
    }
    assert_finite_json(summary)
    return summary


def _validate_current_event_contracts(
    events: list[dict[str, Any]],
    *,
    split_result: dict[str, Any],
    governance_fp: str,
    current_market_date: date,
) -> None:
    for event in events:
        if event["split_manifest_fingerprint"] != split_result["split_manifest_fingerprint"]:
            raise LiveReviewForwardLedgerError("live-review ledger belongs to a different split")
        if event["model_governance_fingerprint"] != governance_fp:
            raise LiveReviewForwardLedgerError("live-review ledger belongs to different model governance")
        if event["event_type"] == "LIVE_REVIEW_PREDICTION":
            observation = _parse_required_date(
                event["prediction"]["observation_market_date"],
                "existing live-review observation date",
            )
            if observation > current_market_date:
                raise LiveReviewForwardLedgerError(
                    "current market report is older than an existing live-review prediction"
                )


def build_live_review_forward_ledger(
    *,
    context: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    packet_path: Path = DEFAULT_DECISION_PACKET,
    report_path: Path = DEFAULT_REPORT,
    governance_path: Path = DEFAULT_MODEL_GOVERNANCE,
    split_path: Path = DEFAULT_MANIFEST,
    ledger_path: Path = DEFAULT_LEDGER,
    summary_path: Path | None = DEFAULT_SUMMARY,
    prices_dir: Path = DEFAULT_PRICES_DIR,
    anchor_path: Path | None = None,
    as_of_market_date: date | None = None,
    validation_now: datetime | None = None,
) -> dict[str, Any]:
    supplied_private_inputs = [context is not None, request is not None, response is not None]
    if any(supplied_private_inputs) and not all(supplied_private_inputs):
        raise LiveReviewForwardLedgerError(
            "private context, request, and response must be supplied together"
        )
    manifest = load_strict_json(split_path)
    anchor_dates = load_anchor_market_dates(
        anchor_path or prices_dir / "SPY_daily.csv"
    )
    split_result = validate_split_manifest(manifest, anchor_dates=anchor_dates)
    packet = load_public_json(packet_path, "decision_packet")
    report = load_public_json(report_path, "market_report")
    governance = load_public_json(governance_path, "model_governance")
    configured_governance = load_governance_config()
    governance_fp = governance_fingerprint(configured_governance)
    if governance.get("model_governance_fingerprint") != governance_fp:
        raise LiveReviewForwardLedgerError("model_governance report fingerprint mismatch")
    as_of = as_of_market_date or latest_completed_us_market_weekday(validation_now)
    observation, metadata = _validate_market_inputs(packet, report, split_result, as_of)

    events = load_live_review_ledger(ledger_path)
    _validate_current_event_contracts(
        events,
        split_result=split_result,
        governance_fp=governance_fp,
        current_market_date=observation,
    )
    event_by_id = {event["event_id"]: event for event in events}
    new_outcomes = 0
    prediction_events = [
        event for event in events if event["event_type"] == "LIVE_REVIEW_PREDICTION"
    ]
    for prediction_event in prediction_events:
        if prediction_event["prediction"]["evaluation_status"] != "PENDING":
            continue
        outcome_id = _outcome_event_id(prediction_event["event_id"])
        calculated = _calculate_outcome(
            prediction_event,
            report=report,
            metadata=metadata,
            prices_dir=prices_dir,
            as_of_market_date=as_of,
        )
        existing = event_by_id.get(outcome_id)
        if calculated is None:
            if existing is not None:
                raise LiveReviewForwardLedgerError(
                    f"existing matured live-review outcome is no longer reproducible: {outcome_id}"
                )
            continue
        candidate = _outcome_event(prediction_event, calculated, metadata=metadata)
        if existing is not None:
            if existing.get("event_type") != "LIVE_REVIEW_OUTCOME" or not _equivalent_outcome(
                existing, candidate
            ):
                raise LiveReviewForwardLedgerError(
                    f"immutable matured live-review outcome changed: {outcome_id}"
                )
            continue
        events.append(candidate)
        event_by_id[outcome_id] = candidate
        new_outcomes += 1

    new_predictions = 0
    if context is not None and request is not None and response is not None:
        reference = validation_now or datetime.now(timezone.utc)
        policy = request.get("validation_policy")
        if not isinstance(policy, dict):
            raise LiveReviewForwardLedgerError("live-review validation policy is missing")
        expected_request = build_live_review_request(
            context,
            packet,
            governance,
            generated_at=_parse_live_review_timestamp(
                request.get("generated_at_utc"),
                "request.generated_at_utc",
            ),
            max_account_age_seconds=policy.get("max_account_snapshot_age_seconds"),
            max_context_age_seconds=policy.get("max_context_age_seconds"),
            request_ttl_seconds=policy.get("request_ttl_seconds"),
            response_max_age_seconds=policy.get("response_max_age_seconds"),
        )
        if request != expected_request:
            raise LiveReviewForwardLedgerError(
                "live-review request does not match the supplied current private/public evidence"
            )
        validate_live_review_response(request, response, now=reference)
        _validate_public_source_binding(request, packet, governance)
        candidate = _prediction_event(
            request,
            response,
            packet=packet,
            report=report,
            metadata=metadata,
            governance=configured_governance,
            split_result=split_result,
            events=events,
        )
        if candidate["prediction"]["evaluation_status"] == "PENDING":
            immediate_outcome = _calculate_outcome(
                candidate,
                report=report,
                metadata=metadata,
                prices_dir=prices_dir,
                as_of_market_date=as_of,
            )
            if immediate_outcome is not None:
                raise LiveReviewForwardLedgerError(
                    "a newly recorded live review cannot already have a prospective outcome"
                )
        existing = event_by_id.get(candidate["event_id"])
        if existing is not None:
            if existing.get("event_type") != "LIVE_REVIEW_PREDICTION" or not _equivalent_prediction(
                existing, candidate
            ):
                raise LiveReviewForwardLedgerError(
                    f"immutable sanitized live-review prediction changed: {candidate['event_id']}"
                )
        else:
            events.append(candidate)
            event_by_id[candidate["event_id"]] = candidate
            new_predictions = 1

    if new_predictions or new_outcomes:
        _write_jsonl_atomic(ledger_path, events)
    load_live_review_ledger(ledger_path)
    summary = _build_summary(
        events,
        metadata=metadata,
        split_result=split_result,
        governance_fp=governance_fp,
        new_prediction_events=new_predictions,
        new_outcome_events=new_outcomes,
    )
    if summary_path is not None:
        _write_json_atomic(summary_path, summary)
    return summary


def record_private_review_from_files(
    *,
    context_path: Path = DEFAULT_PRIVATE_CONTEXT,
    request_path: Path = DEFAULT_REQUEST,
    response_path: Path = DEFAULT_RESPONSE,
    packet_path: Path = DEFAULT_DECISION_PACKET,
    report_path: Path = DEFAULT_REPORT,
    governance_path: Path = DEFAULT_MODEL_GOVERNANCE,
    split_path: Path = DEFAULT_MANIFEST,
    ledger_path: Path = DEFAULT_LEDGER,
    summary_path: Path = DEFAULT_SUMMARY,
    prices_dir: Path = DEFAULT_PRICES_DIR,
    anchor_path: Path | None = None,
) -> dict[str, Any]:
    response = validate_response_from_files(
        request_path=request_path,
        response_path=response_path,
        context_path=context_path,
        decision_path=packet_path,
        governance_path=governance_path,
    )
    context = _load_object(context_path, "private ChatGPT account context")
    request = _load_object(request_path, "live review request")
    return build_live_review_forward_ledger(
        context=context,
        request=request,
        response=response,
        packet_path=packet_path,
        report_path=report_path,
        governance_path=governance_path,
        split_path=split_path,
        ledger_path=ledger_path,
        summary_path=summary_path,
        prices_dir=prices_dir,
        anchor_path=anchor_path,
    )


def validate_live_review_forward_artifacts(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    summary_path: Path = DEFAULT_SUMMARY,
    report_path: Path = DEFAULT_REPORT,
    split_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    events = load_live_review_ledger(ledger_path)
    summary = load_public_json(summary_path, "live_review_forward_status")
    report = load_public_json(report_path, "market_report")
    metadata = extract_market_data_metadata(report)
    if metadata.get("metadata_status") != "COMPLETE":
        raise LiveReviewForwardLedgerError("market_report metadata is invalid")
    manifest = load_strict_json(split_path)
    split_result = validate_split_manifest(manifest)
    governance_fp = governance_fingerprint(load_governance_config())
    predictions = [event for event in events if event["event_type"] == "LIVE_REVIEW_PREDICTION"]
    outcomes = [event for event in events if event["event_type"] == "LIVE_REVIEW_OUTCOME"]
    counts = summary.get("ledger_counts")
    if not isinstance(counts, dict):
        raise LiveReviewForwardLedgerError("live-review forward status ledger_counts is missing")
    expected_counts = {
        "prediction_events": len(predictions),
        "evaluable_prediction_events": sum(
            event["prediction"]["evaluation_status"] == "PENDING" for event in predictions
        ),
        "non_evaluable_prediction_events": sum(
            event["prediction"]["evaluation_status"] != "PENDING" for event in predictions
        ),
        "matured_outcome_events": len(outcomes),
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise LiveReviewForwardLedgerError("live-review status counts do not match its ledger")
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise LiveReviewForwardLedgerError("unsupported live-review forward status schema")
    if summary.get("contains_private_account_data") is not False:
        raise LiveReviewForwardLedgerError("live-review forward status contains private data")
    if summary.get("automatic_order_allowed") is not False:
        raise LiveReviewForwardLedgerError("live-review forward status permits automatic orders")
    if summary.get("human_confirmation_required") is not True:
        raise LiveReviewForwardLedgerError("live-review forward status omits human confirmation")
    expected = _build_summary(
        events,
        metadata=metadata,
        split_result=split_result,
        governance_fp=governance_fp,
        new_prediction_events=_non_negative_summary_count(
            counts.get("new_prediction_events"),
            "new_prediction_events",
        ),
        new_outcome_events=_non_negative_summary_count(
            counts.get("new_outcome_events"),
            "new_outcome_events",
        ),
    )
    if summary != expected:
        raise LiveReviewForwardLedgerError(
            "live-review forward status is not reproducible from its ledger and current report"
        )
    return {
        "status": "VALID",
        "prediction_event_count": len(predictions),
        "outcome_event_count": len(outcomes),
    }


def _non_negative_summary_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LiveReviewForwardLedgerError(
            f"live-review status {field} must be a non-negative integer"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record sanitized private-live-review evidence and mature public outcomes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "record-private-review",
        help="Local-only: validate private request/response and append one sanitized public event.",
    )
    update_parser = subparsers.add_parser(
        "update-outcomes",
        help="Public-safe: mature existing sanitized events from validated EOD price caches.",
    )
    update_parser.add_argument("--as-of-market-date", type=str)
    subparsers.add_parser("validate")
    args = parser.parse_args(argv)
    try:
        if args.command == "record-private-review":
            result = record_private_review_from_files()
        elif args.command == "update-outcomes":
            as_of = None
            if args.as_of_market_date is not None:
                as_of = parse_market_date(args.as_of_market_date)
                if as_of is None or as_of.isoformat() != args.as_of_market_date:
                    raise LiveReviewForwardLedgerError(
                        "--as-of-market-date must be YYYY-MM-DD"
                    )
            result = build_live_review_forward_ledger(as_of_market_date=as_of)
        else:
            result = validate_live_review_forward_artifacts()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (
        LiveReviewForwardLedgerError,
        LiveReviewError,
        ForwardLedgerError,
        ValidationSplitError,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"live-review forward ledger FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
