from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.build_forward_ledger import _write_json_atomic
from scripts.build_local_ibkr_context import (
    CONTEXT_SCHEMA_VERSION,
    DEFAULT_PRIVATE_CONTEXT,
    validate_private_context,
)
from scripts.ibkr_readonly import IbkrReadonlyError, ROOT
from scripts.live_review_contract import (
    DEFAULT_DECISION_PACKET,
    DEFAULT_MODEL_GOVERNANCE,
    LiveReviewError,
    _assert_private_path,
    _canonical_hash,
    _held_symbols,
    _load_object,
    _parse_timestamp,
    _positive_seconds,
    _schema_validate,
    _verify_context_freshness,
    _verify_public_market_freshness,
    _verify_source_alignment,
    _write_private_json,
)


REQUEST_SCHEMA_VERSION = "private-holding-review-request-v1"
RESPONSE_SCHEMA_VERSION = "private-holding-review-response-v1"
PUBLIC_STATUS_SCHEMA_VERSION = "public-holding-review-status-v1"

DEFAULT_REQUEST_SCHEMA = ROOT / "schemas" / "holding_review_request.schema.json"
DEFAULT_RESPONSE_SCHEMA = ROOT / "schemas" / "holding_review_response.schema.json"
DEFAULT_PUBLIC_STATUS_SCHEMA = ROOT / "schemas" / "holding_review_status.schema.json"
DEFAULT_REQUEST = ROOT / "private" / "ibkr" / "holding_review_request.json"
DEFAULT_RESPONSE = ROOT / "private" / "ibkr" / "holding_review_response.json"
DEFAULT_PUBLIC_STATUS = ROOT / "docs" / "holding_review_status.json"

ALLOWED_ACTIONS = ("HOLD", "REDUCE_REVIEW", "EXIT_REVIEW", "NO_ACTION")
ACTIONABLE_INTRADAY_KINDS = {
    "INTRADAY_REALTIME_BID_ASK",
    "INTRADAY_REALTIME_LAST",
    "INTRADAY_DELAYED",
}
REQUIRED_CHECK_IDS = (
    "market_data",
    "technical_risk",
    "fresh_news_events",
    "earnings_fundamentals",
    "actual_account_risk",
)
EXPECTED_DATA_KINDS = {
    "market_data": {
        "INTRADAY_REALTIME_BID_ASK",
        "INTRADAY_REALTIME_LAST",
        "INTRADAY_DELAYED",
        "END_OF_DAY_ADJUSTED",
        "END_OF_DAY_UNADJUSTED",
        "GITHUB_QUANT_EVIDENCE",
    },
    "technical_risk": {
        "INTRADAY_REALTIME_BID_ASK",
        "INTRADAY_REALTIME_LAST",
        "INTRADAY_DELAYED",
        "END_OF_DAY_ADJUSTED",
        "END_OF_DAY_UNADJUSTED",
        "GITHUB_QUANT_EVIDENCE",
    },
    "fresh_news_events": {"NEWS_OR_EVENT"},
    "earnings_fundamentals": {"FUNDAMENTAL_OR_EARNINGS"},
    "actual_account_risk": {"PRIVATE_ACCOUNT_SNAPSHOT"},
}


class HoldingReviewError(IbkrReadonlyError):
    """Raised when a complete private holding review cannot be trusted."""


def _request_identity_payload(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": request.get("schema_version"),
        "source_context": request.get("source_context"),
        "review_scope": request.get("review_scope"),
        "required_check_ids": request.get("required_check_ids"),
        "validation_policy": request.get("validation_policy"),
    }


def _source_context(context: dict[str, Any]) -> dict[str, Any]:
    ibkr_source = context["sources"]["ibkr"]
    github_source = context["sources"]["github_strategy_evidence"]
    return {
        "context_schema_version": context["schema_version"],
        "context_generated_at_utc": context["generated_at_utc"],
        "context_sha256": _canonical_hash(context),
        "ibkr": {
            "source": ibkr_source["source"],
            "account_data_timezone": ibkr_source["account_data_timezone"],
            "data_timestamp": ibkr_source["data_timestamp"],
            "status": ibkr_source["status"],
            "portfolio_price_temporality": ibkr_source[
                "portfolio_price_temporality"
            ],
        },
        "github": dict(github_source),
    }


def build_holding_review_request(
    context: dict[str, Any],
    decision_packet: dict[str, Any],
    model_governance: dict[str, Any],
    *,
    generated_at: datetime | None = None,
    max_account_age_seconds: float = 300.0,
    max_context_age_seconds: float = 300.0,
    request_ttl_seconds: float = 300.0,
    response_max_age_seconds: float = 300.0,
) -> dict[str, Any]:
    validate_private_context(context)
    now = generated_at or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise HoldingReviewError("holding-review clock must include a timezone")
    now = now.astimezone(timezone.utc)
    try:
        _verify_context_freshness(
            context,
            now=now,
            max_account_age_seconds=max_account_age_seconds,
            max_context_age_seconds=max_context_age_seconds,
        )
        _verify_source_alignment(context, decision_packet, model_governance)
        _verify_public_market_freshness(decision_packet, now=now)
    except LiveReviewError as exc:
        raise HoldingReviewError(str(exc)) from exc

    ttl = _positive_seconds(request_ttl_seconds, "request_ttl_seconds")
    response_age = _positive_seconds(
        response_max_age_seconds,
        "response_max_age_seconds",
    )
    held = _held_symbols(context)
    quant_action = decision_packet.get("decision", {}).get("final_action")
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": (
            "AWAITING_HOLDING_REVIEW"
            if held
            else "NO_OPEN_HOLDINGS_REVIEW_NOT_REQUIRED"
        ),
        "request_id": "",
        "generated_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(seconds=ttl)).isoformat(),
        "privacy": {
            "classification": "PRIVATE_LOCAL_ONLY",
            "git_publish_allowed": False,
            "contains_private_position_symbols": bool(held),
            "contains_exact_account_values": False,
        },
        "required_input_files": [
            "private/ibkr/chatgpt_account_context.json",
            "docs/decision_packet.json",
            "docs/model_governance.json",
        ],
        "source_context": _source_context(context),
        "review_scope": {
            "layer_type": "HOLDING_REVIEW",
            "scope_type": "ALL_CURRENT_PRIVATE_IBKR_HOLDINGS",
            "selection_policy": "COMPLETE_UNRANKED_UNCAPPED_DISTINCT_SYMBOLS",
            "held_symbol_count": len(held),
            "held_symbols": held,
            "required_assessment_for_every_held_symbol": True,
            "allowed_actions": list(ALLOWED_ACTIONS),
            "can_create_buy_candidate": False,
            "affects_quant_final_action": False,
            "quant_final_action_at_request": quant_action,
        },
        "required_check_ids": list(REQUIRED_CHECK_IDS),
        "validation_policy": {
            "max_account_snapshot_age_seconds": float(max_account_age_seconds),
            "max_context_age_seconds": float(max_context_age_seconds),
            "request_ttl_seconds": ttl,
            "response_max_age_seconds": response_age,
            "complete_coverage_required": True,
            "incomplete_evidence_action": "NO_ACTION",
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "order_payload": None,
    }
    request["request_id"] = _canonical_hash(_request_identity_payload(request))
    validate_holding_review_request(request, now=now)
    return request


def validate_holding_review_request(
    request: dict[str, Any],
    *,
    now: datetime | None = None,
    require_unexpired: bool = True,
    schema_path: Path = DEFAULT_REQUEST_SCHEMA,
) -> dict[str, Any]:
    schema = _load_object(schema_path, "holding review request schema")
    try:
        _schema_validate(request, schema, "holding review request")
    except LiveReviewError as exc:
        raise HoldingReviewError(str(exc)) from exc
    if request.get("request_id") != _canonical_hash(_request_identity_payload(request)):
        raise HoldingReviewError("holding review request fingerprint mismatch")
    if request.get("required_input_files") != [
        "private/ibkr/chatgpt_account_context.json",
        "docs/decision_packet.json",
        "docs/model_governance.json",
    ]:
        raise HoldingReviewError("holding review input-file contract changed")
    if request.get("required_check_ids") != list(REQUIRED_CHECK_IDS):
        raise HoldingReviewError("holding review check order or membership changed")
    if request.get("source_context", {}).get("context_schema_version") != CONTEXT_SCHEMA_VERSION:
        raise HoldingReviewError("holding review context version is unsupported")

    scope = request.get("review_scope", {})
    held = scope.get("held_symbols", [])
    if held != sorted(held):
        raise HoldingReviewError("holding review symbols must be deterministic and sorted")
    if scope.get("held_symbol_count") != len(held):
        raise HoldingReviewError("holding review symbol count is inconsistent")
    expected_status = (
        "AWAITING_HOLDING_REVIEW"
        if held
        else "NO_OPEN_HOLDINGS_REVIEW_NOT_REQUIRED"
    )
    if request.get("status") != expected_status:
        raise HoldingReviewError("holding review request status is inconsistent")
    if request.get("privacy", {}).get("contains_private_position_symbols") != bool(held):
        raise HoldingReviewError("holding review privacy metadata is inconsistent")
    if request.get("automatic_order_allowed") is not False or request.get("order_payload") is not None:
        raise HoldingReviewError("holding review request must not contain an order")
    if request.get("human_confirmation_required") is not True:
        raise HoldingReviewError("holding review requires human confirmation")

    generated = _parse_timestamp(request.get("generated_at_utc"), "request.generated_at_utc")
    expires = _parse_timestamp(request.get("expires_at_utc"), "request.expires_at_utc")
    if expires <= generated:
        raise HoldingReviewError("holding review expiry must follow generation")
    policy = request.get("validation_policy", {})
    ttl = _positive_seconds(
        policy.get("request_ttl_seconds"),
        "request.validation_policy.request_ttl_seconds",
    )
    if abs((expires - generated).total_seconds() - ttl) > 0.001:
        raise HoldingReviewError("holding review expiry does not match its frozen TTL")
    for label, observed, limit in (
        (
            "embedded private context",
            _parse_timestamp(
                request.get("source_context", {}).get("context_generated_at_utc"),
                "request.source_context.context_generated_at_utc",
            ),
            _positive_seconds(
                policy.get("max_context_age_seconds"),
                "request.validation_policy.max_context_age_seconds",
            ),
        ),
        (
            "embedded IBKR account snapshot",
            _parse_timestamp(
                request.get("source_context", {}).get("ibkr", {}).get("data_timestamp"),
                "request.source_context.ibkr.data_timestamp",
            ),
            _positive_seconds(
                policy.get("max_account_snapshot_age_seconds"),
                "request.validation_policy.max_account_snapshot_age_seconds",
            ),
        ),
    ):
        age = (generated - observed).total_seconds()
        if age < -5:
            raise HoldingReviewError(f"{label} timestamp is in the future")
        if age > limit:
            raise HoldingReviewError(f"{label} is stale at request generation")
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if generated > reference + timedelta(seconds=5):
        raise HoldingReviewError("holding review request timestamp is in the future")
    if require_unexpired and reference > expires:
        raise HoldingReviewError("holding review request is expired")
    try:
        _verify_public_market_freshness(
            {"market_data": {"data_timestamp": request["source_context"]["github"]["data_timestamp"]}},
            now=generated,
        )
    except LiveReviewError as exc:
        raise HoldingReviewError(str(exc)) from exc
    return request


def validate_holding_review_response(
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    now: datetime | None = None,
    response_schema_path: Path = DEFAULT_RESPONSE_SCHEMA,
) -> dict[str, Any]:
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    validate_holding_review_request(request, now=reference, require_unexpired=True)
    schema = _load_object(response_schema_path, "holding review response schema")
    try:
        _schema_validate(response, schema, "holding review response")
    except LiveReviewError as exc:
        raise HoldingReviewError(str(exc)) from exc
    if response.get("request_id") != request["request_id"]:
        raise HoldingReviewError("holding review response request_id mismatch")
    if response.get("input_context_sha256") != request["source_context"]["context_sha256"]:
        raise HoldingReviewError("holding review response context fingerprint mismatch")
    if response.get("automatic_order_allowed") is not False or response.get("order_payload") is not None:
        raise HoldingReviewError("holding review response must not contain an order")
    if response.get("human_confirmation_required") is not True:
        raise HoldingReviewError("holding review response requires human confirmation")

    generated = _parse_timestamp(request["generated_at_utc"], "request.generated_at_utc")
    expires = _parse_timestamp(request["expires_at_utc"], "request.expires_at_utc")
    reviewed = _parse_timestamp(response.get("reviewed_at_utc"), "response.reviewed_at_utc")
    if reviewed < generated or reviewed > expires:
        raise HoldingReviewError("holding review was not completed inside the request window")
    if reviewed > reference + timedelta(seconds=5):
        raise HoldingReviewError("holding review response timestamp is in the future")
    max_response_age = _positive_seconds(
        request["validation_policy"]["response_max_age_seconds"],
        "request.validation_policy.response_max_age_seconds",
    )
    if (reference - reviewed).total_seconds() > max_response_age:
        raise HoldingReviewError("holding review response is stale")

    required_symbols = request["review_scope"]["held_symbols"]
    assessments = response["assessments"]
    assessed_symbols = [row["symbol"] for row in assessments]
    if assessed_symbols != required_symbols:
        raise HoldingReviewError(
            "holding review must assess every held symbol exactly once in frozen order"
        )
    coverage = response["coverage"]
    if (
        coverage["required_holding_count"] != len(required_symbols)
        or coverage["assessed_holding_count"] != len(assessments)
        or coverage["complete"] is not True
    ):
        raise HoldingReviewError("holding review coverage is incomplete")

    account_timestamp = _parse_timestamp(
        request["source_context"]["ibkr"]["data_timestamp"],
        "request.source_context.ibkr.data_timestamp",
    )
    for assessment in assessments:
        checks = assessment["checks"]
        check_by_id = {check["check_id"]: check for check in checks}
        if len(check_by_id) != len(checks) or list(check_by_id) != list(REQUIRED_CHECK_IDS):
            raise HoldingReviewError(
                f"{assessment['symbol']} must contain every holding-review check in order"
            )
        observed_by_check: dict[str, list[tuple[dict[str, Any], datetime]]] = {}
        for check in checks:
            check_id = check["check_id"]
            if check["status"] == "PASS" and not check["evidence_sources"]:
                raise HoldingReviewError(
                    f"passed holding-review check {check_id} has no evidence"
                )
            observed_by_check[check_id] = []
            for source in check["evidence_sources"]:
                observed = _parse_timestamp(
                    source["observed_at_utc"],
                    f"response.{assessment['symbol']}.{check_id}.observed_at_utc",
                )
                if observed > reviewed + timedelta(seconds=5):
                    raise HoldingReviewError("holding-review evidence is after the review")
                if source["source_type"].startswith("PUBLIC_") and source["source_url"] is None:
                    raise HoldingReviewError("public holding-review evidence requires a URL")
                observed_by_check[check_id].append((source, observed))
            if check["status"] == "PASS" and not any(
                source["data_kind"] in EXPECTED_DATA_KINDS[check_id]
                for source, _ in observed_by_check[check_id]
            ):
                raise HoldingReviewError(
                    f"passed holding-review check {check_id} has no compatible evidence"
                )
        account_check = check_by_id["actual_account_risk"]
        if account_check["status"] == "PASS" and not any(
            source["data_kind"] == "PRIVATE_ACCOUNT_SNAPSHOT"
            and observed == account_timestamp
            for source, observed in observed_by_check["actual_account_risk"]
        ):
            raise HoldingReviewError(
                "holding actual_account_risk does not bind to the requested IBKR snapshot"
            )
        if any(check["status"] != "PASS" for check in checks) and assessment["action"] != "NO_ACTION":
            raise HoldingReviewError(
                "failed or unavailable holding evidence requires NO_ACTION"
            )
        if assessment["action"] in {"REDUCE_REVIEW", "EXIT_REVIEW"} and not any(
            source["data_kind"] in ACTIONABLE_INTRADAY_KINDS
            and observed >= generated
            for source, observed in observed_by_check["market_data"]
        ):
            raise HoldingReviewError(
                f"{assessment['action']} requires current intraday market evidence"
            )

    confirmation = response["quant_boundary_confirmation"]
    if confirmation["original_quant_final_action"] != request["review_scope"][
        "quant_final_action_at_request"
    ]:
        raise HoldingReviewError("holding review changed the quant final action")
    return response


def _reviewer_fingerprint(response: dict[str, Any]) -> str:
    return _canonical_hash(response["reviewer"])


def build_public_holding_review_status(
    request: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    validate_holding_review_response(
        request,
        response,
        now=_parse_timestamp(response["reviewed_at_utc"], "response.reviewed_at_utc"),
    )
    count = len(response["assessments"])
    action_counts = Counter(row["action"] for row in response["assessments"])
    status = "HOLDING_REVIEW_COMPLETE" if count else "NO_OPEN_HOLDINGS"
    payload = {
        "schema_version": PUBLIC_STATUS_SCHEMA_VERSION,
        "status": status,
        "generated_at_utc": response["reviewed_at_utc"],
        "observation_market_date": request["source_context"]["github"]["data_timestamp"],
        "privacy": {
            "contains_private_position_symbols": False,
            "contains_exact_account_values": False,
            "contains_account_identifiers": False,
        },
        "scope": {
            "layer_type": "HOLDING_REVIEW",
            "scope_type": "ALL_CURRENT_PRIVATE_IBKR_HOLDINGS",
            "selection_policy": "COMPLETE_UNRANKED_UNCAPPED_DISTINCT_SYMBOLS",
            "allowed_actions": list(ALLOWED_ACTIONS),
            "tradingagents_adapter_compatible": True,
        },
        "latest_review": {
            "reviewed_at_utc": response["reviewed_at_utc"],
            "reviewed_holding_count": count,
            "coverage_complete": True,
            "action_counts": {
                action: action_counts.get(action, 0) for action in ALLOWED_ACTIONS
            },
            "reviewer_fingerprint": _reviewer_fingerprint(response),
        },
        "quant_boundary": {
            "quant_final_action": request["review_scope"]["quant_final_action_at_request"],
            "quant_final_action_unchanged": True,
            "buy_standard_modified": False,
            "can_create_buy_candidate": False,
            "affects_shadow_evidence_gate": False,
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }
    validate_public_holding_review_status(payload)
    return payload


def build_private_context_required_status(
    decision_packet: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": PUBLIC_STATUS_SCHEMA_VERSION,
        "status": "PRIVATE_IBKR_CONTEXT_REQUIRED",
        "generated_at_utc": decision_packet["generated_at_utc"],
        "observation_market_date": decision_packet["market_data"]["data_timestamp"],
        "privacy": {
            "contains_private_position_symbols": False,
            "contains_exact_account_values": False,
            "contains_account_identifiers": False,
        },
        "scope": {
            "layer_type": "HOLDING_REVIEW",
            "scope_type": "ALL_CURRENT_PRIVATE_IBKR_HOLDINGS",
            "selection_policy": "COMPLETE_UNRANKED_UNCAPPED_DISTINCT_SYMBOLS",
            "allowed_actions": list(ALLOWED_ACTIONS),
            "tradingagents_adapter_compatible": True,
        },
        "latest_review": None,
        "quant_boundary": {
            "quant_final_action": decision_packet["decision"]["final_action"],
            "quant_final_action_unchanged": True,
            "buy_standard_modified": False,
            "can_create_buy_candidate": False,
            "affects_shadow_evidence_gate": False,
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }
    validate_public_holding_review_status(payload)
    return payload


def validate_public_holding_review_status(
    payload: dict[str, Any],
    *,
    schema_path: Path = DEFAULT_PUBLIC_STATUS_SCHEMA,
) -> dict[str, Any]:
    schema = _load_object(schema_path, "holding review public status schema")
    try:
        _schema_validate(payload, schema, "holding review public status")
    except LiveReviewError as exc:
        raise HoldingReviewError(str(exc)) from exc
    latest = payload.get("latest_review")
    status = payload.get("status")
    if status == "PRIVATE_IBKR_CONTEXT_REQUIRED" and latest is not None:
        raise HoldingReviewError(
            "private-context-required status cannot claim a completed holding review"
        )
    if status in {"NO_OPEN_HOLDINGS", "HOLDING_REVIEW_COMPLETE"} and latest is None:
        raise HoldingReviewError("completed holding-review status requires review metadata")
    if latest is not None:
        if sum(latest["action_counts"].values()) != latest["reviewed_holding_count"]:
            raise HoldingReviewError("public holding-review action counts are inconsistent")
        if status == "NO_OPEN_HOLDINGS" and latest["reviewed_holding_count"] != 0:
            raise HoldingReviewError("no-open-holdings status must have zero reviewed holdings")
        if status == "HOLDING_REVIEW_COMPLETE" and latest["reviewed_holding_count"] <= 0:
            raise HoldingReviewError("completed holding review must cover at least one holding")
    if payload.get("automatic_order_allowed") is not False:
        raise HoldingReviewError("public holding-review status must prohibit orders")
    return payload


def build_request_from_files(
    *,
    context_path: Path = DEFAULT_PRIVATE_CONTEXT,
    decision_path: Path = DEFAULT_DECISION_PACKET,
    governance_path: Path = DEFAULT_MODEL_GOVERNANCE,
    output_path: Path = DEFAULT_REQUEST,
    max_account_age_seconds: float = 300.0,
    max_context_age_seconds: float = 300.0,
) -> dict[str, Any]:
    _assert_private_path(context_path)
    context = _load_object(context_path, "private ChatGPT account context")
    decision = _load_object(decision_path, "decision packet")
    governance = _load_object(governance_path, "model governance")
    request = build_holding_review_request(
        context,
        decision,
        governance,
        max_account_age_seconds=max_account_age_seconds,
        max_context_age_seconds=max_context_age_seconds,
    )
    _write_private_json(output_path, request)
    return request


def validate_response_from_files(
    *,
    request_path: Path = DEFAULT_REQUEST,
    response_path: Path = DEFAULT_RESPONSE,
    context_path: Path = DEFAULT_PRIVATE_CONTEXT,
    decision_path: Path = DEFAULT_DECISION_PACKET,
    governance_path: Path = DEFAULT_MODEL_GOVERNANCE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _assert_private_path(request_path)
    _assert_private_path(response_path)
    request = _load_object(request_path, "holding review request")
    response = _load_object(response_path, "holding review response")
    context = _load_object(context_path, "private ChatGPT account context")
    decision = _load_object(decision_path, "decision packet")
    governance = _load_object(governance_path, "model governance")
    policy = request.get("validation_policy", {})
    expected = build_holding_review_request(
        context,
        decision,
        governance,
        generated_at=_parse_timestamp(request.get("generated_at_utc"), "request.generated_at_utc"),
        max_account_age_seconds=policy.get("max_account_snapshot_age_seconds"),
        max_context_age_seconds=policy.get("max_context_age_seconds"),
        request_ttl_seconds=policy.get("request_ttl_seconds"),
        response_max_age_seconds=policy.get("response_max_age_seconds"),
    )
    if request != expected:
        raise HoldingReviewError(
            "holding review request no longer matches current private/public evidence"
        )
    validate_holding_review_response(request, response)
    return request, response


def record_review_from_files(
    *,
    request_path: Path = DEFAULT_REQUEST,
    response_path: Path = DEFAULT_RESPONSE,
    context_path: Path = DEFAULT_PRIVATE_CONTEXT,
    decision_path: Path = DEFAULT_DECISION_PACKET,
    governance_path: Path = DEFAULT_MODEL_GOVERNANCE,
    output_path: Path = DEFAULT_PUBLIC_STATUS,
) -> dict[str, Any]:
    request, response = validate_response_from_files(
        request_path=request_path,
        response_path=response_path,
        context_path=context_path,
        decision_path=decision_path,
        governance_path=governance_path,
    )
    public = build_public_holding_review_status(request, response)
    _write_json_atomic(output_path, public)
    return public


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, validate, and sanitize the private all-holdings review contract."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-request")
    build_parser.add_argument("--max-account-age-seconds", type=float, default=300.0)
    build_parser.add_argument("--max-context-age-seconds", type=float, default=300.0)
    subparsers.add_parser("validate-response")
    subparsers.add_parser("record-review")
    subparsers.add_parser("validate-public-status")
    args = parser.parse_args(argv)
    try:
        if args.command == "build-request":
            result = build_request_from_files(
                max_account_age_seconds=args.max_account_age_seconds,
                max_context_age_seconds=args.max_context_age_seconds,
            )
            output = {
                "status": result["status"],
                "held_symbol_count": result["review_scope"]["held_symbol_count"],
                "private_output": str(DEFAULT_REQUEST.relative_to(ROOT)),
                "automatic_order_allowed": False,
            }
        elif args.command == "validate-response":
            request, response = validate_response_from_files()
            output = {
                "status": response["status"],
                "reviewed_holding_count": len(response["assessments"]),
                "quant_final_action": request["review_scope"]["quant_final_action_at_request"],
                "automatic_order_allowed": False,
            }
        elif args.command == "record-review":
            result = record_review_from_files()
            output = {
                "status": result["status"],
                "public_output": str(DEFAULT_PUBLIC_STATUS.relative_to(ROOT)),
                "automatic_order_allowed": False,
            }
        else:
            result = _load_object(DEFAULT_PUBLIC_STATUS, "holding review public status")
            validate_public_holding_review_status(result)
            output = {
                "status": result["status"],
                "public_output": str(DEFAULT_PUBLIC_STATUS.relative_to(ROOT)),
                "automatic_order_allowed": False,
            }
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (HoldingReviewError, LiveReviewError, KeyError, TypeError, AssertionError) as exc:
        print(
            json.dumps(
                {
                    "status": "HOLDING_REVIEW_UNAVAILABLE",
                    "reason": str(exc),
                    "automatic_order_allowed": False,
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
