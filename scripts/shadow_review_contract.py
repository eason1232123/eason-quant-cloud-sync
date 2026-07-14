from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jsonschema

from scripts.build_forward_ledger import load_public_json
from scripts.market_clock import latest_completed_us_market_weekday, parse_timestamp
from scripts.shadow_evidence_policy import (
    ALLOWED_ASSESSMENTS,
    load_shadow_evidence_policy,
    prospective_start_market_date,
)
from scripts.validate_decision_packet import (
    DEFAULT_SCHEMA as DEFAULT_DECISION_SCHEMA,
    FORMAT_CHECKER,
    validate_invariants,
    validate_schema as validate_decision_schema,
)
from scripts.validate_validation_split import canonical_fingerprint


ROOT = Path(__file__).resolve().parents[1]
REQUEST_SCHEMA_VERSION = "shadow-review-request-v1"
RESPONSE_SCHEMA_VERSION = "shadow-review-response-v1"
DEFAULT_DECISION_PACKET = ROOT / "docs" / "decision_packet.json"
DEFAULT_REQUEST_SCHEMA = ROOT / "schemas" / "shadow_review_request.schema.json"
DEFAULT_RESPONSE_SCHEMA = ROOT / "schemas" / "shadow_review_response.schema.json"
DEFAULT_REQUEST = ROOT / "private" / "shadow" / "shadow_review_request.json"
DEFAULT_RESPONSE = ROOT / "private" / "shadow" / "shadow_review_response.json"
REQUIRED_CHECK_IDS = (
    "live_price_context",
    "fresh_news_macro",
    "earnings_valuation",
    "risk_reward",
)
MAX_REVIEW_DELAY = timedelta(hours=36)


class ShadowReviewContractError(ValueError):
    """Raised when counterfactual review evidence is incomplete or retrospective."""


def _aware_utc(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.utcoffset() is None:
        raise ShadowReviewContractError("shadow review clock must include a timezone")
    return result.astimezone(timezone.utc)


def _required_timestamp(value: Any, field: str) -> datetime:
    parsed = parse_timestamp(value)
    if parsed is None or parsed.utcoffset() is None:
        raise ShadowReviewContractError(f"{field} must be a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc)


def _load_schema(path: Path, label: str) -> dict[str, Any]:
    return load_public_json(path, label)


def _schema_validate(payload: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ShadowReviewContractError(f"invalid {label} schema: {exc.message}") from exc
    validator = validator_class(schema, format_checker=FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:10]
        )
        raise ShadowReviewContractError(f"{label} schema validation failed: {details}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _request_identity(request: dict[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(request)
    identity.pop("request_id", None)
    return identity


def _candidate_id(
    *,
    decision_fingerprint: str,
    observation_market_date: str,
    symbol: str,
    rule: str,
) -> str:
    identity = {
        "namespace": "eason-shadow-candidate-v1",
        "decision_packet_fingerprint": decision_fingerprint,
        "observation_market_date": observation_market_date,
        "symbol": symbol,
        "rule": rule,
    }
    return f"shc_{canonical_fingerprint(identity)}"


def _validate_public_packet(packet: dict[str, Any]) -> None:
    schema = _load_schema(DEFAULT_DECISION_SCHEMA, "decision packet schema")
    validate_decision_schema(packet, schema)
    validate_invariants(packet)


def build_shadow_review_request(
    packet: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    _validate_public_packet(packet)
    policy = load_shadow_evidence_policy()
    current = _aware_utc(now)
    shadow = packet["candidates"]["shadow"]
    observation = packet["market_data"]["data_timestamp"]
    if observation is None:
        raise ShadowReviewContractError("decision packet market date is unavailable")
    if observation < prospective_start_market_date(policy).isoformat():
        raise ShadowReviewContractError("shadow review cannot start before the frozen prospective date")
    if shadow["collection_status"] != "READY_FOR_SAME_DAY_SHADOW_REVIEW":
        raise ShadowReviewContractError("no same-day shadow candidate cohort is reviewable")
    if latest_completed_us_market_weekday(current).isoformat() != observation:
        raise ShadowReviewContractError(
            "historical shadow-review backfill is prohibited; packet must be the latest completed market date"
        )
    rows = shadow["top"]
    if not rows or any(row["prospective_evidence_eligible"] is not True for row in rows):
        raise ShadowReviewContractError("shadow candidates are not prospective-evidence eligible")
    decision_fingerprint = canonical_fingerprint(packet)
    candidates = []
    for row in rows:
        candidates.append(
            {
                "candidate_id": _candidate_id(
                    decision_fingerprint=decision_fingerprint,
                    observation_market_date=observation,
                    symbol=row["ticker"],
                    rule=row["rule"],
                ),
                "candidate_type": "SHADOW_CANDIDATE",
                "selection_rank": row["selection_rank"],
                "symbol": row["ticker"],
                "rule": row["rule"],
                "signal_status": row["status"],
                "benchmark": row["benchmark"],
                "latest_price": row["latest_price"],
                "evidence_score_0_100": row["evidence_score_0_100"],
                "quant_evidence_sufficient": row["status"]
                == "QUANT_PASS_NEEDS_CHATGPT_REVIEW",
                "counterfactual_only": True,
                "execution_eligible": False,
                "automatic_order_allowed": False,
            }
        )
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_id": "",
        "generated_at_utc": current.isoformat(),
        "observation_market_date": observation,
        "decision_packet_fingerprint": decision_fingerprint,
        "quant_final_action": packet["decision"]["final_action"],
        "selection_policy": shadow["selection_policy"],
        "candidates": candidates,
        "required_check_ids": list(REQUIRED_CHECK_IDS),
        "allowed_assessments": list(ALLOWED_ASSESSMENTS),
        "review_contract": {
            "assess_every_candidate": True,
            "missing_assessment_rejects_entire_cohort": True,
            "historical_backfill_allowed": False,
            "assessment_changes_execution_decision": False,
        },
        "counterfactual_only": True,
        "execution_eligible": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }
    request["request_id"] = f"shr_{canonical_fingerprint(_request_identity(request))}"
    validate_shadow_review_request(request, now=current)
    return request


def validate_shadow_review_request(
    request: dict[str, Any],
    *,
    now: datetime | None = None,
    require_same_market_date: bool = True,
    schema_path: Path = DEFAULT_REQUEST_SCHEMA,
) -> dict[str, Any]:
    _schema_validate(request, _load_schema(schema_path, "shadow request schema"), "shadow request")
    expected_id = f"shr_{canonical_fingerprint(_request_identity(request))}"
    if request["request_id"] != expected_id:
        raise ShadowReviewContractError("shadow review request fingerprint mismatch")
    if request["required_check_ids"] != list(REQUIRED_CHECK_IDS):
        raise ShadowReviewContractError("shadow review check order or membership changed")
    if request["allowed_assessments"] != list(ALLOWED_ASSESSMENTS):
        raise ShadowReviewContractError("shadow review assessments changed")
    ranks = [row["selection_rank"] for row in request["candidates"]]
    if ranks != list(range(1, len(ranks) + 1)):
        raise ShadowReviewContractError("shadow candidate ranks must be consecutive and ordered")
    if len({row["candidate_id"] for row in request["candidates"]}) != len(ranks):
        raise ShadowReviewContractError("shadow candidates must be unique")
    for row in request["candidates"]:
        expected_candidate_id = _candidate_id(
            decision_fingerprint=request["decision_packet_fingerprint"],
            observation_market_date=request["observation_market_date"],
            symbol=row["symbol"],
            rule=row["rule"],
        )
        if row["candidate_id"] != expected_candidate_id:
            raise ShadowReviewContractError("shadow candidate identity is invalid")
        expected_sufficient = row["signal_status"] == "QUANT_PASS_NEEDS_CHATGPT_REVIEW"
        if row["quant_evidence_sufficient"] is not expected_sufficient:
            raise ShadowReviewContractError("shadow candidate quant evidence flag is inconsistent")
    policy = load_shadow_evidence_policy()
    if request["observation_market_date"] < prospective_start_market_date(policy).isoformat():
        raise ShadowReviewContractError("shadow request predates the frozen prospective start")
    generated = _required_timestamp(request["generated_at_utc"], "request.generated_at_utc")
    current = _aware_utc(now)
    if generated > current + timedelta(minutes=5):
        raise ShadowReviewContractError("shadow request timestamp is in the future")
    if require_same_market_date and (
        latest_completed_us_market_weekday(current).isoformat()
        != request["observation_market_date"]
    ):
        raise ShadowReviewContractError("historical shadow-review request is not allowed")
    return request


def validate_shadow_review_response(
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    now: datetime | None = None,
    response_schema_path: Path = DEFAULT_RESPONSE_SCHEMA,
) -> dict[str, Any]:
    current = _aware_utc(now)
    validate_shadow_review_request(request, now=current)
    _schema_validate(
        response,
        _load_schema(response_schema_path, "shadow response schema"),
        "shadow response",
    )
    if response["request_id"] != request["request_id"]:
        raise ShadowReviewContractError("shadow response is bound to a different request")
    generated = _required_timestamp(request["generated_at_utc"], "request.generated_at_utc")
    reviewed = _required_timestamp(response["reviewed_at_utc"], "response.reviewed_at_utc")
    if reviewed < generated or reviewed > current + timedelta(minutes=5):
        raise ShadowReviewContractError("shadow response timing is invalid")
    if reviewed - generated > MAX_REVIEW_DELAY:
        raise ShadowReviewContractError("shadow response arrived after the prospective review window")
    if latest_completed_us_market_weekday(reviewed).isoformat() != request[
        "observation_market_date"
    ]:
        raise ShadowReviewContractError("shadow response would create retrospective evidence")

    expected = [
        (row["candidate_id"], row["symbol"], row["rule"])
        for row in request["candidates"]
    ]
    actual = [
        (row["candidate_id"], row["symbol"], row["rule"])
        for row in response["assessments"]
    ]
    if actual != expected:
        raise ShadowReviewContractError(
            "every selected shadow candidate must be assessed exactly once in frozen order"
        )
    for assessment in response["assessments"]:
        check_ids = [check["check_id"] for check in assessment["checks"]]
        if check_ids != list(REQUIRED_CHECK_IDS):
            raise ShadowReviewContractError("shadow assessment check order or membership changed")
        if assessment["assessment"] == "BUY_REVIEW" and any(
            check["status"] != "PASS" for check in assessment["checks"]
        ):
            raise ShadowReviewContractError("BUY_REVIEW requires every shadow check to pass")
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate a counterfactual shadow review.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-request")
    build_parser.add_argument("--packet", type=Path, default=DEFAULT_DECISION_PACKET)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_REQUEST)
    validate_parser = subparsers.add_parser("validate-response")
    validate_parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    validate_parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-request":
            request = build_shadow_review_request(load_public_json(args.packet, "decision packet"))
            _write_json(args.output, request)
            print(json.dumps(request, ensure_ascii=False, allow_nan=False, sort_keys=True))
        else:
            request = load_public_json(args.request, "shadow review request")
            response = load_public_json(args.response, "shadow review response")
            validate_shadow_review_response(request, response)
            print("shadow review response: VALID")
    except (ShadowReviewContractError, ValueError, AssertionError) as exc:
        print(f"shadow review contract FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
