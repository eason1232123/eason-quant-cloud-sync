from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jsonschema

from scripts.build_local_ibkr_context import (
    CONTEXT_SCHEMA_VERSION,
    DEFAULT_PRIVATE_CONTEXT,
    validate_private_context,
)
from scripts.ibkr_readonly import IbkrReadonlyConfig, IbkrReadonlyError, ROOT, validate_config
from scripts.market_clock import latest_completed_us_market_weekday
from scripts.validate_decision_packet import (
    DEFAULT_SCHEMA as DEFAULT_DECISION_SCHEMA,
    FORMAT_CHECKER,
    validate_invariants,
    validate_schema as validate_decision_schema,
)
from scripts.validate_generated_json import load_strict_json
from scripts.validate_model_artifacts import validate_model_artifacts


REQUEST_SCHEMA_VERSION = "v6-private-live-review-request-v1"
RESPONSE_SCHEMA_VERSION = "v6-private-live-review-response-v1"
DEFAULT_REQUEST_SCHEMA = ROOT / "schemas" / "live_review_request.schema.json"
DEFAULT_RESPONSE_SCHEMA = ROOT / "schemas" / "live_review_response.schema.json"
DEFAULT_DECISION_PACKET = ROOT / "docs" / "decision_packet.json"
DEFAULT_MODEL_GOVERNANCE = ROOT / "docs" / "model_governance.json"
DEFAULT_REQUEST = ROOT / "private" / "ibkr" / "live_review_request.json"
DEFAULT_RESPONSE = ROOT / "private" / "ibkr" / "live_review_response.json"
REQUIRED_CHECK_IDS = (
    "live_quote",
    "market_regime",
    "fresh_news_macro",
    "earnings_valuation",
    "actual_account_risk",
    "execution_feasibility",
)
REALTIME_QUOTE_KINDS = {
    "INTRADAY_REALTIME_BID_ASK",
    "INTRADAY_REALTIME_LAST",
}
ACTION_CEILINGS = {
    "DATA_REVIEW_REQUIRED": ("NO_TRADE", "WAIT"),
    "NO_TRADE": ("NO_TRADE", "WAIT"),
    "RISK_REVIEW_REQUIRED": ("NO_TRADE", "WAIT", "REDUCE_REVIEW"),
    "BUY_CANDIDATE_REVIEW_REQUIRED": (
        "NO_TRADE",
        "WAIT",
        "REDUCE_REVIEW",
        "BUY_REVIEW",
    ),
}
MAXIMUM_LIVE_ACTION = {
    "DATA_REVIEW_REQUIRED": "WAIT_ONLY",
    "NO_TRADE": "WAIT_ONLY",
    "RISK_REVIEW_REQUIRED": "REDUCE_REVIEW_ONLY",
    "BUY_CANDIDATE_REVIEW_REQUIRED": "BUY_REVIEW_ONLY",
}


class LiveReviewError(IbkrReadonlyError):
    """Raised when a private live review cannot be trusted or traced."""


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise LiveReviewError(f"{field} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LiveReviewError(f"{field} must be a valid timestamp") from exc
    if parsed.utcoffset() is None:
        raise LiveReviewError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _aware_now(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.utcoffset() is None:
        raise LiveReviewError("validation clock must include a timezone")
    return result.astimezone(timezone.utc)


def _positive_seconds(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveReviewError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise LiveReviewError(f"{field} must be positive and finite")
    return parsed


def _schema_validate(payload: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise LiveReviewError(f"invalid {label} schema: {exc.message}") from exc
    validator = validator_class(schema, format_checker=FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:10]
        )
        raise LiveReviewError(f"{label} schema validation failed: {details}")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        return load_strict_json(path)
    except AssertionError as exc:
        raise LiveReviewError(f"invalid {label}: {exc}") from exc


def _validate_current_public_evidence(decision: dict[str, Any]) -> None:
    decision_schema = _load_object(DEFAULT_DECISION_SCHEMA, "decision packet schema")
    validate_decision_schema(decision, decision_schema)
    validate_invariants(decision)
    try:
        validate_model_artifacts()
    except AssertionError as exc:
        raise LiveReviewError(f"current GitHub model artifacts are invalid: {exc}") from exc


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_private_path(path: Path) -> Path:
    try:
        validate_config(IbkrReadonlyConfig(private_snapshot_path=path))
    except IbkrReadonlyError as exc:
        raise LiveReviewError(str(exc)) from exc
    return path


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    _assert_private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _held_symbols(context: dict[str, Any]) -> list[str]:
    symbols: set[str] = set()
    for account in context.get("private_accounts", []):
        for currency_slice in account.get("currency_slices", []):
            for position in currency_slice.get("positions", []):
                symbol = position.get("symbol")
                quantity = position.get("quantity")
                if (
                    isinstance(symbol, str)
                    and symbol
                    and isinstance(quantity, (int, float))
                    and not isinstance(quantity, bool)
                    and math.isfinite(quantity)
                    and quantity != 0
                ):
                    symbols.add(symbol)
    return sorted(symbols)


def _candidate_symbols(decision_packet: dict[str, Any]) -> list[str]:
    candidates = decision_packet.get("candidates", {})
    execution = candidates.get("execution", {}) if isinstance(candidates, dict) else {}
    rows = execution.get("top", []) if isinstance(execution, dict) else []
    if not isinstance(rows, list):
        raise LiveReviewError("decision packet execution candidates must be a list")
    symbols: set[str] = set()
    for row in rows:
        symbol = row.get("ticker") if isinstance(row, dict) else None
        if not isinstance(symbol, str) or not symbol:
            raise LiveReviewError("decision packet candidate ticker is invalid")
        symbols.add(symbol)
    return sorted(symbols)


def _verify_source_alignment(
    context: dict[str, Any],
    decision_packet: dict[str, Any],
    model_governance: dict[str, Any],
) -> None:
    if context.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise LiveReviewError("private context schema version is unsupported")
    if context.get("public_decision") != decision_packet.get("decision"):
        raise LiveReviewError("private context decision does not match current GitHub evidence")
    github_source = context.get("sources", {}).get("github_strategy_evidence", {})
    market_data = decision_packet.get("market_data", {})
    comparisons = {
        "source": market_data.get("source"),
        "market_timezone": market_data.get("market_timezone"),
        "report_generated_at_utc": market_data.get("report_generated_at_utc"),
        "data_timestamp": market_data.get("data_timestamp"),
        "price_frequency": market_data.get("price_frequency"),
        "price_adjustment_policy": market_data.get("price_adjustment_policy"),
        "decision_schema_version": decision_packet.get("schema_version"),
        "model_governance_fingerprint": model_governance.get(
            "model_governance_fingerprint"
        ),
    }
    if any(github_source.get(key) != value for key, value in comparisons.items()):
        raise LiveReviewError("private context GitHub lineage does not match current artifacts")
    expected_governance = {
        "status": model_governance.get("status"),
        "current_market_regime": model_governance.get("current_market_regime"),
        "allocation_decision": model_governance.get("allocation_decision"),
        "effective_target_weights": model_governance.get("effective_target_weights"),
    }
    if context.get("public_model_governance") != expected_governance:
        raise LiveReviewError("private context governance content does not match current artifacts")
    if decision_packet.get("decision", {}).get("automatic_order_allowed") is not False:
        raise LiveReviewError("decision packet must prohibit automatic orders")
    if model_governance.get("automatic_order_allowed") is not False:
        raise LiveReviewError("model governance must prohibit automatic orders")


def _verify_context_freshness(
    context: dict[str, Any],
    *,
    now: datetime,
    max_account_age_seconds: float,
    max_context_age_seconds: float,
) -> None:
    account_age_limit = _positive_seconds(max_account_age_seconds, "max_account_age_seconds")
    context_age_limit = _positive_seconds(max_context_age_seconds, "max_context_age_seconds")
    context_time = _parse_timestamp(context.get("generated_at_utc"), "context.generated_at_utc")
    account_time = _parse_timestamp(
        context.get("sources", {}).get("ibkr", {}).get("data_timestamp"),
        "context.sources.ibkr.data_timestamp",
    )
    for label, observed, limit in (
        ("private context", context_time, context_age_limit),
        ("IBKR account snapshot", account_time, account_age_limit),
    ):
        age = (now - observed).total_seconds()
        if age < -5:
            raise LiveReviewError(f"{label} timestamp is in the future")
        if age > limit:
            raise LiveReviewError(f"{label} is stale")


def _verify_public_market_freshness(
    decision_packet: dict[str, Any],
    *,
    now: datetime,
) -> None:
    observed = decision_packet.get("market_data", {}).get("data_timestamp")
    expected = latest_completed_us_market_weekday(now).isoformat()
    if observed != expected:
        raise LiveReviewError(
            f"public GitHub EOD evidence is stale or future: expected {expected}, observed {observed}"
        )


def _request_identity_payload(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": request.get("schema_version"),
        "source_context": request.get("source_context"),
        "quant_boundary": request.get("quant_boundary"),
        "candidate_symbols": request.get("candidate_symbols"),
        "held_symbols": request.get("held_symbols"),
        "required_check_ids": request.get("required_check_ids"),
        "validation_policy": request.get("validation_policy"),
    }


def _allowed_live_conclusions(
    quant_action: str,
    *,
    held_symbols: list[str],
    candidate_symbols: list[str],
) -> list[str]:
    if quant_action not in ACTION_CEILINGS:
        raise LiveReviewError(f"unsupported quant final action: {quant_action}")
    allowed = list(ACTION_CEILINGS[quant_action])
    if not held_symbols and "REDUCE_REVIEW" in allowed:
        allowed.remove("REDUCE_REVIEW")
    if not candidate_symbols and "BUY_REVIEW" in allowed:
        allowed.remove("BUY_REVIEW")
    return allowed


def build_live_review_request(
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
    now = _aware_now(generated_at)
    _verify_context_freshness(
        context,
        now=now,
        max_account_age_seconds=max_account_age_seconds,
        max_context_age_seconds=max_context_age_seconds,
    )
    _verify_source_alignment(context, decision_packet, model_governance)
    _verify_public_market_freshness(decision_packet, now=now)

    ttl = _positive_seconds(request_ttl_seconds, "request_ttl_seconds")
    response_age = _positive_seconds(response_max_age_seconds, "response_max_age_seconds")
    action = decision_packet.get("decision", {}).get("final_action")
    held = _held_symbols(context)
    candidates = _candidate_symbols(decision_packet)
    allowed = _allowed_live_conclusions(
        action,
        held_symbols=held,
        candidate_symbols=candidates,
    )

    ibkr_source = context["sources"]["ibkr"]
    github_source = context["sources"]["github_strategy_evidence"]
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": "AWAITING_CHATGPT_LIVE_REVIEW",
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
        "source_context": {
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
        },
        "quant_boundary": {
            "quant_final_action": action,
            "quant_buy_permission": decision_packet["decision"].get("buy_permission"),
            "maximum_live_action": MAXIMUM_LIVE_ACTION[action],
            "allowed_live_conclusions": allowed,
            "monotonic_safety_rule": (
                "ChatGPT may preserve or tighten the quant decision; it may not create a buy "
                "review unless GitHub supplied a current eligible candidate."
            ),
        },
        "candidate_symbols": candidates,
        "held_symbols": held,
        "required_check_ids": list(REQUIRED_CHECK_IDS),
        "validation_policy": {
            "max_account_snapshot_age_seconds": float(max_account_age_seconds),
            "max_context_age_seconds": float(max_context_age_seconds),
            "request_ttl_seconds": ttl,
            "response_max_age_seconds": response_age,
            "buy_review_requires_all_checks_pass": True,
            "buy_review_requires_realtime_quote": True,
            "missing_or_failed_check_action": "WAIT",
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "order_payload": None,
    }
    request["request_id"] = _canonical_hash(_request_identity_payload(request))
    validate_live_review_request(request, now=now, require_unexpired=True)
    return request


def validate_live_review_request(
    request: dict[str, Any],
    *,
    now: datetime | None = None,
    require_unexpired: bool = True,
    schema_path: Path = DEFAULT_REQUEST_SCHEMA,
) -> dict[str, Any]:
    schema = _load_object(schema_path, "live review request schema")
    _schema_validate(request, schema, "live review request")
    if request.get("request_id") != _canonical_hash(_request_identity_payload(request)):
        raise LiveReviewError("live review request fingerprint mismatch")
    if request.get("required_check_ids") != list(REQUIRED_CHECK_IDS):
        raise LiveReviewError("live review request check order or membership changed")
    if request.get("required_input_files") != [
        "private/ibkr/chatgpt_account_context.json",
        "docs/decision_packet.json",
        "docs/model_governance.json",
    ]:
        raise LiveReviewError("live review request input-file contract changed")
    if request.get("source_context", {}).get("context_schema_version") != CONTEXT_SCHEMA_VERSION:
        raise LiveReviewError("live review request private-context version is unsupported")
    boundary = request.get("quant_boundary", {})
    action = boundary.get("quant_final_action")
    expected_allowed = _allowed_live_conclusions(
        action,
        held_symbols=request.get("held_symbols", []),
        candidate_symbols=request.get("candidate_symbols", []),
    )
    if boundary.get("allowed_live_conclusions") != expected_allowed:
        raise LiveReviewError("live review request quant boundary is inconsistent")
    if boundary.get("maximum_live_action") != MAXIMUM_LIVE_ACTION[action]:
        raise LiveReviewError("live review request maximum action is inconsistent")
    if request.get("privacy", {}).get("contains_private_position_symbols") != bool(
        request.get("held_symbols")
    ):
        raise LiveReviewError("live review request privacy metadata is inconsistent")
    if request.get("automatic_order_allowed") is not False or request.get("order_payload") is not None:
        raise LiveReviewError("live review request must not contain an executable order")
    if request.get("human_confirmation_required") is not True:
        raise LiveReviewError("live review request must require human confirmation")
    generated = _parse_timestamp(request.get("generated_at_utc"), "request.generated_at_utc")
    expires = _parse_timestamp(request.get("expires_at_utc"), "request.expires_at_utc")
    policy = request.get("validation_policy", {})
    context_time = _parse_timestamp(
        request.get("source_context", {}).get("context_generated_at_utc"),
        "request.source_context.context_generated_at_utc",
    )
    account_time = _parse_timestamp(
        request.get("source_context", {}).get("ibkr", {}).get("data_timestamp"),
        "request.source_context.ibkr.data_timestamp",
    )
    for label, observed, limit in (
        (
            "embedded private context",
            context_time,
            _positive_seconds(
                policy.get("max_context_age_seconds"),
                "request.validation_policy.max_context_age_seconds",
            ),
        ),
        (
            "embedded IBKR account snapshot",
            account_time,
            _positive_seconds(
                policy.get("max_account_snapshot_age_seconds"),
                "request.validation_policy.max_account_snapshot_age_seconds",
            ),
        ),
    ):
        age = (generated - observed).total_seconds()
        if age < -5:
            raise LiveReviewError(f"{label} timestamp is in the future")
        if age > limit:
            raise LiveReviewError(f"{label} is stale")
    observed_market_date = request.get("source_context", {}).get("github", {}).get(
        "data_timestamp"
    )
    expected_market_date = latest_completed_us_market_weekday(generated).isoformat()
    if observed_market_date != expected_market_date:
        raise LiveReviewError(
            "embedded public GitHub EOD evidence is stale or future: "
            f"expected {expected_market_date}, observed {observed_market_date}"
        )
    ttl = _positive_seconds(
        policy.get("request_ttl_seconds"),
        "request.validation_policy.request_ttl_seconds",
    )
    if abs((expires - generated).total_seconds() - ttl) > 0.001:
        raise LiveReviewError("live review request expiry does not match its frozen TTL")
    reference = _aware_now(now)
    if generated > reference + timedelta(seconds=5):
        raise LiveReviewError("live review request timestamp is in the future")
    if require_unexpired and reference > expires:
        raise LiveReviewError("live review request has expired")
    return request


def validate_live_review_response(
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    now: datetime | None = None,
    request_schema_path: Path = DEFAULT_REQUEST_SCHEMA,
    response_schema_path: Path = DEFAULT_RESPONSE_SCHEMA,
) -> dict[str, Any]:
    reference = _aware_now(now)
    validate_live_review_request(
        request,
        now=reference,
        require_unexpired=True,
        schema_path=request_schema_path,
    )
    response_schema = _load_object(response_schema_path, "live review response schema")
    _schema_validate(response, response_schema, "live review response")
    if response.get("request_id") != request["request_id"]:
        raise LiveReviewError("live review response request_id mismatch")
    if response.get("input_context_sha256") != request["source_context"]["context_sha256"]:
        raise LiveReviewError("live review response context fingerprint mismatch")
    if response.get("automatic_order_allowed") is not False or response.get("order_payload") is not None:
        raise LiveReviewError("live review response must not contain an executable order")
    if response.get("human_confirmation_required") is not True:
        raise LiveReviewError("live review response must require human confirmation")

    generated = _parse_timestamp(request["generated_at_utc"], "request.generated_at_utc")
    expires = _parse_timestamp(request["expires_at_utc"], "request.expires_at_utc")
    reviewed = _parse_timestamp(response.get("reviewed_at_utc"), "response.reviewed_at_utc")
    if reviewed < generated or reviewed > expires:
        raise LiveReviewError("live review response was not completed within the request window")
    if reviewed > reference + timedelta(seconds=5):
        raise LiveReviewError("live review response timestamp is in the future")
    max_response_age = _positive_seconds(
        request["validation_policy"]["response_max_age_seconds"],
        "request.validation_policy.response_max_age_seconds",
    )
    if (reference - reviewed).total_seconds() > max_response_age:
        raise LiveReviewError("live review response is stale")

    checks = response["checks"]
    check_by_id = {item["check_id"]: item for item in checks}
    if len(check_by_id) != len(checks) or set(check_by_id) != set(REQUIRED_CHECK_IDS):
        raise LiveReviewError("live review response must contain each required check exactly once")
    observed_by_check: dict[str, list[tuple[dict[str, Any], datetime]]] = {}
    for check in checks:
        if check["status"] == "PASS" and not check["evidence_sources"]:
            raise LiveReviewError(f"passed check {check['check_id']} has no evidence source")
        observed_by_check[check["check_id"]] = []
        for source in check["evidence_sources"]:
            observed = _parse_timestamp(
                source["observed_at_utc"],
                f"response.checks.{check['check_id']}.source.observed_at_utc",
            )
            if observed > reviewed + timedelta(seconds=5):
                raise LiveReviewError("live review evidence timestamp is after the review")
            if source["source_type"].startswith("PUBLIC_") and source["source_url"] is None:
                raise LiveReviewError("public live-review evidence must include a source URL")
            observed_by_check[check["check_id"]].append((source, observed))

    expected_kinds = {
        "live_quote": {
            "INTRADAY_REALTIME_BID_ASK",
            "INTRADAY_REALTIME_LAST",
            "INTRADAY_DELAYED",
        },
        "market_regime": {
            "INTRADAY_REALTIME_BID_ASK",
            "INTRADAY_REALTIME_LAST",
            "INTRADAY_DELAYED",
            "END_OF_DAY_ADJUSTED",
            "END_OF_DAY_UNADJUSTED",
            "GITHUB_QUANT_EVIDENCE",
        },
        "fresh_news_macro": {"NEWS_OR_MACRO"},
        "earnings_valuation": {"FUNDAMENTAL_OR_VALUATION"},
        "actual_account_risk": {"PRIVATE_ACCOUNT_SNAPSHOT"},
        "execution_feasibility": {
            "INTRADAY_REALTIME_BID_ASK",
            "INTRADAY_DELAYED",
        },
    }
    for check_id, check in check_by_id.items():
        if check["status"] == "PASS" and not any(
            source["data_kind"] in expected_kinds[check_id]
            for source, _ in observed_by_check[check_id]
        ):
            raise LiveReviewError(f"passed check {check_id} has no compatible evidence type")
    for check_id in ("live_quote", "execution_feasibility"):
        if check_by_id[check_id]["status"] == "PASS" and not any(
            observed >= generated for _, observed in observed_by_check[check_id]
        ):
            raise LiveReviewError(f"passed check {check_id} has no evidence from this request window")
    if check_by_id["actual_account_risk"]["status"] == "PASS":
        account_timestamp = _parse_timestamp(
            request["source_context"]["ibkr"]["data_timestamp"],
            "request.source_context.ibkr.data_timestamp",
        )
        if not any(
            source["data_kind"] == "PRIVATE_ACCOUNT_SNAPSHOT"
            and observed == account_timestamp
            for source, observed in observed_by_check["actual_account_risk"]
        ):
            raise LiveReviewError("actual_account_risk does not bind to the requested IBKR snapshot")

    conclusion = response["conclusion"]
    action = conclusion["action"]
    symbol = conclusion["symbol"]
    allowed = request["quant_boundary"]["allowed_live_conclusions"]
    if action not in allowed:
        raise LiveReviewError("live review conclusion exceeds the frozen quant boundary")
    if any(check["status"] != "PASS" for check in checks) and action not in {"NO_TRADE", "WAIT"}:
        raise LiveReviewError("failed or unavailable checks require NO_TRADE or WAIT")
    if action in {"BUY_REVIEW", "REDUCE_REVIEW"}:
        live_quote_sources = observed_by_check["live_quote"]
        execution_sources = observed_by_check["execution_feasibility"]
        if not any(
            source["data_kind"] in REALTIME_QUOTE_KINDS and observed >= generated
            for source, observed in live_quote_sources
        ):
            raise LiveReviewError(f"{action} requires current real-time quote evidence")
        if not any(
            source["data_kind"] == "INTRADAY_REALTIME_BID_ASK" and observed >= generated
            for source, observed in execution_sources
        ):
            raise LiveReviewError(f"{action} requires current real-time bid/ask evidence")
    if action == "BUY_REVIEW":
        if symbol not in request["candidate_symbols"]:
            raise LiveReviewError("BUY_REVIEW symbol is not a current GitHub candidate")
    elif action == "REDUCE_REVIEW":
        if symbol not in request["held_symbols"]:
            raise LiveReviewError("REDUCE_REVIEW symbol is not present in the private holdings context")
    elif symbol is not None:
        raise LiveReviewError("NO_TRADE and WAIT conclusions must not select a symbol")
    return response


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
    _validate_current_public_evidence(decision)
    request = build_live_review_request(
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
) -> dict[str, Any]:
    _assert_private_path(request_path)
    _assert_private_path(response_path)
    request = _load_object(request_path, "live review request")
    response = _load_object(response_path, "live review response")
    context = _load_object(context_path, "private ChatGPT account context")
    decision = _load_object(decision_path, "decision packet")
    governance = _load_object(governance_path, "model governance")
    _validate_current_public_evidence(decision)
    reference = datetime.now(timezone.utc)
    policy = request.get("validation_policy", {})
    _verify_context_freshness(
        context,
        now=reference,
        max_account_age_seconds=policy.get("max_account_snapshot_age_seconds"),
        max_context_age_seconds=policy.get("max_context_age_seconds"),
    )
    expected_request = build_live_review_request(
        context,
        decision,
        governance,
        generated_at=_parse_timestamp(request.get("generated_at_utc"), "request.generated_at_utc"),
        max_account_age_seconds=policy.get("max_account_snapshot_age_seconds"),
        max_context_age_seconds=policy.get("max_context_age_seconds"),
        request_ttl_seconds=policy.get("request_ttl_seconds"),
        response_max_age_seconds=policy.get("response_max_age_seconds"),
    )
    if request != expected_request:
        raise LiveReviewError("live review request no longer matches current private/public evidence")
    return validate_live_review_response(request, response)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate private ChatGPT live-review contracts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-request")
    build_parser.add_argument("--max-account-age-seconds", type=float, default=300.0)
    build_parser.add_argument("--max-context-age-seconds", type=float, default=300.0)
    subparsers.add_parser("validate-response")
    args = parser.parse_args()
    try:
        if args.command == "build-request":
            result = build_request_from_files(
                max_account_age_seconds=args.max_account_age_seconds,
                max_context_age_seconds=args.max_context_age_seconds,
            )
            output = {
                "status": result["status"],
                "request_id": result["request_id"],
                "expires_at_utc": result["expires_at_utc"],
                "private_output": str(DEFAULT_REQUEST.relative_to(ROOT)),
                "automatic_order_allowed": False,
            }
        else:
            result = validate_response_from_files()
            output = {
                "status": result["status"],
                "request_id": result["request_id"],
                "conclusion": result["conclusion"]["action"],
                "human_confirmation_required": True,
                "automatic_order_allowed": False,
            }
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (LiveReviewError, KeyError, TypeError, AssertionError) as exc:
        print(
            json.dumps(
                {
                    "status": "PRIVATE_LIVE_REVIEW_UNAVAILABLE",
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
