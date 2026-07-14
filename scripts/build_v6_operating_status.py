from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_v6_release import (  # noqa: E402
    BLOCKER_BY_GATE,
    CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES,
    DEFAULT_DOCS,
    HUMAN_PILOT_REQUIRED_GATES,
    RELEASE_GATE_ORDER,
    V6ReleaseAuditError,
    audit_v6_release,
    validate_v6_release_status,
)
from scripts.build_forward_ledger import _write_json_atomic, load_public_json  # noqa: E402
from scripts.build_live_review_forward_ledger import (  # noqa: E402
    DEFAULT_LEDGER as DEFAULT_LIVE_REVIEW_LEDGER,
    load_live_review_ledger,
)
from scripts.validate_validation_split import assert_finite_json  # noqa: E402


DEFAULT_OUTPUT = DEFAULT_DOCS / "v6_operating_status.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "v6_operating_status.schema.json"
SCHEMA_VERSION = "v6-operating-status-v2"
SHADOW_REQUIRED_GATES = (
    "model_artifacts_valid",
    "live_review_forward_artifacts_valid",
    "ibkr_to_chatgpt_contract_evidenced",
)
FINAL_EXECUTION_LAYER = "IBKR_MANUAL_WITH_EXPLICIT_HUMAN_CONFIRMATION"


class V6OperatingStatusError(ValueError):
    """Raised when v6 operating-mode evidence is malformed or inconsistent."""


def _schema_validate(payload: dict[str, Any], schema_path: Path) -> None:
    schema = load_public_json(schema_path, "v6 operating status schema")
    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise V6OperatingStatusError(
            f"invalid v6 operating status schema: {exc.message}"
        ) from exc
    validator = validator_class(schema, format_checker=jsonschema.FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise V6OperatingStatusError(
            f"v6 operating status schema validation failed: {details}"
        )


def _failed_gates(
    release_gates: dict[str, bool],
    required: tuple[str, ...],
) -> list[str]:
    return [name for name in required if not release_gates[name]]


def _latest_evidence_timestamp(
    release_generated_at_utc: str,
    live_events: list[dict[str, Any]],
) -> str:
    raw_values = [release_generated_at_utc] + [
        event["recorded_at_utc"] for event in live_events
    ]
    parsed: list[datetime] = []
    for raw in raw_values:
        try:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise V6OperatingStatusError("v6 evidence timestamp is invalid") from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise V6OperatingStatusError("v6 evidence timestamp must include a timezone")
        parsed.append(value)
    return max(parsed).isoformat()


def _validate_mode_gate(
    gate: Any,
    *,
    label: str,
    required: tuple[str, ...],
    release_gates: dict[str, bool],
) -> None:
    if not isinstance(gate, dict):
        raise V6OperatingStatusError(f"mode_gates.{label} must be an object")
    if gate.get("required_release_gates") != list(required):
        raise V6OperatingStatusError(f"mode_gates.{label} required gates changed")
    expected_failed = _failed_gates(release_gates, required)
    if gate.get("failed_release_gates") != expected_failed:
        raise V6OperatingStatusError(f"mode_gates.{label} failed gates are inconsistent")
    if gate.get("passed") is not (not expected_failed):
        raise V6OperatingStatusError(f"mode_gates.{label} pass state is inconsistent")


def validate_v6_operating_status(
    payload: dict[str, Any],
    *,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    _schema_validate(payload, schema_path)
    release_gates = payload["release_gates"]
    if set(release_gates) != set(RELEASE_GATE_ORDER):
        raise V6OperatingStatusError("v6 operating release gate membership changed")

    mode_gates = payload["mode_gates"]
    _validate_mode_gate(
        mode_gates["read_only_shadow"],
        label="read_only_shadow",
        required=SHADOW_REQUIRED_GATES,
        release_gates=release_gates,
    )
    _validate_mode_gate(
        mode_gates["human_pilot_review"],
        label="human_pilot_review",
        required=HUMAN_PILOT_REQUIRED_GATES,
        release_gates=release_gates,
    )
    _validate_mode_gate(
        mode_gates["challenger_model_promotion_evidence"],
        label="challenger_model_promotion_evidence",
        required=CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES,
        release_gates=release_gates,
    )
    automatic = mode_gates["automatic_execution"]
    if automatic != {
        "passed": False,
        "reason_code": "AUTOMATIC_EXECUTION_PERMANENTLY_PROHIBITED",
    }:
        raise V6OperatingStatusError("automatic execution gate must remain permanently closed")

    shadow_ready = mode_gates["read_only_shadow"]["passed"]
    pilot_ready = mode_gates["human_pilot_review"]["passed"]
    promotion_evidence_ready = mode_gates[
        "challenger_model_promotion_evidence"
    ]["passed"]
    expected_mode = (
        "HUMAN_PILOT_REVIEW_READY"
        if pilot_ready
        else "READ_ONLY_SHADOW"
        if shadow_ready
        else "UNAVAILABLE"
    )
    expected_next = (
        None
        if pilot_ready
        else "HUMAN_PILOT_REVIEW_READY"
        if shadow_ready
        else "READ_ONLY_SHADOW"
    )
    if payload["operating_mode"] != expected_mode:
        raise V6OperatingStatusError("v6 operating mode does not match its gates")
    if payload["next_operating_mode"] != expected_next:
        raise V6OperatingStatusError("v6 next operating mode does not match its gates")

    capabilities = payload["capabilities"]
    expected_capabilities = {
        "github_public_evidence_available": release_gates["model_artifacts_valid"],
        "chatgpt_live_review_available": (
            release_gates["live_review_forward_artifacts_valid"]
            and release_gates["ibkr_to_chatgpt_contract_evidenced"]
        ),
        "ibkr_readonly_context_evidenced": release_gates[
            "ibkr_to_chatgpt_contract_evidenced"
        ],
        "human_decision_support_available": shadow_ready,
        "human_pilot_review_ready": pilot_ready,
        "challenger_model_promotion_evidence_ready": promotion_evidence_ready,
        "automatic_order_allowed": False,
    }
    if capabilities != expected_capabilities:
        raise V6OperatingStatusError("v6 operating capabilities do not match release evidence")
    if payload["source_release_status"] != (
        "READY_FOR_HUMAN_PILOT_REVIEW"
        if pilot_ready
        else "PROSPECTIVE_VALIDATION_IN_PROGRESS"
    ):
        raise V6OperatingStatusError("source release status is inconsistent")
    expected_release_blockers = [
        BLOCKER_BY_GATE[name]
        for name in HUMAN_PILOT_REQUIRED_GATES
        if not release_gates[name]
    ]
    if payload["release_blockers"] != expected_release_blockers:
        raise V6OperatingStatusError("human-pilot release blockers are inconsistent")
    expected_promotion_blockers = [
        BLOCKER_BY_GATE[name]
        for name in CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES
        if not release_gates[name]
    ]
    if (
        payload["challenger_model_promotion_blockers"]
        != expected_promotion_blockers
    ):
        raise V6OperatingStatusError(
            "challenger model promotion blockers are inconsistent"
        )
    if payload["automatic_order_allowed"] is not False:
        raise V6OperatingStatusError("v6 operating status must prohibit automatic orders")
    if payload["human_confirmation_required"] is not True:
        raise V6OperatingStatusError("v6 operating status must require human confirmation")
    if payload["contains_private_account_data"] is not False:
        raise V6OperatingStatusError("v6 operating status must not contain private account data")
    assert_finite_json(payload)
    return payload


def derive_v6_operating_status(
    release: dict[str, Any],
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    validate_v6_release_status(release)
    release_gates = release["release_gates"]
    if set(release_gates) != set(RELEASE_GATE_ORDER):
        raise V6OperatingStatusError("unsupported v6 release gate generation")
    release_gates = {name: release_gates[name] for name in RELEASE_GATE_ORDER}
    for field in (
        "generated_at_utc",
        "data_source",
        "market_timezone",
        "data_timestamp",
        "price_frequency",
        "price_adjustment_policy",
        "current_data_quality_status",
    ):
        if not isinstance(release.get(field), str) or not release[field]:
            raise V6OperatingStatusError(f"release status {field} is unavailable")

    shadow_failed = _failed_gates(release_gates, SHADOW_REQUIRED_GATES)
    human_failed = _failed_gates(release_gates, HUMAN_PILOT_REQUIRED_GATES)
    promotion_failed = _failed_gates(
        release_gates,
        CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES,
    )
    shadow_ready = not shadow_failed
    pilot_ready = not human_failed
    promotion_evidence_ready = not promotion_failed
    operating_mode = (
        "HUMAN_PILOT_REVIEW_READY"
        if pilot_ready
        else "READ_ONLY_SHADOW"
        if shadow_ready
        else "UNAVAILABLE"
    )
    next_mode = (
        None
        if pilot_ready
        else "HUMAN_PILOT_REVIEW_READY"
        if shadow_ready
        else "READ_ONLY_SHADOW"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or release["generated_at_utc"],
        "operating_mode": operating_mode,
        "next_operating_mode": next_mode,
        "source_release_status": release["status"],
        "data_source": release["data_source"],
        "market_timezone": release["market_timezone"],
        "data_timestamp": release["data_timestamp"],
        "price_frequency": release["price_frequency"],
        "price_adjustment_policy": release["price_adjustment_policy"],
        "current_data_quality_status": release["current_data_quality_status"],
        "capabilities": {
            "github_public_evidence_available": release_gates["model_artifacts_valid"],
            "chatgpt_live_review_available": (
                release_gates["live_review_forward_artifacts_valid"]
                and release_gates["ibkr_to_chatgpt_contract_evidenced"]
            ),
            "ibkr_readonly_context_evidenced": release_gates[
                "ibkr_to_chatgpt_contract_evidenced"
            ],
            "human_decision_support_available": shadow_ready,
            "human_pilot_review_ready": pilot_ready,
            "challenger_model_promotion_evidence_ready": (
                promotion_evidence_ready
            ),
            "automatic_order_allowed": False,
        },
        "mode_gates": {
            "read_only_shadow": {
                "passed": shadow_ready,
                "required_release_gates": list(SHADOW_REQUIRED_GATES),
                "failed_release_gates": shadow_failed,
            },
            "human_pilot_review": {
                "passed": pilot_ready,
                "required_release_gates": list(HUMAN_PILOT_REQUIRED_GATES),
                "failed_release_gates": human_failed,
            },
            "challenger_model_promotion_evidence": {
                "passed": promotion_evidence_ready,
                "required_release_gates": list(
                    CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES
                ),
                "failed_release_gates": promotion_failed,
            },
            "automatic_execution": {
                "passed": False,
                "reason_code": "AUTOMATIC_EXECUTION_PERMANENTLY_PROHIBITED",
            },
        },
        "release_gates": dict(release_gates),
        "release_blockers": list(release["blockers"]),
        "challenger_model_promotion_blockers": list(
            release["challenger_model_promotion_blockers"]
        ),
        "evidence_thresholds": dict(release["evidence_thresholds"]),
        "evidence_counts": dict(release["evidence_counts"]),
        "contract_fingerprints": dict(release["contract_fingerprints"]),
        "final_execution_layer": FINAL_EXECUTION_LAYER,
        "important_limit": (
            "READ_ONLY_SHADOW is decision support, not validated pilot performance or an order. "
            "Human-pilot readiness and challenger-model promotion evidence are separate; "
            "neither authorizes model replacement or automatic execution."
        ),
        "contains_private_account_data": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }
    return validate_v6_operating_status(payload)


def build_v6_operating_status(
    *,
    docs: Path = DEFAULT_DOCS,
    output_path: Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    current_release = load_public_json(docs / "v6_release_status.json", "v6 release status")
    validate_v6_release_status(current_release)
    expected_release = audit_v6_release(docs=docs, output_path=None)
    if current_release != expected_release:
        raise V6OperatingStatusError(
            "committed v6 release status is stale relative to current evidence"
        )
    live_events = load_live_review_ledger(docs / DEFAULT_LIVE_REVIEW_LEDGER.name)
    payload = derive_v6_operating_status(
        current_release,
        generated_at_utc=_latest_evidence_timestamp(
            current_release["generated_at_utc"],
            live_events,
        ),
    )
    if output_path is not None:
        _write_json_atomic(output_path, payload)
    return payload


def validate_v6_operating_status_file(
    *,
    docs: Path = DEFAULT_DOCS,
    path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    actual = load_public_json(path, "v6 operating status")
    validate_v6_operating_status(actual)
    expected = build_v6_operating_status(docs=docs, output_path=None)
    if actual != expected:
        raise V6OperatingStatusError(
            "committed v6 operating status is stale relative to current release evidence"
        )
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the v6 operating-mode readiness artifact."
    )
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            payload = validate_v6_operating_status_file(
                docs=args.docs,
                path=args.output,
            )
        else:
            payload = build_v6_operating_status(
                docs=args.docs,
                output_path=args.output,
            )
    except (AssertionError, V6OperatingStatusError, V6ReleaseAuditError, ValueError) as exc:
        print(f"v6 operating status FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
