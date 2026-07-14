from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_forward_ledger import (  # noqa: E402
    DEFAULT_LEDGER as DEFAULT_SIGNAL_LEDGER,
    _write_json_atomic,
    load_ledger,
    load_public_json,
)
from scripts.build_live_review_forward_ledger import (  # noqa: E402
    DEFAULT_LEDGER as DEFAULT_LIVE_REVIEW_LEDGER,
    DEFAULT_SUMMARY as DEFAULT_LIVE_REVIEW_SUMMARY,
    LiveReviewForwardLedgerError,
    load_live_review_ledger,
    validate_live_review_forward_artifacts,
)
from scripts.model_governance import load_governance_config  # noqa: E402
from scripts.strategy_contract import (  # noqa: E402
    MIN_EFFECTIVE_SAMPLE,
    PRIMARY_EVALUATION_HORIZON,
)
from scripts.validate_model_artifacts import validate_model_artifacts  # noqa: E402
from scripts.validate_validation_split import assert_finite_json  # noqa: E402


DEFAULT_DOCS = ROOT / "docs"
DEFAULT_OUTPUT = DEFAULT_DOCS / "v6_release_status.json"
SCHEMA_VERSION = "v6-release-readiness-status-v2"
RELEASE_GATE_ORDER = (
    "model_artifacts_valid",
    "live_review_forward_artifacts_valid",
    "public_signal_minimum_sample_reached",
    "model_governance_promotion_sample_reached",
    "live_review_minimum_sample_reached",
    "ibkr_to_chatgpt_contract_evidenced",
)
HUMAN_PILOT_REQUIRED_GATES = (
    "model_artifacts_valid",
    "live_review_forward_artifacts_valid",
    "public_signal_minimum_sample_reached",
    "live_review_minimum_sample_reached",
    "ibkr_to_chatgpt_contract_evidenced",
)
CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES = (
    "model_artifacts_valid",
    "model_governance_promotion_sample_reached",
)
EVIDENCE_COUNT_FIELDS = (
    "public_signal_primary_horizon_outcomes",
    "minimum_governance_paired_samples",
    "sanitized_live_review_predictions",
    "sanitized_live_review_matured_outcomes",
)
EVIDENCE_THRESHOLD_FIELDS = (
    "public_signal_primary_horizon_outcomes",
    "governance_paired_samples_per_challenger",
    "sanitized_live_review_matured_outcomes",
    "ibkr_bound_sanitized_live_reviews",
)
COUNT_GATE_REQUIREMENTS = {
    "public_signal_minimum_sample_reached": (
        "public_signal_primary_horizon_outcomes",
        "public_signal_primary_horizon_outcomes",
    ),
    "model_governance_promotion_sample_reached": (
        "minimum_governance_paired_samples",
        "governance_paired_samples_per_challenger",
    ),
    "live_review_minimum_sample_reached": (
        "sanitized_live_review_matured_outcomes",
        "sanitized_live_review_matured_outcomes",
    ),
    "ibkr_to_chatgpt_contract_evidenced": (
        "sanitized_live_review_predictions",
        "ibkr_bound_sanitized_live_reviews",
    ),
}
BLOCKER_BY_GATE = {
    "model_artifacts_valid": "MODEL_ARTIFACT_VALIDATION_FAILED",
    "live_review_forward_artifacts_valid": "LIVE_REVIEW_ARTIFACT_VALIDATION_FAILED",
    "public_signal_minimum_sample_reached": (
        "PUBLIC_SIGNAL_PRIMARY_OUTCOMES_BELOW_MINIMUM"
    ),
    "model_governance_promotion_sample_reached": (
        "MODEL_GOVERNANCE_PAIRED_SAMPLES_BELOW_PROMOTION_THRESHOLD"
    ),
    "live_review_minimum_sample_reached": (
        "LIVE_REVIEW_MATURED_SAMPLES_BELOW_MINIMUM"
    ),
    "ibkr_to_chatgpt_contract_evidenced": (
        "IBKR_TO_CHATGPT_RUNTIME_NOT_YET_EVIDENCED"
    ),
}


class V6ReleaseAuditError(ValueError):
    """Raised when release evidence is malformed or internally inconsistent."""


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V6ReleaseAuditError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise V6ReleaseAuditError(f"{field} must be a positive integer")
    return value


def _canonical_evidence_thresholds() -> dict[str, int]:
    governance_config = load_governance_config()
    promotion_threshold = _positive_int(
        governance_config["allocation_gates"][
            "minimum_paired_samples_for_promotion_review"
        ],
        "model governance promotion threshold",
    )
    return {
        "public_signal_primary_horizon_outcomes": MIN_EFFECTIVE_SAMPLE,
        "governance_paired_samples_per_challenger": promotion_threshold,
        "sanitized_live_review_matured_outcomes": MIN_EFFECTIVE_SAMPLE,
        "ibkr_bound_sanitized_live_reviews": 1,
    }


def validate_v6_release_status(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise V6ReleaseAuditError("unsupported v6 release status schema")
    gates = payload.get("release_gates")
    evidence = payload.get("evidence_counts")
    thresholds = payload.get("evidence_thresholds")
    if (
        not isinstance(gates, dict)
        or not isinstance(evidence, dict)
        or not isinstance(thresholds, dict)
    ):
        raise V6ReleaseAuditError(
            "v6 release gates, evidence counts, and evidence thresholds are required"
        )
    if set(gates) != set(RELEASE_GATE_ORDER):
        raise V6ReleaseAuditError("v6 release gate membership changed")
    if set(evidence) != set(EVIDENCE_COUNT_FIELDS):
        raise V6ReleaseAuditError("v6 release evidence count membership changed")
    if set(thresholds) != set(EVIDENCE_THRESHOLD_FIELDS):
        raise V6ReleaseAuditError("v6 release evidence threshold membership changed")
    gate_values = [gates[name] for name in RELEASE_GATE_ORDER]
    if any(not isinstance(value, bool) for value in gate_values):
        raise V6ReleaseAuditError("v6 release gates must be booleans")
    for field in EVIDENCE_COUNT_FIELDS:
        _non_negative_int(evidence[field], f"evidence_counts.{field}")
    for field in EVIDENCE_THRESHOLD_FIELDS:
        _positive_int(thresholds[field], f"evidence_thresholds.{field}")
    if thresholds != _canonical_evidence_thresholds():
        raise V6ReleaseAuditError(
            "v6 release evidence thresholds changed from the frozen contract"
        )
    for gate, (count_field, threshold_field) in COUNT_GATE_REQUIREMENTS.items():
        expected_passed = evidence[count_field] >= thresholds[threshold_field]
        if gates[gate] is not expected_passed:
            raise V6ReleaseAuditError(
                f"v6 release gate {gate} does not match its evidence count and threshold"
            )
    expected_ready = all(gates[name] for name in HUMAN_PILOT_REQUIRED_GATES)
    if payload.get("ready_for_human_pilot_review") is not expected_ready:
        raise V6ReleaseAuditError("v6 release readiness does not match its gates")
    expected_promotion_evidence_ready = all(
        gates[name]
        for name in CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES
    )
    if (
        payload.get("challenger_model_promotion_evidence_ready")
        is not expected_promotion_evidence_ready
    ):
        raise V6ReleaseAuditError(
            "v6 challenger model promotion evidence readiness does not match its gates"
        )
    expected_status = (
        "READY_FOR_HUMAN_PILOT_REVIEW"
        if expected_ready
        else "PROSPECTIVE_VALIDATION_IN_PROGRESS"
    )
    if payload.get("status") != expected_status:
        raise V6ReleaseAuditError("v6 release status does not match its gates")
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item for item in blockers
    ):
        raise V6ReleaseAuditError("v6 release blockers must be a string list")
    expected_blockers = [
        BLOCKER_BY_GATE[name]
        for name in HUMAN_PILOT_REQUIRED_GATES
        if not gates[name]
    ]
    if blockers != expected_blockers:
        raise V6ReleaseAuditError(
            "v6 human-pilot blocker list does not match failed gates"
        )
    promotion_blockers = payload.get("challenger_model_promotion_blockers")
    if not isinstance(promotion_blockers, list) or any(
        not isinstance(item, str) or not item for item in promotion_blockers
    ):
        raise V6ReleaseAuditError(
            "v6 challenger model promotion blockers must be a string list"
        )
    expected_promotion_blockers = [
        BLOCKER_BY_GATE[name]
        for name in CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES
        if not gates[name]
    ]
    if promotion_blockers != expected_promotion_blockers:
        raise V6ReleaseAuditError(
            "v6 challenger model promotion blocker list does not match failed gates"
        )
    if payload.get("automatic_order_allowed") is not False:
        raise V6ReleaseAuditError("v6 release status must not allow automatic orders")
    if payload.get("human_confirmation_required") is not True:
        raise V6ReleaseAuditError("v6 release status must require human confirmation")
    if payload.get("contains_private_account_data") is not False:
        raise V6ReleaseAuditError("v6 release status must not contain private account data")
    assert_finite_json(payload)
    return payload


def audit_v6_release(
    *,
    docs: Path = DEFAULT_DOCS,
    signal_ledger_path: Path | None = None,
    live_review_ledger_path: Path | None = None,
    live_review_summary_path: Path | None = None,
    output_path: Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    signal_path = signal_ledger_path or docs / DEFAULT_SIGNAL_LEDGER.name
    live_path = live_review_ledger_path or docs / DEFAULT_LIVE_REVIEW_LEDGER.name
    live_summary_path = live_review_summary_path or docs / DEFAULT_LIVE_REVIEW_SUMMARY.name

    model_validation = validate_model_artifacts(docs=docs)
    live_validation = validate_live_review_forward_artifacts(
        ledger_path=live_path,
        summary_path=live_summary_path,
        report_path=docs / "market_report.json",
    )
    signal_events = load_ledger(signal_path)
    live_events = load_live_review_ledger(live_path)
    governance_report = load_public_json(docs / "model_governance.json", "model_governance")
    market_report = load_public_json(docs / "market_report.json", "market_report")
    decision_packet = load_public_json(docs / "decision_packet.json", "decision_packet")

    primary_signal_outcomes = sum(
        event["event_type"] == "OUTCOME"
        and event["outcome"]["horizon_bars"] == PRIMARY_EVALUATION_HORIZON
        for event in signal_events
    )
    challenger_metrics = governance_report.get("challenger_metrics")
    if not isinstance(challenger_metrics, dict) or not challenger_metrics:
        raise V6ReleaseAuditError("model governance has no challenger metrics")
    paired_counts = [
        _non_negative_int(value.get("paired_sample_count"), f"challenger_metrics.{model_id}")
        for model_id, value in challenger_metrics.items()
        if isinstance(value, dict)
    ]
    if len(paired_counts) != len(challenger_metrics):
        raise V6ReleaseAuditError("invalid model governance challenger metric payload")
    minimum_paired = min(paired_counts)
    live_predictions = sum(
        event["event_type"] == "LIVE_REVIEW_PREDICTION" for event in live_events
    )
    live_outcomes = sum(
        event["event_type"] == "LIVE_REVIEW_OUTCOME" for event in live_events
    )

    thresholds = _canonical_evidence_thresholds()
    gates = {
        "model_artifacts_valid": model_validation.get("status") == "VALID",
        "live_review_forward_artifacts_valid": live_validation.get("status") == "VALID",
        "public_signal_minimum_sample_reached": primary_signal_outcomes
        >= thresholds["public_signal_primary_horizon_outcomes"],
        "model_governance_promotion_sample_reached": minimum_paired
        >= thresholds["governance_paired_samples_per_challenger"],
        "live_review_minimum_sample_reached": live_outcomes
        >= thresholds["sanitized_live_review_matured_outcomes"],
        "ibkr_to_chatgpt_contract_evidenced": live_predictions
        >= thresholds["ibkr_bound_sanitized_live_reviews"],
    }
    blockers = [
        BLOCKER_BY_GATE[name]
        for name in HUMAN_PILOT_REQUIRED_GATES
        if not gates[name]
    ]
    promotion_blockers = [
        BLOCKER_BY_GATE[name]
        for name in CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES
        if not gates[name]
    ]
    ready = not blockers
    promotion_evidence_ready = not promotion_blockers

    packet_quality = decision_packet.get("data_quality")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": market_report.get("generated_at_utc"),
        "status": (
            "READY_FOR_HUMAN_PILOT_REVIEW"
            if ready
            else "PROSPECTIVE_VALIDATION_IN_PROGRESS"
        ),
        "ready_for_human_pilot_review": ready,
        "challenger_model_promotion_evidence_ready": promotion_evidence_ready,
        "release_scope": "V6_DECISION_SUPPORT_HUMAN_PILOT_NOT_AUTOMATIC_EXECUTION",
        "data_source": market_report.get("data_source"),
        "market_timezone": market_report.get("market_timezone"),
        "data_timestamp": market_report.get("data_timestamp"),
        "price_frequency": market_report.get("price_frequency"),
        "price_adjustment_policy": market_report.get("price_adjustment_policy"),
        "current_data_quality_status": (
            packet_quality.get("data_status") if isinstance(packet_quality, dict) else None
        ),
        "contract_fingerprints": {
            "strategy_fingerprint": model_validation["strategy_fingerprint"],
            "split_manifest_fingerprint": model_validation["split_manifest_fingerprint"],
            "portfolio_contract_fingerprint": model_validation[
                "portfolio_contract_fingerprint"
            ],
            "full_model_fingerprint": model_validation["full_model_fingerprint"],
            "model_governance_fingerprint": model_validation[
                "model_governance_fingerprint"
            ],
        },
        "evidence_thresholds": thresholds,
        "evidence_counts": {
            "public_signal_primary_horizon_outcomes": primary_signal_outcomes,
            "minimum_governance_paired_samples": minimum_paired,
            "sanitized_live_review_predictions": live_predictions,
            "sanitized_live_review_matured_outcomes": live_outcomes,
        },
        "evidence_scope": {
            "public_signal_count": (
                "raw prospective primary-horizon outcomes; supporting evidence only, "
                "not an independence claim"
            ),
            "model_governance_count": (
                "paired, chronologically non-overlapping incumbent/challenger cohorts; "
                "sample sufficiency alone is not a promotion approval"
            ),
            "live_review_count": (
                "sanitized outcomes with non-overlap enforced per public symbol"
            ),
            "ibkr_runtime_evidence": (
                "a sanitized prediction is created only after the private read-only IBKR "
                "context and live-review response validate; this is application-level, not "
                "broker-independent attestation"
            ),
        },
        "release_gates": gates,
        "blockers": blockers,
        "challenger_model_promotion_blockers": promotion_blockers,
        "known_limitations": [
            "SURVIVORSHIP_BIAS_REMAINS_UNCONTROLLED_FOR_THE_CURRENT_FIXED_ASSET_SET",
            "CONFIGURED_COSTS_ARE_NOT_OBSERVED_LIVE_EXECUTION_COSTS",
            "SANITIZED_RUNTIME_EVIDENCE_IS_NOT_AN_INDEPENDENT_BROKER_ATTESTATION",
            "BACKTEST_OR_PROSPECTIVE_RESULTS_DO_NOT_GUARANTEE_FUTURE_PERFORMANCE",
        ],
        "contains_private_account_data": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }
    validate_v6_release_status(payload)
    if output_path is not None:
        _write_json_atomic(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit v6 human-pilot and challenger-model promotion evidence readiness."
        )
    )
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = audit_v6_release(docs=args.docs, output_path=args.output)
    except (AssertionError, LiveReviewForwardLedgerError, V6ReleaseAuditError, ValueError) as exc:
        print(f"v6 release audit FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
