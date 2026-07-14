from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "config" / "shadow_evidence_policy.json"
POLICY_SCHEMA_VERSION = "shadow-evidence-policy-v1"
ALLOWED_ASSESSMENTS = ("BUY_REVIEW", "WAIT", "REJECT", "NO_TRADE")


class ShadowEvidencePolicyError(ValueError):
    """Raised when the frozen shadow-evidence policy is malformed."""


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ShadowEvidencePolicyError(f"non-finite JSON value is not allowed: {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowEvidencePolicyError(f"cannot load shadow evidence policy: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShadowEvidencePolicyError("shadow evidence policy must be a JSON object")
    return payload


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShadowEvidencePolicyError(f"{field} must be a positive integer")
    return value


def load_shadow_evidence_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    payload = _strict_json(path)
    expected_top = {
        "schema_version",
        "frozen_at_utc",
        "prospective_start_market_date",
        "allow_historical_backfill",
        "selection",
        "review",
        "evaluation",
        "execution_boundary",
    }
    if set(payload) != expected_top:
        raise ShadowEvidencePolicyError("shadow evidence policy fields changed")
    if payload["schema_version"] != POLICY_SCHEMA_VERSION:
        raise ShadowEvidencePolicyError("unsupported shadow evidence policy version")
    try:
        frozen = datetime.fromisoformat(payload["frozen_at_utc"].replace("Z", "+00:00"))
        start = date.fromisoformat(payload["prospective_start_market_date"])
    except (AttributeError, ValueError) as exc:
        raise ShadowEvidencePolicyError("shadow evidence policy dates are invalid") from exc
    if frozen.utcoffset() is None or start <= frozen.date():
        raise ShadowEvidencePolicyError(
            "prospective start must be strictly after the timezone-aware policy freeze date"
        )
    if payload["allow_historical_backfill"] is not False:
        raise ShadowEvidencePolicyError("historical shadow-evidence backfill must remain disabled")

    selection = payload["selection"]
    review = payload["review"]
    evaluation = payload["evaluation"]
    boundary = payload["execution_boundary"]
    if not all(isinstance(value, dict) for value in (selection, review, evaluation, boundary)):
        raise ShadowEvidencePolicyError("shadow evidence policy sections must be objects")
    if set(selection) != {
        "maximum_candidates_per_market_date",
        "required_signal_state",
        "require_fresh_market_data",
        "ranking_policy",
    }:
        raise ShadowEvidencePolicyError("shadow selection contract changed")
    maximum = _positive_int(
        selection["maximum_candidates_per_market_date"],
        "selection.maximum_candidates_per_market_date",
    )
    if maximum > 3:
        raise ShadowEvidencePolicyError("shadow selection may not exceed three daily candidates")
    if (
        selection["required_signal_state"] != "ACTIVE"
        or selection["require_fresh_market_data"] is not True
        or selection["ranking_policy"]
        != "QUANT_PASS_FIRST_THEN_EVIDENCE_SCORE_SAMPLES_TICKER_RULE_V1"
    ):
        raise ShadowEvidencePolicyError("shadow selection safety rules changed")
    if set(review) != {
        "required_assessment_for_every_selected_candidate",
        "allowed_assessments",
    }:
        raise ShadowEvidencePolicyError("shadow review contract changed")
    if (
        review["required_assessment_for_every_selected_candidate"] is not True
        or review["allowed_assessments"] != list(ALLOWED_ASSESSMENTS)
    ):
        raise ShadowEvidencePolicyError("shadow review labels or completeness rule changed")
    if set(evaluation) != {
        "entry_close_offset_bars",
        "horizon_bars",
        "minimum_matured_samples_for_human_pilot",
    }:
        raise ShadowEvidencePolicyError("shadow evaluation contract changed")
    if (
        evaluation["entry_close_offset_bars"] != 1
        or evaluation["horizon_bars"] != 20
        or _positive_int(
            evaluation["minimum_matured_samples_for_human_pilot"],
            "evaluation.minimum_matured_samples_for_human_pilot",
        )
        != 20
    ):
        raise ShadowEvidencePolicyError("shadow evaluation thresholds changed")
    if boundary != {
        "counterfactual_only": True,
        "execution_eligible": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }:
        raise ShadowEvidencePolicyError("shadow execution boundary changed")
    return payload


def prospective_start_market_date(policy: dict[str, Any]) -> date:
    return date.fromisoformat(policy["prospective_start_market_date"])
