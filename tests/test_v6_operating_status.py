from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone
from typing import Any

from scripts.audit_v6_release import BLOCKER_BY_GATE, audit_v6_release
from scripts.build_v6_operating_status import (
    V6OperatingStatusError,
    build_v6_operating_status,
    derive_v6_operating_status,
    validate_v6_operating_status,
)


def _human_ready_release(*, promotion_evidence_ready: bool) -> dict[str, Any]:
    release = copy.deepcopy(audit_v6_release(output_path=None))
    thresholds = release["evidence_thresholds"]
    counts = release["evidence_counts"]
    counts["public_signal_primary_horizon_outcomes"] = thresholds[
        "public_signal_primary_horizon_outcomes"
    ]
    counts["sanitized_live_review_matured_outcomes"] = thresholds[
        "sanitized_live_review_matured_outcomes"
    ]
    counts["sanitized_live_review_predictions"] = thresholds[
        "ibkr_bound_sanitized_live_reviews"
    ]
    counts["minimum_governance_paired_samples"] = thresholds[
        "governance_paired_samples_per_challenger"
    ] - (0 if promotion_evidence_ready else 1)
    release["release_gates"].update(
        {
            "model_artifacts_valid": True,
            "live_review_forward_artifacts_valid": True,
            "public_signal_minimum_sample_reached": True,
            "model_governance_promotion_sample_reached": promotion_evidence_ready,
            "live_review_minimum_sample_reached": True,
            "ibkr_to_chatgpt_contract_evidenced": True,
        }
    )
    release["blockers"] = []
    release["ready_for_human_pilot_review"] = True
    release["status"] = "READY_FOR_HUMAN_PILOT_REVIEW"
    release["challenger_model_promotion_evidence_ready"] = (
        promotion_evidence_ready
    )
    release["challenger_model_promotion_blockers"] = (
        []
        if promotion_evidence_ready
        else [BLOCKER_BY_GATE["model_governance_promotion_sample_reached"]]
    )
    return release


class V6OperatingStatusTests(unittest.TestCase):
    def test_current_evidence_is_read_only_shadow_not_human_pilot(self) -> None:
        payload = build_v6_operating_status(output_path=None)

        self.assertEqual(payload["operating_mode"], "READ_ONLY_SHADOW")
        self.assertEqual(
            payload["next_operating_mode"],
            "HUMAN_PILOT_REVIEW_READY",
        )
        self.assertTrue(payload["capabilities"]["human_decision_support_available"])
        self.assertFalse(payload["capabilities"]["human_pilot_review_ready"])
        self.assertFalse(
            payload["capabilities"][
                "challenger_model_promotion_evidence_ready"
            ]
        )
        self.assertNotIn(
            "model_governance_promotion_sample_reached",
            payload["mode_gates"]["human_pilot_review"][
                "required_release_gates"
            ],
        )
        self.assertFalse(payload["automatic_order_allowed"])
        self.assertTrue(payload["human_confirmation_required"])
        self.assertFalse(payload["contains_private_account_data"])
        self.assertGreaterEqual(
            datetime.fromisoformat(payload["generated_at_utc"]),
            datetime(2026, 7, 13, 14, 44, 26, tzinfo=timezone.utc),
        )

    def test_human_pilot_does_not_require_challenger_promotion_sample(self) -> None:
        payload = derive_v6_operating_status(
            _human_ready_release(promotion_evidence_ready=False)
        )

        self.assertEqual(payload["operating_mode"], "HUMAN_PILOT_REVIEW_READY")
        self.assertIsNone(payload["next_operating_mode"])
        self.assertTrue(payload["mode_gates"]["human_pilot_review"]["passed"])
        self.assertFalse(
            payload["mode_gates"]["challenger_model_promotion_evidence"][
                "passed"
            ]
        )
        self.assertEqual(
            payload["challenger_model_promotion_blockers"],
            [BLOCKER_BY_GATE["model_governance_promotion_sample_reached"]],
        )
        self.assertFalse(payload["mode_gates"]["automatic_execution"]["passed"])

    def test_all_release_gates_make_both_readiness_tracks_pass(self) -> None:
        payload = derive_v6_operating_status(
            _human_ready_release(promotion_evidence_ready=True)
        )

        self.assertTrue(payload["mode_gates"]["human_pilot_review"]["passed"])
        self.assertTrue(
            payload["mode_gates"]["challenger_model_promotion_evidence"][
                "passed"
            ]
        )
        self.assertTrue(
            payload["capabilities"][
                "challenger_model_promotion_evidence_ready"
            ]
        )
        self.assertEqual(payload["challenger_model_promotion_blockers"], [])

    def test_tampered_mode_or_execution_claim_fails(self) -> None:
        payload = build_v6_operating_status(output_path=None)
        wrong_mode = copy.deepcopy(payload)
        wrong_mode["operating_mode"] = "HUMAN_PILOT_REVIEW_READY"
        with self.assertRaisesRegex(V6OperatingStatusError, "operating mode"):
            validate_v6_operating_status(wrong_mode)

        executable = copy.deepcopy(payload)
        executable["automatic_order_allowed"] = True
        with self.assertRaisesRegex(V6OperatingStatusError, "schema validation"):
            validate_v6_operating_status(executable)


if __name__ == "__main__":
    unittest.main()
