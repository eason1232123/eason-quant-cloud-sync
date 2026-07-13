from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone

from scripts.audit_v6_release import audit_v6_release
from scripts.build_v6_operating_status import (
    V6OperatingStatusError,
    build_v6_operating_status,
    derive_v6_operating_status,
    validate_v6_operating_status,
)


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
        self.assertFalse(payload["automatic_order_allowed"])
        self.assertTrue(payload["human_confirmation_required"])
        self.assertFalse(payload["contains_private_account_data"])
        self.assertGreaterEqual(
            datetime.fromisoformat(payload["generated_at_utc"]),
            datetime(2026, 7, 13, 14, 44, 26, tzinfo=timezone.utc),
        )

    def test_all_frozen_release_gates_are_required_for_human_pilot(self) -> None:
        release = audit_v6_release(output_path=None)
        ready = copy.deepcopy(release)
        ready["release_gates"] = {name: True for name in ready["release_gates"]}
        ready["blockers"] = []
        ready["ready_for_human_pilot_review"] = True
        ready["status"] = "READY_FOR_HUMAN_PILOT_REVIEW"
        for field, threshold in ready["evidence_thresholds"].items():
            count_field = {
                "governance_paired_samples_per_challenger": "minimum_governance_paired_samples",
                "ibkr_bound_sanitized_live_reviews": "sanitized_live_review_predictions",
            }.get(field, field)
            ready["evidence_counts"][count_field] = threshold

        payload = derive_v6_operating_status(ready)

        self.assertEqual(payload["operating_mode"], "HUMAN_PILOT_REVIEW_READY")
        self.assertIsNone(payload["next_operating_mode"])
        self.assertTrue(payload["mode_gates"]["human_pilot_review"]["passed"])
        self.assertFalse(payload["mode_gates"]["automatic_execution"]["passed"])

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
