from __future__ import annotations

import copy
import unittest

from scripts.audit_v6_release import (
    V6ReleaseAuditError,
    audit_v6_release,
    validate_v6_release_status,
)


class V6ReleaseAuditTests(unittest.TestCase):
    def test_release_gates_are_derived_from_current_evidence(self) -> None:
        payload = audit_v6_release(output_path=None)
        counts = payload["evidence_counts"]
        thresholds = payload["evidence_thresholds"]
        gates = payload["release_gates"]

        self.assertEqual(
            gates["public_signal_minimum_sample_reached"],
            counts["public_signal_primary_horizon_outcomes"]
            >= thresholds["public_signal_primary_horizon_outcomes"],
        )
        self.assertEqual(
            gates["model_governance_promotion_sample_reached"],
            counts["minimum_governance_paired_samples"]
            >= thresholds["governance_paired_samples_per_challenger"],
        )
        self.assertEqual(
            gates["live_review_minimum_sample_reached"],
            counts["sanitized_live_review_matured_outcomes"]
            >= thresholds["sanitized_live_review_matured_outcomes"],
        )
        self.assertEqual(payload["ready_for_human_pilot_review"], all(gates.values()))
        self.assertEqual(bool(payload["blockers"]), not payload["ready_for_human_pilot_review"])
        self.assertFalse(payload["automatic_order_allowed"])
        self.assertTrue(payload["human_confirmation_required"])

    def test_inconsistent_or_executable_release_claim_fails(self) -> None:
        payload = audit_v6_release(output_path=None)
        inconsistent = copy.deepcopy(payload)
        inconsistent["ready_for_human_pilot_review"] = not all(
            inconsistent["release_gates"].values()
        )
        with self.assertRaisesRegex(V6ReleaseAuditError, "readiness"):
            validate_v6_release_status(inconsistent)

        executable = copy.deepcopy(payload)
        executable["automatic_order_allowed"] = True
        with self.assertRaisesRegex(V6ReleaseAuditError, "automatic orders"):
            validate_v6_release_status(executable)


if __name__ == "__main__":
    unittest.main()
