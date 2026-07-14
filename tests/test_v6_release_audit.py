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

    def test_count_gates_must_match_frozen_thresholds(self) -> None:
        payload = audit_v6_release(output_path=None)
        requirements = {
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

        for gate, (count_field, threshold_field) in requirements.items():
            with self.subTest(gate=gate):
                tampered = copy.deepcopy(payload)
                tampered["evidence_counts"][count_field] = (
                    tampered["evidence_thresholds"][threshold_field] - 1
                )
                tampered["release_gates"][gate] = True
                with self.assertRaisesRegex(V6ReleaseAuditError, "does not match"):
                    validate_v6_release_status(tampered)

        lowered = copy.deepcopy(payload)
        lowered["evidence_thresholds"][
            "public_signal_primary_horizon_outcomes"
        ] = 1
        with self.assertRaisesRegex(V6ReleaseAuditError, "frozen contract"):
            validate_v6_release_status(lowered)

    def test_gate_membership_and_blockers_must_match_the_contract(self) -> None:
        payload = audit_v6_release(output_path=None)

        extra_gate = copy.deepcopy(payload)
        extra_gate["release_gates"]["unreviewed_gate"] = True
        with self.assertRaisesRegex(V6ReleaseAuditError, "gate membership"):
            validate_v6_release_status(extra_gate)

        wrong_blocker = copy.deepcopy(payload)
        wrong_blocker["blockers"] = ["UNRELATED_BLOCKER"]
        with self.assertRaisesRegex(V6ReleaseAuditError, "failed gates"):
            validate_v6_release_status(wrong_blocker)


if __name__ == "__main__":
    unittest.main()
