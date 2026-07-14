from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import scripts.audit_v6_release as release_audit

from scripts.audit_v6_release import (
    BLOCKER_BY_GATE,
    CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES,
    HUMAN_PILOT_REQUIRED_GATES,
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
            gates["shadow_review_minimum_sample_reached"],
            counts["shadow_review_matured_outcomes"]
            >= thresholds["shadow_review_matured_outcomes"],
        )
        self.assertEqual(
            payload["ready_for_human_pilot_review"],
            all(gates[name] for name in HUMAN_PILOT_REQUIRED_GATES),
        )
        self.assertEqual(
            payload["challenger_model_promotion_evidence_ready"],
            all(
                gates[name]
                for name in CHALLENGER_MODEL_PROMOTION_EVIDENCE_REQUIRED_GATES
            ),
        )
        self.assertEqual(bool(payload["blockers"]), not payload["ready_for_human_pilot_review"])
        self.assertEqual(
            bool(payload["challenger_model_promotion_blockers"]),
            not payload["challenger_model_promotion_evidence_ready"],
        )
        self.assertFalse(payload["automatic_order_allowed"])
        self.assertTrue(payload["human_confirmation_required"])
        self.assertIn(
            "RETROSPECTIVE_BACKTEST_SURVIVORSHIP_BIAS_REMAINS_UNCONTROLLED",
            payload["known_limitations"],
        )
        self.assertIn(
            "PROSPECTIVE_FROZEN_UNIVERSE_DOES_NOT_SUPPORT_MARKET_WIDE_GENERALIZATION",
            payload["known_limitations"],
        )

    def test_public_signal_gate_counts_only_active_primary_outcomes(self) -> None:
        events: list[dict] = []
        for index in range(20):
            prediction_id = f"no_signal_{index}"
            events.extend(
                [
                    {
                        "event_type": "PREDICTION",
                        "event_id": prediction_id,
                        "prediction": {"state": "NO_SIGNAL"},
                    },
                    {
                        "event_type": "OUTCOME",
                        "prediction_event_id": prediction_id,
                        "outcome": {"horizon_bars": 20},
                    },
                ]
            )

        with patch.object(release_audit, "load_ledger", return_value=events):
            payload = audit_v6_release(output_path=None)

        self.assertEqual(
            payload["evidence_counts"]["public_signal_primary_horizon_outcomes"],
            0,
        )
        self.assertFalse(
            payload["release_gates"]["public_signal_minimum_sample_reached"]
        )
        self.assertIn("ACTIVE", payload["evidence_scope"]["public_signal_count"])

    def test_human_pilot_and_challenger_promotion_tracks_are_independent(self) -> None:
        payload = audit_v6_release(output_path=None)
        separated = copy.deepcopy(payload)
        separated["evidence_counts"]["public_signal_primary_horizon_outcomes"] = (
            separated["evidence_thresholds"]["public_signal_primary_horizon_outcomes"]
        )
        separated["evidence_counts"]["shadow_review_matured_outcomes"] = (
            separated["evidence_thresholds"]["shadow_review_matured_outcomes"]
        )
        separated["evidence_counts"]["minimum_governance_paired_samples"] = (
            separated["evidence_thresholds"][
                "governance_paired_samples_per_challenger"
            ]
            - 1
        )
        for gate in HUMAN_PILOT_REQUIRED_GATES:
            separated["release_gates"][gate] = True
        separated["release_gates"][
            "model_governance_promotion_sample_reached"
        ] = False
        separated["ready_for_human_pilot_review"] = True
        separated["status"] = "READY_FOR_HUMAN_PILOT_REVIEW"
        separated["blockers"] = []
        separated["challenger_model_promotion_evidence_ready"] = False
        separated["challenger_model_promotion_blockers"] = [
            BLOCKER_BY_GATE["model_governance_promotion_sample_reached"]
        ]

        validate_v6_release_status(separated)

        self.assertTrue(separated["ready_for_human_pilot_review"])
        self.assertFalse(separated["challenger_model_promotion_evidence_ready"])

    def test_inconsistent_or_executable_release_claim_fails(self) -> None:
        payload = audit_v6_release(output_path=None)
        inconsistent = copy.deepcopy(payload)
        inconsistent["ready_for_human_pilot_review"] = not inconsistent[
            "ready_for_human_pilot_review"
        ]
        with self.assertRaisesRegex(V6ReleaseAuditError, "readiness"):
            validate_v6_release_status(inconsistent)

        executable = copy.deepcopy(payload)
        executable["automatic_order_allowed"] = True
        with self.assertRaisesRegex(V6ReleaseAuditError, "automatic orders"):
            validate_v6_release_status(executable)

        hidden_limit = copy.deepcopy(payload)
        hidden_limit["known_limitations"] = hidden_limit["known_limitations"][1:]
        with self.assertRaisesRegex(V6ReleaseAuditError, "known limitations"):
            validate_v6_release_status(hidden_limit)

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
            "shadow_review_minimum_sample_reached": (
                "shadow_review_matured_outcomes",
                "shadow_review_matured_outcomes",
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

        wrong_promotion_blocker = copy.deepcopy(payload)
        wrong_promotion_blocker["challenger_model_promotion_blockers"] = [
            "UNRELATED_BLOCKER"
        ]
        with self.assertRaisesRegex(V6ReleaseAuditError, "promotion blocker"):
            validate_v6_release_status(wrong_promotion_blocker)


if __name__ == "__main__":
    unittest.main()
