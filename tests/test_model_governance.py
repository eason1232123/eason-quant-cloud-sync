from __future__ import annotations

import copy
import unittest

import pandas as pd

from scripts.model_governance import (
    ModelGovernanceError,
    allocation_decision,
    blend_model_weights,
    choose_regime,
    governance_fingerprint,
    incumbent_model,
    load_governance_config,
    paired_metrics,
    regime_evidence,
    validate_governance_config,
)


def metrics(
    count: int,
    *,
    mean_excess: float = 0.01,
    win_rate: float = 0.75,
    recent_excess: float = 0.02,
    drawdown: float = -0.05,
) -> dict:
    return {
        "paired_sample_count": count,
        "mean_excess_return": mean_excess,
        "median_excess_return": mean_excess,
        "paired_win_rate": win_rate,
        "cumulative_incumbent_return": 0.1,
        "cumulative_challenger_return": 0.2,
        "cumulative_excess_return": 0.1,
        "recent_cumulative_excess_return": recent_excess,
        "challenger_max_drawdown": drawdown,
    }


class ModelGovernanceConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_governance_config()
        self.incumbent_id = incumbent_model(self.config)["model_id"]
        self.challenger_ids = [
            model["model_id"]
            for model in self.config["models"]
            if model["role"] == "CHALLENGER"
        ]

    def test_frozen_config_is_strict_and_fingerprinted(self) -> None:
        first = governance_fingerprint(self.config)
        same = governance_fingerprint(copy.deepcopy(self.config))
        changed = copy.deepcopy(self.config)
        changed["models"][1]["regime_weights"]["base"]["QQQ"] -= 0.01
        changed["models"][1]["regime_weights"]["base"]["CASH"] += 0.01

        self.assertEqual(first, same)
        self.assertNotEqual(first, governance_fingerprint(changed))
        self.assertEqual(len(first), 64)

    def test_invalid_weights_and_retrospective_promotion_fail(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["models"][0]["regime_weights"]["base"]["QQQ"] = 0.31
        with self.assertRaisesRegex(ModelGovernanceError, "sum to exactly one"):
            validate_governance_config(invalid)

        contaminated = copy.deepcopy(self.config)
        contaminated["evidence_policy"]["retrospective_evidence_may_promote"] = True
        with self.assertRaisesRegex(ModelGovernanceError, "must never promote"):
            validate_governance_config(contaminated)

        invalid_win_rate = copy.deepcopy(self.config)
        invalid_win_rate["allocation_gates"]["minimum_paired_win_rate"] = 1.1
        with self.assertRaisesRegex(ModelGovernanceError, "must be in"):
            validate_governance_config(invalid_win_rate)

        invalid_rollback = copy.deepcopy(self.config)
        invalid_rollback["rollback_gates"]["recent_paired_sample_window"] = 0
        with self.assertRaisesRegex(ModelGovernanceError, "positive integer"):
            validate_governance_config(invalid_rollback)

    def test_regime_evidence_is_explicit_and_warmup_is_not_silently_defensive(self) -> None:
        warmup = pd.Series({"QQQ": 100.0, "SMH": 100.0, "SPY": 100.0})
        self.assertEqual(choose_regime(warmup), "warmup_base")
        self.assertEqual(regime_evidence(warmup)["reason_codes"], ["MA200_UNAVAILABLE"])

        severe = pd.Series(
            {
                "QQQ": 80.0,
                "QQQ_ma200": 100.0,
                "SMH": 70.0,
                "SMH_ma200": 100.0,
                "SPY": 90.0,
                "SPY_ma200": 100.0,
                "QQQ_ret20": -0.10,
                "SMH_ret20": -0.15,
            }
        )
        evidence = regime_evidence(severe)
        self.assertEqual(evidence["regime"], "severe_defensive")
        self.assertIn("SPY_QQQ_SMH_BELOW_MA200", evidence["reason_codes"])

    def test_dynamic_blend_is_deterministic_and_sums_to_one(self) -> None:
        challenger_id = self.challenger_ids[0]
        blended = blend_model_weights(
            self.config,
            {self.incumbent_id: 0.75, challenger_id: 0.25},
            "defensive",
        )
        self.assertAlmostEqual(sum(blended.values()), 1.0)
        expected_qqq = (
            self.config["models"][0]["regime_weights"]["defensive"]["QQQ"] * 0.75
            + self.config["models"][1]["regime_weights"]["defensive"]["QQQ"] * 0.25
        )
        self.assertAlmostEqual(blended["QQQ"], expected_qqq)

    def test_no_samples_or_weak_samples_hold_incumbent(self) -> None:
        empty = allocation_decision(self.config, {})
        self.assertEqual(
            empty["status"],
            "HOLD_INCUMBENT_INSUFFICIENT_OR_WEAK_PROSPECTIVE_EVIDENCE",
        )
        self.assertEqual(empty["model_allocations"], {self.incumbent_id: 1.0})
        self.assertFalse(empty["automatic_order_allowed"])

        weak = allocation_decision(
            self.config,
            {self.challenger_ids[0]: metrics(48, mean_excess=-0.001)},
        )
        self.assertEqual(weak["challenger_allocation"], 0.0)

    def test_frozen_sample_gates_cap_dynamic_allocation(self) -> None:
        challenger_id = self.challenger_ids[0]
        cases = (
            (11, 0.0, "HOLD_INCUMBENT_INSUFFICIENT_OR_WEAK_PROSPECTIVE_EVIDENCE"),
            (12, 0.10, "PILOT_PROSPECTIVE_BLEND"),
            (24, 0.25, "LIMITED_PROSPECTIVE_BLEND"),
            (48, 0.50, "PROMOTION_REVIEW_REQUIRED_MAXIMUM_CAPPED_BLEND"),
        )
        for count, expected_allocation, expected_status in cases:
            with self.subTest(count=count):
                decision = allocation_decision(
                    self.config,
                    {challenger_id: metrics(count)},
                )
                self.assertEqual(decision["status"], expected_status)
                self.assertEqual(decision["challenger_allocation"], expected_allocation)
                self.assertLessEqual(decision["challenger_allocation"], 0.50)
                self.assertFalse(decision["automatic_order_allowed"])

    def test_rollback_and_data_failure_return_to_incumbent(self) -> None:
        challenger_id = self.challenger_ids[0]
        rollback = allocation_decision(
            self.config,
            {challenger_id: metrics(48, recent_excess=-0.031)},
        )
        self.assertEqual(rollback["status"], "ROLLBACK_TO_INCUMBENT")
        self.assertEqual(rollback["model_allocations"], {self.incumbent_id: 1.0})

        failed = allocation_decision(
            self.config,
            {challenger_id: metrics(48)},
            data_integrity_ok=False,
        )
        self.assertEqual(failed["status"], "ROLLBACK_DATA_INTEGRITY_FAILURE")
        self.assertEqual(failed["challenger_allocation"], 0.0)

    def test_paired_metrics_reject_unpaired_or_invalid_samples(self) -> None:
        with self.assertRaisesRegex(ModelGovernanceError, "must match"):
            paired_metrics([0.01], [], recent_window=6)
        with self.assertRaisesRegex(ModelGovernanceError, "greater than -100%"):
            paired_metrics([0.01], [-1.0], recent_window=6)
        with self.assertRaisesRegex(ModelGovernanceError, "positive integer"):
            paired_metrics([0.01], [0.02], recent_window=0)

        result = paired_metrics(
            [0.01, -0.02, 0.01],
            [0.02, -0.01, 0.03],
            recent_window=2,
        )
        self.assertEqual(result["paired_sample_count"], 3)
        self.assertGreater(result["mean_excess_return"], 0)
        self.assertEqual(result["paired_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
