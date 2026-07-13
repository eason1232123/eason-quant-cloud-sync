from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import build_report
from scripts import strategy_contract as contract


class StrategyContractTests(unittest.TestCase):
    def test_forward_return_uses_next_close_and_round_trip_costs(self) -> None:
        close = pd.Series([100.0, 110.0, 121.0, 120.0])
        gross = contract.next_close_forward_return(close, 1)

        self.assertAlmostEqual(float(gross.iloc[0]), 0.10, places=12)
        self.assertNotAlmostEqual(float(gross.iloc[0]), 0.21, places=12)
        self.assertTrue(pd.isna(gross.iloc[-2]))

        net = contract.net_return_after_round_trip_costs(gross.iloc[0])
        expected = (
            1.10
            * (1 - contract.EXECUTION_IMPACT_RATE)
            * (1 - contract.COMMISSION_RATE)
            / ((1 + contract.EXECUTION_IMPACT_RATE) * (1 + contract.COMMISSION_RATE))
            - 1
        )
        self.assertAlmostEqual(float(net), expected, places=12)
        self.assertLess(float(net), float(gross.iloc[0]))

    def test_non_overlapping_samples_allow_next_signal_on_prior_exit_bar(self) -> None:
        signal = pd.Series([True, True, True, True, True, True, True])
        selected = contract.non_overlapping_signal_mask(signal, horizon=2)

        self.assertEqual(selected[selected].index.tolist(), [0, 3, 6])

    def test_mae_is_unavailable_when_any_required_low_is_missing(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        low = pd.Series([99.0, 100.0, float("nan"), 102.0])

        mae = contract.next_close_forward_mae(close, low, horizon=2)

        self.assertTrue(pd.isna(mae.iloc[0]))

    def test_costs_are_explicit_configured_assumptions(self) -> None:
        costs = contract.execution_cost_assumptions()

        self.assertEqual(costs["assumption_source"], "configured_not_observed")
        self.assertEqual(costs["spread_observation_status"], "NOT_OBSERVED")
        self.assertAlmostEqual(
            costs["vectorbt_fees_rate"],
            costs["commission_bps_per_side"] / 10_000,
        )
        self.assertAlmostEqual(
            costs["vectorbt_slippage_plus_half_spread_rate"],
            (
                costs["slippage_bps_per_side"]
                + costs["half_spread_bps_per_side"]
            )
            / 10_000,
        )

    def test_anchor_benchmarks_never_compare_with_themselves(self) -> None:
        for ticker in ("SPY", "QQQ", "SMH", "SOXX"):
            with self.subTest(ticker=ticker):
                self.assertNotEqual(contract.benchmark_for_ticker(ticker), ticker)

    def test_rule_parameter_change_changes_both_fingerprints(self) -> None:
        original_rule = contract.rule_fingerprint()
        original_strategy = contract.strategy_fingerprint()

        with patch.dict(
            contract.RULE_PARAMETERS["pullback_reclaim_5dma"],
            {"ret_5d_max": -0.05},
        ):
            self.assertNotEqual(contract.rule_fingerprint(), original_rule)
            self.assertNotEqual(contract.strategy_fingerprint(), original_strategy)

        implementation = contract.rule_contract_payload()["implementation_ast"]
        self.assertIn("rule_signals", implementation)
        self.assertIn("benchmark_for_ticker", implementation)
        execution_implementation = contract.strategy_contract_payload()[
            "execution_implementation_ast"
        ]
        self.assertIn("next_close_forward_mae", execution_implementation)
        self.assertIn("non_overlapping_signal_mask", execution_implementation)

    def test_report_evidence_uses_completed_independent_next_close_window(self) -> None:
        dates = pd.bdate_range("2026-06-01", periods=5)
        frame = pd.DataFrame(
            {
                "date": dates,
                "price": [100.0, 110.0, 121.0, 120.0, 119.0],
                "low_price": [99.0, 109.0, 105.0, 118.0, 117.0],
                "rule_fixture": [True, False, False, False, True],
            }
        )

        result = build_report.summarize_rule(
            "FIXTURE",
            frame,
            "rule_fixture",
            {},
            horizon=1,
        )

        self.assertEqual(result["raw_signal_count"], 2)
        self.assertEqual(result["completed_signal_count"], 1)
        self.assertEqual(result["effective_sample_count"], 1)
        self.assertEqual(result["samples"], 1)
        self.assertAlmostEqual(result["avg_gross_return"], 0.10, places=4)
        self.assertLess(result["avg_return"], result["avg_gross_return"])

    def test_missing_evidence_score_inputs_remain_unavailable(self) -> None:
        self.assertIsNone(build_report.evidence_score({"samples": 20, "win_rate": None}))

    def test_vectorbt_validation_preserves_native_calendar_gaps(self) -> None:
        module_path = Path("scripts/build_vectorbt_validation.py").resolve()
        spec = importlib.util.spec_from_file_location("_validation_calendar_test", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        fake_vectorbt = types.ModuleType("vectorbt")

        with patch.dict(sys.modules, {"vectorbt": fake_vectorbt}):
            spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            module.OUT = Path(temp_dir)
            module.VALIDATION_TICKERS = ["SPY", "AMD"]
            pd.DataFrame(
                {
                    "date": ["2026-07-08", "2026-07-09", "2026-07-10"],
                    "adjClose": [100.0, 101.0, 102.0],
                }
            ).to_csv(module.OUT / "SPY_daily.csv", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-07-08", "2026-07-09"],
                    "adjClose": [50.0, 51.0],
                }
            ).to_csv(module.OUT / "AMD_daily.csv", index=False)

            matrix = module.load_price_matrix()
            prices, price_basis, missing = module.load_native_price_map()

        self.assertTrue(pd.isna(matrix.loc[pd.Timestamp("2026-07-10"), "AMD"]))
        self.assertEqual(prices["AMD"].index.max(), pd.Timestamp("2026-07-09"))
        self.assertEqual(price_basis, {"SPY": "adjusted", "AMD": "adjusted"})
        self.assertEqual(missing, [])

    def test_vectorbt_validation_maps_every_shared_entry_to_shared_exit(self) -> None:
        module_path = Path("scripts/build_vectorbt_validation.py").resolve()
        spec = importlib.util.spec_from_file_location("_validation_rule_test", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"vectorbt": types.ModuleType("vectorbt")}):
            spec.loader.exec_module(module)

        index = pd.bdate_range("2024-01-02", periods=260)
        close = pd.DataFrame({"NVDA": [100 + (i % 17) - i * 0.01 for i in range(260)]}, index=index)
        relative = pd.DataFrame({"NVDA": [0.01] * 260}, index=index)
        expected_entries = contract.rule_signals(close, relative)
        expected_exits = contract.exit_signals(close)
        rules = module.make_rules(close, relative)

        self.assertEqual(set(rules), set(contract.ENTRY_RULE_SPECS))
        for rule_name, (entries, exits) in rules.items():
            pd.testing.assert_frame_equal(entries, expected_entries[rule_name])
            exit_name = contract.ENTRY_RULE_SPECS[rule_name]["exit_rule"]
            pd.testing.assert_frame_equal(exits, expected_exits[exit_name])


if __name__ == "__main__":
    unittest.main()
