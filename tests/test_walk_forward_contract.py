from __future__ import annotations

import unittest

import pandas as pd

from scripts import build_walk_forward_report as walk


METADATA = {
    "generated_at_utc": "2026-07-12T00:00:00+00:00",
    "data_source": "fixture",
    "market_timezone": "America/New_York",
    "data_timestamp": "2026-07-10",
    "data_timestamp_granularity": "market_date",
    "data_timestamp_status": "AVAILABLE",
    "price_frequency": "end_of_day_daily",
    "price_adjustment_policy": "adjusted_ohlc_when_available_else_unadjusted",
    "strategy_contract_version": "fixture",
    "rule_fingerprint": "a" * 64,
    "strategy_fingerprint": "b" * 64,
    "split_manifest_fingerprint": "c" * 64,
}


class WalkForwardContractTests(unittest.TestCase):
    def test_period_final_values_are_rebased_at_period_start(self) -> None:
        values = pd.Series([20_000.0, 21_000.0, 22_000.0])
        dates = pd.Series(pd.bdate_range("2026-01-02", periods=3))

        result = walk.metrics(values, dates)

        self.assertEqual(result["source_series_start_value"], 20_000.0)
        self.assertEqual(result["final_value"], 11_000.0)
        self.assertEqual(result["final_value_basis"], "period_rebased_to_10000")

    def test_partial_period_is_never_eligible_as_validation_claim(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=80)
        frame = pd.DataFrame(
            {
                "date": dates,
                "strategy_value": range(10_000, 10_080),
                "buy_hold_SPY": range(10_000, 10_080),
            }
        )
        periods = [
            {
                "name": "validation",
                "start": "2019-01-02",
                "end": dates[-1].date().isoformat(),
                "classification": "RETROSPECTIVE_CONTAMINATED",
                "purpose": "fixture",
            }
        ]

        report = walk.build_walk_forward(frame, periods, METADATA)

        self.assertEqual(report["evidence_classification"], "RETROSPECTIVE_CONTAMINATED")
        self.assertFalse(report["eligible_as_pristine_out_of_sample"])
        self.assertEqual(report["periods"][0]["coverage_status"], "PARTIAL")
        self.assertFalse(report["periods"][0]["eligible_for_validation_claim"])

    def test_high_retrospective_score_cannot_become_validated(self) -> None:
        period = {
            "period": {"name": "fixture"},
            "available": True,
            "strategy_metrics": {"cagr_pct": 10.0, "max_drawdown_pct": -10.0, "sharpe": 1.0},
            "strategy_vs_benchmarks": [{"strategy_cagr_minus_benchmark_pct": 2.0}],
        }

        report = walk.overfitting_check({"periods": [period]}, METADATA)

        self.assertEqual(report["raw_heuristic_verdict"], "PASS_STABILITY_CHECK")
        self.assertEqual(report["verdict"], "RETROSPECTIVE_ONLY_NOT_VALIDATED")
        self.assertEqual(report["prospective_validation_status"], "NOT_YET_SUFFICIENT")

    def test_regime_drawdown_resets_for_each_contiguous_episode(self) -> None:
        equity = pd.DataFrame(
            {
                "strategy_value": [100.0, 90.0, 200.0, 180.0, 50.0, 45.0],
                "regime": ["base", "base", "defensive", "defensive", "base", "base"],
                "cash_weight": [0.1] * 6,
                "tech_ai_concentration": [0.5] * 6,
            }
        )

        report = walk.regime_report(equity, METADATA)
        base = next(row for row in report["regimes"] if row["regime"] == "base")

        self.assertEqual(base["max_regime_drawdown_pct"], -10.0)
        self.assertEqual(
            base["drawdown_definition"],
            "minimum drawdown within contiguous regime episodes",
        )


if __name__ == "__main__":
    unittest.main()
