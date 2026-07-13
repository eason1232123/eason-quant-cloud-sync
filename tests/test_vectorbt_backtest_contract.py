from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import build_vectorbt_backtest as backtest
from scripts.strategy_contract import (
    COMMISSION_RATE,
    ENTRY_RULE_SPECS,
    EXECUTION_IMPACT_RATE,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    exit_signals,
    net_return_after_round_trip_costs,
    risk_signals,
    rule_signals,
)


class _FakeMapped:
    def __init__(self, values: list[float]):
        self.values = np.asarray(values, dtype=float)


class _FakeTradeSide:
    def __init__(self, returns: list[float]):
        self.returns = _FakeMapped(returns)
        self._count = len(returns)

    def count(self) -> int:
        return self._count


class _FakeTrades:
    def __init__(self, closed_returns: list[float], open_count: int):
        self.closed = _FakeTradeSide(closed_returns)
        self.open = _FakeTradeSide([0.0] * open_count)


class _FakePortfolio:
    def __init__(self, value: pd.Series, closed_returns: list[float], open_count: int):
        self._value = value
        self.trades = _FakeTrades(closed_returns, open_count)

    def value(self) -> pd.Series:
        return self._value


class VectorbtBacktestContractTests(unittest.TestCase):
    def test_all_entry_and_exit_columns_come_from_shared_contract(self):
        dates = pd.bdate_range("2025-01-02", periods=260)
        close = pd.Series(
            100 + np.linspace(0, 35, len(dates)) + np.sin(np.arange(len(dates)) / 4) * 6,
            dtype=float,
        )
        benchmark_close = pd.Series(100 + np.linspace(0, 18, len(dates)), dtype=float)
        data = pd.DataFrame({"date": dates, "close": close, "low": close - 1})
        benchmark = pd.DataFrame(
            {"date": dates, "close": benchmark_close, "low": benchmark_close - 1}
        )

        actual = backtest.add_signals(data, benchmark)
        expected_entries = rule_signals(actual["close"], actual["relative_20d"])
        expected_exits = exit_signals(actual["close"])
        expected_risks = risk_signals(actual["close"])

        self.assertEqual(set(backtest.ENTRY_RULES), set(ENTRY_RULE_SPECS))
        self.assertEqual(len(backtest.ENTRY_RULES), 6)
        for name in ENTRY_RULE_SPECS:
            pd.testing.assert_series_equal(
                actual[f"entry_{name}"], expected_entries[name], check_names=False
            )
            exit_name = ENTRY_RULE_SPECS[name]["exit_rule"]
            pd.testing.assert_series_equal(
                actual[f"exit_{exit_name}"], expected_exits[exit_name], check_names=False
            )
        for name, expected in expected_risks.items():
            pd.testing.assert_series_equal(
                actual[f"risk_{name}"], expected, check_names=False
            )

    def test_execution_shift_is_exactly_one_trading_bar(self):
        raw = pd.Series([False, True, False, False], dtype=bool)
        shifted = backtest.shift_execution(raw)
        pd.testing.assert_series_equal(
            shifted,
            pd.Series([False, False, True, False], dtype=bool),
        )

    def test_forward_evidence_uses_next_close_costs_and_non_overlapping_samples(self):
        dates = pd.bdate_range("2026-01-02", periods=8)
        close = pd.Series([100, 110, 120, 130, 140, 150, 160, 170], dtype=float)
        data = pd.DataFrame(
            {
                "date": dates,
                "close": close,
                "low": close - 2,
                "entry_fixture": [True, True, False, False, True, False, False, False],
            }
        )
        benchmark = pd.DataFrame(
            {
                "date": dates,
                "close": pd.Series([100, 101, 102, 103, 104, 105, 106, 107], dtype=float),
            }
        )

        row = backtest.forward_evidence_row(
            "TEST",
            data,
            "fixture",
            "entry",
            "entry_fixture",
            2,
            benchmark,
            "fixture rule",
            "SPY",
        )

        gross = pd.Series([130 / 110 - 1, 170 / 150 - 1], dtype=float)
        net = net_return_after_round_trip_costs(gross)
        self.assertEqual(row["raw_signal_count"], 3)
        self.assertEqual(row["completed_signal_count"], 3)
        self.assertEqual(row["effective_samples"], 2)
        self.assertEqual(row["sample_count"], 2)
        self.assertEqual(row["closed_trade_count"], 0)
        self.assertEqual(row["avg_gross_forward_return"], backtest.clean_float(gross.mean()))
        self.assertEqual(row["avg_forward_return"], backtest.clean_float(net.mean()))
        self.assertNotEqual(row["avg_forward_return"], row["avg_gross_forward_return"])
        self.assertEqual(row["benchmark"], "SPY")
        self.assertEqual(row["strategy_contract"], STRATEGY_CONTRACT_VERSION)
        self.assertEqual(row["strategy_fingerprint"], STRATEGY_FINGERPRINT)
        self.assertEqual(
            row["cost_assumption_source"],
            "configured_not_observed",
        )

    def test_missing_benchmark_bar_is_not_forward_filled(self):
        dates = pd.bdate_range("2026-01-02", periods=25)
        data = pd.DataFrame(
            {
                "date": dates,
                "close": np.linspace(100, 124, len(dates)),
                "low": np.linspace(99, 123, len(dates)),
            }
        )
        benchmark = pd.DataFrame(
            {
                "date": dates.delete(20),
                "close": np.delete(np.linspace(200, 224, len(dates)), 20),
            }
        )

        relative = backtest.relative_20d_series(data, benchmark)
        self.assertTrue(pd.isna(relative.iloc[20]))

    def test_open_position_is_not_counted_as_a_closed_trade(self):
        fake = _FakePortfolio(
            pd.Series([10000.0, 10100.0]),
            closed_returns=[0.10],
            open_count=1,
        )

        returns, closed_count, open_count = backtest.engine_trade_statistics(fake)

        self.assertEqual(returns, [0.10])
        self.assertEqual(closed_count, 1)
        self.assertEqual(open_count, 1)

    def test_strategy_row_separates_raw_effective_and_closed_counts(self):
        dates = pd.bdate_range("2025-01-02", periods=60)
        close = pd.Series(np.linspace(100, 130, len(dates)), dtype=float)
        data = pd.DataFrame({"date": dates, "close": close, "low": close - 1})
        data["entry_fixture"] = False
        data.loc[[0, 1, 10], "entry_fixture"] = True
        data["exit_fixture"] = False
        data.loc[5, "exit_fixture"] = True
        fake = _FakePortfolio(
            pd.Series(np.linspace(10000, 10100, len(dates))),
            closed_returns=[0.01],
            open_count=1,
        )

        from_signals = Mock(return_value=fake)
        fake_engine = SimpleNamespace(
            Portfolio=SimpleNamespace(from_signals=from_signals),
        )
        with patch.object(backtest, "vectorbt_module", return_value=fake_engine):
            row = backtest.run_vectorbt_strategy(
                "TEST",
                data,
                "fixture",
                {
                    "entry_col": "entry_fixture",
                    "exit_col": "exit_fixture",
                    "description": "fixture",
                },
            )

        kwargs = from_signals.call_args.kwargs
        self.assertFalse(bool(kwargs["entries"].iloc[0]))
        self.assertTrue(bool(kwargs["entries"].iloc[1]))
        self.assertEqual(kwargs["fees"], COMMISSION_RATE)
        self.assertEqual(kwargs["slippage"], EXECUTION_IMPACT_RATE)
        self.assertEqual(kwargs["upon_long_conflict"], "exit")
        self.assertEqual(row["raw_signal_count"], 3)
        self.assertEqual(row["executed_entry_count"], 2)
        self.assertEqual(row["closed_trade_count"], 1)
        self.assertEqual(row["effective_samples"], 1)
        self.assertEqual(row["sample_count"], 1)
        self.assertEqual(row["trade_count"], 1)
        self.assertTrue(row["latest_position_active"])

    def test_missing_score_inputs_remain_unavailable(self):
        row = {
            "rule_type": "entry",
            "sample_count": 12,
            "valid": False,
            "win_rate": 0.5,
            "median_forward_return": 0.01,
            "alpha_vs_benchmark_same_dates": None,
        }
        self.assertIsNone(backtest.evidence_score(row))


if __name__ == "__main__":
    unittest.main()
