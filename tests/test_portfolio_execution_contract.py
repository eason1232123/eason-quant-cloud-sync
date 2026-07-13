from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import build_portfolio_backtest as portfolio
from scripts import strategy_contract
from scripts.validate_validation_split import canonical_fingerprint


def price_frame(dates: list[str], qqq_prices: list[float] | None = None) -> pd.DataFrame:
    rows = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "QQQ": qqq_prices if qqq_prices is not None else [100.0] * rows,
            "SMH": [100.0] * rows,
            "MSFT": [100.0] * rows,
            "SPY": [100.0] * rows,
        }
    )


def observed_cash_returns(dates: list[str], values: list[float] | None = None) -> pd.Series:
    return pd.Series(
        values if values is not None else [0.0] * len(dates),
        index=pd.to_datetime(dates),
        name="cash_ret",
        dtype=float,
    )


class PortfolioExecutionContractTests(unittest.TestCase):
    def test_initial_monthly_and_drift_signals_execute_on_next_common_bar(self):
        dates = ["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03"]
        prices = price_frame(dates, [100.0, 100.0, 140.0, 140.0])

        with patch.object(portfolio, "choose_regime", new=lambda row: "base"):
            equity, trades, _ = portfolio.simulate_strategy(prices, observed_cash_returns(dates))

        initial_trades = trades[trades["signal_date"] == dates[0]]
        self.assertFalse(initial_trades.empty)
        self.assertEqual(set(initial_trades["execution_date"]), {dates[1]})

        monthly_trades = trades[trades["signal_date"] == dates[2]]
        self.assertFalse(monthly_trades.empty)
        self.assertEqual(set(monthly_trades["execution_date"]), {dates[3]})
        self.assertTrue(monthly_trades["trigger_reasons"].str.contains("monthly_rebalance").all())
        self.assertTrue(monthly_trades["trigger_reasons"].str.contains("drift_rebalance").all())
        self.assertNotIn(dates[2], set(monthly_trades["execution_date"]))
        self.assertEqual(equity.iloc[3]["executed_signal_date"], dates[2])

    def test_regime_signal_executes_on_following_bar(self):
        dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        prices = price_frame(dates)

        def regime_for_row(row: pd.Series) -> str:
            return "defensive" if pd.to_datetime(row["date"]) >= pd.Timestamp(dates[2]) else "base"

        with patch.object(portfolio, "choose_regime", new=regime_for_row):
            _, trades, _ = portfolio.simulate_strategy(prices, observed_cash_returns(dates))

        regime_trades = trades[trades["trigger_reasons"].str.contains("regime_change")]
        self.assertFalse(regime_trades.empty)
        self.assertEqual(set(regime_trades["signal_date"]), {dates[2]})
        self.assertEqual(set(regime_trades["execution_date"]), {dates[3]})
        self.assertEqual(set(regime_trades["regime_used_for_execution"]), {"defensive"})

    def test_drift_only_signal_executes_on_following_bar(self):
        dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        prices = price_frame(dates, [100.0, 100.0, 200.0, 200.0, 200.0])

        with patch.object(portfolio, "choose_regime", new=lambda row: "base"):
            _, trades, _ = portfolio.simulate_strategy(prices, observed_cash_returns(dates))

        drift_trades = trades[
            (trades["signal_date"] == dates[2])
            & trades["trigger_reasons"].str.contains("drift_rebalance")
        ]
        self.assertFalse(drift_trades.empty)
        self.assertEqual(set(drift_trades["execution_date"]), {dates[3]})
        self.assertFalse(drift_trades["trigger_reasons"].str.contains("monthly_rebalance").any())
        self.assertFalse(drift_trades["trigger_reasons"].str.contains("regime_change").any())

    def test_missing_cash_proxy_or_core_asset_data_fails_visibly(self):
        dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
        prices = price_frame(dates)
        with self.assertRaisesRegex(ValueError, "cash-proxy returns are unavailable"):
            portfolio.simulate_strategy(prices, None)

        missing_cash = observed_cash_returns(dates, [0.0, np.nan, 0.0])
        with self.assertRaisesRegex(ValueError, "missing or invalid after proxy start"):
            portfolio.simulate_strategy(prices, missing_cash)

        missing_asset = prices.copy()
        missing_asset.loc[1, "QQQ"] = np.nan
        with self.assertRaisesRegex(ValueError, "QQQ has missing or invalid price bars"):
            portfolio.simulate_strategy(missing_asset, observed_cash_returns(dates))

    def test_price_cache_does_not_silently_drop_invalid_or_duplicate_rows(self):
        with TemporaryDirectory() as temp_dir, patch.object(portfolio, "OUT", Path(temp_dir)):
            cache_path = Path(temp_dir) / "QQQ_daily.csv"
            pd.DataFrame(
                {
                    "date": ["2026-01-05", "2026-01-06"],
                    "adjClose": [100.0, np.nan],
                }
            ).to_csv(cache_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing or invalid prices"):
                portfolio.load_price("QQQ")

            pd.DataFrame(
                {
                    "date": ["2026-01-05", "2026-01-05"],
                    "adjClose": [100.0, 101.0],
                }
            ).to_csv(cache_path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate dates"):
                portfolio.load_price("QQQ")

    def test_cross_asset_alignment_does_not_hide_a_missing_bar(self):
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        frames = {
            ticker: pd.DataFrame({"date": dates, ticker: [100.0, 101.0, 102.0]})
            for ticker in portfolio.CORE_TICKERS
        }
        frames["QQQ"] = frames["QQQ"].drop(index=1).reset_index(drop=True)

        with patch.object(portfolio, "load_price", new=lambda ticker: frames[ticker].copy()):
            with self.assertRaisesRegex(SystemExit, "Missing required asset bars after common start"):
                portfolio.load_prices(portfolio.CORE_TICKERS)

    def test_pre_proxy_rows_are_explicitly_untradeable(self):
        dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        cash_returns = observed_cash_returns(dates, [np.nan, 0.0, 0.0, 0.0])
        with patch.object(portfolio, "choose_regime", new=lambda row: "base"):
            equity, trades, assumptions = portfolio.simulate_strategy(price_frame(dates), cash_returns)

        self.assertEqual(equity.iloc[0]["date"].date().isoformat(), dates[1])
        self.assertEqual(assumptions["cash_proxy"]["excluded_pre_cash_proxy_rows"], 1)
        self.assertEqual(assumptions["cash_proxy"]["cash_proxy_status"], "OBSERVED_COMPLETE_AFTER_START")
        self.assertEqual(set(trades["signal_date"]), {dates[1]})
        self.assertEqual(set(trades["execution_date"]), {dates[2]})

    def test_shared_configured_costs_are_applied_and_traceable(self):
        dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
        with patch.object(portfolio, "choose_regime", new=lambda row: "base"):
            equity, trades, assumptions = portfolio.simulate_strategy(
                price_frame(dates),
                observed_cash_returns(dates),
            )

        self.assertFalse(trades.empty)
        self.assertEqual(set(trades["cost_calibration_status"]), {"configured_not_observed"})
        self.assertTrue((trades["commission_dollars"] > 0).all())
        self.assertTrue((trades["execution_impact_dollars"] > 0).all())
        first_trade = trades.iloc[0]
        self.assertAlmostEqual(
            first_trade["execution_price"],
            first_trade["reference_close"] * (1 + strategy_contract.EXECUTION_IMPACT_RATE),
            places=4,
        )
        self.assertAlmostEqual(
            first_trade["commission_dollars"],
            first_trade["dollars"] * strategy_contract.COMMISSION_RATE,
            places=2,
        )
        self.assertEqual(assumptions["execution_costs"]["calibration_status"], "configured_not_observed")
        self.assertEqual(assumptions["strategy_fingerprint"], strategy_contract.STRATEGY_FINGERPRINT)
        self.assertLess(equity.iloc[1]["strategy_value"], equity.iloc[0]["strategy_value"])

    def test_generated_report_preserves_fields_and_adds_contract_metadata(self):
        dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
        prices = price_frame(dates)
        cash_returns = observed_cash_returns(dates)
        with (
            TemporaryDirectory() as temp_dir,
            patch.object(portfolio, "OUT", Path(temp_dir)),
            patch.object(portfolio, "load_prices", new=lambda tickers: prices.copy()),
            patch.object(portfolio, "build_cash_returns", new=lambda: cash_returns.copy()),
            patch.object(portfolio, "choose_regime", new=lambda row: "base"),
        ):
            portfolio.main()
            report = json.loads((Path(temp_dir) / "portfolio_backtest.json").read_text(encoding="utf-8"))
            trade_log = pd.read_csv(Path(temp_dir) / "portfolio_trades.csv")

        self.assertTrue(report["available"])
        self.assertEqual(report["data_timestamp"], dates[-1])
        self.assertEqual(report["market_timezone"], "America/New_York")
        self.assertEqual(report["cost_calibration_status"], "configured_not_observed")
        self.assertEqual(report["strategy_fingerprint"], strategy_contract.STRATEGY_FINGERPRINT)
        self.assertEqual(len(report["portfolio_contract_fingerprint"]), 64)
        self.assertEqual(len(report["split_manifest_fingerprint"]), 64)
        self.assertEqual(len(report["full_model_fingerprint"]), 64)
        self.assertIn("strategy_metrics", report)
        self.assertIn("signal_date", trade_log.columns)
        self.assertIn("execution_date", trade_log.columns)
        self.assertIn("full_model_fingerprint", trade_log.columns)

    def test_portfolio_weight_change_changes_portfolio_fingerprint(self):
        original = canonical_fingerprint(portfolio.portfolio_contract_payload())

        with patch.dict(portfolio.BASE_WEIGHTS, {"QQQ": 0.31}):
            changed = canonical_fingerprint(portfolio.portfolio_contract_payload())

        self.assertNotEqual(original, changed)


if __name__ == "__main__":
    unittest.main()
