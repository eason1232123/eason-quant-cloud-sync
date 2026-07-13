from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import build_trade_review as review


class TradeReviewContractTests(unittest.TestCase):
    def test_invalid_trade_rows_and_actions_fail_instead_of_dropping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trade_log.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-07-10",
                        "ticker": "NVDA",
                        "action": "UNKNOWN",
                        "fill_price": 100.0,
                        "shares": 1.0,
                        "fees": 0.1,
                    }
                ]
            ).to_csv(path, index=False)
            with patch.object(review, "TRADE_LOG", path):
                with self.assertRaisesRegex(SystemExit, "unsupported actions"):
                    review.load_trade_log()

            frame = pd.read_csv(path)
            frame.loc[0, "action"] = "BUY"
            frame.loc[0, "fill_price"] = None
            frame.to_csv(path, index=False)
            with patch.object(review, "TRADE_LOG", path):
                with self.assertRaisesRegex(SystemExit, "invalid or incomplete"):
                    review.load_trade_log()

    def test_actual_fill_review_uses_unadjusted_close(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(review, "OUT", Path(temp_dir)):
            pd.DataFrame(
                {
                    "date": ["2026-07-09", "2026-07-10"],
                    "close": [50.0, 55.0],
                    "adjClose": [100.0, 110.0],
                }
            ).to_csv(Path(temp_dir) / "NVDA_daily.csv", index=False)

            prices = review.load_price("NVDA")

        self.assertEqual(prices["price"].tolist(), [50.0, 55.0])

    def test_public_review_redacts_exact_shares_and_fees(self) -> None:
        row = pd.Series(
            {
                "date": "2026-07-09",
                "ticker": "NVDA",
                "action": "BUY",
                "fill_price": 50.0,
                "shares": 123.0,
                "fees": 4.56,
            }
        )
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-09", "2026-07-10"]),
                "price": [50.0, 55.0],
            }
        )
        with patch.object(review, "load_price", return_value=prices):
            result = review.review_trade(row)

        self.assertIsNone(result["shares"])
        self.assertIsNone(result["fees"])
        self.assertTrue(result["shares_redacted"])
        self.assertTrue(result["fees_redacted"])
        self.assertEqual(result["price_basis"], "unadjusted_close_matched_to_actual_fill")

    def test_small_actual_sample_remains_unavailable(self) -> None:
        reviews = pd.DataFrame({"outcome_return_20d_pct": [1.0] * 19})

        result = review.actual_vs_backtest(reviews)

        self.assertFalse(result["available"])
        self.assertEqual(result["completed_20d_trade_count"], 19)
        self.assertEqual(result["minimum_completed_20d_sample"], 20)


if __name__ == "__main__":
    unittest.main()
