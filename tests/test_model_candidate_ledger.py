from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts.build_model_candidate_ledger import (
    ModelCandidateLedgerError,
    build_model_candidate_ledger,
)
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
)
from scripts.strategy_contract import (
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
)


ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_PATH = ROOT / "config" / "model_governance.json"
SPLIT_PATH = ROOT / "config" / "validation_split.json"
ANCHOR_PATH = ROOT / "docs" / "SPY_daily.csv"
TICKERS = ("QQQ", "SMH", "MSFT", "SPY", "SGOV")
MASTER_DATES = pd.bdate_range("2025-05-01", "2026-12-31")


def generated_after(market_date: str) -> str:
    value = datetime.combine(
        date.fromisoformat(market_date) + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    return value.isoformat()


def public_inputs(market_date: str) -> tuple[dict, dict]:
    generated_at = generated_after(market_date)
    source = "fixture adjusted EOD cache"
    report = {
        "generated_at_utc": generated_at,
        "data_source": source,
        "market_timezone": "America/New_York",
        "data_timestamp": market_date,
        "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
        "data_timestamp_status": "AVAILABLE",
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "universe": {"expected_latest_market_date": market_date},
    }
    packet = {
        "market_data": {
            "source": source,
            "market_timezone": "America/New_York",
            "report_generated_at_utc": generated_at,
            "data_timestamp": market_date,
            "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
            "data_timestamp_status": "AVAILABLE",
            "price_frequency": PRICE_FREQUENCY,
            "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
            "metadata_status": "COMPLETE",
        },
        "data_quality": {
            "reference_market_date_status": "ALIGNED",
            "reference_market_date": market_date,
            "source_report_generated_at_utc": generated_at,
        },
    }
    return packet, report


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def write_prices(root: Path, market_date: str) -> None:
    selected = MASTER_DATES[MASTER_DATES <= pd.Timestamp(market_date)]
    for ticker_index, ticker in enumerate(TICKERS):
        lines = ["ticker,date,close,adjClose,price\n"]
        for master_index, market_day in enumerate(selected):
            # Stable master-index pricing prevents a later fixture refresh from
            # rewriting prices already sealed in an earlier prediction/outcome.
            price = 80.0 + ticker_index * 10.0 + master_index * (0.05 + ticker_index * 0.01)
            lines.append(
                f"{ticker},{market_day.date().isoformat()},{price:.8f},{price:.8f},{price:.8f}\n"
            )
        (root / f"{ticker}_daily.csv").write_text("".join(lines), encoding="utf-8")


class ModelCandidateLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet_path = self.root / "decision_packet.json"
        self.report_path = self.root / "market_report.json"
        self.ledger_path = self.root / "model_candidate_forward_ledger.jsonl"
        self.summary_path = self.root / "model_governance.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_for(self, market_date: str) -> dict:
        packet, report = public_inputs(market_date)
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        write_prices(self.root, market_date)
        return build_model_candidate_ledger(
            governance_path=GOVERNANCE_PATH,
            split_path=SPLIT_PATH,
            packet_path=self.packet_path,
            report_path=self.report_path,
            ledger_path=self.ledger_path,
            summary_path=self.summary_path,
            prices_dir=self.root,
            anchor_path=ANCHOR_PATH,
            as_of_market_date=date.fromisoformat(market_date),
        )

    def events(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        return [json.loads(line) for line in self.ledger_path.read_text(encoding="utf-8").splitlines()]

    def test_boundary_cohorts_outcomes_non_overlap_and_idempotency(self) -> None:
        waiting = self.run_for("2026-07-10")
        self.assertEqual(waiting["status"], "WAITING_FOR_FIRST_PROSPECTIVE_MARKET_DATE")
        self.assertEqual(waiting["ledger_counts"]["prediction_events"], 0)
        self.assertFalse(self.ledger_path.exists())

        first = self.run_for("2026-07-13")
        self.assertEqual(first["ledger_counts"]["new_prediction_events"], 3)
        self.assertEqual(first["ledger_counts"]["outcome_events"], 0)
        self.assertEqual(first["allocation_decision"]["challenger_allocation"], 0.0)
        self.assertFalse(first["automatic_order_allowed"])
        first_text = self.ledger_path.read_text(encoding="utf-8")

        same = self.run_for("2026-07-13")
        self.assertEqual(same["ledger_counts"]["new_prediction_events"], 0)
        self.assertEqual(self.ledger_path.read_text(encoding="utf-8"), first_text)

        too_soon_date = MASTER_DATES[MASTER_DATES.get_loc(pd.Timestamp("2026-07-13")) + 10]
        too_soon = self.run_for(too_soon_date.date().isoformat())
        self.assertEqual(too_soon["ledger_counts"]["prediction_events"], 3)
        self.assertEqual(too_soon["ledger_counts"]["outcome_events"], 0)

        matured_date = MASTER_DATES[MASTER_DATES.get_loc(pd.Timestamp("2026-07-13")) + 21]
        matured = self.run_for(matured_date.date().isoformat())
        self.assertEqual(matured["ledger_counts"]["prediction_events"], 6)
        self.assertEqual(matured["ledger_counts"]["outcome_events"], 3)
        self.assertEqual(matured["ledger_counts"]["new_prediction_events"], 3)
        self.assertEqual(matured["ledger_counts"]["new_outcome_events"], 3)
        for challenger in matured["challenger_metrics"].values():
            self.assertEqual(challenger["paired_sample_count"], 1)

        first_cohort = [
            event
            for event in self.events()
            if event["event_type"] == "MODEL_PREDICTION"
            and event["prediction"]["observation_market_date"] == "2026-07-13"
        ]
        outcomes = [event for event in self.events() if event["event_type"] == "MODEL_OUTCOME"]
        self.assertEqual(len(first_cohort), 3)
        self.assertEqual(len(outcomes), 3)
        for event in outcomes:
            self.assertEqual(event["outcome"]["entry_market_date"], "2026-07-14")
            self.assertEqual(event["outcome"]["exit_market_date"], matured_date.date().isoformat())
            self.assertEqual(event["evidence_classification"], "PROSPECTIVE")
            self.assertFalse(event["contains_private_account_data"])

    def test_tampering_and_ambiguous_price_basis_fail_visibly(self) -> None:
        self.run_for("2026-07-13")
        events = self.events()
        events[0]["prediction"]["market_regime"] = "severe_defensive"
        self.ledger_path.write_text(
            "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ModelCandidateLedgerError, "hash mismatch"):
            self.run_for("2026-07-13")

        self.ledger_path.unlink()
        packet, report = public_inputs("2026-07-13")
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        write_prices(self.root, "2026-07-13")
        qqq = self.root / "QQQ_daily.csv"
        qqq_rows = pd.read_csv(qqq)
        qqq_rows[["ticker", "date", "price"]].to_csv(qqq, index=False)
        with self.assertRaisesRegex(ModelCandidateLedgerError, "price basis"):
            build_model_candidate_ledger(
                governance_path=GOVERNANCE_PATH,
                split_path=SPLIT_PATH,
                packet_path=self.packet_path,
                report_path=self.report_path,
                ledger_path=self.ledger_path,
                summary_path=self.summary_path,
                prices_dir=self.root,
                anchor_path=ANCHOR_PATH,
                as_of_market_date=date(2026, 7, 13),
            )

    def test_future_price_rows_fail_instead_of_becoming_current_evidence(self) -> None:
        packet, report = public_inputs("2026-07-13")
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        write_prices(self.root, "2026-07-14")
        with self.assertRaisesRegex(ModelCandidateLedgerError, "future rows"):
            build_model_candidate_ledger(
                governance_path=GOVERNANCE_PATH,
                split_path=SPLIT_PATH,
                packet_path=self.packet_path,
                report_path=self.report_path,
                ledger_path=self.ledger_path,
                summary_path=self.summary_path,
                prices_dir=self.root,
                anchor_path=ANCHOR_PATH,
                as_of_market_date=date(2026, 7, 13),
            )


if __name__ == "__main__":
    unittest.main()
