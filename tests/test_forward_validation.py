from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.build_forward_ledger import (
    ForwardLedgerError,
    _outcome_event,
    _seal_event,
    _write_jsonl_atomic,
    build_forward_ledger,
    load_ledger,
)
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
)
from scripts.prospective_universe_contract import (
    AMENDMENT_POLICY,
    PROSPECTIVE_SURVIVORSHIP_BIAS_STATUS,
    PROSPECTIVE_UNIVERSE_STATUS,
    SELECTION_BASIS,
    TICKER_SOURCE,
    ProspectiveUniverseContractError,
    load_and_validate_prospective_universe_contract,
)
from scripts.strategy_contract import (
    ENTRY_RULE_SPECS,
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    net_return_after_round_trip_costs,
)
from scripts.validate_validation_split import (
    ValidationSplitError,
    full_model_fingerprint,
    load_anchor_market_dates,
    load_strict_json,
    split_manifest_fingerprint,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "config" / "validation_split.json"
ANCHOR_PATH = ROOT / "docs" / "SPY_daily.csv"
SOURCE = "test public adjusted EOD source"


def signal_vector(*active_rules: str) -> dict[str, bool]:
    active = set(active_rules)
    return {rule: rule in active for rule in ENTRY_RULE_SPECS} | {"failed_rebound_risk": False}


def public_inputs(
    market_date: str,
    *,
    generated_at: str,
    active_rule: str | None = "pullback_reclaim_5dma",
) -> tuple[dict, dict]:
    ticker_dates = {"AAA": market_date, "BBB": market_date, "CCC": "2026-07-10"}
    report = {
        "generated_at_utc": generated_at,
        "data_source": SOURCE,
        "market_timezone": "America/New_York",
        "data_timestamp": market_date,
        "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
        "data_timestamp_status": "AVAILABLE",
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "universe": {
            "configured_ticker_count": 3,
            "loaded_ticker_count": 3,
            "expected_latest_market_date": market_date,
        },
        "data_timestamp_by_ticker": ticker_dates,
        "price_basis_by_ticker": {"AAA": "adjusted", "BBB": "adjusted", "CCC": "adjusted"},
        "update_log": {ticker: {"latest_date": value} for ticker, value in ticker_dates.items()},
        "technicals": {
            "AAA": {
                "latest_date": market_date,
                "active_signals": signal_vector(*(active_rule,) if active_rule else ()),
            },
            "BBB": {"latest_date": market_date, "active_signals": signal_vector()},
            "CCC": {"latest_date": "2026-07-10", "active_signals": signal_vector()},
        },
    }
    packet_market = {
        "source": SOURCE,
        "market_timezone": "America/New_York",
        "report_generated_at_utc": generated_at,
        "data_timestamp": market_date,
        "data_timestamp_granularity": DATA_TIMESTAMP_GRANULARITY,
        "data_timestamp_status": "AVAILABLE",
        "price_frequency": PRICE_FREQUENCY,
        "price_adjustment_policy": PRICE_ADJUSTMENT_POLICY,
        "metadata_status": "COMPLETE",
    }
    packet = {
        "schema_version": "test-public-packet",
        "market_data": packet_market,
        "data_quality": {
            "reference_market_date_status": "ALIGNED",
            "reference_market_date": market_date,
            "source_report_generated_at_utc": generated_at,
            "data_status": "PARTIAL_STALE_WATCHLIST",
        },
        "decision": {
            "final_action": "NO_TRADE",
            "buy_permission": "NO_QUANT_CANDIDATE",
            "automatic_order_allowed": False,
        },
    }
    return packet, report


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def prospective_universe_contract(tickers: list[str]) -> dict:
    manifest = load_strict_json(SPLIT_PATH)
    return {
        "schema_version": "v6-prospective-universe-v1",
        "status": "FROZEN",
        "frozen_on_date": "2026-07-12",
        "effective_after_market_date": "2026-07-10",
        "selection_basis": SELECTION_BASIS,
        "ticker_source": TICKER_SOURCE,
        "amendment_policy": AMENDMENT_POLICY,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "split_manifest_fingerprint": split_manifest_fingerprint(manifest),
        "ticker_count": len(tickers),
        "tickers": sorted(tickers),
        "historical_survivorship_bias_status": (
            "KNOWN_UNCONTROLLED_RETROSPECTIVE_RESEARCH_UNIVERSE"
        ),
        "prospective_survivorship_bias_status": (
            PROSPECTIVE_SURVIVORSHIP_BIAS_STATUS
        ),
        "private_account_data_allowed": False,
    }


def write_prices(path: Path, ticker: str, rows: list[tuple[str, float]]) -> None:
    text = "ticker,date,price\n" + "".join(f"{ticker},{market_date},{price}\n" for market_date, price in rows)
    path.joinpath(f"{ticker}_daily.csv").write_text(text, encoding="utf-8")


def assert_all_finite(testcase: unittest.TestCase, value: object) -> None:
    if isinstance(value, float):
        testcase.assertTrue(math.isfinite(value))
    elif isinstance(value, dict):
        for item in value.values():
            assert_all_finite(testcase, item)
    elif isinstance(value, list):
        for item in value:
            assert_all_finite(testcase, item)


class ValidationSplitTests(unittest.TestCase):
    def test_frozen_split_is_retrospective_and_purged(self) -> None:
        manifest = load_strict_json(SPLIT_PATH)
        result = validate_split_manifest(
            manifest,
            anchor_dates=load_anchor_market_dates(ANCHOR_PATH),
        )
        self.assertEqual(result["historical_status"], "RETROSPECTIVE_CONTAMINATED")
        self.assertEqual(result["contaminated_through_market_date"], "2026-07-10")
        self.assertEqual(result["prospective_starts_strictly_after"], "2026-07-10")
        self.assertEqual(result["maximum_label_span_bars"], 61)
        self.assertEqual(result["split_manifest_fingerprint"], split_manifest_fingerprint(manifest))

    def test_overlap_or_fingerprint_drift_fails(self) -> None:
        manifest = load_strict_json(SPLIT_PATH)
        overlapping = copy.deepcopy(manifest)
        overlapping["historical_partitions"][1]["signal_start_market_date"] = "2016-12-28"
        with self.assertRaises(ValidationSplitError):
            validate_split_manifest(overlapping, enforce_runtime_contract=False)

        drifted = copy.deepcopy(manifest)
        drifted["strategy_fingerprint"] = "0" * 64
        with self.assertRaises(ValidationSplitError):
            validate_split_manifest(drifted)

    def test_full_model_fingerprint_binds_caller_owned_portfolio(self) -> None:
        manifest = load_strict_json(SPLIT_PATH)
        first = full_model_fingerprint(STRATEGY_FINGERPRINT, {"weights_hash": "a"}, manifest)
        same = full_model_fingerprint(STRATEGY_FINGERPRINT, {"weights_hash": "a"}, manifest)
        changed = full_model_fingerprint(STRATEGY_FINGERPRINT, {"weights_hash": "b"}, manifest)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_frozen_prospective_universe_matches_runtime_config(self) -> None:
        manifest = load_strict_json(SPLIT_PATH)
        result = load_and_validate_prospective_universe_contract(
            split_manifest=manifest,
        )
        self.assertEqual(result["ticker_count"], 94)
        self.assertEqual(result["status"], PROSPECTIVE_UNIVERSE_STATUS)
        self.assertEqual(
            result["prospective_survivorship_bias_status"],
            PROSPECTIVE_SURVIVORSHIP_BIAS_STATUS,
        )

    def test_prospective_universe_membership_drift_fails(self) -> None:
        manifest = load_strict_json(SPLIT_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prospective_universe.json"
            contract = prospective_universe_contract(["AAA", "BBB", "CCC"])
            write_json(path, contract)
            with self.assertRaisesRegex(
                ProspectiveUniverseContractError,
                "differs from config.py:TICKERS",
            ):
                load_and_validate_prospective_universe_contract(
                    path,
                    split_manifest=manifest,
                )

            contract = prospective_universe_contract(["AAA", "BBB", "CCC"])
            contract["account_number"] = "must-not-be-public"
            write_json(path, contract)
            with self.assertRaisesRegex(
                ProspectiveUniverseContractError,
                "field membership changed",
            ):
                load_and_validate_prospective_universe_contract(
                    path,
                    split_manifest=manifest,
                    expected_tickers=["AAA", "BBB", "CCC"],
                )

    def test_prospective_universe_policy_drift_fails(self) -> None:
        manifest = load_strict_json(SPLIT_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prospective_universe.json"
            for field in ("selection_basis", "ticker_source", "amendment_policy"):
                contract = prospective_universe_contract(["AAA", "BBB", "CCC"])
                contract[field] = f"{contract[field]} changed"
                write_json(path, contract)
                with self.subTest(field=field), self.assertRaisesRegex(
                    ProspectiveUniverseContractError,
                    rf"{field} drifted",
                ):
                    load_and_validate_prospective_universe_contract(
                        path,
                        split_manifest=manifest,
                        expected_tickers=["AAA", "BBB", "CCC"],
                    )


class ForwardLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet_path = self.root / "decision_packet.json"
        self.report_path = self.root / "market_report.json"
        self.ledger_path = self.root / "forward_signal_ledger.jsonl"
        self.summary_path = self.root / "forward_validation_status.json"
        self.universe_path = self.root / "prospective_universe.json"
        self.expected_tickers = ["AAA", "BBB", "CCC"]
        write_json(
            self.universe_path,
            prospective_universe_contract(self.expected_tickers),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, market_date: str) -> dict:
        return build_forward_ledger(
            split_path=SPLIT_PATH,
            packet_path=self.packet_path,
            report_path=self.report_path,
            ledger_path=self.ledger_path,
            summary_path=self.summary_path,
            prices_dir=self.root,
            anchor_path=ANCHOR_PATH,
            as_of_market_date=date.fromisoformat(market_date),
            universe_contract_path=self.universe_path,
            expected_tickers=self.expected_tickers,
        )

    def test_before_boundary_is_explicitly_not_prospective(self) -> None:
        packet, report = public_inputs("2026-07-10", generated_at="2026-07-11T01:00:00+00:00")
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        summary = self._run("2026-07-10")
        self.assertEqual(summary["status"], "WAITING_FOR_FIRST_PROSPECTIVE_MARKET_DATE")
        self.assertEqual(summary["historical_evidence_classification"], "RETROSPECTIVE_CONTAMINATED")
        self.assertFalse(self.ledger_path.exists())

    def test_all_tickers_idempotency_pending_and_next_close_outcome(self) -> None:
        first_packet, first_report = public_inputs(
            "2026-07-13",
            generated_at="2026-07-14T01:00:00+00:00",
        )
        write_json(self.packet_path, first_packet)
        write_json(self.report_path, first_report)
        initial_rows = [("2026-07-10", 99.0), ("2026-07-13", 100.0)]
        for ticker in ("AAA", "BBB", "CCC"):
            write_prices(self.root, ticker, initial_rows)

        first_summary = self._run("2026-07-13")
        self.assertEqual(
            first_summary["current_market_date_prediction_counts"],
            {"total": 3, "active": 1, "no_signal": 1, "skipped": 1},
        )
        self.assertEqual(first_summary["ledger_counts"]["pending_outcomes"], 8)
        self.assertEqual(first_summary["prospective_universe_ticker_count"], 3)
        self.assertEqual(
            first_summary["prospective_universe_status"],
            PROSPECTIVE_UNIVERSE_STATUS,
        )
        self.assertEqual(first_summary["validated_prediction_cohort_count"], 1)
        first_text = self.ledger_path.read_text(encoding="utf-8")
        first_events = load_ledger(self.ledger_path)
        self.assertEqual(len(first_events), 3)
        self.assertEqual(
            {event["prediction"]["ticker"]: event["prediction"]["state"] for event in first_events},
            {"AAA": "ACTIVE", "BBB": "NO_SIGNAL", "CCC": "SKIPPED"},
        )
        for event in first_events:
            statuses = {row["status"] for row in event["prediction"]["expected_outcomes"]}
            expected = {"NOT_EVALUATED"} if event["prediction"]["state"] == "SKIPPED" else {"PENDING"}
            self.assertEqual(statuses, expected)

        second_summary = self._run("2026-07-13")
        self.assertEqual(second_summary["ledger_counts"]["new_prediction_events"], 0)
        self.assertEqual(self.ledger_path.read_text(encoding="utf-8"), first_text)

        later_packet, later_report = public_inputs(
            "2026-07-21",
            generated_at="2026-07-22T01:00:00+00:00",
        )
        write_json(self.packet_path, later_packet)
        write_json(self.report_path, later_report)
        matured_rows = [
            ("2026-07-10", 99.0),
            ("2026-07-13", 100.0),
            ("2026-07-14", 110.0),
            ("2026-07-15", 111.0),
            ("2026-07-16", 112.0),
            ("2026-07-17", 113.0),
            ("2026-07-20", 114.0),
            ("2026-07-21", 121.0),
        ]
        for ticker in ("AAA", "BBB"):
            write_prices(self.root, ticker, matured_rows)
        write_prices(self.root, "CCC", initial_rows)
        later_summary = self._run("2026-07-21")
        self.assertEqual(later_summary["ledger_counts"]["new_outcome_events"], 2)
        outcomes = [event for event in load_ledger(self.ledger_path) if event["event_type"] == "OUTCOME"]
        old_five_day = [
            event
            for event in outcomes
            if event["outcome"]["observation_market_date"] == "2026-07-13"
            and event["outcome"]["horizon_bars"] == 5
        ]
        self.assertEqual(len(old_five_day), 2)
        for event in old_five_day:
            outcome = event["outcome"]
            self.assertEqual(outcome["entry_market_date"], "2026-07-14")
            self.assertEqual(outcome["exit_market_date"], "2026-07-21")
            self.assertAlmostEqual(outcome["gross_return"], 0.1)
            self.assertAlmostEqual(
                outcome["net_return_after_frozen_costs"],
                net_return_after_round_trip_costs(0.1),
                places=12,
            )
            self.assertEqual(event["market_data"]["source"], SOURCE)
            self.assertEqual(event["market_data"]["market_timezone"], "America/New_York")
            self.assertEqual(event["market_data"]["data_timestamp"], "2026-07-21")
            self.assertEqual(event["strategy_fingerprint"], STRATEGY_FINGERPRINT)

        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line, parse_constant=lambda value: self.fail(value))
            assert_all_finite(self, parsed)
        assert_all_finite(self, json.loads(self.summary_path.read_text(encoding="utf-8")))

    def test_prediction_state_and_outcome_lineage_must_be_consistent(self) -> None:
        source_events = load_ledger(ROOT / "docs" / "forward_signal_ledger.jsonl")
        no_signal = copy.deepcopy(
            next(
                event
                for event in source_events
                if event["event_type"] == "PREDICTION"
                and event["prediction"]["state"] == "NO_SIGNAL"
            )
        )

        forged_state = copy.deepcopy(no_signal)
        forged_state["prediction"]["state"] = "ACTIVE"
        _write_jsonl_atomic(self.ledger_path, [_seal_event(forged_state)])
        with self.assertRaisesRegex(
            ForwardLedgerError,
            "prediction ACTIVE state requires active entry signals",
        ):
            load_ledger(self.ledger_path)

        prediction = no_signal["prediction"]
        metadata = no_signal["market_data"]
        mismatched_outcome = _outcome_event(
            no_signal,
            {
                "ticker": "WRONG",
                "observation_market_date": prediction["observation_market_date"],
                "horizon_bars": 20,
                "status": "MATURED",
                "entry_market_date": "2026-07-14",
                "exit_market_date": "2026-08-11",
                "entry_close": 1.0,
                "exit_close": 1.0,
                "gross_return": 0.0,
                "net_return_after_frozen_costs": net_return_after_round_trip_costs(0.0),
                "return_definition": "close[t+1+h] / close[t+1] - 1",
                "price_basis": prediction["price_basis"],
            },
            metadata=metadata,
            split_result={
                "split_manifest_fingerprint": no_signal[
                    "split_manifest_fingerprint"
                ]
            },
        )
        _write_jsonl_atomic(self.ledger_path, [no_signal, mismatched_outcome])
        with self.assertRaisesRegex(
            ForwardLedgerError,
            "outcome ticker does not match prediction",
        ):
            load_ledger(self.ledger_path)

    def test_complete_same_day_cohort_is_replayed_but_invalid_inputs_fail(self) -> None:
        packet, report = public_inputs("2026-07-13", generated_at="2026-07-14T01:00:00+00:00")
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        rows = [("2026-07-10", 99.0), ("2026-07-13", 100.0)]
        for ticker in ("AAA", "BBB", "CCC"):
            write_prices(self.root, ticker, rows)
        self._run("2026-07-13")
        original_ledger = self.ledger_path.read_text(encoding="utf-8")

        changed_packet, changed_report = public_inputs(
            "2026-07-13",
            generated_at="2026-07-14T02:00:00+00:00",
            active_rule=None,
        )
        changed_packet["data_quality"]["data_status"] = "OK"
        changed_report["data_timestamp_by_ticker"]["CCC"] = "2026-07-13"
        changed_report["update_log"]["CCC"]["latest_date"] = "2026-07-13"
        changed_report["technicals"]["CCC"]["latest_date"] = "2026-07-13"
        write_json(self.packet_path, changed_packet)
        write_json(self.report_path, changed_report)
        replayed = self._run("2026-07-13")
        self.assertEqual(replayed["ledger_counts"]["new_prediction_events"], 0)
        self.assertEqual(self.ledger_path.read_text(encoding="utf-8"), original_ledger)
        replayed_events = load_ledger(self.ledger_path)
        ccc = next(event for event in replayed_events if event["prediction"]["ticker"] == "CCC")
        self.assertEqual(ccc["prediction"]["state"], "SKIPPED")
        self.assertEqual(ccc["prediction"]["ticker_market_date"], "2026-07-10")

        broken_report = copy.deepcopy(report)
        broken_report["strategy_fingerprint"] = "0" * 64
        write_json(self.packet_path, packet)
        write_json(self.report_path, broken_report)
        with self.assertRaisesRegex(ForwardLedgerError, "fingerprint"):
            self._run("2026-07-13")

        future_packet, future_report = public_inputs(
            "2026-07-14",
            generated_at="2026-07-15T01:00:00+00:00",
        )
        write_json(self.packet_path, future_packet)
        write_json(self.report_path, future_report)
        with self.assertRaisesRegex(ForwardLedgerError, "future data"):
            self._run("2026-07-13")

    def test_partial_same_day_prediction_cohort_fails_closed(self) -> None:
        packet, report = public_inputs(
            "2026-07-13",
            generated_at="2026-07-14T01:00:00+00:00",
        )
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        rows = [("2026-07-10", 99.0), ("2026-07-13", 100.0)]
        for ticker in ("AAA", "BBB", "CCC"):
            write_prices(self.root, ticker, rows)
        self._run("2026-07-13")

        lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        self.ledger_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ForwardLedgerError, "frozen prospective universe"):
            self._run("2026-07-13")

    def test_cross_date_ticker_set_drift_fails_closed(self) -> None:
        packet, report = public_inputs(
            "2026-07-13",
            generated_at="2026-07-14T01:00:00+00:00",
        )
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        rows = [("2026-07-10", 99.0), ("2026-07-13", 100.0)]
        for ticker in self.expected_tickers:
            write_prices(self.root, ticker, rows)
        self._run("2026-07-13")

        next_packet, next_report = public_inputs(
            "2026-07-14",
            generated_at="2026-07-15T01:00:00+00:00",
        )
        next_report["data_timestamp_by_ticker"]["DDD"] = next_report[
            "data_timestamp_by_ticker"
        ].pop("CCC")
        next_report["price_basis_by_ticker"]["DDD"] = next_report[
            "price_basis_by_ticker"
        ].pop("CCC")
        next_report["update_log"]["DDD"] = next_report["update_log"].pop("CCC")
        next_report["technicals"]["DDD"] = next_report["technicals"].pop("CCC")
        write_json(self.packet_path, next_packet)
        write_json(self.report_path, next_report)
        with self.assertRaisesRegex(
            ForwardLedgerError,
            "differs from the frozen prospective universe",
        ):
            self._run("2026-07-14")

    def test_prediction_on_or_before_universe_freeze_fails_closed(self) -> None:
        packet, report = public_inputs(
            "2026-07-12",
            generated_at="2026-07-13T01:00:00+00:00",
        )
        write_json(self.packet_path, packet)
        write_json(self.report_path, report)
        rows = [("2026-07-10", 99.0), ("2026-07-12", 100.0)]
        for ticker in self.expected_tickers:
            write_prices(self.root, ticker, rows)
        with self.assertRaisesRegex(
            ForwardLedgerError,
            "not prospective to the frozen universe",
        ):
            self._run("2026-07-12")


if __name__ == "__main__":
    unittest.main()
