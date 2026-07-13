from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts import live_review_contract
from scripts.build_live_review_forward_ledger import (
    LiveReviewForwardLedgerError,
    build_live_review_forward_ledger,
    live_review_due_status,
    load_live_review_ledger,
    validate_live_review_forward_artifacts,
)
from scripts.build_local_ibkr_context import build_private_context
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
)
from scripts.strategy_contract import (
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    net_return_after_round_trip_costs,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "config" / "validation_split.json"
ANCHOR_PATH = ROOT / "docs" / "SPY_daily.csv"
GOVERNANCE = json.loads((ROOT / "docs" / "model_governance.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 7, 14, 1, 0, tzinfo=timezone.utc)
ACCOUNT = "PRIVATE_ACCOUNT_MUST_NOT_PUBLISH"
PRIVATE_HELD_SYMBOL = "PRIVATE1"
SOURCE = "fixture adjusted EOD source"


def _instrument(symbol: str) -> dict:
    return {
        "con_id": 1,
        "symbol": symbol,
        "local_symbol": symbol,
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "multiplier": "",
    }


def _snapshot(*, collected_at: datetime = NOW) -> dict:
    return {
        "schema_version": "v6-ibkr-private-readonly-snapshot-v1",
        "status": "CONNECTED_READ_ONLY_SNAPSHOT",
        "source": "IBKR official TWS API local socket",
        "account_data_timezone": "UTC",
        "collected_at_utc": collected_at.isoformat(),
        "portfolio_price_temporality": "UNKNOWN_TWS_ACCOUNT_UPDATE_NOT_VALIDATED_AS_REALTIME",
        "managed_accounts": [ACCOUNT],
        "positions": [],
        "portfolio": [
            {
                "account_id": ACCOUNT,
                "contract": _instrument(PRIVATE_HELD_SYMBOL),
                "quantity": 2.0,
                "market_price": 100.0,
                "market_value": 200.0,
                "average_cost": 90.0,
                "unrealized_pnl": 20.0,
                "realized_pnl": 0.0,
            }
        ],
        "account_summary": [
            {
                "account_id": ACCOUNT,
                "tag": "NetLiquidation",
                "value": "1000",
                "currency": "USD",
            }
        ],
        "account_values": [],
        "account_readiness": {ACCOUNT: True},
        "automatic_order_allowed": False,
        "order_methods_invoked": [],
        "privacy": {
            "classification": "PRIVATE_LOCAL_ONLY",
            "git_publish_allowed": False,
            "contains_account_ids": True,
            "contains_exact_positions": True,
        },
    }


def _public_inputs(
    market_date: str,
    *,
    action: str = "BUY_CANDIDATE_REVIEW_REQUIRED",
) -> tuple[dict, dict]:
    generated_at = datetime.combine(
        date.fromisoformat(market_date) + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ).isoformat()
    candidate_rows = [{"ticker": "QQQ"}] if action == "BUY_CANDIDATE_REVIEW_REQUIRED" else []
    permission = {
        "BUY_CANDIDATE_REVIEW_REQUIRED": "CHATGPT_LIVE_REVIEW_REQUIRED",
        "RISK_REVIEW_REQUIRED": "BLOCKED_BY_MODEL_PORTFOLIO_RISK",
        "NO_TRADE": "NO_QUANT_CANDIDATE",
    }[action]
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
        "universe": {"expected_latest_market_date": market_date},
        "price_basis_by_ticker": {"QQQ": "adjusted"},
    }
    packet = {
        "schema_version": "decision-packet-v5.0",
        "market_data": {
            "source": SOURCE,
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
        "decision": {
            "final_action": action,
            "buy_permission": permission,
            "automatic_order_allowed": False,
        },
        "candidates": {
            "actionable_count": len(candidate_rows),
            "top_actionable": candidate_rows,
        },
    }
    return packet, report


def _request(packet: dict, *, generated_at: datetime = NOW) -> tuple[dict, dict]:
    context = build_private_context(
        _snapshot(collected_at=generated_at),
        packet,
        GOVERNANCE,
        generated_at=generated_at,
    )
    return context, live_review_contract.build_live_review_request(
        context, packet, GOVERNANCE, generated_at=generated_at
    )


def _source(check_id: str, *, generated_at: datetime) -> dict:
    if check_id == "actual_account_risk":
        source_type = "PRIVATE_CONTEXT"
        data_kind = "PRIVATE_ACCOUNT_SNAPSHOT"
        observed = generated_at
    elif check_id == "live_quote":
        source_type = "IBKR_TWS"
        data_kind = "INTRADAY_REALTIME_BID_ASK"
        observed = generated_at + timedelta(seconds=10)
    elif check_id == "execution_feasibility":
        source_type = "IBKR_TWS"
        data_kind = "INTRADAY_REALTIME_BID_ASK"
        observed = generated_at + timedelta(seconds=10)
    elif check_id == "fresh_news_macro":
        source_type = "PUBLIC_NEWS"
        data_kind = "NEWS_OR_MACRO"
        observed = generated_at + timedelta(seconds=10)
    elif check_id == "earnings_valuation":
        source_type = "PUBLIC_FILINGS"
        data_kind = "FUNDAMENTAL_OR_VALUATION"
        observed = generated_at + timedelta(seconds=10)
    else:
        source_type = "PUBLIC_MARKET_DATA"
        data_kind = "END_OF_DAY_ADJUSTED"
        observed = generated_at + timedelta(seconds=10)
    return {
        "source_name": f"SECRET SOURCE {check_id}",
        "source_type": source_type,
        "source_url": (
            f"https://example.com/{check_id}" if source_type.startswith("PUBLIC_") else None
        ),
        "observed_at_utc": observed.isoformat(),
        "market_timezone": "America/New_York",
        "data_kind": data_kind,
    }


def _response(
    request: dict,
    *,
    action: str,
    symbol: str | None,
    generated_at: datetime = NOW,
) -> dict:
    return {
        "schema_version": live_review_contract.RESPONSE_SCHEMA_VERSION,
        "status": "CHATGPT_LIVE_REVIEW_COMPLETED",
        "request_id": request["request_id"],
        "input_context_sha256": request["source_context"]["context_sha256"],
        "reviewed_at_utc": (generated_at + timedelta(seconds=30)).isoformat(),
        "reviewer": {
            "system": "CODEX",
            "model_id": "PRIVATE_MODEL_LABEL_MUST_NOT_PUBLISH",
        },
        "privacy": {
            "classification": "PRIVATE_LOCAL_ONLY",
            "git_publish_allowed": False,
            "contains_private_account_reasoning": True,
        },
        "checks": [
            {
                "check_id": check_id,
                "status": "PASS",
                "finding": f"SECRET FINDING {check_id}",
                "evidence_sources": [_source(check_id, generated_at=generated_at)],
            }
            for check_id in live_review_contract.REQUIRED_CHECK_IDS
        ],
        "conclusion": {
            "action": action,
            "symbol": symbol,
            "confidence": "MEDIUM",
            "reason_codes": ["PRIVATE_REASON_MUST_NOT_PUBLISH"],
            "summary": "SECRET PRIVATE SUMMARY MUST NOT PUBLISH",
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "order_payload": None,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _write_prices(path: Path, dates: list[str]) -> None:
    rows = ["ticker,date,close,adjClose,price\n"]
    for index, market_date in enumerate(dates):
        price = 100.0 + index
        rows.append(f"QQQ,{market_date},{price},{price},{price}\n")
    (path / "QQQ_daily.csv").write_text("".join(rows), encoding="utf-8")


class LiveReviewForwardLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet_path = self.root / "decision_packet.json"
        self.report_path = self.root / "market_report.json"
        self.governance_path = self.root / "model_governance.json"
        self.ledger_path = self.root / "live_review_forward_ledger.jsonl"
        self.summary_path = self.root / "live_review_forward_status.json"
        _write_json(self.governance_path, GOVERNANCE)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_public(self, market_date: str, *, action: str = "BUY_CANDIDATE_REVIEW_REQUIRED") -> dict:
        packet, report = _public_inputs(market_date, action=action)
        _write_json(self.packet_path, packet)
        _write_json(self.report_path, report)
        return packet

    def _run(
        self,
        *,
        context: dict | None = None,
        request: dict | None = None,
        response: dict | None = None,
        as_of: str,
        now: datetime,
    ) -> dict:
        return build_live_review_forward_ledger(
            context=context,
            request=request,
            response=response,
            packet_path=self.packet_path,
            report_path=self.report_path,
            governance_path=self.governance_path,
            split_path=SPLIT_PATH,
            ledger_path=self.ledger_path,
            summary_path=self.summary_path,
            prices_dir=self.root,
            anchor_path=ANCHOR_PATH,
            as_of_market_date=date.fromisoformat(as_of),
            validation_now=now,
        )

    def test_buy_review_is_sanitized_idempotent_and_matures_next_close(self) -> None:
        packet = self._write_public("2026-07-13")
        _write_prices(self.root, ["2026-07-13"])
        context, request = _request(packet)
        response = _response(request, action="BUY_REVIEW", symbol="QQQ")

        first = self._run(
            context=context,
            request=request,
            response=response,
            as_of="2026-07-13",
            now=NOW + timedelta(seconds=30),
        )
        self.assertEqual(first["ledger_counts"]["new_prediction_events"], 1)
        self.assertEqual(first["ledger_counts"]["pending_outcomes"], 1)
        first_text = self.ledger_path.read_text(encoding="utf-8")
        for secret in (
            ACCOUNT,
            PRIVATE_HELD_SYMBOL,
            request["request_id"],
            request["source_context"]["context_sha256"],
            "PRIVATE_MODEL_LABEL_MUST_NOT_PUBLISH",
            "SECRET SOURCE",
            "SECRET FINDING",
            "SECRET PRIVATE SUMMARY",
            "PRIVATE_REASON_MUST_NOT_PUBLISH",
        ):
            self.assertNotIn(secret, first_text)

        same = self._run(
            context=context,
            request=request,
            response=response,
            as_of="2026-07-13",
            now=NOW + timedelta(seconds=30),
        )
        self.assertEqual(same["ledger_counts"]["new_prediction_events"], 0)
        self.assertEqual(self.ledger_path.read_text(encoding="utf-8"), first_text)

        dates = [value.date().isoformat() for value in pd.bdate_range("2026-07-13", periods=22)]
        exit_date = dates[-1]
        self._write_public(exit_date)
        _write_prices(self.root, dates)
        matured = self._run(
            as_of=exit_date,
            now=datetime.combine(
                date.fromisoformat(exit_date) + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ),
        )
        self.assertEqual(matured["ledger_counts"]["new_outcome_events"], 1)
        events = load_live_review_ledger(self.ledger_path)
        outcome = [event for event in events if event["event_type"] == "LIVE_REVIEW_OUTCOME"][0]
        self.assertEqual(outcome["outcome"]["entry_market_date"], dates[1])
        self.assertEqual(outcome["outcome"]["exit_market_date"], dates[21])
        expected_gross = 121.0 / 101.0 - 1.0
        self.assertAlmostEqual(outcome["outcome"]["gross_counterfactual_return"], expected_gross)
        self.assertAlmostEqual(
            outcome["outcome"]["net_counterfactual_return_after_frozen_costs"],
            net_return_after_round_trip_costs(expected_gross),
        )
        self.assertEqual(outcome["outcome"]["close_path_max_adverse_excursion"], 0.0)
        self.assertIn("NO_EXECUTION_CLAIM", outcome["outcome"]["interpretation"])
        validated = validate_live_review_forward_artifacts(
            ledger_path=self.ledger_path,
            summary_path=self.summary_path,
            report_path=self.report_path,
            split_path=SPLIT_PATH,
        )
        self.assertEqual(validated["outcome_event_count"], 1)

    def test_one_live_review_cohort_per_market_date(self) -> None:
        packet = self._write_public("2026-07-13", action="NO_TRADE")
        _write_prices(self.root, ["2026-07-13"])
        before = live_review_due_status(
            packet_path=self.packet_path,
            ledger_path=self.ledger_path,
        )
        self.assertTrue(before["review_due"])

        context, request = _request(packet)
        response = _response(request, action="NO_TRADE", symbol=None)
        first = self._run(
            context=context,
            request=request,
            response=response,
            as_of="2026-07-13",
            now=NOW + timedelta(seconds=30),
        )
        self.assertEqual(first["ledger_counts"]["new_prediction_events"], 1)
        after = live_review_due_status(
            packet_path=self.packet_path,
            ledger_path=self.ledger_path,
        )
        self.assertFalse(after["review_due"])

        later = NOW + timedelta(seconds=60)
        second_context, second_request = _request(packet, generated_at=later)
        second_response = _response(
            second_request,
            action="NO_TRADE",
            symbol=None,
            generated_at=later,
        )
        with self.assertRaisesRegex(
            LiveReviewForwardLedgerError,
            "cohort already exists",
        ):
            self._run(
                context=second_context,
                request=second_request,
                response=second_response,
                as_of="2026-07-13",
                now=later + timedelta(seconds=30),
            )

        changed_packet = self._write_public(
            "2026-07-13",
            action="RISK_REVIEW_REQUIRED",
        )
        self.assertNotEqual(packet, changed_packet)
        with self.assertRaisesRegex(
            LiveReviewForwardLedgerError,
            "different public decision evidence",
        ):
            live_review_due_status(
                packet_path=self.packet_path,
                ledger_path=self.ledger_path,
            )

    def test_later_overlapping_review_is_recorded_but_not_counted_as_independent(self) -> None:
        packet = self._write_public("2026-07-13")
        _write_prices(self.root, ["2026-07-13"])
        first_context, first_request = _request(packet)
        first_response = _response(first_request, action="BUY_REVIEW", symbol="QQQ")
        self._run(
            context=first_context,
            request=first_request,
            response=first_response,
            as_of="2026-07-13",
            now=NOW + timedelta(seconds=30),
        )

        second_packet = self._write_public("2026-07-14")
        _write_prices(self.root, ["2026-07-13", "2026-07-14"])
        later = datetime(2026, 7, 15, 1, 0, tzinfo=timezone.utc)
        second_context, second_request = _request(second_packet, generated_at=later)
        second_response = _response(
            second_request,
            action="WAIT",
            symbol=None,
            generated_at=later,
        )
        summary = self._run(
            context=second_context,
            request=second_request,
            response=second_response,
            as_of="2026-07-14",
            now=later + timedelta(seconds=30),
        )
        self.assertEqual(summary["ledger_counts"]["prediction_events"], 2)
        self.assertEqual(summary["ledger_counts"]["evaluable_prediction_events"], 1)
        statuses = [
            event["prediction"]["evaluation_status"]
            for event in load_live_review_ledger(self.ledger_path)
            if event["event_type"] == "LIVE_REVIEW_PREDICTION"
        ]
        self.assertEqual(statuses, ["PENDING", "NOT_EVALUATED_OVERLAPPING_WINDOW"])

    def test_private_reduce_symbol_is_redacted_and_not_evaluated(self) -> None:
        packet = self._write_public("2026-07-13", action="RISK_REVIEW_REQUIRED")
        context, request = _request(packet)
        response = _response(
            request,
            action="REDUCE_REVIEW",
            symbol=PRIVATE_HELD_SYMBOL,
        )
        summary = self._run(
            context=context,
            request=request,
            response=response,
            as_of="2026-07-13",
            now=NOW + timedelta(seconds=30),
        )
        event = load_live_review_ledger(self.ledger_path)[0]
        self.assertIsNone(event["prediction"]["public_symbol"])
        self.assertEqual(
            event["prediction"]["evaluation_status"],
            "NOT_EVALUATED_PRIVATE_REDUCTION_REDACTED",
        )
        self.assertNotIn(PRIVATE_HELD_SYMBOL, self.ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["ledger_counts"]["evaluable_prediction_events"], 0)
        self.assertEqual(summary["ledger_counts"]["matured_outcome_events"], 0)

    def test_private_context_is_mandatory_and_bound_to_the_request(self) -> None:
        packet = self._write_public("2026-07-13")
        _write_prices(self.root, ["2026-07-13"])
        context, request = _request(packet)
        response = _response(request, action="BUY_REVIEW", symbol="QQQ")
        with self.assertRaisesRegex(
            LiveReviewForwardLedgerError,
            "context, request, and response",
        ):
            self._run(
                request=request,
                response=response,
                as_of="2026-07-13",
                now=NOW + timedelta(seconds=30),
            )

        mismatched = copy.deepcopy(context)
        mismatched["generated_at_utc"] = (NOW + timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(
            LiveReviewForwardLedgerError,
            "does not match",
        ):
            self._run(
                context=mismatched,
                request=request,
                response=response,
                as_of="2026-07-13",
                now=NOW + timedelta(seconds=30),
            )

    def test_tampering_and_future_price_rows_fail_visibly(self) -> None:
        packet = self._write_public("2026-07-13")
        _write_prices(self.root, ["2026-07-13"])
        context, request = _request(packet)
        response = _response(request, action="BUY_REVIEW", symbol="QQQ")
        self._run(
            context=context,
            request=request,
            response=response,
            as_of="2026-07-13",
            now=NOW + timedelta(seconds=30),
        )
        event = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        event["prediction"]["confidence"] = "HIGH"
        self.ledger_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(LiveReviewForwardLedgerError, "hash mismatch"):
            load_live_review_ledger(self.ledger_path)

        self.ledger_path.unlink()
        _write_prices(self.root, ["2026-07-13", "2026-07-14"])
        with self.assertRaisesRegex(LiveReviewForwardLedgerError, "future rows"):
            self._run(
                context=context,
                request=request,
                response=response,
                as_of="2026-07-13",
                now=NOW + timedelta(seconds=30),
            )


if __name__ == "__main__":
    unittest.main()
