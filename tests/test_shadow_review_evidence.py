from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from scripts.build_decision_report import compile_decision_outputs
from scripts.build_shadow_review_forward_ledger import (
    ShadowReviewForwardLedgerError,
    build_shadow_review_forward_ledger,
    validate_shadow_review_forward_artifacts,
)
from scripts.shadow_review_contract import (
    REQUIRED_CHECK_IDS,
    ShadowReviewContractError,
    build_shadow_review_request,
    validate_shadow_review_response,
)
from tests.test_decision_contract import portfolio, report


OBSERVATION = date(2026, 7, 15)


def _weekday_dates(start: date, count: int) -> list[date]:
    dates: list[date] = []
    cursor = start
    while len(dates) < count:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor = cursor.fromordinal(cursor.toordinal() + 1)
    return dates


def _evidence(market_date: date) -> tuple[dict, dict]:
    market = market_date.isoformat()
    generated = datetime.combine(market_date, time(23, 0), tzinfo=timezone.utc)
    market_report = report(lrcx_date=market, arm_date=market, qqq_date=market)
    market_report["generated_at_utc"] = generated.isoformat()
    market_report["data_timestamp"] = market
    market_report["universe"]["expected_latest_market_date"] = market
    for technical in market_report["technicals"].values():
        technical["latest_date"] = market
    for backtest in market_report["backtests"].values():
        backtest["latest_date"] = market
    market_report["backtests"]["LRCX"]["relative_strength_rebound"]["20d"].pop(
        "worst_mae"
    )
    market_report["price_basis_by_ticker"] = {
        ticker: "adjusted" for ticker in market_report["technicals"]
    }
    outputs = compile_decision_outputs(
        market_report,
        portfolio(),
        generated_at_utc=generated.isoformat(),
        as_of=generated,
    )
    packet = outputs["decision_packet"]
    assert packet["decision"]["final_action"] == "NO_TRADE"
    return packet, market_report


def _response(request: dict, *, assessment: str = "BUY_REVIEW") -> dict:
    reviewed = datetime.fromisoformat(request["generated_at_utc"]) + timedelta(minutes=5)
    return {
        "schema_version": "shadow-review-response-v1",
        "request_id": request["request_id"],
        "reviewed_at_utc": reviewed.isoformat(),
        "reviewer": {"system": "CHATGPT", "model_id": "fixture-shadow-reviewer"},
        "assessments": [
            {
                "candidate_id": candidate["candidate_id"],
                "symbol": candidate["symbol"],
                "rule": candidate["rule"],
                "assessment": assessment,
                "confidence": "MEDIUM",
                "checks": [
                    {"check_id": check_id, "status": "PASS"}
                    for check_id in REQUIRED_CHECK_IDS
                ],
            }
            for candidate in request["candidates"]
        ],
        "execution_impact": "NONE_COUNTERFACTUAL_ONLY",
        "counterfactual_only": True,
        "execution_eligible": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_prices(path: Path, dates: list[date], prices: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "price"])
        writer.writeheader()
        for market_date, price in zip(dates, prices):
            writer.writerow({"date": market_date.isoformat(), "price": price})


class ShadowReviewContractTests(unittest.TestCase):
    def test_no_trade_can_collect_complete_non_executable_shadow_assessment(self) -> None:
        packet, _ = _evidence(OBSERVATION)
        now = datetime(2026, 7, 15, 23, 5, tzinfo=timezone.utc)
        request = build_shadow_review_request(packet, now=now)
        response = _response(request)

        validate_shadow_review_response(request, response, now=now.replace(minute=15))

        self.assertEqual(request["quant_final_action"], "NO_TRADE")
        self.assertTrue(request["counterfactual_only"])
        self.assertFalse(request["execution_eligible"])
        self.assertFalse(request["automatic_order_allowed"])
        self.assertEqual(response["assessments"][0]["assessment"], "BUY_REVIEW")
        self.assertEqual(response["execution_impact"], "NONE_COUNTERFACTUAL_ONLY")

    def test_missing_candidate_or_failed_buy_check_rejects_entire_cohort(self) -> None:
        packet, _ = _evidence(OBSERVATION)
        now = datetime(2026, 7, 15, 23, 5, tzinfo=timezone.utc)
        request = build_shadow_review_request(packet, now=now)
        response = _response(request)
        missing = copy.deepcopy(response)
        missing["assessments"] = []
        with self.assertRaises(ShadowReviewContractError):
            validate_shadow_review_response(request, missing, now=now.replace(minute=15))

        failed = copy.deepcopy(response)
        failed["assessments"][0]["checks"][0]["status"] = "UNAVAILABLE"
        with self.assertRaisesRegex(ShadowReviewContractError, "BUY_REVIEW"):
            validate_shadow_review_response(request, failed, now=now.replace(minute=15))

    def test_historical_backfill_is_rejected(self) -> None:
        packet, _ = _evidence(OBSERVATION)
        retrospective = datetime(2026, 7, 20, 23, 0, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ShadowReviewContractError, "backfill"):
            build_shadow_review_request(packet, now=retrospective)


class ShadowReviewForwardLedgerTests(unittest.TestCase):
    def test_complete_cohort_matures_with_benchmark_and_no_trade_gate(self) -> None:
        dates = _weekday_dates(OBSERVATION, 22)
        exit_date = dates[-1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_path = root / "decision_packet.json"
            report_path = root / "market_report.json"
            ledger_path = root / "shadow_review_forward_ledger.jsonl"
            summary_path = root / "shadow_review_forward_status.json"
            packet, market_report = _evidence(OBSERVATION)
            _write_json(packet_path, packet)
            _write_json(report_path, market_report)
            _write_prices(root / "LRCX_daily.csv", [OBSERVATION], [100.0])
            _write_prices(root / "SMH_daily.csv", [OBSERVATION], [200.0])
            request_now = datetime(2026, 7, 15, 23, 5, tzinfo=timezone.utc)
            request = build_shadow_review_request(packet, now=request_now)
            response = _response(request)

            pending = build_shadow_review_forward_ledger(
                request=request,
                response=response,
                packet_path=packet_path,
                report_path=report_path,
                ledger_path=ledger_path,
                summary_path=summary_path,
                prices_dir=root,
                as_of_market_date=OBSERVATION,
                validation_now=request_now.replace(minute=15),
            )
            self.assertEqual(pending["ledger_counts"]["prediction_events"], 1)
            self.assertEqual(pending["ledger_counts"]["pending_outcomes"], 1)
            self.assertFalse(pending["real_trade_count_used_as_release_prerequisite"])

            matured_packet, matured_report = _evidence(exit_date)
            _write_json(packet_path, matured_packet)
            _write_json(report_path, matured_report)
            _write_prices(
                root / "LRCX_daily.csv",
                dates,
                [100.0 + index for index in range(len(dates))],
            )
            _write_prices(
                root / "SMH_daily.csv",
                dates,
                [200.0 + index * 0.5 for index in range(len(dates))],
            )
            matured = build_shadow_review_forward_ledger(
                packet_path=packet_path,
                report_path=report_path,
                ledger_path=ledger_path,
                summary_path=summary_path,
                prices_dir=root,
                as_of_market_date=exit_date,
            )
            self.assertEqual(matured["ledger_counts"]["matured_outcome_events"], 1)
            self.assertEqual(matured["ledger_counts"]["pending_outcomes"], 0)
            self.assertEqual(matured["matured_metrics"]["buy_review"]["sample_count"], 1)
            self.assertIsNotNone(
                matured["matured_metrics"]["buy_review"][
                    "mean_net_return_after_costs_minus_benchmark"
                ]
            )
            validated = validate_shadow_review_forward_artifacts(
                ledger_path=ledger_path,
                summary_path=summary_path,
                report_path=report_path,
            )
            self.assertEqual(validated["status"], "VALID")

    def test_partial_persisted_cohort_is_rejected_as_cherry_picking(self) -> None:
        packet, market_report = _evidence(OBSERVATION)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_path = root / "packet.json"
            report_path = root / "report.json"
            ledger_path = root / "ledger.jsonl"
            summary_path = root / "summary.json"
            _write_json(packet_path, packet)
            _write_json(report_path, market_report)
            _write_prices(root / "LRCX_daily.csv", [OBSERVATION], [100.0])
            _write_prices(root / "SMH_daily.csv", [OBSERVATION], [200.0])
            now = datetime(2026, 7, 15, 23, 5, tzinfo=timezone.utc)
            request = build_shadow_review_request(packet, now=now)
            response = _response(request, assessment="WAIT")
            build_shadow_review_forward_ledger(
                request=request,
                response=response,
                packet_path=packet_path,
                report_path=report_path,
                ledger_path=ledger_path,
                summary_path=summary_path,
                prices_dir=root,
                as_of_market_date=OBSERVATION,
                validation_now=now.replace(minute=15),
            )
            row = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
            row["prediction"]["cohort_candidate_count"] = 2
            from scripts.build_forward_ledger import _seal_event

            row.pop("immutable_payload_hash")
            row = _seal_event(row)
            ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ShadowReviewForwardLedgerError, "cherry-picking"):
                validate_shadow_review_forward_artifacts(
                    ledger_path=ledger_path,
                    summary_path=summary_path,
                    report_path=report_path,
                )


if __name__ == "__main__":
    unittest.main()
