from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import live_review_contract as contract
from scripts.build_local_ibkr_context import build_private_context


NOW = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)
ACCOUNT = "PRIVATE_ACCOUNT_FIXTURE"


def _instrument(symbol: str, con_id: int) -> dict:
    return {
        "con_id": con_id,
        "symbol": symbol,
        "local_symbol": symbol,
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "multiplier": "",
    }


def _snapshot() -> dict:
    return {
        "schema_version": "v6-ibkr-private-readonly-snapshot-v1",
        "status": "CONNECTED_READ_ONLY_SNAPSHOT",
        "source": "IBKR official TWS API local socket",
        "account_data_timezone": "UTC",
        "collected_at_utc": NOW.isoformat(),
        "portfolio_price_temporality": "UNKNOWN_TWS_ACCOUNT_UPDATE_NOT_VALIDATED_AS_REALTIME",
        "managed_accounts": [ACCOUNT],
        "positions": [],
        "portfolio": [
            {
                "account_id": ACCOUNT,
                "contract": _instrument("QQQ", 1),
                "quantity": 2.0,
                "market_price": 520.0,
                "market_value": 1040.0,
                "average_cost": 500.0,
                "unrealized_pnl": 40.0,
                "realized_pnl": 0.0,
            },
            {
                "account_id": ACCOUNT,
                "contract": _instrument("OTHER", 2),
                "quantity": 1.0,
                "market_price": 100.0,
                "market_value": 100.0,
                "average_cost": 90.0,
                "unrealized_pnl": 10.0,
                "realized_pnl": 0.0,
            },
        ],
        "account_summary": [
            {
                "account_id": ACCOUNT,
                "tag": "NetLiquidation",
                "value": "2000",
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


def _decision(action: str = "BUY_CANDIDATE_REVIEW_REQUIRED") -> dict:
    candidate_rows = [{"ticker": "QQQ"}] if action == "BUY_CANDIDATE_REVIEW_REQUIRED" else []
    permissions = {
        "BUY_CANDIDATE_REVIEW_REQUIRED": "CHATGPT_LIVE_REVIEW_REQUIRED",
        "RISK_REVIEW_REQUIRED": "BLOCKED_BY_MODEL_PORTFOLIO_RISK",
        "NO_TRADE": "NO_QUANT_CANDIDATE",
        "DATA_REVIEW_REQUIRED": "BLOCKED_BY_MODEL_PORTFOLIO_DATA",
    }
    return {
        "schema_version": "decision-packet-v5.1",
        "market_data": {
            "source": "fixture EOD source",
            "market_timezone": "America/New_York",
            "report_generated_at_utc": NOW.isoformat(),
            "data_timestamp": "2026-07-10",
            "price_frequency": "end_of_day_daily",
            "price_adjustment_policy": "adjusted",
            "metadata_status": "COMPLETE",
        },
        "decision": {
            "final_action": action,
            "buy_permission": permissions[action],
            "automatic_order_allowed": False,
        },
        "candidates": {
            "execution": {
                "candidate_count": len(candidate_rows),
                "top": candidate_rows,
            },
        },
    }


def _governance() -> dict:
    return {
        "status": "WAITING_FOR_FIRST_PROSPECTIVE_MARKET_DATE",
        "model_governance_fingerprint": "a" * 64,
        "current_market_regime": {"regime": "base"},
        "allocation_decision": {
            "model_allocations": {"balanced_incumbent_v1": 1.0},
            "automatic_order_allowed": False,
        },
        "effective_target_weights": {"QQQ": 0.3, "CASH_PROXY": 0.7},
        "automatic_order_allowed": False,
    }


def _context(decision: dict, governance: dict) -> dict:
    return build_private_context(
        _snapshot(),
        decision,
        governance,
        generated_at=NOW,
    )


def _request(action: str = "BUY_CANDIDATE_REVIEW_REQUIRED") -> dict:
    decision = _decision(action)
    governance = _governance()
    return contract.build_live_review_request(
        _context(decision, governance),
        decision,
        governance,
        generated_at=NOW,
    )


def _source(check_id: str, *, delayed: bool = False) -> dict:
    if check_id == "live_quote":
        source_type = "IBKR_TWS"
        data_kind = "INTRADAY_DELAYED" if delayed else "INTRADAY_REALTIME_BID_ASK"
    elif check_id == "actual_account_risk":
        source_type = "PRIVATE_CONTEXT"
        data_kind = "PRIVATE_ACCOUNT_SNAPSHOT"
    elif check_id == "fresh_news_macro":
        source_type = "PUBLIC_NEWS"
        data_kind = "NEWS_OR_MACRO"
    elif check_id == "earnings_valuation":
        source_type = "PUBLIC_FILINGS"
        data_kind = "FUNDAMENTAL_OR_VALUATION"
    elif check_id == "execution_feasibility":
        source_type = "IBKR_TWS"
        data_kind = "INTRADAY_REALTIME_BID_ASK"
    else:
        source_type = "PUBLIC_MARKET_DATA"
        data_kind = "INTRADAY_DELAYED"
    return {
        "source_name": f"fixture {check_id}",
        "source_type": source_type,
        "source_url": (
            f"https://example.com/{check_id}" if source_type.startswith("PUBLIC_") else None
        ),
        "observed_at_utc": (
            NOW if check_id == "actual_account_risk" else NOW + timedelta(seconds=10)
        ).isoformat(),
        "market_timezone": "America/New_York",
        "data_kind": data_kind,
    }


def _response(
    request: dict,
    *,
    action: str = "BUY_REVIEW",
    symbol: str | None = "QQQ",
) -> dict:
    return {
        "schema_version": contract.RESPONSE_SCHEMA_VERSION,
        "status": "CHATGPT_LIVE_REVIEW_COMPLETED",
        "request_id": request["request_id"],
        "input_context_sha256": request["source_context"]["context_sha256"],
        "reviewed_at_utc": (NOW + timedelta(seconds=30)).isoformat(),
        "reviewer": {"system": "CODEX", "model_id": "fixture-model"},
        "privacy": {
            "classification": "PRIVATE_LOCAL_ONLY",
            "git_publish_allowed": False,
            "contains_private_account_reasoning": True,
        },
        "checks": [
            {
                "check_id": check_id,
                "status": "PASS",
                "finding": f"fixture {check_id} passed",
                "evidence_sources": [_source(check_id)],
            }
            for check_id in contract.REQUIRED_CHECK_IDS
        ],
        "conclusion": {
            "action": action,
            "symbol": symbol,
            "confidence": "MEDIUM",
            "reason_codes": ["FIXTURE_PASS"],
            "summary": "Fixture-only live review result.",
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "order_payload": None,
    }


class LiveReviewContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (contract.ROOT / "private").mkdir(parents=True, exist_ok=True)

    def test_request_binds_private_context_and_frozen_quant_boundary(self) -> None:
        request = _request()
        self.assertEqual(request["candidate_symbols"], ["QQQ"])
        self.assertEqual(request["held_symbols"], ["OTHER", "QQQ"])
        self.assertIn("BUY_REVIEW", request["quant_boundary"]["allowed_live_conclusions"])
        self.assertFalse(request["automatic_order_allowed"])
        self.assertIsNone(request["order_payload"])
        contract.validate_live_review_request(request, now=NOW)

        mutated = copy.deepcopy(request)
        mutated["quant_boundary"]["allowed_live_conclusions"].append("NO_TRADE")
        with self.assertRaisesRegex(contract.LiveReviewError, "schema|fingerprint"):
            contract.validate_live_review_request(mutated, now=NOW)

    def test_no_trade_cannot_be_upgraded_to_buy(self) -> None:
        request = _request("NO_TRADE")
        self.assertNotIn("BUY_REVIEW", request["quant_boundary"]["allowed_live_conclusions"])
        response = _response(request)
        with self.assertRaisesRegex(contract.LiveReviewError, "quant boundary"):
            contract.validate_live_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )

        forged = copy.deepcopy(request)
        forged["candidate_symbols"] = ["QQQ"]
        forged["quant_boundary"]["maximum_live_action"] = "BUY_REVIEW_ONLY"
        forged["quant_boundary"]["allowed_live_conclusions"] = [
            "NO_TRADE",
            "WAIT",
            "BUY_REVIEW",
        ]
        forged["request_id"] = contract._canonical_hash(
            contract._request_identity_payload(forged)
        )
        with self.assertRaisesRegex(contract.LiveReviewError, "quant boundary|maximum action"):
            contract.validate_live_review_request(forged, now=NOW)

        forged_date = copy.deepcopy(request)
        forged_date["source_context"]["github"]["data_timestamp"] = "2026-07-09"
        forged_date["request_id"] = contract._canonical_hash(
            contract._request_identity_payload(forged_date)
        )
        with self.assertRaisesRegex(contract.LiveReviewError, "public GitHub EOD evidence"):
            contract.validate_live_review_request(forged_date, now=NOW)

    def test_buy_review_requires_all_checks_realtime_and_candidate(self) -> None:
        request = _request()
        response = _response(request)
        validated = contract.validate_live_review_response(
            request,
            response,
            now=NOW + timedelta(seconds=30),
        )
        self.assertEqual(validated["conclusion"]["action"], "BUY_REVIEW")

        delayed = copy.deepcopy(response)
        delayed["checks"][0]["evidence_sources"] = [_source("live_quote", delayed=True)]
        with self.assertRaisesRegex(contract.LiveReviewError, "real-time"):
            contract.validate_live_review_response(
                request,
                delayed,
                now=NOW + timedelta(seconds=30),
            )
        old_quote = copy.deepcopy(response)
        old_quote["checks"][0]["evidence_sources"][0]["observed_at_utc"] = (
            NOW - timedelta(seconds=1)
        ).isoformat()
        with self.assertRaisesRegex(contract.LiveReviewError, "request window"):
            contract.validate_live_review_response(
                request,
                old_quote,
                now=NOW + timedelta(seconds=30),
            )
        wrong_symbol = copy.deepcopy(response)
        wrong_symbol["conclusion"]["symbol"] = "AAPL"
        with self.assertRaisesRegex(contract.LiveReviewError, "current GitHub candidate"):
            contract.validate_live_review_response(
                request,
                wrong_symbol,
                now=NOW + timedelta(seconds=30),
            )

    def test_failed_or_missing_check_forces_wait(self) -> None:
        request = _request("RISK_REVIEW_REQUIRED")
        response = _response(request, action="REDUCE_REVIEW", symbol="QQQ")
        response["checks"][2]["status"] = "UNAVAILABLE"
        response["checks"][2]["evidence_sources"] = []
        with self.assertRaisesRegex(contract.LiveReviewError, "require NO_TRADE or WAIT"):
            contract.validate_live_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )
        response["conclusion"]["action"] = "WAIT"
        response["conclusion"]["symbol"] = None
        contract.validate_live_review_response(
            request,
            response,
            now=NOW + timedelta(seconds=30),
        )

        missing = copy.deepcopy(response)
        missing["checks"].pop()
        with self.assertRaisesRegex(contract.LiveReviewError, "schema|each required check"):
            contract.validate_live_review_response(
                request,
                missing,
                now=NOW + timedelta(seconds=30),
            )

    def test_reduce_review_requires_a_private_held_symbol(self) -> None:
        request = _request("RISK_REVIEW_REQUIRED")
        response = _response(request, action="REDUCE_REVIEW", symbol="OTHER")
        contract.validate_live_review_response(
            request,
            response,
            now=NOW + timedelta(seconds=30),
        )
        response["conclusion"]["symbol"] = "AAPL"
        with self.assertRaisesRegex(contract.LiveReviewError, "private holdings"):
            contract.validate_live_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )

    def test_stale_future_mismatched_or_public_output_fails(self) -> None:
        request = _request()
        response = _response(request)
        with self.assertRaisesRegex(contract.LiveReviewError, "expired"):
            contract.validate_live_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=301),
            )
        decision = _decision()
        governance = _governance()
        with self.assertRaisesRegex(contract.LiveReviewError, "public GitHub EOD evidence"):
            contract.build_live_review_request(
                _context(decision, governance),
                decision,
                governance,
                generated_at=NOW + timedelta(days=1),
                max_account_age_seconds=172800,
                max_context_age_seconds=172800,
            )
        mismatch = copy.deepcopy(response)
        mismatch["input_context_sha256"] = "b" * 64
        with self.assertRaisesRegex(contract.LiveReviewError, "context fingerprint"):
            contract.validate_live_review_response(
                request,
                mismatch,
                now=NOW + timedelta(seconds=30),
            )
        wrong_account_time = copy.deepcopy(response)
        account_check = next(
            row for row in wrong_account_time["checks"] if row["check_id"] == "actual_account_risk"
        )
        account_check["evidence_sources"][0]["observed_at_utc"] = (
            NOW + timedelta(seconds=1)
        ).isoformat()
        with self.assertRaisesRegex(contract.LiveReviewError, "requested IBKR snapshot"):
            contract.validate_live_review_response(
                request,
                wrong_account_time,
                now=NOW + timedelta(seconds=30),
            )
        executable = copy.deepcopy(response)
        executable["automatic_order_allowed"] = True
        executable["order_payload"] = {"symbol": "QQQ", "action": "BUY"}
        with self.assertRaisesRegex(contract.LiveReviewError, "schema"):
            contract.validate_live_review_response(
                request,
                executable,
                now=NOW + timedelta(seconds=30),
            )

        with tempfile.TemporaryDirectory(dir=contract.ROOT / "private") as temp_dir:
            path = Path(temp_dir) / "request.json"
            contract._write_private_json(path, request)
            self.assertEqual(contract._load_object(path, "request"), request)
        with self.assertRaisesRegex(contract.LiveReviewError, "private/"):
            contract._write_private_json(contract.ROOT / "docs" / "forbidden.json", request)


if __name__ == "__main__":
    unittest.main()
