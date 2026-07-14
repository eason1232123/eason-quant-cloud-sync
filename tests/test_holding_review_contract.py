from __future__ import annotations

import copy
import json
import unittest
from datetime import datetime, timedelta, timezone

from scripts import holding_review_contract as contract
from scripts.audit_v6_release import (
    HUMAN_PILOT_REQUIRED_GATES,
    RELEASE_GATE_ORDER,
)
from scripts.build_action_board_v3 import holding_review_for_current_decision
from scripts.build_local_ibkr_context import build_private_context
from scripts.build_v6_operating_status import SHADOW_REQUIRED_GATES


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


def _snapshot(*, empty: bool = False) -> dict:
    portfolio = [] if empty else [
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
            "contract": _instrument("AAPL", 2),
            "quantity": 1.0,
            "market_price": 210.0,
            "market_value": 210.0,
            "average_cost": 200.0,
            "unrealized_pnl": 10.0,
            "realized_pnl": 0.0,
        },
    ]
    return {
        "schema_version": "v6-ibkr-private-readonly-snapshot-v1",
        "status": "CONNECTED_READ_ONLY_SNAPSHOT",
        "source": "IBKR official TWS API local socket",
        "account_data_timezone": "UTC",
        "collected_at_utc": NOW.isoformat(),
        "portfolio_price_temporality": "UNKNOWN_TWS_ACCOUNT_UPDATE_NOT_VALIDATED_AS_REALTIME",
        "managed_accounts": [ACCOUNT],
        "positions": [],
        "portfolio": portfolio,
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


def _decision() -> dict:
    return {
        "schema_version": "decision-packet-v5.1",
        "generated_at_utc": NOW.isoformat(),
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
            "final_action": "NO_TRADE",
            "buy_permission": "NO_QUANT_CANDIDATE",
            "automatic_order_allowed": False,
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


def _request(*, empty: bool = False) -> dict:
    decision = _decision()
    governance = _governance()
    context = build_private_context(
        _snapshot(empty=empty),
        decision,
        governance,
        generated_at=NOW,
    )
    return contract.build_holding_review_request(
        context,
        decision,
        governance,
        generated_at=NOW,
    )


def _source(check_id: str) -> dict:
    if check_id == "fresh_news_events":
        source_type, data_kind = "PUBLIC_NEWS", "NEWS_OR_EVENT"
    elif check_id == "earnings_fundamentals":
        source_type, data_kind = "PUBLIC_FILINGS", "FUNDAMENTAL_OR_EARNINGS"
    elif check_id == "actual_account_risk":
        source_type, data_kind = "PRIVATE_CONTEXT", "PRIVATE_ACCOUNT_SNAPSHOT"
    elif check_id == "technical_risk":
        source_type, data_kind = "GITHUB_EVIDENCE", "GITHUB_QUANT_EVIDENCE"
    else:
        source_type, data_kind = "PUBLIC_MARKET_DATA", "END_OF_DAY_ADJUSTED"
    return {
        "source_name": f"fixture {check_id}",
        "source_type": source_type,
        "source_url": (
            f"https://example.com/{check_id}"
            if source_type.startswith("PUBLIC_")
            else None
        ),
        "observed_at_utc": (
            NOW if check_id == "actual_account_risk" else NOW + timedelta(seconds=10)
        ).isoformat(),
        "market_timezone": "America/New_York",
        "data_kind": data_kind,
    }


def _response(request: dict) -> dict:
    assessments = []
    for symbol in request["review_scope"]["held_symbols"]:
        assessments.append(
            {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": "MEDIUM",
                "reason_codes": ["FIXTURE_COMPLETE"],
                "summary": "Fixture-only holding review.",
                "checks": [
                    {
                        "check_id": check_id,
                        "status": "PASS",
                        "finding": f"fixture {check_id} complete",
                        "evidence_sources": [_source(check_id)],
                    }
                    for check_id in contract.REQUIRED_CHECK_IDS
                ],
            }
        )
    return {
        "schema_version": contract.RESPONSE_SCHEMA_VERSION,
        "status": "HOLDING_REVIEW_COMPLETED",
        "request_id": request["request_id"],
        "input_context_sha256": request["source_context"]["context_sha256"],
        "reviewed_at_utc": (NOW + timedelta(seconds=30)).isoformat(),
        "reviewer": {
            "system": "TRADINGAGENTS",
            "model_id": "fixture-multi-agent-model",
            "adapter_version": "fixture-adapter-v1",
        },
        "privacy": {
            "classification": "PRIVATE_LOCAL_ONLY",
            "git_publish_allowed": False,
            "contains_private_account_reasoning": True,
        },
        "coverage": {
            "required_holding_count": len(assessments),
            "assessed_holding_count": len(assessments),
            "complete": True,
        },
        "assessments": assessments,
        "quant_boundary_confirmation": {
            "original_quant_final_action": "NO_TRADE",
            "unchanged": True,
            "buy_permission_modified": False,
            "buy_candidate_created": False,
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "order_payload": None,
    }


class HoldingReviewContractTests(unittest.TestCase):
    def test_holding_review_is_not_a_release_or_shadow_evidence_gate(self) -> None:
        gate_names = (
            *RELEASE_GATE_ORDER,
            *HUMAN_PILOT_REQUIRED_GATES,
            *SHADOW_REQUIRED_GATES,
        )
        self.assertFalse(any("holding_review" in name for name in gate_names))

        public = contract.build_private_context_required_status(_decision())
        self.assertFalse(public["quant_boundary"]["affects_shadow_evidence_gate"])
        self.assertFalse(public["quant_boundary"]["buy_standard_modified"])

    def test_public_review_is_usable_only_for_the_same_quant_market_date(self) -> None:
        request = _request()
        public = contract.build_public_holding_review_status(
            request,
            _response(request),
        )
        current = holding_review_for_current_decision(public, _decision())
        self.assertTrue(current["current_for_quant_market_date"])
        self.assertTrue(current["usable_for_current_holding_decision"])

        stale_decision = _decision()
        stale_decision["market_data"]["data_timestamp"] = "2026-07-13"
        stale = holding_review_for_current_decision(public, stale_decision)
        self.assertFalse(stale["current_for_quant_market_date"])
        self.assertFalse(stale["usable_for_current_holding_decision"])

    def test_request_covers_every_holding_without_ranking_or_buy_permission(self) -> None:
        request = _request()
        self.assertEqual(request["review_scope"]["held_symbols"], ["AAPL", "QQQ"])
        self.assertEqual(request["review_scope"]["held_symbol_count"], 2)
        self.assertEqual(
            request["review_scope"]["selection_policy"],
            "COMPLETE_UNRANKED_UNCAPPED_DISTINCT_SYMBOLS",
        )
        self.assertNotIn("BUY", request["review_scope"]["allowed_actions"])
        self.assertFalse(request["review_scope"]["can_create_buy_candidate"])
        self.assertFalse(request["review_scope"]["affects_quant_final_action"])
        self.assertEqual(
            request["review_scope"]["quant_final_action_at_request"],
            "NO_TRADE",
        )
        self.assertFalse(request["automatic_order_allowed"])
        contract.validate_holding_review_request(request, now=NOW)

    def test_complete_tradingagents_response_is_accepted_and_sanitized(self) -> None:
        request = _request()
        response = _response(request)
        contract.validate_holding_review_response(
            request,
            response,
            now=NOW + timedelta(seconds=30),
        )
        public = contract.build_public_holding_review_status(request, response)
        self.assertEqual(public["status"], "HOLDING_REVIEW_COMPLETE")
        self.assertEqual(public["latest_review"]["reviewed_holding_count"], 2)
        self.assertEqual(public["latest_review"]["action_counts"]["HOLD"], 2)
        self.assertTrue(public["quant_boundary"]["quant_final_action_unchanged"])
        self.assertFalse(public["quant_boundary"]["buy_standard_modified"])
        encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("AAPL", encoded)
        self.assertNotIn("QQQ", encoded)
        self.assertNotIn(ACCOUNT, encoded)

    def test_missing_or_reordered_holding_is_rejected(self) -> None:
        request = _request()
        response = _response(request)
        response["assessments"].pop()
        response["coverage"]["assessed_holding_count"] = 1
        with self.assertRaisesRegex(contract.HoldingReviewError, "every held symbol"):
            contract.validate_holding_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )

        response = _response(request)
        response["assessments"].reverse()
        with self.assertRaisesRegex(contract.HoldingReviewError, "frozen order"):
            contract.validate_holding_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )

    def test_incomplete_evidence_forces_no_action(self) -> None:
        request = _request()
        response = _response(request)
        response["assessments"][0]["checks"][2]["status"] = "UNAVAILABLE"
        response["assessments"][0]["checks"][2]["evidence_sources"] = []
        with self.assertRaisesRegex(contract.HoldingReviewError, "requires NO_ACTION"):
            contract.validate_holding_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )
        response["assessments"][0]["action"] = "NO_ACTION"
        contract.validate_holding_review_response(
            request,
            response,
            now=NOW + timedelta(seconds=30),
        )

    def test_reduce_or_exit_review_requires_current_intraday_market_evidence(self) -> None:
        request = _request()
        response = _response(request)
        response["assessments"][0]["action"] = "REDUCE_REVIEW"
        with self.assertRaisesRegex(contract.HoldingReviewError, "current intraday"):
            contract.validate_holding_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )

        market_source = response["assessments"][0]["checks"][0][
            "evidence_sources"
        ][0]
        market_source["data_kind"] = "INTRADAY_REALTIME_BID_ASK"
        market_source["observed_at_utc"] = (NOW + timedelta(seconds=10)).isoformat()
        contract.validate_holding_review_response(
            request,
            response,
            now=NOW + timedelta(seconds=30),
        )

    def test_request_expiry_cannot_be_extended_without_changing_contract(self) -> None:
        request = _request()
        request["expires_at_utc"] = (NOW + timedelta(hours=1)).isoformat()
        with self.assertRaisesRegex(contract.HoldingReviewError, "frozen TTL"):
            contract.validate_holding_review_request(request, now=NOW)

    def test_no_open_holdings_produces_complete_empty_review(self) -> None:
        request = _request(empty=True)
        self.assertEqual(request["status"], "NO_OPEN_HOLDINGS_REVIEW_NOT_REQUIRED")
        self.assertEqual(request["review_scope"]["held_symbols"], [])
        response = _response(request)
        contract.validate_holding_review_response(
            request,
            response,
            now=NOW + timedelta(seconds=30),
        )
        public = contract.build_public_holding_review_status(request, response)
        self.assertEqual(public["status"], "NO_OPEN_HOLDINGS")
        self.assertEqual(public["latest_review"]["reviewed_holding_count"], 0)

    def test_quant_boundary_or_order_payload_cannot_be_changed(self) -> None:
        request = _request()
        response = _response(request)
        response["quant_boundary_confirmation"]["original_quant_final_action"] = (
            "BUY_CANDIDATE_REVIEW_REQUIRED"
        )
        with self.assertRaisesRegex(contract.HoldingReviewError, "quant final action"):
            contract.validate_holding_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )

        response = _response(request)
        response["automatic_order_allowed"] = True
        response["order_payload"] = {"symbol": "AAPL", "action": "SELL"}
        with self.assertRaisesRegex(contract.HoldingReviewError, "schema"):
            contract.validate_holding_review_response(
                request,
                response,
                now=NOW + timedelta(seconds=30),
            )


if __name__ == "__main__":
    unittest.main()
