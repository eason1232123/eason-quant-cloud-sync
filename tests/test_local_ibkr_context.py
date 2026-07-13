from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_local_ibkr_context import (
    LocalIbkrContextError,
    build_private_context,
    validate_private_context,
    write_private_context,
)
from scripts.ibkr_readonly import IbkrReadonlyError, ROOT, SNAPSHOT_SCHEMA_VERSION


NOW = datetime(2026, 7, 13, 6, 0, tzinfo=timezone.utc)
ACCOUNT = "PRIVATE_ACCOUNT_FIXTURE"


def contract(symbol: str, currency: str = "USD", con_id: int = 1) -> dict:
    return {
        "con_id": con_id,
        "symbol": symbol,
        "local_symbol": symbol,
        "security_type": "STK",
        "currency": currency,
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "multiplier": "",
    }


def private_snapshot() -> dict:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "CONNECTED_READ_ONLY_SNAPSHOT",
        "source": "IBKR official TWS API local socket",
        "account_data_timezone": "UTC",
        "collected_at_utc": NOW.isoformat(),
        "portfolio_price_temporality": "UNKNOWN_TWS_ACCOUNT_UPDATE_NOT_VALIDATED_AS_REALTIME",
        "connection": {
            "host_scope": "LOOPBACK_ONLY",
            "port": 7497,
            "client_id": 71,
            "tws_read_only_setting": "REQUIRED_USER_CONFIGURATION_NOT_PROGRAMMATICALLY_VERIFIABLE",
        },
        "managed_accounts": [ACCOUNT],
        "positions": [],
        "portfolio": [
            {
                "account_id": ACCOUNT,
                "contract": contract("QQQ"),
                "quantity": 2.0,
                "market_price": 520.0,
                "market_value": 1040.0,
                "average_cost": 500.0,
                "unrealized_pnl": 40.0,
                "realized_pnl": 0.0,
            },
            {
                "account_id": ACCOUNT,
                "contract": contract("OTHER", con_id=2),
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
            },
            {
                "account_id": ACCOUNT,
                "tag": "TotalCashValue",
                "value": "860",
                "currency": "USD",
            },
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


def decision_packet() -> dict:
    return {
        "schema_version": "decision-packet-v5.0",
        "market_data": {
            "source": "fixture EOD source",
            "market_timezone": "America/New_York",
            "report_generated_at_utc": "2026-07-12T12:00:00+00:00",
            "data_timestamp": "2026-07-10",
            "price_frequency": "end_of_day_daily",
            "price_adjustment_policy": "adjusted",
            "metadata_status": "COMPLETE",
        },
        "decision": {
            "final_action": "NO_TRADE",
            "automatic_order_allowed": False,
        },
    }


def governance() -> dict:
    return {
        "status": "WAITING_FOR_FIRST_PROSPECTIVE_MARKET_DATE",
        "model_governance_fingerprint": "a" * 64,
        "current_market_regime": {"regime": "base"},
        "allocation_decision": {
            "model_allocations": {"balanced_incumbent_v1": 1.0},
            "automatic_order_allowed": False,
        },
        "effective_target_weights": {
            "QQQ": 0.3,
            "SMH": 0.25,
            "MSFT": 0.2,
            "SPY": 0.1,
            "CASH_PROXY": 0.15,
        },
        "automatic_order_allowed": False,
    }


class LocalIbkrContextTests(unittest.TestCase):
    def test_exact_private_exposure_is_traceable_without_order_payload(self) -> None:
        context = build_private_context(
            private_snapshot(),
            decision_packet(),
            governance(),
            generated_at=NOW,
        )
        usd = context["private_accounts"][0]["currency_slices"][0]
        self.assertEqual(usd["currency"], "USD")
        self.assertEqual(usd["net_liquidation"], 2000.0)
        self.assertEqual(usd["total_cash_value"], 860.0)
        self.assertEqual(usd["gross_position_exposure_weight"], 0.57)
        self.assertEqual(usd["QQQ_SMH_MSFT_gross_concentration_weight"], 0.52)
        qqq = next(row for row in usd["positions"] if row["symbol"] == "QQQ")
        self.assertEqual(qqq["net_liquidation_weight"], 0.52)
        self.assertFalse(context["automatic_order_allowed"])
        self.assertTrue(context["human_confirmation_required"])
        self.assertIsNone(context["order_payload"])
        self.assertFalse(context["privacy"]["git_publish_allowed"])
        self.assertEqual(
            context["sources"]["ibkr"]["portfolio_price_temporality"],
            "UNKNOWN_TWS_ACCOUNT_UPDATE_NOT_VALIDATED_AS_REALTIME",
        )
        self.assertEqual(
            context["sources"]["github_strategy_evidence"]["data_timestamp"],
            "2026-07-10",
        )

    def test_missing_net_liquidation_stays_null_not_zero(self) -> None:
        snapshot = private_snapshot()
        snapshot["account_summary"] = []
        context = build_private_context(snapshot, decision_packet(), governance(), generated_at=NOW)
        usd = context["private_accounts"][0]["currency_slices"][0]
        self.assertIsNone(usd["net_liquidation"])
        self.assertIsNone(usd["gross_position_exposure_weight"])
        self.assertIsNone(usd["QQQ_SMH_MSFT_gross_concentration_weight"])
        self.assertEqual(usd["exposure_status"], "MISSING_MARKET_VALUE_OR_NET_LIQUIDATION")

    def test_currencies_are_never_silently_aggregated(self) -> None:
        snapshot = private_snapshot()
        snapshot["portfolio"].append(
            {
                "account_id": ACCOUNT,
                "contract": contract("0700", currency="HKD", con_id=3),
                "quantity": 10.0,
                "market_price": 400.0,
                "market_value": 4000.0,
                "average_cost": 350.0,
                "unrealized_pnl": 500.0,
                "realized_pnl": 0.0,
            }
        )
        snapshot["account_summary"].append(
            {
                "account_id": ACCOUNT,
                "tag": "NetLiquidation",
                "value": "10000",
                "currency": "HKD",
            }
        )
        context = build_private_context(snapshot, decision_packet(), governance(), generated_at=NOW)
        slices = context["private_accounts"][0]["currency_slices"]
        self.assertEqual([item["currency"] for item in slices], ["HKD", "USD"])
        self.assertTrue(all(item["currency_conversion_applied"] is False for item in slices))
        self.assertEqual(
            context["risk_calculation_policy"]["cross_currency_aggregation"],
            "PROHIBITED_WITHOUT_EXPLICIT_FX_RATES",
        )

    def test_public_order_permission_or_private_output_outside_private_fails(self) -> None:
        unsafe_decision = decision_packet()
        unsafe_decision["decision"]["automatic_order_allowed"] = True
        with self.assertRaisesRegex(LocalIbkrContextError, "prohibit automatic orders"):
            build_private_context(private_snapshot(), unsafe_decision, governance(), generated_at=NOW)

        context = build_private_context(
            private_snapshot(), decision_packet(), governance(), generated_at=NOW
        )
        with self.assertRaisesRegex(IbkrReadonlyError, "private/"):
            write_private_context(ROOT / "docs" / "forbidden-private-context.json", context)

    def test_context_validation_rejects_executable_order(self) -> None:
        context = build_private_context(
            private_snapshot(), decision_packet(), governance(), generated_at=NOW
        )
        contaminated = copy.deepcopy(context)
        contaminated["order_payload"] = {"symbol": "QQQ", "action": "BUY"}
        with self.assertRaisesRegex(LocalIbkrContextError, "executable order"):
            validate_private_context(contaminated)

        private_root = ROOT / "private"
        private_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private_root) as temp_dir:
            path = Path(temp_dir) / "context.json"
            write_private_context(path, context)
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["status"], "PRIVATE_LOCAL_ANALYSIS_CONTEXT_READY")


if __name__ == "__main__":
    unittest.main()
