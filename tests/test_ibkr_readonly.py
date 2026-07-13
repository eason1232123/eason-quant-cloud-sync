from __future__ import annotations

import ast
import inspect
import json
import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scripts import ibkr_readonly
from scripts.ibkr_readonly import (
    IbkrReadonlyConfig,
    IbkrReadonlyError,
    OfficialIbapiAdapter,
    capture_private_snapshot,
    config_from_env,
    load_private_snapshot,
    probe_endpoint,
    validate_config,
    validate_private_snapshot,
)


class FakeAdapter:
    def capture(self, config: IbkrReadonlyConfig) -> dict:
        return {
            "managed_accounts": ["PRIVATE_ACCOUNT_FIXTURE"],
            "positions": [
                {
                    "account_id": "PRIVATE_ACCOUNT_FIXTURE",
                    "contract": {
                        "con_id": 1,
                        "symbol": "QQQ",
                        "local_symbol": "QQQ",
                        "security_type": "STK",
                        "currency": "USD",
                        "exchange": "SMART",
                        "primary_exchange": "NASDAQ",
                        "multiplier": "",
                    },
                    "quantity": 2.0,
                    "average_cost": 500.0,
                }
            ],
            "portfolio": [
                {
                    "account_id": "PRIVATE_ACCOUNT_FIXTURE",
                    "contract": {
                        "con_id": 1,
                        "symbol": "QQQ",
                        "local_symbol": "QQQ",
                        "security_type": "STK",
                        "currency": "USD",
                        "exchange": "SMART",
                        "primary_exchange": "NASDAQ",
                        "multiplier": "",
                    },
                    "quantity": 2.0,
                    "market_price": 520.0,
                    "market_value": 1040.0,
                    "average_cost": 500.0,
                    "unrealized_pnl": 40.0,
                    "realized_pnl": 0.0,
                }
            ],
            "account_summary": [
                {
                    "account_id": "PRIVATE_ACCOUNT_FIXTURE",
                    "tag": "NetLiquidation",
                    "value": "2000",
                    "currency": "USD",
                }
            ],
            "account_values": [],
            "account_readiness": {"PRIVATE_ACCOUNT_FIXTURE": True},
        }


class IbkrReadonlyTests(unittest.TestCase):
    def private_path(self, temp_root: str) -> Path:
        # Production validation requires the real repository private/ root.
        # Each test uses a unique subdirectory under that ignored location.
        return ibkr_readonly.ROOT / "private" / "test-fixtures" / Path(temp_root).name / "snapshot.json"

    def tearDown(self) -> None:
        fixture_root = ibkr_readonly.ROOT / "private" / "test-fixtures"
        if fixture_root.exists():
            for path in sorted(fixture_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            fixture_root.rmdir()

    def test_config_is_loopback_nonzero_and_private_only(self) -> None:
        with self.assertRaisesRegex(IbkrReadonlyError, "loopback"):
            validate_config(IbkrReadonlyConfig(host="192.0.2.10"))
        with self.assertRaisesRegex(IbkrReadonlyError, "non-zero"):
            validate_config(IbkrReadonlyConfig(client_id=0))
        with self.assertRaisesRegex(IbkrReadonlyError, "private/"):
            validate_config(
                IbkrReadonlyConfig(
                    private_snapshot_path=ibkr_readonly.ROOT / "docs" / "forbidden.json"
                )
            )

        parsed = config_from_env(
            {
                "IBKR_HOST": "localhost",
                "IBKR_PORT": "4002",
                "IBKR_CLIENT_ID": "72",
                "IBKR_TIMEOUT_SECONDS": "20",
                "IBKR_PRIVATE_SNAPSHOT": "private/ibkr/test.json",
            }
        )
        self.assertEqual(parsed.port, 4002)
        self.assertEqual(parsed.client_id, 72)
        self.assertEqual(
            parsed.private_snapshot_path.relative_to(ibkr_readonly.ROOT).as_posix(),
            "private/ibkr/test.json",
        )

    def test_probe_reports_offline_without_claiming_account_access(self) -> None:
        config = IbkrReadonlyConfig()
        with patch("scripts.ibkr_readonly.socket.create_connection", side_effect=ConnectionRefusedError()):
            result = probe_endpoint(config)
        self.assertEqual(result["status"], "IBKR_ENDPOINT_OFFLINE")
        self.assertFalse(result["reachable"])
        self.assertFalse(result["automatic_order_allowed"])
        self.assertNotIn("account", json.dumps(result).lower())

    def test_private_capture_roundtrip_is_fresh_and_never_public(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.private_path(temp_dir)
            config = IbkrReadonlyConfig(private_snapshot_path=path)
            now = datetime.now(timezone.utc)
            snapshot = capture_private_snapshot(
                config,
                adapter=FakeAdapter(),
                collected_at=now,
            )
            self.assertTrue(path.exists())
            self.assertFalse(snapshot["automatic_order_allowed"])
            self.assertEqual(snapshot["order_methods_invoked"], [])
            self.assertEqual(snapshot["privacy"]["git_publish_allowed"], False)
            self.assertTrue(snapshot["account_readiness"]["PRIVATE_ACCOUNT_FIXTURE"])
            self.assertEqual(
                snapshot["portfolio_price_temporality"],
                ibkr_readonly.PORTFOLIO_PRICE_TEMPORALITY,
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
            validate_private_snapshot(loaded, max_age_seconds=300, now=now + timedelta(seconds=10))

            reloaded = load_private_snapshot(path, max_age_seconds=300)
            self.assertEqual(reloaded["positions"][0]["contract"]["symbol"], "QQQ")

    def test_stale_future_invalid_and_order_fields_fail_closed(self) -> None:
        now = datetime(2026, 7, 13, 5, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            config = IbkrReadonlyConfig(private_snapshot_path=self.private_path(temp_dir))
            snapshot = capture_private_snapshot(config, adapter=FakeAdapter(), collected_at=now)

        with self.assertRaisesRegex(IbkrReadonlyError, "stale"):
            validate_private_snapshot(snapshot, max_age_seconds=60, now=now + timedelta(seconds=61))
        future = dict(snapshot)
        future["collected_at_utc"] = (now + timedelta(seconds=6)).isoformat()
        with self.assertRaisesRegex(IbkrReadonlyError, "future"):
            validate_private_snapshot(future, max_age_seconds=60, now=now)
        contaminated = dict(snapshot)
        contaminated["orders"] = []
        with self.assertRaisesRegex(IbkrReadonlyError, "prohibited field"):
            validate_private_snapshot(contaminated, max_age_seconds=60, now=now)
        not_ready = dict(snapshot)
        not_ready["account_readiness"] = {"PRIVATE_ACCOUNT_FIXTURE": False}
        with self.assertRaisesRegex(IbkrReadonlyError, "readiness"):
            validate_private_snapshot(not_ready, max_age_seconds=60, now=now)
        invalid_market_value = dict(snapshot)
        invalid_market_value["portfolio"] = [dict(snapshot["portfolio"][0])]
        invalid_market_value["portfolio"][0]["market_value"] = "unknown"
        with self.assertRaisesRegex(IbkrReadonlyError, "market_value"):
            validate_private_snapshot(invalid_market_value, max_age_seconds=60, now=now)

    def test_official_adapter_source_cannot_call_order_methods(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(OfficialIbapiAdapter.capture)))
        invoked = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(invoked & ibkr_readonly.FORBIDDEN_ORDER_METHODS)

    def test_gitignore_covers_all_private_snapshots(self) -> None:
        ignore = (ibkr_readonly.ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("private/", ignore.splitlines())
        self.assertNotIn("docs", str(ibkr_readonly.DEFAULT_PRIVATE_SNAPSHOT.relative_to(ibkr_readonly.ROOT)))


if __name__ == "__main__":
    unittest.main()
