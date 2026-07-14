from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.artifact_io import atomic_write_json, atomic_write_text
from scripts.validate_publication_bundle import (
    CORE_ARTIFACTS,
    PublicationBundleError,
    build_manifest,
    validate_manifest,
)


GENERATED_AT = "2026-07-14T14:36:58+00:00"
DATA_DATE = "2026-07-13"


def market_data() -> dict:
    return {
        "source": "fixture",
        "market_timezone": "America/New_York",
        "report_generated_at_utc": GENERATED_AT,
        "data_timestamp": DATA_DATE,
    }


def decision_packet() -> dict:
    return {
        "schema_version": "decision-packet-v5.0",
        "generated_at_utc": GENERATED_AT,
        "market_data": market_data(),
        "decision": {
            "final_action": "NO_TRADE",
            "buy_permission": "NO_QUANT_CANDIDATE",
            "chatgpt_review_required": True,
            "automatic_order_allowed": False,
        },
        "execution_contract": {"automatic_order_allowed": False},
    }


def fixture_payloads() -> dict[str, dict]:
    packet = decision_packet()
    return {
        "market_report.json": {
            "generated_at_utc": GENERATED_AT,
            "data_source": "fixture",
            "market_timezone": "America/New_York",
            "data_timestamp": DATA_DATE,
        },
        "latest_market_summary.json": {
            "generated_at_utc": GENERATED_AT,
            "data_timestamp": DATA_DATE,
        },
        "portfolio_backtest.json": {
            "generated_at_utc": GENERATED_AT,
            "data_timestamp": DATA_DATE,
        },
        "trade_review.json": {
            "generated_at_utc": GENERATED_AT,
            "summary": {"trade_count": 0},
            "actual_vs_backtest": {
                "available": False,
                "reason": "No actual trades logged yet.",
            },
        },
        "forward_validation_status.json": {"generated_at_utc": GENERATED_AT},
        "live_review_forward_status.json": {"generated_at_utc": GENERATED_AT},
        "v6_release_status.json": {
            "generated_at_utc": GENERATED_AT,
            "status": "PROSPECTIVE_VALIDATION_IN_PROGRESS",
            "automatic_order_allowed": False,
        },
        "v6_operating_status.json": {
            "generated_at_utc": GENERATED_AT,
            "operating_mode": "READ_ONLY_SHADOW",
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
            "capabilities": {"human_pilot_review_ready": False},
            "release_blockers": ["SAMPLES_BELOW_MINIMUM"],
            "evidence_counts": {"public_signal_primary_horizon_outcomes": 0},
            "evidence_thresholds": {"public_signal_primary_horizon_outcomes": 20},
        },
        "eason_signal.json": {
            "generated_at_utc": GENERATED_AT,
            "market_data": market_data(),
            "final_action": "NO_TRADE",
            "buy_permission": "NO_QUANT_CANDIDATE",
        },
        "latest_decision_summary.json": {
            "generated_at_utc": GENERATED_AT,
            "market_data": market_data(),
            "final_action": "NO_TRADE",
            "buy_permission": "NO_QUANT_CANDIDATE",
            "chatgpt_final_review_required": True,
        },
        "decision_packet.json": packet,
        "action_board.json": {
            "generated_at_utc": GENERATED_AT,
            "decision_contract": packet,
            "signal_summary": {
                "final_action": "NO_TRADE",
                "buy_permission": "NO_QUANT_CANDIDATE",
            },
            "final_gate": {
                "quant_signal": "NO_TRADE",
                "chatgpt_review_required": True,
            },
        },
    }


class PublicationBundleTests(unittest.TestCase):
    def write_fixture(self, docs: Path) -> None:
        docs.mkdir(parents=True, exist_ok=True)
        payloads = fixture_payloads()
        self.assertEqual(set(payloads), set(CORE_ARTIFACTS))
        for name, payload in payloads.items():
            atomic_write_json(docs / name, payload)

    def test_manifest_proves_large_artifact_and_cross_file_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            self.write_fixture(docs)
            manifest = build_manifest(
                docs,
                generated_at=datetime(2026, 7, 14, 15, 0, tzinfo=timezone.utc),
            )
            result = validate_manifest(manifest, docs)
            market = next(
                row for row in manifest["artifacts"] if row["path"] == "docs/market_report.json"
            )
            self.assertGreater(market["size_bytes"], 0)
            self.assertEqual(len(market["sha256"]), 64)
            self.assertEqual(result["status"], "VALIDATED")
            self.assertEqual(result["operating_mode"], "READ_ONLY_SHADOW")

    def test_manifest_detects_post_validation_artifact_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            self.write_fixture(docs)
            manifest = build_manifest(docs)
            market = json.loads((docs / "market_report.json").read_text(encoding="utf-8"))
            market["unexpected_change"] = True
            atomic_write_json(docs / "market_report.json", market)
            with self.assertRaisesRegex(PublicationBundleError, "size mismatch|SHA-256 mismatch"):
                validate_manifest(manifest, docs)

    def test_cross_file_review_flag_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            self.write_fixture(docs)
            summary_path = docs / "latest_decision_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["chatgpt_final_review_required"] = False
            atomic_write_json(summary_path, summary)
            with self.assertRaisesRegex(PublicationBundleError, "chatgpt_review_required"):
                build_manifest(docs)

    def test_manifest_rejects_stale_embedded_evidence_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            self.write_fixture(docs)
            manifest = build_manifest(docs)
            manifest["evidence_readiness"]["logged_trade_count"] = 999
            with self.assertRaisesRegex(PublicationBundleError, "evidence readiness"):
                validate_manifest(manifest, docs)

    def test_atomic_write_preserves_previous_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "critical.json"
            atomic_write_text(path, "old\n")
            with patch("scripts.artifact_io.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_write_text(path, "new\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(list(path.parent.glob(".critical.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
