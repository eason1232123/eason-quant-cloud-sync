from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.ibkr_readonly import IbkrReadonlyConfig
from scripts.run_v6_live_cycle import (
    V6LiveCycleError,
    finalize_live_cycle,
    main,
    prepare_live_cycle,
)


class V6LiveCycleTests(unittest.TestCase):
    @patch("scripts.run_v6_live_cycle.validate_model_artifacts")
    def test_invalid_age_fails_before_model_or_broker_work(
        self,
        validate_model_artifacts: Mock,
    ) -> None:
        with self.assertRaisesRegex(V6LiveCycleError, "positive and finite"):
            prepare_live_cycle(max_snapshot_age_seconds=0)
        validate_model_artifacts.assert_not_called()

    @patch("scripts.run_v6_live_cycle.capture_private_snapshot")
    @patch("scripts.run_v6_live_cycle.probe_endpoint")
    @patch("scripts.run_v6_live_cycle.validate_model_artifacts")
    def test_prepare_fails_before_capture_when_endpoint_is_offline(
        self,
        validate_model_artifacts: Mock,
        probe_endpoint: Mock,
        capture_private_snapshot: Mock,
    ) -> None:
        validate_model_artifacts.return_value = {"status": "VALID"}
        probe_endpoint.return_value = {
            "status": "IBKR_ENDPOINT_OFFLINE",
            "reachable": False,
        }
        with self.assertRaisesRegex(V6LiveCycleError, "IBKR_ENDPOINT_OFFLINE"):
            prepare_live_cycle(config=IbkrReadonlyConfig())
        capture_private_snapshot.assert_not_called()

    @patch("scripts.run_v6_live_cycle.build_request_from_files")
    @patch("scripts.run_v6_live_cycle.build_local_context")
    @patch("scripts.run_v6_live_cycle.capture_private_snapshot")
    @patch("scripts.run_v6_live_cycle.probe_endpoint")
    @patch("scripts.run_v6_live_cycle.validate_model_artifacts")
    def test_prepare_runs_readonly_chain_and_returns_only_sanitized_status(
        self,
        validate_model_artifacts: Mock,
        probe_endpoint: Mock,
        capture_private_snapshot: Mock,
        build_local_context: Mock,
        build_request_from_files: Mock,
    ) -> None:
        validate_model_artifacts.return_value = {"status": "VALID"}
        probe_endpoint.return_value = {
            "status": "IBKR_ENDPOINT_REACHABLE",
            "reachable": True,
        }
        capture_private_snapshot.return_value = {
            "collected_at_utc": "2026-07-13T13:00:00+00:00",
            "managed_accounts": ["SECRET_ACCOUNT"],
        }
        build_local_context.return_value = {
            "generated_at_utc": "2026-07-13T13:00:01+00:00",
            "private_accounts": [{"secret": "SECRET_POSITION"}],
        }
        build_request_from_files.return_value = {
            "status": "AWAITING_CHATGPT_LIVE_REVIEW",
            "request_id": "SECRET_REQUEST_ID",
            "expires_at_utc": "2026-07-13T13:05:01+00:00",
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
        }
        config = IbkrReadonlyConfig()
        adapter = Mock()

        result = prepare_live_cycle(config=config, adapter=adapter)

        capture_private_snapshot.assert_called_once_with(config, adapter=adapter)
        build_local_context.assert_called_once()
        build_request_from_files.assert_called_once()
        encoded = json.dumps(result, sort_keys=True)
        for secret in ("SECRET_ACCOUNT", "SECRET_POSITION", "SECRET_REQUEST_ID"):
            self.assertNotIn(secret, encoded)
        self.assertEqual(result["status"], "V6_LIVE_CYCLE_AWAITING_CHATGPT_RESPONSE")
        self.assertFalse(result["automatic_order_allowed"])
        self.assertTrue(result["human_confirmation_required"])
        self.assertEqual(
            result["next_command"],
            "python -m scripts.run_v6_live_cycle finalize",
        )

    @patch("scripts.run_v6_live_cycle.audit_v6_release")
    @patch("scripts.run_v6_live_cycle.record_private_review_from_files")
    def test_finalize_records_sanitized_event_then_refreshes_release_audit(
        self,
        record_private_review_from_files: Mock,
        audit_v6_release: Mock,
    ) -> None:
        record_private_review_from_files.return_value = {
            "ledger_counts": {
                "new_prediction_events": 1,
                "new_outcome_events": 0,
                "prediction_events": 1,
                "matured_outcome_events": 0,
            }
        }
        audit_v6_release.return_value = {
            "status": "PROSPECTIVE_VALIDATION_IN_PROGRESS",
            "ready_for_human_pilot_review": False,
        }

        result = finalize_live_cycle()

        record_private_review_from_files.assert_called_once_with()
        audit_v6_release.assert_called_once_with()
        self.assertEqual(result["status"], "V6_SANITIZED_LIVE_REVIEW_RECORDED")
        self.assertEqual(result["new_prediction_events"], 1)
        self.assertFalse(result["ready_for_human_pilot_review"])
        self.assertFalse(result["automatic_order_allowed"])

    @patch("scripts.run_v6_live_cycle.probe_live_cycle")
    def test_cli_probe_returns_nonzero_and_valid_json_when_offline(
        self,
        probe_live_cycle: Mock,
    ) -> None:
        probe_live_cycle.return_value = {
            "status": "V6_LIVE_CYCLE_ENDPOINT_OFFLINE",
            "endpoint": {"reachable": False},
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
        }
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["probe"])
        self.assertNotEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "V6_LIVE_CYCLE_ENDPOINT_OFFLINE")
        self.assertFalse(payload["automatic_order_allowed"])

    def test_orchestrator_has_no_order_api_and_is_not_in_github_workflow(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "run_v6_live_cycle.py").read_text(
            encoding="utf-8"
        )
        workflow = (root / ".github" / "workflows" / "main.yml").read_text(
            encoding="utf-8"
        )
        for prohibited in ("placeOrder", "reqIds", "reqOpenOrders", "cancelOrder"):
            self.assertNotIn(prohibited, source)
        self.assertNotIn("run_v6_live_cycle", workflow)


if __name__ == "__main__":
    unittest.main()
