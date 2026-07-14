from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_v6_release import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_RELEASE_STATUS,
    audit_v6_release,
)
from scripts.build_live_review_forward_ledger import (  # noqa: E402
    DEFAULT_LEDGER as DEFAULT_LIVE_REVIEW_LEDGER,
    DEFAULT_SUMMARY as DEFAULT_LIVE_REVIEW_STATUS,
    live_review_due_status,
    record_private_review_from_files,
)
from scripts.build_local_ibkr_context import (  # noqa: E402
    DEFAULT_PRIVATE_CONTEXT,
    build_local_context,
)
from scripts.build_v6_operating_status import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_OPERATING_STATUS,
    build_v6_operating_status,
)
from scripts.ibkr_readonly import (  # noqa: E402
    IbkrReadonlyConfig,
    IbkrReadonlyError,
    SnapshotAdapter,
    capture_private_snapshot,
    resolve_runtime_endpoint,
)
from scripts.holding_review_contract import (  # noqa: E402
    DEFAULT_PUBLIC_STATUS as DEFAULT_HOLDING_REVIEW_STATUS,
    DEFAULT_REQUEST as DEFAULT_HOLDING_REVIEW_REQUEST,
    DEFAULT_RESPONSE as DEFAULT_HOLDING_REVIEW_RESPONSE,
    build_request_from_files as build_holding_review_request_from_files,
    record_review_from_files as record_holding_review_from_files,
)
from scripts.live_review_contract import (  # noqa: E402
    DEFAULT_REQUEST,
    DEFAULT_RESPONSE,
    build_request_from_files,
)
from scripts.validate_model_artifacts import validate_model_artifacts  # noqa: E402


class V6LiveCycleError(IbkrReadonlyError):
    """Raised when the local read-only live cycle cannot advance safely."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise V6LiveCycleError(f"live-cycle path is outside the repository: {path}") from exc


def _positive_seconds(value: float, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise V6LiveCycleError(f"{field} must be positive and finite")
    return float(value)


def probe_live_cycle(
    *,
    config: IbkrReadonlyConfig | None = None,
) -> dict[str, Any]:
    model_validation = validate_model_artifacts()
    _, endpoint = resolve_runtime_endpoint(config=config)
    return {
        "status": (
            "V6_LIVE_CYCLE_ENDPOINT_READY"
            if endpoint["reachable"]
            else (
                "V6_LIVE_CYCLE_ENDPOINT_AMBIGUOUS"
                if endpoint["status"] == "IBKR_ENDPOINT_AMBIGUOUS"
                else "V6_LIVE_CYCLE_ENDPOINT_OFFLINE"
            )
        ),
        "model_artifacts_status": model_validation["status"],
        "endpoint": endpoint,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }


def prepare_live_cycle(
    *,
    config: IbkrReadonlyConfig | None = None,
    adapter: SnapshotAdapter | None = None,
    max_snapshot_age_seconds: float = 300.0,
    max_account_age_seconds: float = 300.0,
    max_context_age_seconds: float = 300.0,
) -> dict[str, Any]:
    snapshot_age = _positive_seconds(
        max_snapshot_age_seconds,
        "max_snapshot_age_seconds",
    )
    account_age = _positive_seconds(
        max_account_age_seconds,
        "max_account_age_seconds",
    )
    context_age = _positive_seconds(
        max_context_age_seconds,
        "max_context_age_seconds",
    )
    model_validation = validate_model_artifacts()
    due_status = live_review_due_status()
    if not due_status["review_due"]:
        return {
            "status": "V6_LIVE_CYCLE_CURRENT_MARKET_DATE_ALREADY_RECORDED",
            "model_artifacts_status": model_validation["status"],
            "evidence_status": due_status["status"],
            "data_timestamp": due_status["data_timestamp"],
            "broker_snapshot_collected": False,
            "next_command": None,
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
        }
    active_config, endpoint = resolve_runtime_endpoint(config=config)
    if active_config is None or not endpoint["reachable"]:
        raise V6LiveCycleError(
            f"{endpoint['status']}: {endpoint['reason']}"
        )

    snapshot = capture_private_snapshot(active_config, adapter=adapter)
    context = build_local_context(
        snapshot_path=active_config.private_snapshot_path,
        output_path=DEFAULT_PRIVATE_CONTEXT,
        max_snapshot_age_seconds=snapshot_age,
    )
    request = build_request_from_files(
        context_path=DEFAULT_PRIVATE_CONTEXT,
        output_path=DEFAULT_REQUEST,
        max_account_age_seconds=account_age,
        max_context_age_seconds=context_age,
    )
    holding_request = build_holding_review_request_from_files(
        context_path=DEFAULT_PRIVATE_CONTEXT,
        output_path=DEFAULT_HOLDING_REVIEW_REQUEST,
        max_account_age_seconds=account_age,
        max_context_age_seconds=context_age,
    )
    if request.get("automatic_order_allowed") is not False:
        raise V6LiveCycleError("prepared live-review request permits automatic orders")
    if request.get("human_confirmation_required") is not True:
        raise V6LiveCycleError("prepared live-review request omits human confirmation")
    if holding_request.get("automatic_order_allowed") is not False:
        raise V6LiveCycleError("prepared holding-review request permits automatic orders")
    if holding_request.get("human_confirmation_required") is not True:
        raise V6LiveCycleError("prepared holding-review request omits human confirmation")
    return {
        "status": "V6_LIVE_CYCLE_AWAITING_CHATGPT_RESPONSE",
        "model_artifacts_status": model_validation["status"],
        "endpoint_status": endpoint["status"],
        "snapshot_collected_at_utc": snapshot["collected_at_utc"],
        "context_generated_at_utc": context["generated_at_utc"],
        "request_expires_at_utc": request["expires_at_utc"],
        "private_artifacts": [
            _relative(active_config.private_snapshot_path),
            _relative(DEFAULT_PRIVATE_CONTEXT),
            _relative(DEFAULT_REQUEST),
            _relative(DEFAULT_HOLDING_REVIEW_REQUEST),
        ],
        "expected_private_response": _relative(DEFAULT_RESPONSE),
        "expected_private_holding_response": _relative(
            DEFAULT_HOLDING_REVIEW_RESPONSE
        ),
        "holding_review_scope": "ALL_CURRENT_PRIVATE_IBKR_HOLDINGS",
        "holding_review_symbol_count": holding_request["review_scope"][
            "held_symbol_count"
        ],
        "next_command": "python -m scripts.run_v6_live_cycle finalize",
        "next_commands": [
            "python -m scripts.run_v6_live_cycle finalize-holdings",
            "python -m scripts.run_v6_live_cycle finalize",
        ],
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }


def finalize_live_cycle() -> dict[str, Any]:
    summary = record_private_review_from_files()
    release = audit_v6_release()
    operating = build_v6_operating_status()
    counts = summary["ledger_counts"]
    new_predictions = counts["new_prediction_events"]
    return {
        "status": (
            "V6_SANITIZED_LIVE_REVIEW_RECORDED"
            if new_predictions == 1
            else "V6_SANITIZED_LIVE_REVIEW_ALREADY_RECORDED"
        ),
        "new_prediction_events": new_predictions,
        "new_outcome_events": counts["new_outcome_events"],
        "total_prediction_events": counts["prediction_events"],
        "total_matured_outcomes": counts["matured_outcome_events"],
        "release_status": release["status"],
        "ready_for_human_pilot_review": release[
            "ready_for_human_pilot_review"
        ],
        "operating_mode": operating["operating_mode"],
        "public_artifacts": [
            _relative(DEFAULT_LIVE_REVIEW_LEDGER),
            _relative(DEFAULT_LIVE_REVIEW_STATUS),
            _relative(DEFAULT_RELEASE_STATUS),
            _relative(DEFAULT_OPERATING_STATUS),
        ],
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }


def finalize_holding_review() -> dict[str, Any]:
    status = record_holding_review_from_files()
    latest = status.get("latest_review") or {}
    return {
        "status": status["status"],
        "reviewed_holding_count": latest.get("reviewed_holding_count", 0),
        "coverage_complete": latest.get("coverage_complete", True),
        "public_artifact": _relative(DEFAULT_HOLDING_REVIEW_STATUS),
        "quant_final_action": status["quant_boundary"]["quant_final_action"],
        "quant_final_action_unchanged": True,
        "buy_standard_modified": False,
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local read-only IBKR -> ChatGPT -> sanitized evidence cycle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--max-snapshot-age-seconds", type=float, default=300.0)
    prepare.add_argument("--max-account-age-seconds", type=float, default=300.0)
    prepare.add_argument("--max-context-age-seconds", type=float, default=300.0)
    subparsers.add_parser("finalize")
    subparsers.add_parser("finalize-holdings")
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            result = probe_live_cycle()
            exit_code = 0 if result["endpoint"]["reachable"] else 2
        elif args.command == "prepare":
            result = prepare_live_cycle(
                max_snapshot_age_seconds=args.max_snapshot_age_seconds,
                max_account_age_seconds=args.max_account_age_seconds,
                max_context_age_seconds=args.max_context_age_seconds,
            )
            exit_code = 0
        elif args.command == "finalize":
            result = finalize_live_cycle()
            exit_code = 0
        else:
            result = finalize_holding_review()
            exit_code = 0
    except (
        AssertionError,
        IbkrReadonlyError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        result = {
            "status": "V6_LIVE_CYCLE_UNAVAILABLE",
            "stage": args.command,
            "reason": str(exc),
            "automatic_order_allowed": False,
            "human_confirmation_required": True,
        }
        exit_code = 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
