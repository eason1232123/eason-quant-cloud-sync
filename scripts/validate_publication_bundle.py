from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.artifact_io import atomic_write_json
from scripts.validate_generated_json import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = ROOT / "docs"
DEFAULT_MANIFEST = DEFAULT_DOCS / "artifact_manifest.json"
SCHEMA_VERSION = "publication-artifact-manifest-v1"

CORE_ARTIFACTS = (
    "market_report.json",
    "latest_market_summary.json",
    "portfolio_backtest.json",
    "trade_review.json",
    "forward_validation_status.json",
    "live_review_forward_status.json",
    "v6_release_status.json",
    "v6_operating_status.json",
    "eason_signal.json",
    "latest_decision_summary.json",
    "decision_packet.json",
    "action_board.json",
)


class PublicationBundleError(AssertionError):
    """Raised when public decision artifacts cannot be relied on as one bundle."""


def _require_object(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PublicationBundleError(f"{label} must be a JSON object")
    return payload


def _load_core(docs: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name in CORE_ARTIFACTS:
        path = docs / name
        if not path.exists() or path.stat().st_size == 0:
            raise PublicationBundleError(f"missing or empty critical artifact: {path}")
        try:
            loaded[name] = _require_object(load_strict_json(path), name)
        except AssertionError as exc:
            raise PublicationBundleError(str(exc)) from exc
    return loaded


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise PublicationBundleError(f"missing required field: {'/'.join(keys)}")
        value = value[key]
    return value


def validate_cross_file_consistency(
    loaded: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    packet = loaded["decision_packet.json"]
    signal = loaded["eason_signal.json"]
    summary = loaded["latest_decision_summary.json"]
    action_board = loaded["action_board.json"]
    market = loaded["market_report.json"]
    market_summary = loaded["latest_market_summary.json"]
    portfolio = loaded["portfolio_backtest.json"]
    operating = loaded["v6_operating_status.json"]

    action_contract = _nested(action_board, "decision_contract")
    signal_summary = _nested(action_board, "signal_summary")
    final_gate = _nested(action_board, "final_gate")

    checks = {
        "final_action": [
            _nested(packet, "decision", "final_action"),
            signal.get("final_action"),
            summary.get("final_action"),
            signal_summary.get("final_action"),
            _nested(action_contract, "decision", "final_action"),
            final_gate.get("quant_signal"),
        ],
        "buy_permission": [
            _nested(packet, "decision", "buy_permission"),
            signal.get("buy_permission"),
            summary.get("buy_permission"),
            signal_summary.get("buy_permission"),
            _nested(action_contract, "decision", "buy_permission"),
        ],
        "chatgpt_review_required": [
            _nested(packet, "decision", "chatgpt_review_required"),
            summary.get("chatgpt_final_review_required"),
            final_gate.get("chatgpt_review_required"),
            _nested(action_contract, "decision", "chatgpt_review_required"),
        ],
        "decision_generated_at_utc": [
            packet.get("generated_at_utc"),
            signal.get("generated_at_utc"),
            summary.get("generated_at_utc"),
            action_contract.get("generated_at_utc"),
        ],
        "market_data_timestamp": [
            _nested(packet, "market_data", "data_timestamp"),
            _nested(signal, "market_data", "data_timestamp"),
            _nested(summary, "market_data", "data_timestamp"),
            market.get("data_timestamp"),
            market_summary.get("data_timestamp"),
            portfolio.get("data_timestamp"),
        ],
    }
    for field, values in checks.items():
        if any(value is None for value in values) or any(
            value != values[0] for value in values[1:]
        ):
            raise PublicationBundleError(
                f"cross-file field mismatch for {field}: {sorted(map(repr, values))}"
            )

    if packet != action_contract:
        raise PublicationBundleError(
            "decision_packet.json does not exactly match action_board.decision_contract"
        )
    if any(
        _nested(payload, "automatic_order_allowed") is not False
        for payload in (
            loaded["v6_release_status.json"],
            operating,
        )
    ):
        raise PublicationBundleError("release artifacts must permanently prohibit automatic orders")
    if _nested(packet, "execution_contract", "automatic_order_allowed") is not False:
        raise PublicationBundleError("decision packet must prohibit automatic orders")
    if _nested(packet, "decision", "automatic_order_allowed") is not False:
        raise PublicationBundleError("decision payload must prohibit automatic orders")
    if operating.get("human_confirmation_required") is not True:
        raise PublicationBundleError("v6 operating status must require human confirmation")

    return {
        "status": "CONSISTENT",
        "fields_checked": sorted(checks),
        "exact_decision_contract_match": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": f"docs/{path.name}",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "generated_at_utc": payload.get("generated_at_utc"),
    }


def _evidence_readiness(loaded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    operating = loaded["v6_operating_status.json"]
    release = loaded["v6_release_status.json"]
    trade = loaded["trade_review.json"]
    trade_summary = trade.get("summary") if isinstance(trade.get("summary"), dict) else {}
    actual = (
        trade.get("actual_vs_backtest")
        if isinstance(trade.get("actual_vs_backtest"), dict)
        else {}
    )
    return {
        "operating_mode": operating.get("operating_mode"),
        "human_pilot_review_ready": _nested(
            operating, "capabilities", "human_pilot_review_ready"
        ),
        "release_blockers": operating.get("release_blockers", []),
        "evidence_counts": operating.get("evidence_counts", {}),
        "evidence_thresholds": operating.get("evidence_thresholds", {}),
        "release_status": release.get("status"),
        "logged_trade_count": trade_summary.get("trade_count"),
        "actual_vs_backtest_available": actual.get("available", False),
        "actual_vs_backtest_reason": actual.get("reason"),
    }


def _concentration_readiness() -> dict[str, Any]:
    return {
        "public_model_measure": "DIRECT_MODEL_WEIGHTS_QQQ_SMH_MSFT",
        "private_account_measure": "DIRECT_SYMBOLS_ONLY_NO_ETF_LOOKTHROUGH",
        "etf_lookthrough_status": "NOT_AVAILABLE_REQUIRES_CURRENT_FUND_HOLDINGS",
        "interpretation": "Direct-symbol concentration must not be described as ETF look-through validated.",
    }


def _large_artifact_access() -> dict[str, Any]:
    return {
        "market_report_path": "docs/market_report.json",
        "connector_inline_content_may_be_empty": True,
        "verification_method": "Use this manifest's size_bytes and sha256; inspect decision_packet.json or chatgpt_snapshot.json for compact content.",
    }


def build_manifest(
    docs: Path = DEFAULT_DOCS,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    loaded = _load_core(docs)
    consistency = validate_cross_file_consistency(loaded)
    packet = loaded["decision_packet.json"]
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.utcoffset() is None:
        raise PublicationBundleError("manifest generation timestamp must include a timezone")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "status": "VALIDATED",
        "purpose": "Small connector-readable integrity proof for the public decision bundle, including large artifacts that connectors may not return inline.",
        "decision_batch": {
            "decision_generated_at_utc": packet.get("generated_at_utc"),
            "market_data_timestamp": _nested(packet, "market_data", "data_timestamp"),
            "final_action": _nested(packet, "decision", "final_action"),
            "buy_permission": _nested(packet, "decision", "buy_permission"),
            "chatgpt_review_required": _nested(
                packet, "decision", "chatgpt_review_required"
            ),
        },
        "cross_file_consistency": consistency,
        "artifacts": [
            _artifact_record(docs / name, loaded[name]) for name in CORE_ARTIFACTS
        ],
        "evidence_readiness": _evidence_readiness(loaded),
        "concentration_readiness": _concentration_readiness(),
        "large_artifact_access": _large_artifact_access(),
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
    }


def validate_manifest(
    manifest: dict[str, Any],
    docs: Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PublicationBundleError("unsupported publication manifest schema")
    if manifest.get("status") != "VALIDATED":
        raise PublicationBundleError("publication manifest status is not VALIDATED")
    if manifest.get("automatic_order_allowed") is not False:
        raise PublicationBundleError("publication manifest must prohibit automatic orders")
    if manifest.get("human_confirmation_required") is not True:
        raise PublicationBundleError("publication manifest must require human confirmation")
    generated_at = manifest.get("generated_at_utc")
    try:
        parsed_generated_at = datetime.fromisoformat(str(generated_at))
    except ValueError as exc:
        raise PublicationBundleError("manifest generated_at_utc is not ISO-8601") from exc
    if parsed_generated_at.utcoffset() is None:
        raise PublicationBundleError("manifest generated_at_utc must include a timezone")

    loaded = _load_core(docs)
    consistency = validate_cross_file_consistency(loaded)
    if manifest.get("cross_file_consistency") != consistency:
        raise PublicationBundleError("manifest cross-file consistency record is stale")

    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise PublicationBundleError("manifest artifacts must be a list")
    expected_paths = {f"docs/{name}" for name in CORE_ARTIFACTS}
    actual_paths = {
        record.get("path") for record in records if isinstance(record, dict)
    }
    if actual_paths != expected_paths or len(records) != len(CORE_ARTIFACTS):
        raise PublicationBundleError("manifest artifact membership changed")
    for record in records:
        if not isinstance(record, dict):
            raise PublicationBundleError("manifest artifact record must be an object")
        relative = Path(str(record["path"]))
        if relative.parts != ("docs", relative.name):
            raise PublicationBundleError(
                f"manifest artifact path must be a direct docs child: {relative}"
            )
        path = docs / relative.name
        if record.get("size_bytes") != path.stat().st_size:
            raise PublicationBundleError(f"artifact size mismatch: {path}")
        if record.get("sha256") != _sha256(path):
            raise PublicationBundleError(f"artifact SHA-256 mismatch: {path}")
        expected_record = _artifact_record(path, loaded[relative.name])
        if record != expected_record:
            raise PublicationBundleError(f"artifact manifest metadata mismatch: {path}")

    packet = loaded["decision_packet.json"]
    operating = loaded["v6_operating_status.json"]
    decision_batch = manifest.get("decision_batch")
    expected_batch = {
        "decision_generated_at_utc": packet.get("generated_at_utc"),
        "market_data_timestamp": _nested(packet, "market_data", "data_timestamp"),
        "final_action": _nested(packet, "decision", "final_action"),
        "buy_permission": _nested(packet, "decision", "buy_permission"),
        "chatgpt_review_required": _nested(
            packet, "decision", "chatgpt_review_required"
        ),
    }
    if decision_batch != expected_batch:
        raise PublicationBundleError("manifest decision batch is stale")
    if manifest.get("evidence_readiness") != _evidence_readiness(loaded):
        raise PublicationBundleError("manifest evidence readiness is stale")
    if manifest.get("concentration_readiness") != _concentration_readiness():
        raise PublicationBundleError("manifest concentration disclosure is stale")
    if manifest.get("large_artifact_access") != _large_artifact_access():
        raise PublicationBundleError("manifest large-artifact guidance is stale")

    return {
        "status": "VALIDATED",
        "artifact_count": len(records),
        "decision_generated_at_utc": packet.get("generated_at_utc"),
        "market_data_timestamp": _nested(packet, "market_data", "data_timestamp"),
        "operating_mode": operating.get("operating_mode"),
    }


def write_manifest(
    docs: Path = DEFAULT_DOCS,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest = build_manifest(docs)
    atomic_write_json(manifest_path, manifest)
    validate_manifest(load_strict_json(manifest_path), docs)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the public decision bundle and its connector-readable manifest"
    )
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    try:
        if args.write_manifest:
            manifest = write_manifest()
            print(
                {
                    "status": manifest["status"],
                    "artifact_count": len(manifest["artifacts"]),
                    "manifest": str(DEFAULT_MANIFEST.relative_to(ROOT)),
                }
            )
        else:
            if not DEFAULT_MANIFEST.exists() or DEFAULT_MANIFEST.stat().st_size == 0:
                raise PublicationBundleError(
                    f"missing or empty publication manifest: {DEFAULT_MANIFEST}"
                )
            print(validate_manifest(load_strict_json(DEFAULT_MANIFEST)))
        return 0
    except (AssertionError, OSError, UnicodeError, ValueError) as exc:
        print(f"publication bundle validation FAILED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
