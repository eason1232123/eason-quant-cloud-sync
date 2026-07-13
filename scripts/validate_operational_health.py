from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_decision_packet import (
    DEFAULT_PACKET,
    DEFAULT_SCHEMA,
    load_json,
    validate_packet,
)


HEALTHY_DATA_STATUSES = {"FRESH", "PARTIAL_STALE_WATCHLIST"}
DATA_BLOCKING_PERMISSIONS = {
    "BLOCKED_BY_MODEL_PORTFOLIO_DATA",
    "BLOCKED_BY_MARKET_DATE_CONTEXT",
    "BLOCKED_BY_SOURCE_METADATA",
    "BLOCKED_BY_PORTFOLIO_CONTEXT",
}


def validate_operational_state(packet: dict[str, Any]) -> dict[str, Any]:
    market_data = packet["market_data"]
    quality = packet["data_quality"]
    gate = packet["evidence_gate"]
    decision = packet["decision"]

    failures: list[str] = []
    if market_data["metadata_status"] != "COMPLETE":
        failures.append("market-data metadata is incomplete")
    if quality["data_quality_metadata_status"] != "COMPLETE":
        failures.append("data-quality counts or source metadata are incomplete")
    if quality["reference_market_date_status"] != "ALIGNED":
        failures.append("reported and runtime reference market dates are not aligned")
    if quality["data_status"] not in HEALTHY_DATA_STATUSES:
        failures.append(f"market-data status is {quality['data_status']}")
    if decision["final_action"] == "DATA_REVIEW_REQUIRED":
        failures.append("decision is blocked for data review")
    if decision["buy_permission"] in DATA_BLOCKING_PERMISSIONS:
        failures.append(f"buy permission is {decision['buy_permission']}")
    if gate["recommended_default_action"] == "REFRESH_DATA_BEFORE_DECISION":
        failures.append("evidence gate requires a data or evidence refresh")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise AssertionError(f"operational health gate failed:\n{details}")

    return {
        "status": "HEALTHY",
        "data_status": quality["data_status"],
        "reference_market_date": quality["reference_market_date"],
        "final_action": decision["final_action"],
        "recommended_default_action": gate["recommended_default_action"],
    }


def validate_operational_health(
    packet_path: Path = DEFAULT_PACKET,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    validate_packet(packet_path, schema_path)
    return validate_operational_state(load_json(packet_path))


if __name__ == "__main__":
    print(validate_operational_health())
