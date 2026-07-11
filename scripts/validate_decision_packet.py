from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_clock import MARKET_TIMEZONE
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
)

DEFAULT_PACKET = ROOT / "docs" / "decision_packet.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "decision_packet.schema.json"


def reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_non_finite)


FORMAT_CHECKER = jsonschema.FormatChecker()


@FORMAT_CHECKER.checks("date")
def is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    try:
        return len(value) == 10 and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


@FORMAT_CHECKER.checks("date-time")
def is_rfc3339_datetime(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 20 or value[10] != "T":
        return False
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except ValueError:
        return False


def validate_schema(packet: dict[str, Any], schema: dict[str, Any]) -> None:
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema, format_checker=FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(packet), key=lambda item: list(item.absolute_path))
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise AssertionError(f"decision packet schema validation failed:\n{details}")


def validate_invariants(packet: dict[str, Any]) -> dict[str, Any]:
    market_data = packet["market_data"]
    quality = packet["data_quality"]
    scope = packet["portfolio_scope"]
    gate = packet["evidence_gate"]
    decision = packet["decision"]
    candidates = packet["candidates"]
    risks = packet["risks"]
    execution = packet["execution_contract"]

    assert gate["automatic_order_allowed"] is False
    assert decision["automatic_order_allowed"] is False
    assert execution["automatic_order_allowed"] is False
    assert gate["quant_signal"] == decision["final_action"]
    assert gate["recommended_default_action"] == decision["default_human_action"]

    assert quality["stale_tickers_in_packet_count"] == len(quality["stale_tickers"])
    assert quality["stale_tickers_in_packet_count"] <= quality["stale_ticker_count"]
    counts = (
        quality["loaded_ticker_count"],
        quality["configured_ticker_count"],
        quality["fresh_request_count"],
        quality["errors_count"],
    )
    counts_complete = all(isinstance(value, int) and not isinstance(value, bool) for value in counts)
    market_metadata_complete = market_data["metadata_status"] == "COMPLETE"
    quality_metadata_complete = quality["data_quality_metadata_status"] == "COMPLETE"
    assert quality_metadata_complete == (counts_complete and market_metadata_complete)
    if quality_metadata_complete:
        assert quality["source_report_generated_at_utc"] == market_data["report_generated_at_utc"]
        assert quality["loaded_ticker_count"] > 0, "complete market metadata requires loaded ticker data"
        assert (
            quality["fresh_ticker_count"] + quality["stale_ticker_count"]
            == quality["loaded_ticker_count"]
        ), "fresh and stale ticker counts must partition loaded tickers"
        assert quality["loaded_ticker_count"] <= quality["configured_ticker_count"]
        assert (
            quality["latest_price_date_max"] == market_data["data_timestamp"]
        ), "observed latest price date must equal the market-data timestamp"

    if quality["data_status"] == "FRESH":
        assert quality["fresh_ticker_count"] > 0
        assert quality["stale_ticker_count"] == 0
    elif quality["data_status"] == "PARTIAL_STALE_WATCHLIST":
        assert quality["fresh_ticker_count"] > 0
        assert quality["stale_ticker_count"] > 0

    if market_metadata_complete:
        assert market_data["source"]
        assert market_data["market_timezone"] == MARKET_TIMEZONE
        assert market_data["report_generated_at_utc"]
        assert market_data["data_timestamp"]
        assert market_data["data_timestamp_granularity"] == DATA_TIMESTAMP_GRANULARITY
        assert market_data["data_timestamp_status"] == "AVAILABLE"
        assert market_data["price_frequency"] == PRICE_FREQUENCY
        assert market_data["price_adjustment_policy"] == PRICE_ADJUSTMENT_POLICY

    reference_status = quality["reference_market_date_status"]
    reported_date = quality["reported_expected_market_date"]
    reference_date = quality["reference_market_date"]
    if reference_status == "ALIGNED":
        assert reported_date == reference_date
    elif reference_status == "REPORTED_EXPECTED_DATE_MISSING":
        assert reported_date is None
    elif reference_status == "REPORTED_EXPECTED_DATE_MISMATCH":
        assert reported_date is not None and reported_date != reference_date
    else:
        raise AssertionError(f"unsupported reference market date status: {reference_status}")
    reference = date.fromisoformat(reference_date)

    assert len(candidates["top_actionable"]) == min(candidates["actionable_count"], 5)
    if candidates["actionable_count"]:
        assert quality["fresh_ticker_count"] > 0
    assert all(
        row["data_fresh"]
        and row["decision_eligible"]
        and row["status"] == "QUANT_PASS_NEEDS_CHATGPT_REVIEW"
        and row["latest_date"] == reference_date
        for row in candidates["top_actionable"]
    )

    assert "CASH" not in scope["tickers"]
    assert len(scope["tickers"]) == len(set(scope["tickers"]))
    assert scope["contains_private_shares_or_cash"] is False
    assert scope["actual_holdings_check_required_by_chatgpt"] is True
    assert (scope["portfolio_context_reason"] is None) == scope["portfolio_context_available"]
    model_regime = packet["market_context"]["model_regime"]
    if scope["portfolio_context_available"]:
        assert model_regime in {"base", "defensive", "severe_defensive"}
    else:
        assert model_regime is None

    assert all(
        row["risk_scope"] == "model_portfolio"
        and row["severity_0_100"] >= 65
        and row["data_fresh"]
        and row["decision_eligible"]
        and row["latest_date"] == reference_date
        for row in risks["model_portfolio_high"]
    )
    assert all(
        row["risk_scope"] == "watchlist_advisory"
        and row["severity_0_100"] >= 65
        and row["data_fresh"]
        and row["decision_eligible"]
        and row["latest_date"] == reference_date
        for row in risks["watchlist_high_advisory"]
    )
    assert all(
        not row["data_fresh"]
        and not row["decision_eligible"]
        and (
            row["latest_date"] is None
            or date.fromisoformat(row["latest_date"]) < reference
        )
        for row in risks["stale_excluded"]
    )

    final_action = decision["final_action"]
    if final_action != "NO_TRADE":
        assert decision["chatgpt_review_required"] is True
    if reference_status != "ALIGNED":
        assert final_action == "DATA_REVIEW_REQUIRED"
        assert decision["buy_permission"] == "BLOCKED_BY_MARKET_DATE_CONTEXT"
        assert quality["data_status"] == "EXPECTED_MARKET_DATE_MISSING_OR_INVALID"
    elif not quality_metadata_complete:
        assert final_action == "DATA_REVIEW_REQUIRED"
        assert decision["buy_permission"] == "BLOCKED_BY_SOURCE_METADATA"
        assert quality["data_status"] == "MARKET_DATA_METADATA_MISSING"
    elif not scope["portfolio_context_available"]:
        assert final_action == "DATA_REVIEW_REQUIRED"
        assert decision["buy_permission"] == "BLOCKED_BY_PORTFOLIO_CONTEXT"
        assert quality["data_status"] == "PORTFOLIO_CONTEXT_MISSING"
    elif final_action == "DATA_REVIEW_REQUIRED":
        assert decision["buy_permission"] == "BLOCKED_BY_MODEL_PORTFOLIO_DATA"
        assert quality["data_status"] == "STALE_MODEL_PORTFOLIO_DATA"
    elif final_action == "RISK_REVIEW_REQUIRED":
        assert quality["data_status"] in {"FRESH", "PARTIAL_STALE_WATCHLIST"}
        assert decision["buy_permission"] == "BLOCKED_BY_MODEL_PORTFOLIO_RISK"
        assert risks["model_portfolio_high"]
    elif final_action == "BUY_CANDIDATE_REVIEW_REQUIRED":
        assert quality["data_status"] in {"FRESH", "PARTIAL_STALE_WATCHLIST"}
        assert decision["buy_permission"] == "CHATGPT_LIVE_REVIEW_REQUIRED"
        assert candidates["actionable_count"] > 0 and candidates["top_actionable"]
        assert not risks["model_portfolio_high"]
    elif final_action == "NO_TRADE":
        assert quality["data_status"] in {"FRESH", "PARTIAL_STALE_WATCHLIST"}
        assert decision["buy_permission"] == "NO_QUANT_CANDIDATE"
        assert candidates["actionable_count"] == 0
        assert not risks["model_portfolio_high"]
    else:
        raise AssertionError(f"unsupported final action: {final_action}")

    if market_data["data_timestamp"] is not None:
        assert date.fromisoformat(market_data["data_timestamp"]) <= reference
    for key in ("latest_price_date_max", "latest_price_date_min"):
        if quality[key] is not None:
            assert date.fromisoformat(quality[key]) <= reference

    return {
        "schema_version": packet["schema_version"],
        "final_action": final_action,
        "data_status": quality["data_status"],
        "actionable_count": candidates["actionable_count"],
        "model_high_risk_count": len(risks["model_portfolio_high"]),
    }


def validate_packet(
    packet_path: Path = DEFAULT_PACKET,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    packet = load_json(packet_path)
    schema = load_json(schema_path)
    validate_schema(packet, schema)
    return validate_invariants(packet)


if __name__ == "__main__":
    print(validate_packet())
