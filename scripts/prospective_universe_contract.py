from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from config import TICKERS
from scripts.strategy_contract import STRATEGY_FINGERPRINT
from scripts.validate_validation_split import (
    canonical_fingerprint,
    load_strict_json,
    split_manifest_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE_CONTRACT = ROOT / "config" / "prospective_universe.json"
UNIVERSE_CONTRACT_VERSION = "v6-prospective-universe-v1"
PROSPECTIVE_UNIVERSE_STATUS = "FROZEN_COMPLETE_DAILY_COHORTS"
PROSPECTIVE_SURVIVORSHIP_BIAS_STATUS = (
    "CONTROLLED_FOR_FROZEN_UNIVERSE_COHORT_MEMBERSHIP"
)
HISTORICAL_SURVIVORSHIP_BIAS_STATUS = (
    "KNOWN_UNCONTROLLED_RETROSPECTIVE_RESEARCH_UNIVERSE"
)
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,19}$")
UNIVERSE_CONTRACT_FIELDS = {
    "schema_version",
    "status",
    "frozen_on_date",
    "effective_after_market_date",
    "selection_basis",
    "ticker_source",
    "amendment_policy",
    "strategy_fingerprint",
    "split_manifest_fingerprint",
    "ticker_count",
    "tickers",
    "historical_survivorship_bias_status",
    "prospective_survivorship_bias_status",
    "private_account_data_allowed",
}


class ProspectiveUniverseContractError(ValueError):
    """Raised when the frozen prospective universe is incomplete or has drifted."""


def _required_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ProspectiveUniverseContractError(f"{field} must be an ISO market date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ProspectiveUniverseContractError(
            f"{field} must be an ISO market date"
        ) from exc
    if parsed.isoformat() != value:
        raise ProspectiveUniverseContractError(f"{field} must use YYYY-MM-DD")
    return parsed


def _normalized_tickers(values: Any, field: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ProspectiveUniverseContractError(f"{field} must be a non-empty ticker list")
    tickers: list[str] = []
    for raw in values:
        if (
            not isinstance(raw, str)
            or raw != raw.strip().upper()
            or raw == "CASH"
            or not TICKER_PATTERN.fullmatch(raw)
        ):
            raise ProspectiveUniverseContractError(
                f"{field} contains an invalid public ticker: {raw!r}"
            )
        tickers.append(raw)
    if tickers != sorted(tickers) or len(tickers) != len(set(tickers)):
        raise ProspectiveUniverseContractError(
            f"{field} must be sorted, unique, and deterministic"
        )
    return tickers


def _expected_runtime_tickers(values: Iterable[str] = TICKERS) -> list[str]:
    return _normalized_tickers(sorted(values), "config.py:TICKERS")


def validate_prospective_universe_contract(
    contract: dict[str, Any],
    *,
    split_manifest: dict[str, Any],
    expected_tickers: Iterable[str] = TICKERS,
) -> dict[str, Any]:
    if set(contract) != UNIVERSE_CONTRACT_FIELDS:
        raise ProspectiveUniverseContractError(
            "prospective universe contract field membership changed"
        )
    if contract.get("schema_version") != UNIVERSE_CONTRACT_VERSION:
        raise ProspectiveUniverseContractError(
            "unsupported prospective universe contract schema_version"
        )
    if contract.get("status") != "FROZEN":
        raise ProspectiveUniverseContractError(
            "prospective universe contract must be FROZEN"
        )
    frozen_on = _required_iso_date(contract.get("frozen_on_date"), "frozen_on_date")
    effective_after = _required_iso_date(
        contract.get("effective_after_market_date"),
        "effective_after_market_date",
    )
    prospective = split_manifest.get("prospective_partition")
    if not isinstance(prospective, dict):
        raise ProspectiveUniverseContractError(
            "validation split prospective_partition is missing"
        )
    split_boundary = _required_iso_date(
        prospective.get("last_seen_market_date"),
        "prospective_partition.last_seen_market_date",
    )
    split_frozen_on = _required_iso_date(
        split_manifest.get("frozen_on_date"),
        "validation_split.frozen_on_date",
    )
    if effective_after != split_boundary:
        raise ProspectiveUniverseContractError(
            "prospective universe boundary does not match the validation split"
        )
    if frozen_on != split_frozen_on:
        raise ProspectiveUniverseContractError(
            "prospective universe freeze date does not match the validation split"
        )

    for field in ("selection_basis", "ticker_source", "amendment_policy"):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise ProspectiveUniverseContractError(f"{field} must be non-empty")
    if contract.get("strategy_fingerprint") != STRATEGY_FINGERPRINT:
        raise ProspectiveUniverseContractError(
            "prospective universe strategy fingerprint drifted"
        )
    expected_split_fingerprint = split_manifest_fingerprint(split_manifest)
    if contract.get("split_manifest_fingerprint") != expected_split_fingerprint:
        raise ProspectiveUniverseContractError(
            "prospective universe validation split fingerprint drifted"
        )

    tickers = _normalized_tickers(contract.get("tickers"), "tickers")
    runtime_tickers = _expected_runtime_tickers(expected_tickers)
    if tickers != runtime_tickers:
        missing = sorted(set(runtime_tickers) - set(tickers))
        extra = sorted(set(tickers) - set(runtime_tickers))
        raise ProspectiveUniverseContractError(
            "frozen prospective universe differs from config.py:TICKERS: "
            f"missing={missing}, extra={extra}"
        )
    ticker_count = contract.get("ticker_count")
    if (
        isinstance(ticker_count, bool)
        or not isinstance(ticker_count, int)
        or ticker_count != len(tickers)
    ):
        raise ProspectiveUniverseContractError(
            "prospective universe ticker_count does not match tickers"
        )
    if (
        contract.get("historical_survivorship_bias_status")
        != HISTORICAL_SURVIVORSHIP_BIAS_STATUS
    ):
        raise ProspectiveUniverseContractError(
            "historical survivorship-bias status changed"
        )
    if (
        contract.get("prospective_survivorship_bias_status")
        != PROSPECTIVE_SURVIVORSHIP_BIAS_STATUS
    ):
        raise ProspectiveUniverseContractError(
            "prospective survivorship-bias status changed"
        )
    if contract.get("private_account_data_allowed") is not False:
        raise ProspectiveUniverseContractError(
            "prospective universe contract must prohibit private account data"
        )

    return {
        "schema_version": UNIVERSE_CONTRACT_VERSION,
        "status": PROSPECTIVE_UNIVERSE_STATUS,
        "frozen_on_date": frozen_on.isoformat(),
        "effective_after_market_date": effective_after.isoformat(),
        "ticker_count": len(tickers),
        "tickers": tickers,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "split_manifest_fingerprint": expected_split_fingerprint,
        "prospective_universe_fingerprint": canonical_fingerprint(contract),
        "historical_survivorship_bias_status": HISTORICAL_SURVIVORSHIP_BIAS_STATUS,
        "prospective_survivorship_bias_status": PROSPECTIVE_SURVIVORSHIP_BIAS_STATUS,
    }


def load_and_validate_prospective_universe_contract(
    path: Path = DEFAULT_UNIVERSE_CONTRACT,
    *,
    split_manifest: dict[str, Any],
    expected_tickers: Iterable[str] = TICKERS,
) -> dict[str, Any]:
    try:
        contract = load_strict_json(path)
    except ValueError as exc:
        raise ProspectiveUniverseContractError(
            f"invalid prospective universe contract {path}: {exc}"
        ) from exc
    return validate_prospective_universe_contract(
        contract,
        split_manifest=split_manifest,
        expected_tickers=expected_tickers,
    )
