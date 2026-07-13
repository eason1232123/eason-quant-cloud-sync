from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_clock import (  # noqa: E402
    MARKET_TIMEZONE,
    latest_completed_us_market_weekday,
    parse_market_date,
    parse_timestamp,
)
from scripts.market_data_contract import (  # noqa: E402
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
    extract_market_data_metadata,
)
from scripts.strategy_contract import (  # noqa: E402
    ENTRY_RULE_SPECS,
    FORWARD_HORIZONS,
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    execution_cost_assumptions,
    net_return_after_round_trip_costs,
)
from scripts.validate_validation_split import (  # noqa: E402
    ValidationSplitError,
    assert_finite_json,
    canonical_fingerprint,
    load_anchor_market_dates,
    load_strict_json,
    validate_split_manifest,
)


DEFAULT_SPLIT = ROOT / "config" / "validation_split.json"
DEFAULT_PACKET = ROOT / "docs" / "decision_packet.json"
DEFAULT_REPORT = ROOT / "docs" / "market_report.json"
DEFAULT_LEDGER = ROOT / "docs" / "forward_signal_ledger.jsonl"
DEFAULT_SUMMARY = ROOT / "docs" / "forward_validation_status.json"
DEFAULT_PRICES_DIR = ROOT / "docs"

LEDGER_SCHEMA_VERSION = "v6-forward-ledger-event-v1"
SUMMARY_SCHEMA_VERSION = "v6-forward-validation-status-v1"
PREDICTION_NAMESPACE = "eason-v6-public-forward-prediction-v1"
OUTCOME_NAMESPACE = "eason-v6-public-forward-outcome-v1"
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,19}$")
FORBIDDEN_PRIVATE_KEYS = {
    "account",
    "account_id",
    "account_number",
    "cash",
    "cash_balance",
    "holdings",
    "net_liquidation",
    "order_id",
    "position_size",
    "quantity",
    "shares",
}


class ForwardLedgerError(ValueError):
    """Raised when prospective evidence cannot be recorded without ambiguity."""


def _reject_constant(value: str) -> None:
    raise ForwardLedgerError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForwardLedgerError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def load_public_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise ForwardLedgerError(f"missing or empty {label}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForwardLedgerError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ForwardLedgerError(f"{label} must be a JSON object")
    try:
        assert_finite_json(value)
    except ValidationSplitError as exc:
        raise ForwardLedgerError(f"invalid {label}: {exc}") from exc
    return value


def _parse_required_date(value: Any, context: str) -> date:
    parsed = parse_market_date(value)
    if parsed is None or not isinstance(value, str) or parsed.isoformat() != value:
        raise ForwardLedgerError(f"{context} must be an ISO market date")
    return parsed


def _parse_required_timestamp(value: Any, context: str) -> datetime:
    parsed = parse_timestamp(value)
    if parsed is None or parsed.utcoffset() is None:
        raise ForwardLedgerError(f"{context} must be a timezone-aware timestamp")
    return parsed


def _assert_no_private_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_PRIVATE_KEYS:
                raise ForwardLedgerError(f"private account field is prohibited in forward ledger: {path}.{key}")
            _assert_no_private_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_private_keys(item, f"{path}[{index}]")


def _event_hash(event: dict[str, Any]) -> str:
    payload = copy.deepcopy(event)
    payload.pop("immutable_payload_hash", None)
    return canonical_fingerprint(payload)


def _seal_event(event: dict[str, Any]) -> dict[str, Any]:
    assert_finite_json(event)
    _assert_no_private_keys(event)
    sealed = copy.deepcopy(event)
    sealed["immutable_payload_hash"] = _event_hash(sealed)
    return sealed


def _prediction_event_id(
    *,
    split_fingerprint: str,
    strategy_fingerprint: str,
    observation_date: str,
    ticker: str,
) -> str:
    identity = {
        "namespace": PREDICTION_NAMESPACE,
        "split_manifest_fingerprint": split_fingerprint,
        "strategy_fingerprint": strategy_fingerprint,
        "observation_market_date": observation_date,
        "ticker": ticker,
    }
    return f"pred_{canonical_fingerprint(identity)}"


def _outcome_event_id(prediction_event_id: str, horizon_bars: int) -> str:
    identity = {
        "namespace": OUTCOME_NAMESPACE,
        "prediction_event_id": prediction_event_id,
        "horizon_bars": horizon_bars,
    }
    return f"out_{canonical_fingerprint(identity)}"


def _validate_market_inputs(
    packet: dict[str, Any],
    report: dict[str, Any],
    split_result: dict[str, Any],
    as_of_market_date: date,
) -> tuple[date, dict[str, Any]]:
    metadata = extract_market_data_metadata(report)
    if metadata.get("metadata_status") != "COMPLETE":
        raise ForwardLedgerError("market_report market-data metadata is missing or invalid")
    report_timestamp = _parse_required_timestamp(
        metadata.get("report_generated_at_utc"),
        "market_report.generated_at_utc",
    )
    observation = _parse_required_date(metadata.get("data_timestamp"), "market_report.data_timestamp")
    if observation > as_of_market_date:
        raise ForwardLedgerError(
            f"market_report contains future data ({observation}) relative to as-of market date {as_of_market_date}"
        )
    if report_timestamp.date() < observation:
        raise ForwardLedgerError("market_report generation timestamp predates its market-data timestamp")

    expected_report_contract = {
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    actual_report_contract = {
        "strategy_contract_version": report.get("strategy_contract_version"),
        "rule_fingerprint": report.get("rule_fingerprint"),
        "strategy_fingerprint": report.get("strategy_fingerprint"),
    }
    if actual_report_contract != expected_report_contract:
        raise ForwardLedgerError("market_report strategy contract or fingerprint is missing or mismatched")
    for key in ("strategy_contract_version", "rule_fingerprint", "strategy_fingerprint"):
        if actual_report_contract[key] != split_result[key]:
            raise ForwardLedgerError(f"market_report {key} does not match the frozen validation split")

    packet_market = packet.get("market_data")
    packet_quality = packet.get("data_quality")
    if not isinstance(packet_market, dict) or not isinstance(packet_quality, dict):
        raise ForwardLedgerError("decision_packet market_data and data_quality objects are required")
    if packet_market.get("metadata_status") != "COMPLETE":
        raise ForwardLedgerError("decision_packet market metadata is not COMPLETE")
    comparisons = {
        "source": metadata["source"],
        "market_timezone": metadata["market_timezone"],
        "report_generated_at_utc": metadata["report_generated_at_utc"],
        "data_timestamp": metadata["data_timestamp"],
        "data_timestamp_granularity": metadata["data_timestamp_granularity"],
        "data_timestamp_status": metadata["data_timestamp_status"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
    }
    if any(packet_market.get(key) != expected for key, expected in comparisons.items()):
        raise ForwardLedgerError("decision_packet market metadata does not match market_report")
    if packet_quality.get("reference_market_date_status") != "ALIGNED":
        raise ForwardLedgerError("decision_packet reference market date is not ALIGNED")
    if packet_quality.get("reference_market_date") != observation.isoformat():
        raise ForwardLedgerError("decision_packet reference date does not match market_report data timestamp")
    if packet_quality.get("source_report_generated_at_utc") != metadata["report_generated_at_utc"]:
        raise ForwardLedgerError("decision_packet was not derived from this market_report generation")

    universe = report.get("universe")
    if not isinstance(universe, dict):
        raise ForwardLedgerError("market_report.universe is required")
    if universe.get("expected_latest_market_date") != observation.isoformat():
        raise ForwardLedgerError("market_report expected market date does not match observed data timestamp")

    return observation, metadata


def _ticker_universe(report: dict[str, Any]) -> list[str]:
    technicals = report.get("technicals")
    update_log = report.get("update_log")
    date_by_ticker = report.get("data_timestamp_by_ticker")
    price_basis = report.get("price_basis_by_ticker")
    if not isinstance(technicals, dict) or not isinstance(update_log, dict):
        raise ForwardLedgerError("market_report technicals and update_log must be objects")
    if not isinstance(date_by_ticker, dict) or not isinstance(price_basis, dict):
        raise ForwardLedgerError("market_report per-ticker date and price-basis metadata are required")
    tickers = set(technicals) | set(update_log) | set(date_by_ticker) | set(price_basis)
    normalized: set[str] = set()
    for raw in tickers:
        ticker = str(raw).strip().upper()
        if ticker != raw or not TICKER_PATTERN.fullmatch(ticker) or ticker == "CASH":
            raise ForwardLedgerError(f"invalid public-model ticker in market_report: {raw!r}")
        normalized.add(ticker)
    universe = report.get("universe", {})
    configured_count = universe.get("configured_ticker_count")
    loaded_count = universe.get("loaded_ticker_count")
    if (
        isinstance(configured_count, bool)
        or not isinstance(configured_count, int)
        or configured_count <= 0
        or isinstance(loaded_count, bool)
        or not isinstance(loaded_count, int)
        or loaded_count < 0
    ):
        raise ForwardLedgerError("market_report universe counts are missing or invalid")
    if loaded_count > configured_count or len(normalized) != configured_count:
        raise ForwardLedgerError(
            "cannot record all configured tickers: public report ticker names or universe counts differ"
        )
    return sorted(normalized)


def _ticker_prediction(
    ticker: str,
    report: dict[str, Any],
    observation: date,
) -> dict[str, Any]:
    technicals = report["technicals"]
    date_by_ticker = report["data_timestamp_by_ticker"]
    price_basis_by_ticker = report["price_basis_by_ticker"]
    technical = technicals.get(ticker)
    ticker_timestamp_raw = date_by_ticker.get(ticker)
    ticker_timestamp = parse_market_date(ticker_timestamp_raw)
    if ticker_timestamp is not None and ticker_timestamp > observation:
        raise ForwardLedgerError(f"{ticker} has future per-ticker data relative to report timestamp")

    skip_reasons: list[str] = []
    active_signals: dict[str, Any] | None = None
    technical_latest: date | None = None
    if not isinstance(technical, dict):
        skip_reasons.append("TECHNICAL_ANALYSIS_MISSING")
    else:
        technical_latest = parse_market_date(technical.get("latest_date"))
        if technical_latest is not None and technical_latest > observation:
            raise ForwardLedgerError(f"{ticker} technical analysis is future-dated")
        active_signals = technical.get("active_signals")
        if not isinstance(active_signals, dict):
            skip_reasons.append("ACTIVE_SIGNALS_MISSING_OR_INVALID")

    if ticker_timestamp is None or technical_latest is None:
        skip_reasons.append("TICKER_MARKET_DATE_MISSING")
    elif ticker_timestamp != technical_latest:
        raise ForwardLedgerError(f"{ticker} technical date and per-ticker market timestamp disagree")
    elif ticker_timestamp != observation:
        skip_reasons.append("STALE_TICKER_MARKET_DATA")

    signal_vector: dict[str, bool] = {}
    if active_signals is not None:
        for rule in ENTRY_RULE_SPECS:
            value = active_signals.get(rule)
            if not isinstance(value, bool):
                skip_reasons.append("ENTRY_SIGNAL_VECTOR_INCOMPLETE")
                signal_vector = {}
                break
            signal_vector[rule] = value

    price_basis = price_basis_by_ticker.get(ticker)
    if price_basis not in {"adjusted", "unadjusted"}:
        skip_reasons.append("PRICE_BASIS_MISSING_OR_INVALID")

    skip_reasons = sorted(set(skip_reasons))
    if skip_reasons:
        state = "SKIPPED"
        active_entry_rules: list[str] = []
    else:
        active_entry_rules = sorted(rule for rule, active in signal_vector.items() if active)
        state = "ACTIVE" if active_entry_rules else "NO_SIGNAL"
    active_risk_rules = sorted(
        rule
        for rule, active in (active_signals or {}).items()
        if rule not in ENTRY_RULE_SPECS and isinstance(active, bool) and active
    )
    outcomes = [
        {
            "horizon_bars": horizon,
            "status": "PENDING" if state != "SKIPPED" else "NOT_EVALUATED",
            "return_definition": "close[t+1+h] / close[t+1] - 1",
            "reason": None if state != "SKIPPED" else "prediction_skipped",
        }
        for horizon in FORWARD_HORIZONS
    ]
    return {
        "ticker": ticker,
        "observation_market_date": observation.isoformat(),
        "state": state,
        "decision_eligible": state != "SKIPPED",
        "ticker_market_date": ticker_timestamp.isoformat() if ticker_timestamp else None,
        "price_basis": price_basis if price_basis in {"adjusted", "unadjusted"} else None,
        "signal_vector": signal_vector,
        "active_entry_signals": active_entry_rules,
        "active_risk_signals": active_risk_rules,
        "skip_reasons": skip_reasons,
        "expected_outcomes": outcomes,
    }


def _decision_context(packet: dict[str, Any]) -> dict[str, Any]:
    decision = packet.get("decision")
    quality = packet.get("data_quality")
    if not isinstance(decision, dict) or not isinstance(quality, dict):
        raise ForwardLedgerError("decision_packet decision and data_quality objects are required")
    final_action = decision.get("final_action")
    buy_permission = decision.get("buy_permission")
    data_status = quality.get("data_status")
    if not all(isinstance(value, str) and value for value in (final_action, buy_permission, data_status)):
        raise ForwardLedgerError("decision_packet public decision context is incomplete")
    return {
        "final_action": final_action,
        "buy_permission": buy_permission,
        "data_status": data_status,
        "automatic_order_allowed": False,
    }


def _prediction_event(
    prediction: dict[str, Any],
    *,
    metadata: dict[str, Any],
    decision_context: dict[str, Any],
    split_result: dict[str, Any],
) -> dict[str, Any]:
    event_id = _prediction_event_id(
        split_fingerprint=split_result["split_manifest_fingerprint"],
        strategy_fingerprint=STRATEGY_FINGERPRINT,
        observation_date=prediction["observation_market_date"],
        ticker=prediction["ticker"],
    )
    return _seal_event(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": "PREDICTION",
            "event_id": event_id,
            "recorded_at_utc": metadata["report_generated_at_utc"],
            "evidence_classification": "PROSPECTIVE",
            "split_manifest_fingerprint": split_result["split_manifest_fingerprint"],
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "market_data": {
                "source": metadata["source"],
                "market_timezone": metadata["market_timezone"],
                "report_generated_at_utc": metadata["report_generated_at_utc"],
                "data_timestamp": metadata["data_timestamp"],
                "data_timestamp_granularity": metadata["data_timestamp_granularity"],
                "price_frequency": metadata["price_frequency"],
                "price_adjustment_policy": metadata["price_adjustment_policy"],
            },
            "decision_context": decision_context,
            "prediction": prediction,
            "contains_private_account_data": False,
        }
    )


def _validate_event(event: dict[str, Any]) -> None:
    try:
        assert_finite_json(event)
    except ValidationSplitError as exc:
        raise ForwardLedgerError(str(exc)) from exc
    _assert_no_private_keys(event)
    if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ForwardLedgerError("unsupported event schema in forward ledger")
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ForwardLedgerError("forward ledger event_id is missing")
    if event.get("immutable_payload_hash") != _event_hash(event):
        raise ForwardLedgerError(f"immutable forward ledger event was modified: {event_id}")
    if event.get("strategy_fingerprint") != STRATEGY_FINGERPRINT:
        raise ForwardLedgerError(f"ledger event fingerprint does not match frozen strategy: {event_id}")
    if event.get("rule_fingerprint") != RULE_FINGERPRINT:
        raise ForwardLedgerError(f"ledger event rule fingerprint does not match frozen rules: {event_id}")
    event_type = event.get("event_type")
    if event_type == "PREDICTION":
        prediction = event.get("prediction")
        if not isinstance(prediction, dict):
            raise ForwardLedgerError(f"prediction payload is missing: {event_id}")
        expected_id = _prediction_event_id(
            split_fingerprint=str(event.get("split_manifest_fingerprint")),
            strategy_fingerprint=str(event.get("strategy_fingerprint")),
            observation_date=str(prediction.get("observation_market_date")),
            ticker=str(prediction.get("ticker")),
        )
        if event_id != expected_id or event.get("evidence_classification") != "PROSPECTIVE":
            raise ForwardLedgerError(f"prediction event identity is invalid: {event_id}")
    elif event_type == "OUTCOME":
        horizon = event.get("outcome", {}).get("horizon_bars") if isinstance(event.get("outcome"), dict) else None
        prediction_event_id = event.get("prediction_event_id")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon not in FORWARD_HORIZONS
            or not isinstance(prediction_event_id, str)
            or event_id != _outcome_event_id(prediction_event_id, horizon)
        ):
            raise ForwardLedgerError(f"outcome event identity is invalid: {event_id}")
    else:
        raise ForwardLedgerError(f"unsupported forward ledger event type: {event_type}")


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ForwardLedgerError(f"could not read forward ledger {path}: {exc}") from exc
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ForwardLedgerError(f"blank JSONL record at {path}:{line_number}")
        try:
            event = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except json.JSONDecodeError as exc:
            raise ForwardLedgerError(f"invalid JSONL record at {path}:{line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ForwardLedgerError(f"JSONL record must be an object at {path}:{line_number}")
        _validate_event(event)
        event_id = event["event_id"]
        if event_id in seen:
            raise ForwardLedgerError(f"duplicate forward ledger event_id: {event_id}")
        seen.add(event_id)
        events.append(event)
    prediction_ids = {event["event_id"] for event in events if event["event_type"] == "PREDICTION"}
    for event in events:
        if event["event_type"] == "OUTCOME" and event["prediction_event_id"] not in prediction_ids:
            raise ForwardLedgerError(f"outcome references unknown prediction: {event['event_id']}")
    return events


def _equivalent_prediction(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    stable_keys = (
        "evidence_classification",
        "split_manifest_fingerprint",
        "strategy_contract_version",
        "rule_fingerprint",
        "strategy_fingerprint",
        "decision_context",
        "prediction",
        "contains_private_account_data",
    )
    if any(existing.get(key) != candidate.get(key) for key in stable_keys):
        return False
    existing_market = dict(existing.get("market_data", {}))
    candidate_market = dict(candidate.get("market_data", {}))
    existing_market.pop("report_generated_at_utc", None)
    candidate_market.pop("report_generated_at_utc", None)
    return existing_market == candidate_market


def _price_path(prices_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "-").replace(".", "-")
    return prices_dir / f"{safe}_daily.csv"


def _load_prices(
    ticker: str,
    prices_dir: Path,
    *,
    report_data_timestamp: date,
    as_of_market_date: date,
) -> tuple[list[date], list[float]]:
    path = _price_path(prices_dir, ticker)
    if not path.exists() or path.stat().st_size == 0:
        raise ForwardLedgerError(f"missing price history for eligible prediction: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"date", "price"}.issubset(reader.fieldnames):
                raise ForwardLedgerError(f"price history lacks date/price columns: {path}")
            rows = [(row.get("date"), row.get("price")) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ForwardLedgerError(f"could not read price history {path}: {exc}") from exc
    dates: list[date] = []
    prices: list[float] = []
    for row_number, (raw_date, raw_price) in enumerate(rows, start=2):
        try:
            market_date = date.fromisoformat(str(raw_date))
            price = float(str(raw_price))
        except (ValueError, TypeError) as exc:
            raise ForwardLedgerError(f"invalid price row at {path}:{row_number}") from exc
        if not math.isfinite(price) or price <= 0:
            raise ForwardLedgerError(f"non-positive or non-finite price at {path}:{row_number}")
        dates.append(market_date)
        prices.append(price)
    if not dates or dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ForwardLedgerError(f"price dates must be non-empty, unique, and increasing: {path}")
    if dates[-1] > report_data_timestamp or dates[-1] > as_of_market_date:
        raise ForwardLedgerError(f"price history contains future rows relative to validated report/as-of date: {path}")
    return dates, prices


def _calculate_outcome(
    prediction_event: dict[str, Any],
    horizon: int,
    *,
    prices_dir: Path,
    metadata: dict[str, Any],
    report: dict[str, Any],
    as_of_market_date: date,
) -> dict[str, Any] | None:
    prediction = prediction_event["prediction"]
    if prediction["state"] == "SKIPPED":
        return None
    ticker = prediction["ticker"]
    report_date = _parse_required_date(metadata["data_timestamp"], "market_report.data_timestamp")
    dates, prices = _load_prices(
        ticker,
        prices_dir,
        report_data_timestamp=report_date,
        as_of_market_date=as_of_market_date,
    )
    observation = _parse_required_date(
        prediction["observation_market_date"],
        f"prediction {prediction_event['event_id']} observation date",
    )
    try:
        observation_index = dates.index(observation)
    except ValueError as exc:
        raise ForwardLedgerError(
            f"eligible prediction observation date is absent from {ticker} price history"
        ) from exc
    entry_index = observation_index + 1
    exit_index = entry_index + horizon
    if exit_index >= len(dates):
        return None
    entry_price = prices[entry_index]
    exit_price = prices[exit_index]
    gross_return = exit_price / entry_price - 1
    net_return = net_return_after_round_trip_costs(gross_return)
    if not math.isfinite(gross_return) or not math.isfinite(net_return):
        raise ForwardLedgerError(f"non-finite matured outcome for {ticker} at {horizon} bars")
    price_basis = report.get("price_basis_by_ticker", {}).get(ticker)
    if price_basis != prediction.get("price_basis"):
        raise ForwardLedgerError(f"{ticker} price basis changed since its immutable prediction")
    return {
        "ticker": ticker,
        "observation_market_date": observation.isoformat(),
        "horizon_bars": horizon,
        "status": "MATURED",
        "entry_market_date": dates[entry_index].isoformat(),
        "exit_market_date": dates[exit_index].isoformat(),
        "entry_close": round(entry_price, 10),
        "exit_close": round(exit_price, 10),
        "gross_return": round(gross_return, 12),
        "net_return_after_frozen_costs": round(net_return, 12),
        "return_definition": "close[t+1+h] / close[t+1] - 1",
        "price_basis": price_basis,
    }


def _outcome_event(
    prediction_event: dict[str, Any],
    outcome: dict[str, Any],
    *,
    metadata: dict[str, Any],
    split_result: dict[str, Any],
) -> dict[str, Any]:
    horizon = outcome["horizon_bars"]
    return _seal_event(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": "OUTCOME",
            "event_id": _outcome_event_id(prediction_event["event_id"], horizon),
            "prediction_event_id": prediction_event["event_id"],
            "recorded_at_utc": metadata["report_generated_at_utc"],
            "evidence_classification": "PROSPECTIVE",
            "split_manifest_fingerprint": split_result["split_manifest_fingerprint"],
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "cost_contract": execution_cost_assumptions(),
            "market_data": {
                "source": metadata["source"],
                "market_timezone": metadata["market_timezone"],
                "data_timestamp": outcome["exit_market_date"],
                "source_report_generated_at_utc": metadata["report_generated_at_utc"],
                "report_available_through_market_date": metadata["data_timestamp"],
                "price_frequency": metadata["price_frequency"],
                "price_adjustment_policy": metadata["price_adjustment_policy"],
            },
            "outcome": outcome,
            "contains_private_account_data": False,
        }
    )


def _write_jsonl_atomic(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")
    temp_path.replace(path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    assert_finite_json(value)
    _assert_no_private_keys(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def build_forward_ledger(
    *,
    split_path: Path = DEFAULT_SPLIT,
    packet_path: Path = DEFAULT_PACKET,
    report_path: Path = DEFAULT_REPORT,
    ledger_path: Path = DEFAULT_LEDGER,
    summary_path: Path | None = DEFAULT_SUMMARY,
    prices_dir: Path = DEFAULT_PRICES_DIR,
    anchor_path: Path | None = None,
    as_of_market_date: date | None = None,
) -> dict[str, Any]:
    manifest = load_strict_json(split_path)
    anchor_dates = load_anchor_market_dates(anchor_path or _price_path(prices_dir, "SPY"))
    split_result = validate_split_manifest(manifest, anchor_dates=anchor_dates)
    packet = load_public_json(packet_path, "decision_packet")
    report = load_public_json(report_path, "market_report")
    as_of = as_of_market_date or latest_completed_us_market_weekday()
    observation, metadata = _validate_market_inputs(packet, report, split_result, as_of)
    last_seen = _parse_required_date(
        manifest["prospective_partition"]["last_seen_market_date"],
        "prospective_partition.last_seen_market_date",
    )

    events = load_ledger(ledger_path)
    event_by_id = {event["event_id"]: event for event in events}
    for event in events:
        if event.get("split_manifest_fingerprint") != split_result["split_manifest_fingerprint"]:
            raise ForwardLedgerError("existing ledger belongs to a different frozen validation split")
        if event["event_type"] == "PREDICTION":
            event_date = _parse_required_date(
                event["prediction"]["observation_market_date"],
                f"ledger prediction {event['event_id']}",
            )
            if event_date <= last_seen:
                raise ForwardLedgerError("existing prospective ledger contains a retrospective prediction")
            if event_date > observation:
                raise ForwardLedgerError("market_report is older than an existing prospective prediction")

    new_prediction_count = 0
    current_predictions: list[dict[str, Any]] = []
    if observation > last_seen:
        tickers = _ticker_universe(report)
        decision_context = _decision_context(packet)
        for ticker in tickers:
            prediction = _ticker_prediction(ticker, report, observation)
            candidate = _prediction_event(
                prediction,
                metadata=metadata,
                decision_context=decision_context,
                split_result=split_result,
            )
            existing = event_by_id.get(candidate["event_id"])
            if existing is not None:
                if existing.get("event_type") != "PREDICTION" or not _equivalent_prediction(existing, candidate):
                    raise ForwardLedgerError(
                        f"immutable prediction changed for {ticker} on {observation.isoformat()}"
                    )
                current_predictions.append(existing)
            else:
                events.append(candidate)
                event_by_id[candidate["event_id"]] = candidate
                current_predictions.append(candidate)
                new_prediction_count += 1

    prediction_events = [event for event in events if event["event_type"] == "PREDICTION"]
    new_outcome_count = 0
    for prediction_event in prediction_events:
        if prediction_event["prediction"]["state"] == "SKIPPED":
            continue
        for horizon in FORWARD_HORIZONS:
            outcome_id = _outcome_event_id(prediction_event["event_id"], horizon)
            calculated = _calculate_outcome(
                prediction_event,
                horizon,
                prices_dir=prices_dir,
                metadata=metadata,
                report=report,
                as_of_market_date=as_of,
            )
            existing_outcome = event_by_id.get(outcome_id)
            if calculated is None:
                if existing_outcome is not None:
                    raise ForwardLedgerError(
                        f"existing matured outcome is no longer reproducible: {outcome_id}"
                    )
                continue
            if existing_outcome is not None:
                if existing_outcome.get("event_type") != "OUTCOME" or existing_outcome.get("outcome") != calculated:
                    raise ForwardLedgerError(f"immutable matured outcome changed: {outcome_id}")
                continue
            candidate_outcome = _outcome_event(
                prediction_event,
                calculated,
                metadata=metadata,
                split_result=split_result,
            )
            events.append(candidate_outcome)
            event_by_id[outcome_id] = candidate_outcome
            new_outcome_count += 1

    if new_prediction_count or new_outcome_count:
        _write_jsonl_atomic(ledger_path, events)

    total_predictions = [event for event in events if event["event_type"] == "PREDICTION"]
    total_outcomes = [event for event in events if event["event_type"] == "OUTCOME"]
    evaluable_predictions = [
        event for event in total_predictions if event["prediction"]["state"] != "SKIPPED"
    ]
    expected_outcome_count = len(evaluable_predictions) * len(FORWARD_HORIZONS)
    current_state_counts = Counter(event["prediction"]["state"] for event in current_predictions)
    status = (
        "WAITING_FOR_FIRST_PROSPECTIVE_MARKET_DATE"
        if observation <= last_seen
        else "PROSPECTIVE_EVENTS_RECORDED"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": metadata["report_generated_at_utc"],
        "status": status,
        "historical_evidence_classification": "RETROSPECTIVE_CONTAMINATED",
        "historical_data_through_market_date": last_seen.isoformat(),
        "prospective_start_rule": "STRICTLY_AFTER_LAST_SEEN_MARKET_DATE",
        "current_report_market_date": observation.isoformat(),
        "data_source": metadata["source"],
        "market_timezone": metadata["market_timezone"],
        "data_timestamp": metadata["data_timestamp"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
        "split_manifest_fingerprint": split_result["split_manifest_fingerprint"],
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "current_market_date_prediction_counts": {
            "total": len(current_predictions),
            "active": current_state_counts.get("ACTIVE", 0),
            "no_signal": current_state_counts.get("NO_SIGNAL", 0),
            "skipped": current_state_counts.get("SKIPPED", 0),
        },
        "ledger_counts": {
            "prediction_events": len(total_predictions),
            "matured_outcome_events": len(total_outcomes),
            "pending_outcomes": expected_outcome_count - len(total_outcomes),
            "new_prediction_events": new_prediction_count,
            "new_outcome_events": new_outcome_count,
        },
        "contains_private_account_data": False,
        "automatic_order_allowed": False,
        "important_limit": "Prospective observations are evidence, not a guarantee of future performance or an automatic trading instruction.",
    }
    if summary["ledger_counts"]["pending_outcomes"] < 0:
        raise ForwardLedgerError("ledger contains more matured outcomes than frozen prediction horizons")
    if summary_path is not None:
        _write_json_atomic(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record immutable public-model predictions and next-close prospective outcomes."
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prices-dir", type=Path, default=DEFAULT_PRICES_DIR)
    parser.add_argument("--anchor-csv", type=Path)
    parser.add_argument("--as-of-market-date", type=str)
    args = parser.parse_args(argv)
    as_of = None
    if args.as_of_market_date is not None:
        as_of = parse_market_date(args.as_of_market_date)
        if as_of is None or as_of.isoformat() != args.as_of_market_date:
            print("forward ledger FAILED: --as-of-market-date must be YYYY-MM-DD", file=sys.stderr)
            return 1
    try:
        summary = build_forward_ledger(
            split_path=args.split,
            packet_path=args.packet,
            report_path=args.report,
            ledger_path=args.ledger,
            summary_path=args.summary,
            prices_dir=args.prices_dir,
            anchor_path=args.anchor_csv,
            as_of_market_date=as_of,
        )
    except (ForwardLedgerError, ValidationSplitError) as exc:
        print(f"forward ledger FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
