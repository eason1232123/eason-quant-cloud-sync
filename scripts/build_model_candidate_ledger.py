from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_forward_ledger import (  # noqa: E402
    ForwardLedgerError,
    _event_hash,
    _load_prices,
    _seal_event,
    _validate_market_inputs,
    _write_json_atomic,
    _write_jsonl_atomic,
    load_public_json,
)
from scripts.market_clock import latest_completed_us_market_weekday  # noqa: E402
from scripts.model_governance import (  # noqa: E402
    ASSETS,
    GOVERNANCE_CONTRACT_VERSION,
    ModelGovernanceError,
    allocation_decision,
    blend_model_weights,
    challenger_models,
    governance_fingerprint,
    incumbent_model,
    load_governance_config,
    model_fingerprint,
    paired_metrics,
    regime_evidence,
    weights_for_model_regime,
)
from scripts.strategy_contract import (  # noqa: E402
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
    execution_cost_assumptions,
    net_return_after_round_trip_costs,
)
from scripts.validate_validation_split import (  # noqa: E402
    DEFAULT_MANIFEST,
    assert_finite_json,
    canonical_fingerprint,
    load_anchor_market_dates,
    load_strict_json,
    split_manifest_fingerprint,
    validate_split_manifest,
)


DEFAULT_CONFIG = ROOT / "config" / "model_governance.json"
DEFAULT_PACKET = ROOT / "docs" / "decision_packet.json"
DEFAULT_REPORT = ROOT / "docs" / "market_report.json"
DEFAULT_LEDGER = ROOT / "docs" / "model_candidate_forward_ledger.jsonl"
DEFAULT_SUMMARY = ROOT / "docs" / "model_governance.json"
DEFAULT_PRICES_DIR = ROOT / "docs"

LEDGER_SCHEMA_VERSION = "v6-model-candidate-ledger-event-v1"
SUMMARY_SCHEMA_VERSION = "v6-model-governance-status-v1"
PREDICTION_NAMESPACE = "eason-v6-model-candidate-prediction-v1"
OUTCOME_NAMESPACE = "eason-v6-model-candidate-outcome-v1"
CASH_PROXY_TICKER = "SGOV"


class ModelCandidateLedgerError(ModelGovernanceError):
    """Raised when candidate evidence cannot be appended without ambiguity."""


def _public_asset_name(asset: str) -> str:
    return "CASH_PROXY" if asset == "CASH" else asset


def _public_asset_map(values: dict[str, Any]) -> dict[str, Any]:
    return {_public_asset_name(asset): value for asset, value in values.items()}


def _prediction_id(governance_fp: str, model_fp: str, observation: str) -> str:
    return "model_pred_" + canonical_fingerprint(
        {
            "namespace": PREDICTION_NAMESPACE,
            "governance_fingerprint": governance_fp,
            "model_fingerprint": model_fp,
            "observation_market_date": observation,
        }
    )


def _outcome_id(prediction_id: str) -> str:
    return "model_out_" + canonical_fingerprint(
        {"namespace": OUTCOME_NAMESPACE, "prediction_event_id": prediction_id}
    )


def _read_candidate_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    events: list[dict[str, Any]] = []
    ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ModelCandidateLedgerError(f"could not read candidate ledger {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ModelCandidateLedgerError(f"blank candidate ledger line at {line_number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelCandidateLedgerError(f"invalid candidate ledger JSON at line {line_number}") from exc
        if not isinstance(event, dict):
            raise ModelCandidateLedgerError(f"candidate ledger line {line_number} must be an object")
        assert_finite_json(event)
        if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ModelCandidateLedgerError(f"unsupported candidate ledger schema at line {line_number}")
        if event.get("event_type") not in {"MODEL_PREDICTION", "MODEL_OUTCOME"}:
            raise ModelCandidateLedgerError(f"invalid candidate event_type at line {line_number}")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in ids:
            raise ModelCandidateLedgerError(f"missing or duplicate candidate event_id at line {line_number}")
        if event.get("immutable_payload_hash") != _event_hash(event):
            raise ModelCandidateLedgerError(f"candidate event hash mismatch at line {line_number}")
        if event.get("contains_private_account_data") is not False:
            raise ModelCandidateLedgerError("candidate ledger must not contain private account data")
        ids.add(event_id)
        events.append(event)
    prediction_ids = {
        event["event_id"] for event in events if event["event_type"] == "MODEL_PREDICTION"
    }
    for event in events:
        if (
            event["event_type"] == "MODEL_OUTCOME"
            and event.get("prediction_event_id") not in prediction_ids
        ):
            raise ModelCandidateLedgerError("candidate outcome references an unknown prediction")
    return events


def _price_file(prices_dir: Path, ticker: str) -> Path:
    return prices_dir / f"{ticker.replace('/', '-').replace('.', '-')}_daily.csv"


def _verified_price_basis(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if "price" not in fields:
                raise ModelCandidateLedgerError(f"price column is missing: {path}")
            comparison = "adjClose" if "adjClose" in fields else "close" if "close" in fields else None
            if comparison is None:
                raise ModelCandidateLedgerError(
                    f"price basis is ambiguous because adjClose/close is missing: {path}"
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    price = float(row["price"])
                    reference = float(row[comparison])
                except (TypeError, ValueError) as exc:
                    raise ModelCandidateLedgerError(
                        f"invalid price-basis row at {path}:{row_number}"
                    ) from exc
                if not math.isclose(price, reference, rel_tol=1e-10, abs_tol=1e-10):
                    raise ModelCandidateLedgerError(
                        f"price does not match declared {comparison} basis at {path}:{row_number}"
                    )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ModelCandidateLedgerError(f"could not verify price basis for {path}: {exc}") from exc
    return "adjusted" if comparison == "adjClose" else "unadjusted"


def _aligned_prices(
    prices_dir: Path,
    *,
    report_data_timestamp: date,
    as_of_market_date: date,
) -> tuple[pd.DataFrame, dict[str, str]]:
    ticker_by_asset = {asset: asset for asset in ASSETS if asset != "CASH"} | {
        "CASH": CASH_PROXY_TICKER
    }
    dates_by_asset: dict[str, list[date]] = {}
    values_by_asset: dict[str, list[float]] = {}
    basis_by_asset: dict[str, str] = {}
    for asset, ticker in ticker_by_asset.items():
        try:
            dates, values = _load_prices(
                ticker,
                prices_dir,
                report_data_timestamp=report_data_timestamp,
                as_of_market_date=as_of_market_date,
            )
        except ForwardLedgerError as exc:
            raise ModelCandidateLedgerError(str(exc)) from exc
        dates_by_asset[asset] = dates
        values_by_asset[asset] = values
        basis_by_asset[asset] = _verified_price_basis(_price_file(prices_dir, ticker))

    common_start = max(values[0] for values in dates_by_asset.values())
    expected_dates: list[date] | None = None
    frame = pd.DataFrame()
    for asset in ASSETS:
        pairs = [
            (market_date, value)
            for market_date, value in zip(dates_by_asset[asset], values_by_asset[asset])
            if market_date >= common_start
        ]
        asset_dates = [item[0] for item in pairs]
        if expected_dates is None:
            expected_dates = asset_dates
            frame["date"] = pd.to_datetime(asset_dates)
        elif asset_dates != expected_dates:
            raise ModelCandidateLedgerError(
                f"candidate assets have missing or misaligned market bars after {common_start}"
            )
        frame[asset] = [item[1] for item in pairs]
    if frame.empty or expected_dates is None:
        raise ModelCandidateLedgerError("candidate price frame is empty")
    if expected_dates[-1] != report_data_timestamp:
        raise ModelCandidateLedgerError(
            "every candidate asset, including SGOV, must be fresh through the report market date"
        )
    return frame, basis_by_asset


def _regime_at(frame: pd.DataFrame, observation: date) -> dict[str, Any]:
    d = frame.copy()
    for ticker in ("QQQ", "SMH", "SPY"):
        d[f"{ticker}_ma200"] = d[ticker].rolling(200).mean()
        d[f"{ticker}_ret20"] = d[ticker].pct_change(20, fill_method=None)
    rows = d[d["date"] == pd.Timestamp(observation)]
    if len(rows) != 1:
        raise ModelCandidateLedgerError("observation market date is absent from aligned prices")
    return regime_evidence(rows.iloc[0])


def _prediction_event(
    *,
    model: dict[str, Any],
    config: dict[str, Any],
    split_fp: str,
    observation: date,
    regime: dict[str, Any],
    metadata: dict[str, Any],
    price_basis: dict[str, str],
) -> dict[str, Any]:
    governance_fp = governance_fingerprint(config)
    model_fp = model_fingerprint(model)
    target = weights_for_model_regime(config, model["model_id"], regime["regime"])
    return _seal_event(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": "MODEL_PREDICTION",
            "event_id": _prediction_id(governance_fp, model_fp, observation.isoformat()),
            "recorded_at_utc": metadata["report_generated_at_utc"],
            "evidence_classification": "PROSPECTIVE",
            "split_manifest_fingerprint": split_fp,
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "model_governance_fingerprint": governance_fp,
            "model_id": model["model_id"],
            "model_role": model["role"],
            "model_fingerprint": model_fp,
            "prediction": {
                "observation_market_date": observation.isoformat(),
                "market_regime": regime["regime"],
                "regime_reason_codes": regime["reason_codes"],
                "target_weights": _public_asset_map(target),
                "entry_close_offset_bars": config["evidence_policy"]["entry_close_offset_bars"],
                "evaluation_horizon_bars": config["evidence_policy"]["evaluation_horizon_bars"],
                "non_overlapping_sample": True,
            },
            "market_data": {
                "source": metadata["source"],
                "market_timezone": metadata["market_timezone"],
                "data_timestamp": observation.isoformat(),
                "source_report_generated_at_utc": metadata["report_generated_at_utc"],
                "price_frequency": metadata["price_frequency"],
                "price_adjustment_policy": metadata["price_adjustment_policy"],
                "price_basis_by_asset": _public_asset_map(price_basis),
            },
            "contains_private_account_data": False,
        }
    )


def _equivalent_prediction(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    stable_fields = (
        "evidence_classification",
        "split_manifest_fingerprint",
        "strategy_contract_version",
        "rule_fingerprint",
        "strategy_fingerprint",
        "model_governance_fingerprint",
        "model_id",
        "model_role",
        "model_fingerprint",
        "prediction",
        "contains_private_account_data",
    )
    if any(existing.get(field) != candidate.get(field) for field in stable_fields):
        return False
    old_market = dict(existing.get("market_data", {}))
    new_market = dict(candidate.get("market_data", {}))
    old_market.pop("source_report_generated_at_utc", None)
    new_market.pop("source_report_generated_at_utc", None)
    return old_market == new_market


def _candidate_outcome(
    prediction: dict[str, Any],
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    observation = date.fromisoformat(prediction["prediction"]["observation_market_date"])
    positions = frame.index[frame["date"] == pd.Timestamp(observation)].tolist()
    if len(positions) != 1:
        raise ModelCandidateLedgerError("prediction observation date is missing from aligned prices")
    entry_offset = int(config["evidence_policy"]["entry_close_offset_bars"])
    horizon = int(config["evidence_policy"]["evaluation_horizon_bars"])
    entry_index = positions[0] + entry_offset
    exit_index = entry_index + horizon
    if exit_index >= len(frame):
        return None
    weights = prediction["prediction"]["target_weights"]
    gross_by_asset: dict[str, float] = {}
    net_by_asset: dict[str, float] = {}
    for asset in ASSETS:
        gross = float(frame.iloc[exit_index][asset] / frame.iloc[entry_index][asset] - 1.0)
        if not math.isfinite(gross):
            raise ModelCandidateLedgerError("non-finite candidate asset return")
        public_asset = _public_asset_name(asset)
        gross_by_asset[public_asset] = round(gross, 12)
        net_by_asset[public_asset] = round(
            gross if asset == "CASH" else net_return_after_round_trip_costs(gross),
            12,
        )
    gross_portfolio = sum(
        float(weights[_public_asset_name(asset)]) * gross_by_asset[_public_asset_name(asset)]
        for asset in ASSETS
    )
    net_portfolio = sum(
        float(weights[_public_asset_name(asset)]) * net_by_asset[_public_asset_name(asset)]
        for asset in ASSETS
    )
    return {
        "observation_market_date": observation.isoformat(),
        "entry_market_date": frame.iloc[entry_index]["date"].date().isoformat(),
        "exit_market_date": frame.iloc[exit_index]["date"].date().isoformat(),
        "horizon_bars": horizon,
        "status": "MATURED",
        "gross_return_by_asset": gross_by_asset,
        "net_return_after_frozen_costs_by_asset": net_by_asset,
        "gross_portfolio_return": round(gross_portfolio, 12),
        "net_portfolio_return_after_frozen_costs": round(net_portfolio, 12),
        "return_definition": "frozen target-weight return from close[t+1] to close[t+1+h]; configured round-trip costs apply to non-cash assets",
    }


def _outcome_event(
    prediction: dict[str, Any],
    outcome: dict[str, Any],
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return _seal_event(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": "MODEL_OUTCOME",
            "event_id": _outcome_id(prediction["event_id"]),
            "prediction_event_id": prediction["event_id"],
            "recorded_at_utc": metadata["report_generated_at_utc"],
            "evidence_classification": "PROSPECTIVE",
            "split_manifest_fingerprint": prediction["split_manifest_fingerprint"],
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "model_governance_fingerprint": governance_fingerprint(config),
            "model_id": prediction["model_id"],
            "model_role": prediction["model_role"],
            "model_fingerprint": prediction["model_fingerprint"],
            "cost_contract": execution_cost_assumptions(),
            "outcome": outcome,
            "market_data": {
                "source": metadata["source"],
                "market_timezone": metadata["market_timezone"],
                "data_timestamp": outcome["exit_market_date"],
                "source_report_generated_at_utc": metadata["report_generated_at_utc"],
                "report_available_through_market_date": metadata["data_timestamp"],
                "price_frequency": metadata["price_frequency"],
                "price_adjustment_policy": metadata["price_adjustment_policy"],
            },
            "contains_private_account_data": False,
        }
    )


def _equivalent_outcome(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    stable = copy.deepcopy(existing)
    proposed = copy.deepcopy(candidate)
    for value in (stable, proposed):
        value.pop("recorded_at_utc", None)
        value.pop("immutable_payload_hash", None)
        if isinstance(value.get("market_data"), dict):
            value["market_data"].pop("source_report_generated_at_utc", None)
    return stable == proposed


def _prediction_groups(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event["event_type"] != "MODEL_PREDICTION":
            continue
        observation = event["prediction"]["observation_market_date"]
        groups.setdefault(observation, []).append(event)
    return groups


def _validate_prediction_schedule(
    groups: dict[str, list[dict[str, Any]]],
    *,
    config: dict[str, Any],
    expected_model_ids: set[str],
    frame: pd.DataFrame,
) -> None:
    boundary = date.fromisoformat(config["evidence_policy"]["strictly_after_market_date"])
    horizon = int(config["evidence_policy"]["evaluation_horizon_bars"])
    required_gap = horizon + int(config["evidence_policy"]["entry_close_offset_bars"])
    market_dates = [value.date() for value in frame["date"]]
    prior_index: int | None = None
    for observation_text in sorted(groups):
        group = groups[observation_text]
        if {event["model_id"] for event in group} != expected_model_ids:
            raise ModelCandidateLedgerError(
                f"candidate prediction cohort is incomplete at {observation_text}"
            )
        if len(group) != len(expected_model_ids):
            raise ModelCandidateLedgerError(f"duplicate model prediction at {observation_text}")
        observation = date.fromisoformat(observation_text)
        if observation <= boundary:
            raise ModelCandidateLedgerError("candidate prediction predates the prospective boundary")
        try:
            current_index = market_dates.index(observation)
        except ValueError as exc:
            raise ModelCandidateLedgerError(
                "existing candidate prediction date is absent from current price history"
            ) from exc
        if prior_index is not None and current_index < prior_index + required_gap:
            raise ModelCandidateLedgerError("candidate prediction cohorts overlap")
        prior_index = current_index
        if any(event["prediction"]["evaluation_horizon_bars"] != horizon for event in group):
            raise ModelCandidateLedgerError("candidate prediction horizon drifted")


def _may_create_cohort(
    observation: date,
    frame: pd.DataFrame,
    groups: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> bool:
    if observation.isoformat() in groups:
        return True
    if not groups:
        return True
    last_observation = date.fromisoformat(max(groups))
    dates = [value.date() for value in frame["date"]]
    try:
        last_index = dates.index(last_observation)
        current_index = dates.index(observation)
    except ValueError as exc:
        raise ModelCandidateLedgerError("existing cohort date is absent from current price history") from exc
    required_gap = (
        int(config["evidence_policy"]["entry_close_offset_bars"])
        + int(config["evidence_policy"]["evaluation_horizon_bars"])
    )
    return current_index >= last_index + required_gap


def _governance_metrics(
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    predictions = {
        event["event_id"]: event
        for event in events
        if event["event_type"] == "MODEL_PREDICTION"
    }
    outcomes = {
        event["prediction_event_id"]: event
        for event in events
        if event["event_type"] == "MODEL_OUTCOME"
    }
    by_model_date: dict[tuple[str, str], dict[str, Any]] = {}
    for prediction_id, outcome in outcomes.items():
        prediction = predictions[prediction_id]
        key = (prediction["model_id"], prediction["prediction"]["observation_market_date"])
        if key in by_model_date:
            raise ModelCandidateLedgerError("duplicate matured candidate outcome")
        by_model_date[key] = outcome

    incumbent_id = incumbent_model(config)["model_id"]
    recent_window = int(config["rollback_gates"]["recent_paired_sample_window"])
    result: dict[str, dict[str, Any]] = {}
    for challenger in challenger_models(config):
        challenger_id = challenger["model_id"]
        dates = sorted(
            observation
            for model_id, observation in by_model_date
            if model_id == challenger_id and (incumbent_id, observation) in by_model_date
        )
        incumbent_returns: list[float] = []
        challenger_returns: list[float] = []
        prior_exit: date | None = None
        for observation in dates:
            incumbent_outcome = by_model_date[(incumbent_id, observation)]["outcome"]
            challenger_outcome = by_model_date[(challenger_id, observation)]["outcome"]
            if (
                incumbent_outcome["entry_market_date"] != challenger_outcome["entry_market_date"]
                or incumbent_outcome["exit_market_date"] != challenger_outcome["exit_market_date"]
            ):
                raise ModelCandidateLedgerError("paired candidate outcomes use different windows")
            entry = date.fromisoformat(challenger_outcome["entry_market_date"])
            exit_date = date.fromisoformat(challenger_outcome["exit_market_date"])
            if prior_exit is not None and entry <= prior_exit:
                raise ModelCandidateLedgerError("matured governance samples overlap")
            prior_exit = exit_date
            incumbent_returns.append(
                float(incumbent_outcome["net_portfolio_return_after_frozen_costs"])
            )
            challenger_returns.append(
                float(challenger_outcome["net_portfolio_return_after_frozen_costs"])
            )
        result[challenger_id] = paired_metrics(
            incumbent_returns,
            challenger_returns,
            recent_window=recent_window,
        )
    return result


def build_model_candidate_ledger(
    *,
    governance_path: Path = DEFAULT_CONFIG,
    split_path: Path = DEFAULT_MANIFEST,
    packet_path: Path = DEFAULT_PACKET,
    report_path: Path = DEFAULT_REPORT,
    ledger_path: Path = DEFAULT_LEDGER,
    summary_path: Path | None = DEFAULT_SUMMARY,
    prices_dir: Path = DEFAULT_PRICES_DIR,
    anchor_path: Path | None = None,
    as_of_market_date: date | None = None,
) -> dict[str, Any]:
    config = load_governance_config(governance_path)
    governance_fp = governance_fingerprint(config)
    manifest = load_strict_json(split_path)
    anchor = anchor_path or _price_file(prices_dir, "SPY")
    split_result = validate_split_manifest(
        manifest,
        anchor_dates=load_anchor_market_dates(anchor),
    )
    split_fp = split_manifest_fingerprint(manifest)
    boundary = date.fromisoformat(config["evidence_policy"]["strictly_after_market_date"])
    split_boundary = date.fromisoformat(
        manifest["prospective_partition"]["last_seen_market_date"]
    )
    if boundary != split_boundary:
        raise ModelCandidateLedgerError("governance and strategy prospective boundaries differ")

    packet = load_public_json(packet_path, "decision_packet")
    report = load_public_json(report_path, "market_report")
    as_of = as_of_market_date or latest_completed_us_market_weekday()
    try:
        observation, metadata = _validate_market_inputs(
            packet,
            report,
            split_result,
            as_of,
        )
    except ForwardLedgerError as exc:
        raise ModelCandidateLedgerError(str(exc)) from exc

    events = _read_candidate_ledger(ledger_path)
    configured_models = config["models"]
    model_ids = {model["model_id"] for model in configured_models}
    expected_fingerprints = {
        model["model_id"]: model_fingerprint(model) for model in configured_models
    }
    for event in events:
        if event.get("model_governance_fingerprint") != governance_fp:
            raise ModelCandidateLedgerError("existing candidate ledger uses another governance generation")
        if event.get("split_manifest_fingerprint") != split_fp:
            raise ModelCandidateLedgerError("existing candidate ledger uses another validation split")
        model_id = event.get("model_id")
        if model_id not in model_ids or event.get("model_fingerprint") != expected_fingerprints[model_id]:
            raise ModelCandidateLedgerError("existing candidate ledger model identity drifted")

    frame, price_basis = _aligned_prices(
        prices_dir,
        report_data_timestamp=observation,
        as_of_market_date=as_of,
    )
    groups = _prediction_groups(events)
    _validate_prediction_schedule(
        groups,
        config=config,
        expected_model_ids=model_ids,
        frame=frame,
    )
    event_by_id = {event["event_id"]: event for event in events}
    new_predictions = 0
    if observation > boundary and _may_create_cohort(observation, frame, groups, config):
        regime = _regime_at(frame, observation)
        for model in configured_models:
            candidate = _prediction_event(
                model=model,
                config=config,
                split_fp=split_fp,
                observation=observation,
                regime=regime,
                metadata=metadata,
                price_basis=price_basis,
            )
            existing = event_by_id.get(candidate["event_id"])
            if existing is not None:
                if not _equivalent_prediction(existing, candidate):
                    raise ModelCandidateLedgerError("immutable candidate prediction changed")
            else:
                events.append(candidate)
                event_by_id[candidate["event_id"]] = candidate
                new_predictions += 1

    new_outcomes = 0
    for prediction in [
        event for event in events if event["event_type"] == "MODEL_PREDICTION"
    ]:
        outcome = _candidate_outcome(prediction, frame, config)
        if outcome is None:
            continue
        candidate = _outcome_event(
            prediction,
            outcome,
            config=config,
            metadata=metadata,
        )
        existing = event_by_id.get(candidate["event_id"])
        if existing is not None:
            if not _equivalent_outcome(existing, candidate):
                raise ModelCandidateLedgerError("immutable candidate outcome changed")
        else:
            events.append(candidate)
            event_by_id[candidate["event_id"]] = candidate
            new_outcomes += 1

    events.sort(
        key=lambda event: (
            event.get("prediction", {}).get("observation_market_date")
            or event.get("outcome", {}).get("observation_market_date"),
            0 if event["event_type"] == "MODEL_PREDICTION" else 1,
            event["model_id"],
        )
    )
    metrics = _governance_metrics(events, config)
    decision = allocation_decision(config, metrics, data_integrity_ok=True)
    current_regime = _regime_at(frame, observation)
    effective_weights = blend_model_weights(
        config,
        decision["model_allocations"],
        current_regime["regime"],
    )
    counts = Counter(event["event_type"] for event in events)
    status = (
        "WAITING_FOR_FIRST_PROSPECTIVE_MARKET_DATE"
        if observation <= boundary
        else decision["status"]
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": metadata["report_generated_at_utc"],
        "data_source": metadata["source"],
        "market_timezone": metadata["market_timezone"],
        "data_timestamp": metadata["data_timestamp"],
        "data_timestamp_granularity": metadata["data_timestamp_granularity"],
        "data_timestamp_status": metadata["data_timestamp_status"],
        "price_frequency": metadata["price_frequency"],
        "price_adjustment_policy": metadata["price_adjustment_policy"],
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "split_manifest_fingerprint": split_fp,
        "model_governance_fingerprint": governance_fp,
        "model_governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
        "status": status,
        "evidence_classification": "PROSPECTIVE",
        "prospective_starts_strictly_after": boundary.isoformat(),
        "evaluation_horizon_bars": config["evidence_policy"]["evaluation_horizon_bars"],
        "non_overlapping_samples_required": True,
        "paired_incumbent_challenger_samples_required": True,
        "retrospective_evidence_may_promote": False,
        "ledger_counts": {
            "prediction_events": int(counts.get("MODEL_PREDICTION", 0)),
            "outcome_events": int(counts.get("MODEL_OUTCOME", 0)),
            "new_prediction_events": new_predictions,
            "new_outcome_events": new_outcomes,
        },
        "model_fingerprints": expected_fingerprints,
        "candidate_registry": [
            {
                "model_id": model["model_id"],
                "role": model["role"],
                "model_fingerprint": expected_fingerprints[model["model_id"]],
                "design_basis": model["design_basis"],
            }
            for model in configured_models
        ],
        "allocation_gates": config["allocation_gates"],
        "rollback_gates": config["rollback_gates"],
        "challenger_metrics": metrics,
        "allocation_decision": decision,
        "current_market_regime": current_regime,
        "effective_target_weights": _public_asset_map(effective_weights),
        "cost_contract": execution_cost_assumptions(),
        "automatic_order_allowed": False,
        "contains_private_account_data": False,
        "survivorship_bias_status": "KNOWN_UNCONTROLLED_CURRENT_FIXED_ASSET_SET",
        "selection_bias_status": "CANDIDATES_PREDECLARED_BEFORE_PROSPECTIVE_EVIDENCE_NOT_SELECTED_FROM_REPORTED_RETURNS",
        "important_limit": "Governance uses only paired, non-overlapping prospective public-model samples. It cannot place an order and cannot infer current private IBKR holdings.",
    }
    assert_finite_json(summary)
    if events or ledger_path.exists():
        _write_jsonl_atomic(ledger_path, events)
    if summary_path is not None:
        _write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable prospective model-candidate evidence")
    parser.add_argument("--as-of-market-date", type=date.fromisoformat)
    args = parser.parse_args()
    summary = build_model_candidate_ledger(as_of_market_date=args.as_of_market_date)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
