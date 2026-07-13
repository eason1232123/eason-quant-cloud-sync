from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.build_portfolio_backtest import portfolio_contract_payload
from scripts.build_model_candidate_ledger import (
    ModelCandidateLedgerError,
    _read_candidate_ledger,
)
from scripts.market_clock import MARKET_TIMEZONE, parse_market_date, parse_timestamp
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
)
from scripts.model_governance import (
    GOVERNANCE_CONTRACT_VERSION,
    governance_fingerprint,
    load_governance_config,
    model_fingerprint as governed_model_fingerprint,
)
from scripts.strategy_contract import (
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
)
from scripts.validate_validation_split import (
    DEFAULT_MANIFEST,
    canonical_fingerprint,
    full_model_fingerprint,
    load_strict_json,
    split_manifest_fingerprint,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

MODEL_REPORTS = (
    "market_report.json",
    "vectorbt_validation.json",
    "vectorbt_report.json",
    "portfolio_backtest.json",
    "walk_forward_report.json",
    "market_regime_report.json",
    "overfitting_check.json",
    "forward_validation_status.json",
    "model_governance.json",
)
METADATA_REPORTS = MODEL_REPORTS + (
    "trade_review.json",
    "actual_vs_backtest.json",
)
MODEL_CSV_REPORTS = (
    "backtest_summary.csv",
    "rule_evidence_ranking.csv",
    "vectorbt_signal_stats.csv",
    "vectorbt_strategy_summary.csv",
    "vectorbt_forward_evidence.csv",
    "portfolio_equity_curve.csv",
    "portfolio_trades.csv",
    "portfolio_vs_benchmark.csv",
)
METADATA_CSV_REPORTS = MODEL_CSV_REPORTS + ("trade_review.csv",)
LEDGER_REPORTS = ("model_candidate_forward_ledger.jsonl",)
METADATA_FIELDS = (
    "data_source",
    "market_timezone",
    "data_timestamp",
    "price_frequency",
    "price_adjustment_policy",
)
MODEL_FIELDS = (
    "strategy_contract_version",
    "rule_fingerprint",
    "strategy_fingerprint",
)


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"missing or empty model artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssertionError(f"model artifact must be a JSON object: {path}")
    return value


def validate_metadata(report: dict[str, Any], name: str) -> None:
    for field in METADATA_FIELDS:
        value = report.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AssertionError(f"{name} missing required metadata field {field}")
    if report["market_timezone"] != MARKET_TIMEZONE:
        raise AssertionError(f"{name} market_timezone mismatch")
    if parse_market_date(report["data_timestamp"]) is None:
        raise AssertionError(f"{name} data_timestamp is invalid")
    if report["price_frequency"] != PRICE_FREQUENCY:
        raise AssertionError(f"{name} price_frequency mismatch")
    if report["price_adjustment_policy"] != PRICE_ADJUSTMENT_POLICY:
        raise AssertionError(f"{name} price_adjustment_policy mismatch")
    generated = report.get("generated_at_utc")
    if generated is not None and parse_timestamp(generated) is None:
        raise AssertionError(f"{name} generated_at_utc is invalid")
    granularity = report.get("data_timestamp_granularity")
    if granularity is not None and granularity != DATA_TIMESTAMP_GRANULARITY:
        raise AssertionError(f"{name} data_timestamp_granularity mismatch")


def validate_strategy_contract(report: dict[str, Any], name: str) -> None:
    expected = {
        "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": RULE_FINGERPRINT,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    actual = {field: report.get(field) for field in MODEL_FIELDS}
    if actual != expected:
        raise AssertionError(f"{name} strategy contract mismatch: {actual}")


def validate_csv_contract(path: Path, *, require_model: bool = True) -> int:
    if not path.exists():
        raise AssertionError(f"missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = set(METADATA_FIELDS) | {"report_generated_at_utc"}
        if require_model:
            required |= set(MODEL_FIELDS)
        missing = sorted(required - fields)
        if missing:
            raise AssertionError(f"{path.name} missing metadata columns: {missing}")
        rows = list(reader)
    for row_number, row in enumerate(rows, start=2):
        if row.get("market_timezone") != MARKET_TIMEZONE:
            raise AssertionError(f"{path.name}:{row_number} timezone mismatch")
        if require_model:
            if row.get("strategy_contract_version") != STRATEGY_CONTRACT_VERSION:
                raise AssertionError(f"{path.name}:{row_number} strategy version mismatch")
            if row.get("rule_fingerprint") != RULE_FINGERPRINT:
                raise AssertionError(f"{path.name}:{row_number} rule fingerprint mismatch")
            if row.get("strategy_fingerprint") != STRATEGY_FINGERPRINT:
                raise AssertionError(f"{path.name}:{row_number} strategy fingerprint mismatch")
        if parse_market_date(row.get("data_timestamp")) is None:
            raise AssertionError(f"{path.name}:{row_number} invalid data timestamp")
    return len(rows)


def validate_model_artifacts(
    docs: Path = DOCS,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest = load_strict_json(manifest_path)
    validate_split_manifest(manifest)
    split_fingerprint = split_manifest_fingerprint(manifest)
    portfolio_payload = portfolio_contract_payload()
    portfolio_fingerprint = canonical_fingerprint(portfolio_payload)
    model_fingerprint = full_model_fingerprint(
        STRATEGY_FINGERPRINT,
        portfolio_payload,
        manifest,
    )
    governance = load_governance_config()
    governance_fp = governance_fingerprint(governance)
    expected_model_fingerprints = {
        model["model_id"]: governed_model_fingerprint(model) for model in governance["models"]
    }

    reports = {name: load_json_object(docs / name) for name in METADATA_REPORTS}
    for name, report in reports.items():
        validate_metadata(report, name)
    for name in MODEL_REPORTS:
        validate_strategy_contract(reports[name], name)

    market = reports["market_report.json"]
    rule_engine = market.get("rule_engine")
    if not isinstance(rule_engine, dict):
        raise AssertionError("market_report.json missing rule_engine")
    validate_strategy_contract(rule_engine, "market_report.json.rule_engine")

    portfolio_bound_reports = (
        "portfolio_backtest.json",
        "walk_forward_report.json",
        "market_regime_report.json",
        "overfitting_check.json",
    )
    expected_portfolio = {
        "portfolio_contract_fingerprint": portfolio_fingerprint,
        "split_manifest_fingerprint": split_fingerprint,
        "full_model_fingerprint": model_fingerprint,
        "model_governance_fingerprint": governance_fp,
    }
    for name in portfolio_bound_reports:
        actual = {field: reports[name].get(field) for field in expected_portfolio}
        if actual != expected_portfolio:
            raise AssertionError(f"{name} full-model fingerprint mismatch: {actual}")

    forward = reports["forward_validation_status.json"]
    if forward.get("split_manifest_fingerprint") != split_fingerprint:
        raise AssertionError("forward_validation_status.json split fingerprint mismatch")
    if forward.get("automatic_order_allowed") is not False:
        raise AssertionError("forward validation must never allow automatic orders")

    governance_report = reports["model_governance.json"]
    if governance_report.get("model_governance_fingerprint") != governance_fp:
        raise AssertionError("model_governance.json governance fingerprint mismatch")
    if governance_report.get("model_governance_contract_version") != GOVERNANCE_CONTRACT_VERSION:
        raise AssertionError("model_governance.json governance contract version mismatch")
    if governance_report.get("model_fingerprints") != expected_model_fingerprints:
        raise AssertionError("model_governance.json candidate fingerprints mismatch")
    if governance_report.get("retrospective_evidence_may_promote") is not False:
        raise AssertionError("retrospective evidence must not promote a governed model")
    if governance_report.get("automatic_order_allowed") is not False:
        raise AssertionError("model governance must never allow automatic orders")
    if governance_report.get("contains_private_account_data") is not False:
        raise AssertionError("model governance report must not contain private account data")
    try:
        candidate_events = _read_candidate_ledger(docs / LEDGER_REPORTS[0])
    except ModelCandidateLedgerError as exc:
        raise AssertionError(f"invalid model candidate ledger: {exc}") from exc
    reported_counts = governance_report.get("ledger_counts")
    if not isinstance(reported_counts, dict):
        raise AssertionError("model_governance.json ledger_counts is missing")
    actual_prediction_count = sum(
        event["event_type"] == "MODEL_PREDICTION" for event in candidate_events
    )
    actual_outcome_count = sum(
        event["event_type"] == "MODEL_OUTCOME" for event in candidate_events
    )
    if reported_counts.get("prediction_events") != actual_prediction_count:
        raise AssertionError("model candidate prediction count mismatch")
    if reported_counts.get("outcome_events") != actual_outcome_count:
        raise AssertionError("model candidate outcome count mismatch")

    csv_rows = {
        name: validate_csv_contract(docs / name)
        for name in MODEL_CSV_REPORTS
    }
    csv_rows["trade_review.csv"] = validate_csv_contract(
        docs / "trade_review.csv",
        require_model=False,
    )
    return {
        "status": "VALID",
        "model_report_count": len(MODEL_REPORTS),
        "metadata_report_count": len(METADATA_REPORTS),
        "csv_rows": csv_rows,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
        "split_manifest_fingerprint": split_fingerprint,
        "portfolio_contract_fingerprint": portfolio_fingerprint,
        "full_model_fingerprint": model_fingerprint,
        "model_governance_fingerprint": governance_fp,
        "model_candidate_event_count": len(candidate_events),
    }


def main() -> None:
    print(json.dumps(validate_model_artifacts(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
