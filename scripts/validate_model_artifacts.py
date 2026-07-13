from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.build_portfolio_backtest import portfolio_contract_payload
from scripts.market_clock import MARKET_TIMEZONE, parse_market_date, parse_timestamp
from scripts.market_data_contract import (
    DATA_TIMESTAMP_GRANULARITY,
    PRICE_ADJUSTMENT_POLICY,
    PRICE_FREQUENCY,
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
    }


def main() -> None:
    print(json.dumps(validate_model_artifacts(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
