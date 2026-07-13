from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.strategy_contract import (  # noqa: E402
    EXECUTION_SHIFT_BARS,
    FORWARD_HORIZONS,
    RULE_FINGERPRINT,
    STRATEGY_CONTRACT_VERSION,
    STRATEGY_FINGERPRINT,
)


DEFAULT_MANIFEST = ROOT / "config" / "validation_split.json"
DEFAULT_ANCHOR = ROOT / "docs" / "SPY_daily.csv"
HISTORICAL_CLASSIFICATION = "RETROSPECTIVE_CONTAMINATED"
PROSPECTIVE_CLASSIFICATION = "PROSPECTIVE"


class ValidationSplitError(ValueError):
    """Raised when the frozen validation split is incomplete or inconsistent."""


def _reject_constant(value: str) -> None:
    raise ValidationSplitError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationSplitError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValidationSplitError(f"missing or empty validation split manifest: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationSplitError(f"invalid validation split JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationSplitError("validation split manifest must be a JSON object")
    assert_finite_json(value)
    return value


def assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationSplitError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")


def canonical_fingerprint(value: Any) -> str:
    assert_finite_json(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Return the stable fingerprint used to bind prospective events to one frozen split."""
    return canonical_fingerprint(manifest)


def full_model_fingerprint(
    strategy_fingerprint: str,
    portfolio_contract_payload: dict[str, Any],
    split_manifest: dict[str, Any],
) -> str:
    """Bind strategy, caller-owned portfolio contract, and split without duplicating weights here."""
    if not isinstance(strategy_fingerprint, str) or len(strategy_fingerprint) != 64:
        raise ValidationSplitError("strategy_fingerprint must be a SHA-256 fingerprint")
    try:
        int(strategy_fingerprint, 16)
    except ValueError as exc:
        raise ValidationSplitError("strategy_fingerprint must be hexadecimal") from exc
    if not isinstance(portfolio_contract_payload, dict) or not portfolio_contract_payload:
        raise ValidationSplitError("portfolio_contract_payload must be a non-empty object")
    validate_split_manifest(split_manifest)
    return canonical_fingerprint(
        {
            "strategy_fingerprint": strategy_fingerprint,
            "portfolio_contract_payload": portfolio_contract_payload,
            "split_manifest_fingerprint": split_manifest_fingerprint(split_manifest),
        }
    )


def _require_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValidationSplitError(f"{key} must be an object")
    return value


def _require_iso_date(parent: dict[str, Any], key: str, context: str) -> date:
    value = parent.get(key)
    if not isinstance(value, str):
        raise ValidationSplitError(f"{context}.{key} must be an ISO market date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationSplitError(f"{context}.{key} is not a valid ISO market date: {value}") from exc
    if parsed.isoformat() != value:
        raise ValidationSplitError(f"{context}.{key} must use YYYY-MM-DD format")
    return parsed


def _require_sha256(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationSplitError(f"{key} must be a 64-character SHA-256 fingerprint")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValidationSplitError(f"{key} must be hexadecimal") from exc
    return value


def load_anchor_market_dates(path: Path) -> list[date]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValidationSplitError(f"missing or empty purge anchor data: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "date" not in reader.fieldnames:
                raise ValidationSplitError(f"purge anchor {path} is missing the date column")
            raw_dates = [row.get("date") for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationSplitError(f"could not read purge anchor {path}: {exc}") from exc
    if not raw_dates:
        raise ValidationSplitError(f"purge anchor {path} has no market rows")
    parsed: list[date] = []
    for row_number, raw in enumerate(raw_dates, start=2):
        try:
            market_date = date.fromisoformat(str(raw))
        except ValueError as exc:
            raise ValidationSplitError(
                f"purge anchor {path} has an invalid date on row {row_number}: {raw}"
            ) from exc
        parsed.append(market_date)
    if parsed != sorted(parsed) or len(parsed) != len(set(parsed)):
        raise ValidationSplitError("purge anchor dates must be unique and strictly increasing")
    return parsed


def _anchor_index(anchor_dates: list[date], value: date, context: str) -> int:
    try:
        return anchor_dates.index(value)
    except ValueError as exc:
        raise ValidationSplitError(f"{context} date {value.isoformat()} is absent from purge anchor") from exc


def validate_split_manifest(
    manifest: dict[str, Any],
    *,
    anchor_dates: Iterable[date] | None = None,
    enforce_runtime_contract: bool = True,
) -> dict[str, Any]:
    assert_finite_json(manifest)
    if manifest.get("schema_version") != "v6-validation-split-v1":
        raise ValidationSplitError("unsupported validation split schema_version")
    if manifest.get("status") != "FROZEN":
        raise ValidationSplitError("validation split status must be FROZEN")
    _require_iso_date(manifest, "frozen_on_date", "manifest")
    rule_fingerprint = _require_sha256(manifest, "rule_fingerprint")
    strategy_fingerprint = _require_sha256(manifest, "strategy_fingerprint")
    strategy_version = manifest.get("strategy_contract_version")
    if not isinstance(strategy_version, str) or not strategy_version:
        raise ValidationSplitError("strategy_contract_version is required")

    if enforce_runtime_contract:
        expected = {
            "strategy_contract_version": STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": RULE_FINGERPRINT,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
        }
        actual = {
            "strategy_contract_version": strategy_version,
            "rule_fingerprint": rule_fingerprint,
            "strategy_fingerprint": strategy_fingerprint,
        }
        if actual != expected:
            raise ValidationSplitError(
                "frozen validation fingerprints do not match the running strategy contract; "
                "start a reviewed validation generation instead of silently reusing this ledger"
            )

    historical = _require_object(manifest, "historical_classification")
    if historical.get("status") != HISTORICAL_CLASSIFICATION:
        raise ValidationSplitError("all history visible at freeze must be RETROSPECTIVE_CONTAMINATED")
    if historical.get("prohibited_claim") != "PRISTINE_OUT_OF_SAMPLE":
        raise ValidationSplitError("historical classification must explicitly prohibit pristine OOS claims")
    contaminated_through = _require_iso_date(
        historical,
        "contaminated_through_market_date",
        "historical_classification",
    )

    label = _require_object(manifest, "label_contract")
    entry_offset = label.get("entry_close_offset_bars")
    horizons = label.get("forward_horizons_bars")
    maximum_span = label.get("maximum_label_span_bars")
    if entry_offset != EXECUTION_SHIFT_BARS:
        raise ValidationSplitError("entry_close_offset_bars does not match the strategy execution shift")
    if (
        not isinstance(horizons, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in horizons)
        or horizons != sorted(set(horizons))
    ):
        raise ValidationSplitError("forward_horizons_bars must be sorted unique positive integers")
    if tuple(horizons) != tuple(FORWARD_HORIZONS):
        raise ValidationSplitError("forward_horizons_bars do not match the running strategy contract")
    if maximum_span != entry_offset + max(horizons):
        raise ValidationSplitError("maximum_label_span_bars must equal entry offset plus maximum horizon")

    partitions = manifest.get("historical_partitions")
    if not isinstance(partitions, list) or len(partitions) != 3:
        raise ValidationSplitError("historical_partitions must contain train, validation, and retrospective_test")
    names = [item.get("name") if isinstance(item, dict) else None for item in partitions]
    if names != ["train", "validation", "retrospective_test"]:
        raise ValidationSplitError("historical partitions must be ordered train, validation, retrospective_test")

    parsed_partitions: list[dict[str, Any]] = []
    for item in partitions:
        if not isinstance(item, dict):
            raise ValidationSplitError("each historical partition must be an object")
        name = str(item["name"])
        if item.get("classification") != HISTORICAL_CLASSIFICATION:
            raise ValidationSplitError(f"{name} must be labeled {HISTORICAL_CLASSIFICATION}")
        start = _require_iso_date(item, "signal_start_market_date", name)
        end = _require_iso_date(item, "signal_end_market_date", name)
        label_end = _require_iso_date(item, "label_maturation_end_market_date", name)
        purge = _require_object(item, "boundary_purge")
        purge_start = _require_iso_date(purge, "start_market_date", f"{name}.boundary_purge")
        purge_end = _require_iso_date(purge, "end_market_date", f"{name}.boundary_purge")
        if not start <= end < purge_start <= purge_end:
            raise ValidationSplitError(f"{name} signal and purge dates are not strictly separated")
        if label_end != purge_end:
            raise ValidationSplitError(f"{name} label maturation must end at the purge end")
        if purge.get("trading_bars") != maximum_span:
            raise ValidationSplitError(f"{name} purge bars must equal maximum_label_span_bars")
        parsed_partitions.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "label_end": label_end,
                "purge_start": purge_start,
                "purge_end": purge_end,
            }
        )

    for earlier, later in zip(parsed_partitions, parsed_partitions[1:]):
        if earlier["label_end"] >= later["start"]:
            raise ValidationSplitError(
                f"{earlier['name']} labels overlap the {later['name']} signal partition"
            )
    if parsed_partitions[-1]["label_end"] != contaminated_through:
        raise ValidationSplitError("retrospective labels must end on contaminated_through_market_date")

    anchor = list(anchor_dates) if anchor_dates is not None else None
    if anchor is not None:
        if anchor != sorted(anchor) or len(anchor) != len(set(anchor)):
            raise ValidationSplitError("purge anchor dates must be unique and strictly increasing")
        for item in parsed_partitions:
            _anchor_index(anchor, item["start"], f"{item['name']} signal start")
            end_index = _anchor_index(anchor, item["end"], f"{item['name']} signal end")
            purge_start_index = _anchor_index(anchor, item["purge_start"], f"{item['name']} purge start")
            purge_end_index = _anchor_index(anchor, item["purge_end"], f"{item['name']} purge end")
            if purge_start_index != end_index + 1:
                raise ValidationSplitError(f"{item['name']} purge must start on the next anchor bar")
            if purge_end_index - end_index != maximum_span:
                raise ValidationSplitError(
                    f"{item['name']} purge does not cover exactly {maximum_span} trading bars"
                )

    prospective = _require_object(manifest, "prospective_partition")
    if prospective.get("classification") != PROSPECTIVE_CLASSIFICATION:
        raise ValidationSplitError("prospective_partition must be labeled PROSPECTIVE")
    if prospective.get("start_rule") != "STRICTLY_AFTER_LAST_SEEN_MARKET_DATE":
        raise ValidationSplitError("prospective start rule must be strictly after the last seen market date")
    last_seen = _require_iso_date(prospective, "last_seen_market_date", "prospective_partition")
    earliest = _require_iso_date(prospective, "earliest_calendar_date", "prospective_partition")
    if last_seen != contaminated_through or earliest != last_seen + timedelta(days=1):
        raise ValidationSplitError("prospective boundary must begin after all data visible at freeze")
    if prospective.get("private_account_data_allowed") is not False:
        raise ValidationSplitError("prospective ledger must prohibit private account data")

    return {
        "schema_version": manifest["schema_version"],
        "status": manifest["status"],
        "historical_status": HISTORICAL_CLASSIFICATION,
        "contaminated_through_market_date": contaminated_through.isoformat(),
        "prospective_starts_strictly_after": last_seen.isoformat(),
        "maximum_label_span_bars": maximum_span,
        "partition_names": names,
        "strategy_contract_version": strategy_version,
        "rule_fingerprint": rule_fingerprint,
        "strategy_fingerprint": strategy_fingerprint,
        "split_manifest_fingerprint": split_manifest_fingerprint(manifest),
    }


def validate_manifest_file(
    manifest_path: Path = DEFAULT_MANIFEST,
    anchor_path: Path = DEFAULT_ANCHOR,
) -> dict[str, Any]:
    manifest = load_strict_json(manifest_path)
    anchor_dates = load_anchor_market_dates(anchor_path)
    return validate_split_manifest(manifest, anchor_dates=anchor_dates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen v6 historical/prospective split.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--anchor-csv", type=Path, default=DEFAULT_ANCHOR)
    args = parser.parse_args(argv)
    try:
        result = validate_manifest_file(args.manifest, args.anchor_csv)
    except ValidationSplitError as exc:
        print(f"validation split FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
