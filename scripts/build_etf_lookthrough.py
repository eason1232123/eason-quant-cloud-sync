from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_forward_ledger import _write_json_atomic, load_public_json  # noqa: E402
from scripts.market_clock import parse_market_date  # noqa: E402
from scripts.validate_validation_split import assert_finite_json  # noqa: E402


DEFAULT_POLICY = ROOT / "config" / "etf_lookthrough.json"
DEFAULT_REPORT = ROOT / "docs" / "market_report.json"
DEFAULT_PORTFOLIO = ROOT / "docs" / "portfolio_backtest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "etf_lookthrough_status.json"
STATUS_SCHEMA_VERSION = "official-etf-lookthrough-status-v1"
CACHE_COLUMNS = (
    "fund",
    "as_of_date",
    "symbol",
    "name",
    "weight_pct",
    "source_provider",
    "source_url",
)
TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


class EtfLookthroughError(ValueError):
    """Raised when an official ETF holdings cache is malformed."""


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"td", "th"} and self._row is not None and self._cell is not None:
            value = " ".join("".join(self._cell).split())
            self._row.append(value)
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _weekday_age(older: date, newer: date) -> int | None:
    if older > newer:
        return None
    count = 0
    cursor = older
    while cursor < newer:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            count += 1
    return count


def _policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    payload = load_public_json(path, "ETF lookthrough policy")
    if set(payload) != {"schema_version", "maximum_staleness_weekdays", "funds"}:
        raise EtfLookthroughError("ETF lookthrough policy fields changed")
    if payload["schema_version"] != "official-etf-lookthrough-policy-v1":
        raise EtfLookthroughError("unsupported ETF lookthrough policy")
    maximum = payload["maximum_staleness_weekdays"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise EtfLookthroughError("ETF holdings staleness limit is invalid")
    funds = payload["funds"]
    if not isinstance(funds, dict) or set(funds) != {"QQQ", "SMH"}:
        raise EtfLookthroughError("ETF lookthrough must cover exactly QQQ and SMH")
    for fund, config in funds.items():
        if not isinstance(config, dict) or set(config) != {
            "provider",
            "official_url",
            "cache_file",
        }:
            raise EtfLookthroughError(f"{fund} ETF source contract changed")
        host = urlparse(config["official_url"]).hostname
        expected = "invesco.com" if fund == "QQQ" else "vaneck.com"
        if host is None or not (host == expected or host.endswith(f".{expected}")):
            raise EtfLookthroughError(f"{fund} source is not on the official provider domain")
    return payload


def _parse_official_html(
    *,
    fund: str,
    provider: str,
    url: str,
    html: str,
) -> tuple[date, list[dict[str, Any]]]:
    dates = [
        date(int(year), int(month), int(day))
        for month, day, year in re.findall(
            r"(?:holdings(?:\s+\(%\))?\s+as\s+of|portfolio holdings as of)\s+"
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
            html,
            flags=re.IGNORECASE,
        )
    ]
    if not dates:
        raise EtfLookthroughError(f"{fund} official page has no unambiguous holdings date")
    as_of = max(dates)
    parser = _TableParser()
    parser.feed(html)
    holdings: dict[str, dict[str, Any]] = {}
    for row in parser.rows:
        cleaned = [value.strip() for value in row if value.strip()]
        if len(cleaned) < 3:
            continue
        symbol_index = next(
            (index for index, value in enumerate(cleaned[:3]) if TICKER.fullmatch(value)),
            None,
        )
        if symbol_index is None:
            continue
        symbol = cleaned[symbol_index]
        if "CASH" in symbol or symbol in {"USD", "--"}:
            continue
        weight = None
        for value in cleaned[symbol_index + 1 :]:
            normalized = value.replace("%", "").replace(",", "").strip()
            try:
                candidate = float(normalized)
            except ValueError:
                continue
            if 0 < candidate <= 100:
                weight = candidate
                break
        if weight is None:
            continue
        name = cleaned[symbol_index + 1] if symbol_index + 1 < len(cleaned) else symbol
        holdings[symbol] = {
            "fund": fund,
            "as_of_date": as_of.isoformat(),
            "symbol": symbol,
            "name": name,
            "weight_pct": weight,
            "source_provider": provider,
            "source_url": url,
        }
    minimum = 80 if fund == "QQQ" else 20
    if len(holdings) < minimum:
        raise EtfLookthroughError(
            f"{fund} official page yielded only {len(holdings)} holdings; refusing partial data"
        )
    total = sum(row["weight_pct"] for row in holdings.values())
    if not 95 <= total <= 105:
        raise EtfLookthroughError(f"{fund} official holding weights do not reconcile")
    return as_of, sorted(holdings.values(), key=lambda row: (-row["weight_pct"], row["symbol"]))


def _write_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def refresh_official_cache(
    fund: str,
    config: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    try:
        response = requests.get(
            config["official_url"],
            timeout=timeout_seconds,
            headers={"User-Agent": "eason-quant-official-etf-lookthrough/1.0"},
        )
        response.raise_for_status()
        as_of, rows = _parse_official_html(
            fund=fund,
            provider=config["provider"],
            url=config["official_url"],
            html=response.text,
        )
        _write_cache(ROOT / config["cache_file"], rows)
        return {"status": "REFRESHED", "as_of_date": as_of.isoformat(), "row_count": len(rows)}
    except (requests.RequestException, EtfLookthroughError, OSError) as exc:
        return {"status": "UNAVAILABLE", "error": str(exc)}


def _load_cache(
    fund: str,
    config: dict[str, Any],
    *,
    reference_date: date,
    maximum_staleness: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT / config["cache_file"]
    base = {
        "provider": config["provider"],
        "official_url": config["official_url"],
        "cache_file": config["cache_file"],
        "holdings_as_of_date": None,
        "staleness_weekdays": None,
        "constituent_count": 0,
        "weight_sum_pct": None,
        "status": "UNAVAILABLE",
        "reason": "OFFICIAL_HOLDINGS_CACHE_MISSING",
    }
    if not path.exists() or path.stat().st_size == 0:
        return base, []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CACHE_COLUMNS:
                raise EtfLookthroughError(f"{fund} cache columns changed")
            rows = list(reader)
        parsed = []
        dates: set[date] = set()
        symbols: set[str] = set()
        for row in rows:
            if row["fund"] != fund or row["source_provider"] != config["provider"]:
                raise EtfLookthroughError(f"{fund} cache lineage is invalid")
            if row["source_url"] != config["official_url"] or not TICKER.fullmatch(row["symbol"]):
                raise EtfLookthroughError(f"{fund} cache source or symbol is invalid")
            observed = date.fromisoformat(row["as_of_date"])
            weight = float(row["weight_pct"])
            if not math.isfinite(weight) or not 0 < weight <= 100 or row["symbol"] in symbols:
                raise EtfLookthroughError(f"{fund} cache holding is invalid")
            symbols.add(row["symbol"])
            dates.add(observed)
            parsed.append({"symbol": row["symbol"], "name": row["name"], "weight_pct": weight})
        if len(dates) != 1 or not parsed:
            raise EtfLookthroughError(f"{fund} cache date or holdings are incomplete")
        observed = next(iter(dates))
        age = _weekday_age(observed, reference_date)
        total = sum(row["weight_pct"] for row in parsed)
        if age is None:
            raise EtfLookthroughError(f"{fund} holdings date is after the market report")
        status = "AVAILABLE" if age <= maximum_staleness and 95 <= total <= 105 else "UNAVAILABLE"
        reason = None
        if age > maximum_staleness:
            reason = "OFFICIAL_HOLDINGS_STALE"
        elif not 95 <= total <= 105:
            reason = "OFFICIAL_HOLDING_WEIGHTS_DO_NOT_RECONCILE"
        base.update(
            {
                "holdings_as_of_date": observed.isoformat(),
                "staleness_weekdays": age,
                "constituent_count": len(parsed),
                "weight_sum_pct": round(total, 8),
                "status": status,
                "reason": reason,
            }
        )
        return base, parsed if status == "AVAILABLE" else []
    except (OSError, ValueError, EtfLookthroughError) as exc:
        base["reason"] = "OFFICIAL_HOLDINGS_CACHE_INVALID"
        base["validation_error"] = str(exc)
        return base, []


def _analysis(
    holdings: dict[str, list[dict[str, Any]]],
    portfolio: dict[str, Any],
) -> dict[str, Any]:
    maps = {
        fund: {row["symbol"]: row["weight_pct"] for row in rows}
        for fund, rows in holdings.items()
    }
    shared = sorted(set(maps["QQQ"]) & set(maps["SMH"]))
    shared_rows = [
        {
            "symbol": symbol,
            "qqq_weight_pct": round(maps["QQQ"][symbol], 8),
            "smh_weight_pct": round(maps["SMH"][symbol], 8),
            "minimum_overlap_weight_pct": round(
                min(maps["QQQ"][symbol], maps["SMH"][symbol]), 8
            ),
        }
        for symbol in shared
    ]
    shared_rows.sort(key=lambda row: (-row["minimum_overlap_weight_pct"], row["symbol"]))
    assumptions = portfolio.get("assumptions", {}) if isinstance(portfolio, dict) else {}
    weights = assumptions.get("base_weights", {}) if isinstance(assumptions, dict) else {}
    model_exposure: dict[str, float] = {}
    if isinstance(weights, dict):
        for symbol, direct_weight in weights.items():
            if symbol not in {"QQQ", "SMH", "CASH"} and isinstance(direct_weight, (int, float)):
                model_exposure[symbol] = model_exposure.get(symbol, 0.0) + float(direct_weight)
        for fund in ("QQQ", "SMH"):
            fund_weight = weights.get(fund, 0.0)
            if isinstance(fund_weight, (int, float)):
                for symbol, constituent_pct in maps[fund].items():
                    model_exposure[symbol] = model_exposure.get(symbol, 0.0) + float(
                        fund_weight
                    ) * constituent_pct / 100.0
    top_model = [
        {"symbol": symbol, "model_weight_from_direct_qqq_smh": round(weight, 10)}
        for symbol, weight in sorted(model_exposure.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]
    return {
        "qqq_smh_shared_constituent_count": len(shared_rows),
        "qqq_smh_minimum_weight_overlap_pct": round(
            sum(row["minimum_overlap_weight_pct"] for row in shared_rows), 8
        ),
        "top_shared_constituents": shared_rows[:20],
        "model_portfolio_top_direct_qqq_smh_lookthrough": top_model,
        "private_account_lookthrough_status": "REQUIRES_PRIVATE_LOCAL_CONTEXT",
    }


def build_etf_lookthrough(
    *,
    policy_path: Path = DEFAULT_POLICY,
    report_path: Path = DEFAULT_REPORT,
    portfolio_path: Path = DEFAULT_PORTFOLIO,
    output_path: Path | None = DEFAULT_OUTPUT,
    refresh: bool = False,
    generated_at_utc: datetime | None = None,
) -> dict[str, Any]:
    policy = _policy(policy_path)
    report = load_public_json(report_path, "market report")
    portfolio = load_public_json(portfolio_path, "portfolio backtest")
    reference = parse_market_date(report.get("data_timestamp"))
    if reference is None:
        raise EtfLookthroughError("market report data_timestamp is invalid")
    refresh_results: dict[str, Any] = {}
    if refresh:
        for fund, config in policy["funds"].items():
            refresh_results[fund] = refresh_official_cache(fund, config)
    fund_status: dict[str, Any] = {}
    available_holdings: dict[str, list[dict[str, Any]]] = {}
    for fund, config in policy["funds"].items():
        status, rows = _load_cache(
            fund,
            config,
            reference_date=reference,
            maximum_staleness=policy["maximum_staleness_weekdays"],
        )
        status["refresh_result"] = refresh_results.get(fund)
        fund_status[fund] = status
        if status["status"] == "AVAILABLE":
            available_holdings[fund] = rows
    overall = "AVAILABLE" if set(available_holdings) == {"QQQ", "SMH"} else "UNAVAILABLE"
    payload = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "generated_at_utc": (
            generated_at_utc or datetime.now(timezone.utc)
        ).astimezone(timezone.utc).isoformat(),
        "market_data_timestamp": reference.isoformat(),
        "status": overall,
        "maximum_staleness_weekdays": policy["maximum_staleness_weekdays"],
        "funds": fund_status,
        "overlap_analysis": (
            _analysis(available_holdings, portfolio) if overall == "AVAILABLE" else None
        ),
        "shadow_evidence_collection_blocked": False,
        "human_pilot_release_gate": False,
        "execution_review_requirement": (
            "UNAVAILABLE or stale lookthrough must be disclosed during actual-account review; "
            "it never upgrades or creates a buy candidate."
        ),
        "contains_private_account_data": False,
        "automatic_order_allowed": False,
    }
    assert_finite_json(payload)
    if output_path is not None:
        _write_json_atomic(output_path, payload)
    return payload


def validate_etf_lookthrough(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "generated_at_utc",
        "market_data_timestamp",
        "status",
        "maximum_staleness_weekdays",
        "funds",
        "overlap_analysis",
        "shadow_evidence_collection_blocked",
        "human_pilot_release_gate",
        "execution_review_requirement",
        "contains_private_account_data",
        "automatic_order_allowed",
    }
    if set(payload) != required or payload["schema_version"] != STATUS_SCHEMA_VERSION:
        raise EtfLookthroughError("ETF lookthrough status contract changed")
    if set(payload["funds"]) != {"QQQ", "SMH"}:
        raise EtfLookthroughError("ETF lookthrough fund coverage changed")
    expected = (
        "AVAILABLE"
        if all(payload["funds"][fund]["status"] == "AVAILABLE" for fund in ("QQQ", "SMH"))
        else "UNAVAILABLE"
    )
    if payload["status"] != expected or (payload["overlap_analysis"] is None) is (expected == "AVAILABLE"):
        raise EtfLookthroughError("ETF lookthrough availability or analysis is inconsistent")
    if (
        payload["shadow_evidence_collection_blocked"] is not False
        or payload["human_pilot_release_gate"] is not False
        or payload["contains_private_account_data"] is not False
        or payload["automatic_order_allowed"] is not False
    ):
        raise EtfLookthroughError("ETF lookthrough safety boundary changed")
    assert_finite_json(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build official QQQ/SMH lookthrough status.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = (
            load_public_json(args.output, "ETF lookthrough status")
            if args.validate_only
            else build_etf_lookthrough(output_path=args.output, refresh=args.refresh)
        )
        validate_etf_lookthrough(payload)
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    except (EtfLookthroughError, ValueError, AssertionError) as exc:
        print(f"ETF lookthrough FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
