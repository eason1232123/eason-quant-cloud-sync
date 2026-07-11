from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "docs" / "market_universe.csv"
DEFAULT_SUMMARY = ROOT / "docs" / "market_universe_summary.json"
MINIMUM_ELIGIBLE_TICKERS = 1000
MINIMUM_SOURCE_TICKERS = 100
REQUIRED_SOURCES = ("NASDAQ", "OTHER")


def validate_market_universe(
    csv_path: Path = DEFAULT_CSV,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    errors = summary.get("errors")
    sources = summary.get("sources")
    tickers = [str(row.get("ticker") or "").strip().upper() for row in rows]
    source_counts = {
        source: sum(1 for row in rows if row.get("source") == source)
        for source in REQUIRED_SOURCES
    }
    assert summary.get("version") == "market-universe-v1"
    assert isinstance(errors, list) and not errors, errors
    assert isinstance(sources, dict) and all(sources.get(key) for key in REQUIRED_SOURCES), sources
    assert all(count >= MINIMUM_SOURCE_TICKERS for count in source_counts.values()), source_counts
    assert summary.get("source_counts") == source_counts, (summary.get("source_counts"), source_counts)
    assert len(rows) >= MINIMUM_ELIGIBLE_TICKERS, len(rows)
    assert summary.get("eligible_ticker_count") == len(rows), summary
    assert len(tickers) == len(set(tickers)) and all(tickers)
    assert all(row.get("eligibility") == "eligible" for row in rows)
    return {
        "eligible_ticker_count": len(rows),
        "source_count": len(sources),
        "source_counts": source_counts,
    }


if __name__ == "__main__":
    print(validate_market_universe())
