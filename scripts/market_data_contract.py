from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.market_clock import (
    MARKET_TIMEZONE,
    latest_completed_us_market_weekday,
    parse_market_date,
    parse_timestamp,
)


DATA_TIMESTAMP_GRANULARITY = "market_date"
PRICE_FREQUENCY = "end_of_day_daily"
PRICE_ADJUSTMENT_POLICY = "adjusted_ohlc_when_available_else_unadjusted"


def daily_csv_path(ticker: str, out: Path) -> Path:
    safe = ticker.replace("/", "-").replace(".", "-")
    return out / f"{safe}_daily.csv"


def read_checked_daily_csv(
    ticker: str,
    out: Path,
    expected_market_date: Any = None,
) -> pd.DataFrame:
    path = daily_csv_path(ticker, out)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    if "date" not in frame.columns:
        raise ValueError(f"{path} is missing required date column")
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"{path} contains invalid market dates")
    latest = parsed_dates.max().date()
    expected = parse_market_date(expected_market_date) or latest_completed_us_market_weekday()
    if latest > expected:
        raise ValueError(
            f"{ticker} price cache is future-dated ({latest}) relative to expected market date {expected}"
        )
    return frame


def extract_market_data_metadata(report: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "source": report.get("data_source"),
        "market_timezone": report.get("market_timezone"),
        "report_generated_at_utc": report.get("generated_at_utc"),
        "data_timestamp": report.get("data_timestamp"),
        "data_timestamp_granularity": report.get("data_timestamp_granularity"),
        "data_timestamp_status": report.get("data_timestamp_status"),
        "price_frequency": report.get("price_frequency"),
        "price_adjustment_policy": report.get("price_adjustment_policy"),
    }
    complete = bool(
        isinstance(metadata["source"], str)
        and metadata["source"].strip()
        and metadata["market_timezone"] == MARKET_TIMEZONE
        and parse_timestamp(metadata["report_generated_at_utc"]) is not None
        and parse_market_date(metadata["data_timestamp"]) is not None
        and metadata["data_timestamp_granularity"] == DATA_TIMESTAMP_GRANULARITY
        and metadata["data_timestamp_status"] == "AVAILABLE"
        and metadata["price_frequency"] == PRICE_FREQUENCY
        and metadata["price_adjustment_policy"] == PRICE_ADJUSTMENT_POLICY
    )
    metadata["metadata_status"] = "COMPLETE" if complete else "MISSING_OR_INVALID"
    return metadata
