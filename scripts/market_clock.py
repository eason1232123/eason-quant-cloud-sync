from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


MARKET_TIMEZONE = "America/New_York"
NEW_YORK = ZoneInfo(MARKET_TIMEZONE)


def parse_market_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None


def parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and "T" in value:
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def latest_completed_us_market_weekday(as_of_utc: Any = None) -> date:
    """Conservative weekday clock; exchange-holiday support is a later contract revision."""
    as_of = parse_timestamp(as_of_utc) or datetime.now(timezone.utc)
    ny = as_of.astimezone(NEW_YORK)
    candidate = ny.date()
    if ny.weekday() >= 5 or ny.hour < 18:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def reference_market_context(
    reported_expected_market_date: Any,
    *,
    as_of_utc: Any = None,
    as_of_market_date: Any = None,
) -> dict[str, str | None]:
    explicit_market_date = parse_market_date(as_of_market_date)
    if explicit_market_date is None and isinstance(as_of_utc, str) and "T" not in as_of_utc:
        explicit_market_date = parse_market_date(as_of_utc)
    runtime_expected = explicit_market_date or latest_completed_us_market_weekday(as_of_utc)
    reported_expected = parse_market_date(reported_expected_market_date)

    if reported_expected is None:
        status = "REPORTED_EXPECTED_DATE_MISSING"
    elif reported_expected != runtime_expected:
        status = "REPORTED_EXPECTED_DATE_MISMATCH"
    else:
        status = "ALIGNED"

    return {
        "reference_market_date": runtime_expected.isoformat(),
        "reference_source": "runtime_us_market_weekday_clock",
        "reported_expected_market_date": reported_expected.isoformat() if reported_expected else None,
        "reference_market_date_status": status,
        "date_basis": "weekday_business_days_not_exchange_sessions",
    }


def weekday_lag(latest: Any, reference: Any) -> int | None:
    latest_date = parse_market_date(latest)
    reference_date = parse_market_date(reference)
    if latest_date is None or reference_date is None or latest_date > reference_date:
        return None
    lag = 0
    cursor = latest_date
    while cursor < reference_date:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            lag += 1
    return lag
