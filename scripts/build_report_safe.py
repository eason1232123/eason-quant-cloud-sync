from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from scripts import build_report as br
from scripts.artifact_io import atomic_write_csv, atomic_write_text
from scripts.market_clock import latest_completed_us_market_weekday, weekday_lag


APPEND_ONLY_MODE = True
COVERAGE_GAPS_FIRST = True
WATCH_FORCE_REFRESH_LAG_BUSINESS_DAYS = 2
LONG_TAIL_FORCE_REFRESH_LAG_BUSINESS_DAYS = 5
CRITICAL_REFRESH_ANCHORS = {"SPY", "QQQ", "SMH", "MSFT"}
FRESHNESS_CONTRACT_VERSION = "v5-freshness-contract"

# API-saving refresh tiers. Existing cached data is always loaded into the report.
# Important: no-cache tickers bypass tier rotation so coverage can move toward 94/94.
CORE_DAILY_REFRESH = {
    "SPY", "QQQ", "SMH", "MSFT", "SGOV", "NVDA",
    "SOXX", "AVGO", "AAPL", "GOOGL", "AMZN", "META",
    "GLD", "TLT", "IEF",
}
WATCH_EVERY_3D_REFRESH = {
    "VGT", "XLK", "IYW", "IGV", "IGM", "TSLA",
    "AMD", "ASML", "TSM", "ARM", "MU", "LRCX", "AMAT", "KLAC", "INTC",
    "MRVL", "QCOM", "ON", "ADI", "TXN", "NXPI", "MCHP", "MPWR",
    "ORCL", "PLTR", "CRM", "ADBE", "SNOW", "DDOG", "NET", "NOW", "PANW", "CRWD",
    "VRT", "ETN", "PWR", "CEG", "NRG", "ANET",
}

CACHE_OK_STATUSES = {
    "cache_fresh_enough_no_request",
    "cache_only_tier_rotation",
}
CACHE_WARNING_STATUSES = {
    "cache_only_after_tiingo_circuit_breaker",
    "cache_after_fetch_error_append_only",
}
COVERAGE_GAP_STATUSES = {
    "deferred_new_ticker_rotation",
    "deferred_no_cache",
    "fetch_failed_no_cache_rate_limit",
    "fetch_failed_no_cache",
}


def stable_bucket(ticker: str, modulo: int) -> int:
    return sum(ord(c) for c in ticker.upper()) % modulo


def expected_latest_market_date(now_utc: datetime | None = None) -> datetime.date:
    return latest_completed_us_market_weekday(now_utc)


def refresh_tier(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in CORE_DAILY_REFRESH:
        return "core_daily"
    if ticker in WATCH_EVERY_3D_REFRESH:
        return "watch_every_3d"
    return "long_tail_weekly"


def tier_due_today(ticker: str, now_utc: datetime | None = None) -> bool:
    """Deterministic rotation so the large universe does not hit Tiingo every run."""
    now_utc = now_utc or datetime.now(timezone.utc)
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    ticker = ticker.upper()
    tier = refresh_tier(ticker)

    if tier == "core_daily":
        return True
    if tier == "watch_every_3d":
        return ny.toordinal() % 3 == stable_bucket(ticker, 3)
    return ny.weekday() == stable_bucket(ticker, 5)


def business_day_lag(latest_date, expected_date) -> int:
    """Weekday-only lag; this is not an exchange-session calendar."""
    lag = weekday_lag(latest_date, expected_date)
    return 0 if lag is None else lag


def force_refresh_lag_limit_business_days(ticker: str) -> int:
    tier = refresh_tier(ticker)
    if tier == "core_daily":
        return 0
    if tier == "watch_every_3d":
        return WATCH_FORCE_REFRESH_LAG_BUSINESS_DAYS
    return LONG_TAIL_FORCE_REFRESH_LAG_BUSINESS_DAYS


def request_order_key(ticker: str, expected_date) -> tuple:
    """Prioritize critical anchors, coverage gaps, and forced-overdue caches."""
    ticker = ticker.upper()
    last = br.cached_latest_date(ticker)
    if ticker in CRITICAL_REFRESH_ANCHORS and (last is None or last < expected_date):
        priority = 0
    elif last is None:
        priority = 1
    else:
        lag = business_day_lag(last, expected_date) if last <= expected_date else 0
        priority = 2 if lag > force_refresh_lag_limit_business_days(ticker) else 3
    tier_rank = {"core_daily": 0, "watch_every_3d": 1, "long_tail_weekly": 2}[refresh_tier(ticker)]
    last_for_sort = last if last is not None else datetime(1900, 1, 1).date()
    lag_sort = -business_day_lag(last, expected_date) if last is not None and last <= expected_date else 0
    return (priority, lag_sort, tier_rank, last_for_sort, ticker)


def is_tiingo_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return "http 429" in text or "hourly request allocation" in text or "run over your hourly" in text


def get_incremental_fetch_start(existing: pd.DataFrame) -> str:
    """Fetch only rows after the latest cached date to preserve append-only history."""
    if existing is None or existing.empty:
        return br.START_DATE
    latest = pd.to_datetime(existing["date"]).max().date()
    return (latest + timedelta(days=1)).isoformat()


def clean_float(value: Any, digits: int = 4) -> Any:
    return br.clean_float(value, digits)


def save_append_only(ticker: str, existing: pd.DataFrame, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Append only new dates; never replace existing date rows."""
    downloaded_rows = int(len(new_df)) if new_df is not None else 0

    if existing is None or existing.empty:
        merged = br.merge_and_save(ticker, new_df)
        return merged, downloaded_rows, int(len(merged))

    old = existing.copy()
    old["date"] = pd.to_datetime(old["date"]).dt.tz_localize(None)
    latest_cached = old["date"].max()

    if new_df is None or new_df.empty:
        return old.sort_values("date").reset_index(drop=True), downloaded_rows, 0

    new = new_df.copy()
    new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None)
    new_only = new[new["date"] > latest_cached].copy()

    if new_only.empty:
        return old.sort_values("date").reset_index(drop=True), downloaded_rows, 0

    merged = pd.concat([old, new_only], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    merged = merged.sort_values("date")
    merged = merged.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    preferred = [
        "ticker", "date", "open", "high", "low", "close", "volume",
        "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
        "divCash", "splitFactor", "price", "low_price", "high_price",
    ]
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    merged = merged[cols]
    merged.to_csv(br.csv_path(ticker), index=False)
    return merged, downloaded_rows, int(len(new_only))


def record_warning_error_or_gap(
    ticker: str,
    status: str,
    reason: str,
    warnings: dict,
    errors: dict,
    coverage_gaps: dict,
    has_cache: bool,
) -> None:
    """Keep normal cache skips out of errors so errors_count stays meaningful."""
    if status in CACHE_OK_STATUSES and has_cache:
        return
    if status in COVERAGE_GAP_STATUSES or (not has_cache and "deferred" in status):
        coverage_gaps[ticker] = reason
        return
    if has_cache:
        warnings[ticker] = reason
    else:
        errors[ticker] = reason


def use_cache_or_defer(
    ticker: str,
    existing: pd.DataFrame,
    price_map: dict,
    warnings: dict,
    errors: dict,
    coverage_gaps: dict,
    update_log: dict,
    reason: str,
    status: str | None = None,
    expected_date=None,
    quarantine: bool = False,
) -> None:
    if existing is not None and not existing.empty:
        latest = pd.to_datetime(existing["date"]).max().date()
        latest_date = latest.isoformat()
        effective_status = status or ("cache_only_after_tiingo_circuit_breaker" if "429" in reason.lower() else "cache_only")
        if not quarantine:
            price_map[ticker] = existing
        lag_business_days = (
            business_day_lag(latest, expected_date)
            if expected_date is not None and latest <= expected_date
            else None
        )
        update_log[ticker] = {
            "status": effective_status,
            "reason": reason,
            "refresh_tier": refresh_tier(ticker),
            "append_only_mode": APPEND_ONLY_MODE,
            "coverage_gaps_first": COVERAGE_GAPS_FIRST,
            "latest_date": latest_date,
            "expected_latest_market_date": expected_date.isoformat() if expected_date is not None else None,
            "lag_business_days": lag_business_days,
            "exact_date_fresh": bool(expected_date is not None and latest == expected_date and not quarantine),
            "decision_eligible": bool(expected_date is not None and latest == expected_date and not quarantine),
            "quarantined": quarantine,
            "total_rows_loaded": int(len(existing)),
            "tiingo_request_spent": False,
        }
        record_warning_error_or_gap(ticker, effective_status, reason, warnings, errors, coverage_gaps, has_cache=True)
        print(f"[CACHE] {ticker}: {reason}")
    else:
        effective_status = status or "deferred_no_cache"
        update_log[ticker] = {
            "status": effective_status,
            "reason": reason,
            "refresh_tier": refresh_tier(ticker),
            "append_only_mode": APPEND_ONLY_MODE,
            "coverage_gaps_first": COVERAGE_GAPS_FIRST,
            "expected_latest_market_date": expected_date.isoformat() if expected_date is not None else None,
            "exact_date_fresh": False,
            "decision_eligible": False,
            "quarantined": quarantine,
            "tiingo_request_spent": False,
        }
        record_warning_error_or_gap(ticker, effective_status, reason, warnings, errors, coverage_gaps, has_cache=False)
        print(f"[DEFER] {ticker}: {reason}")


def should_fetch_today(ticker: str, existing: pd.DataFrame, expected_date, now_utc: datetime) -> tuple[bool, str, str]:
    """Return should_fetch, reason, status_if_skipped."""
    has_cache = existing is not None and not existing.empty

    # Critical coverage fix: any no-cache ticker is eligible immediately, regardless of tier rotation.
    if not has_cache:
        return True, "no cache; coverage gap priority full history download", "fetch"

    latest = pd.to_datetime(existing["date"]).max().date()
    if latest > expected_date:
        return (
            False,
            f"future-dated cache {latest} is beyond expected market date {expected_date}; quarantine until rebuilt",
            "cache_future_dated_quarantine",
        )
    if latest == expected_date:
        return False, f"cache already covers expected latest market date {expected_date}", "cache_fresh_enough_no_request"

    lag_business_days = business_day_lag(latest, expected_date)
    force_limit = force_refresh_lag_limit_business_days(ticker)
    if lag_business_days > force_limit:
        return (
            True,
            f"forced refresh: cache lags {lag_business_days} business days, above {refresh_tier(ticker)} limit {force_limit}",
            "fetch",
        )

    if not tier_due_today(ticker, now_utc):
        return (
            False,
            f"not due today by {refresh_tier(ticker)} rotation; cache lag {lag_business_days} business days within limit {force_limit}",
            "cache_only_tier_rotation",
        )

    return True, f"due today by {refresh_tier(ticker)} rotation and cache is stale", "fetch"


def main() -> None:
    price_map: dict[str, pd.DataFrame] = {}
    warnings: dict[str, str] = {}
    errors: dict[str, str] = {}
    coverage_gaps: dict[str, str] = {}
    update_log: dict[str, dict] = {}

    requested = 0
    successful_fresh_requests = 0
    skipped_fresh_enough = 0
    skipped_by_rotation = 0
    new_full_downloads = 0
    rows_appended_total = 0
    forced_refresh_due = 0
    forced_refresh_requested = 0
    forced_refresh_succeeded = 0
    forced_refresh_deferred = 0
    future_quarantined = 0
    tiingo_circuit_open = False
    tiingo_circuit_reason = ""
    now_utc = datetime.now(timezone.utc)
    expected_date = expected_latest_market_date(now_utc)

    ordered_tickers = sorted(br.TICKERS, key=lambda ticker: request_order_key(ticker, expected_date))

    for ticker in ordered_tickers:
        existing = br.load_existing(ticker)
        has_cache = not existing.empty
        is_new_full_download = not has_cache

        should_fetch, skip_reason, skip_status = should_fetch_today(ticker, existing, expected_date, now_utc)
        is_forced_refresh = bool(should_fetch and skip_reason.startswith("forced refresh:"))
        if is_forced_refresh:
            forced_refresh_due += 1

        if not should_fetch:
            if skip_status == "cache_fresh_enough_no_request":
                skipped_fresh_enough += 1
            elif skip_status == "cache_future_dated_quarantine":
                future_quarantined += 1
            else:
                skipped_by_rotation += 1
            use_cache_or_defer(
                ticker,
                existing,
                price_map,
                warnings,
                errors,
                coverage_gaps,
                update_log,
                skip_reason,
                skip_status,
                expected_date=expected_date,
                quarantine=skip_status == "cache_future_dated_quarantine",
            )
            continue

        if tiingo_circuit_open:
            if is_forced_refresh:
                forced_refresh_deferred += 1
                status = "forced_refresh_deferred_circuit_breaker"
            else:
                status = "fetch_failed_no_cache_rate_limit" if not has_cache else "cache_only_after_tiingo_circuit_breaker"
            use_cache_or_defer(
                ticker,
                existing,
                price_map,
                warnings,
                errors,
                coverage_gaps,
                update_log,
                tiingo_circuit_reason,
                status,
                expected_date=expected_date,
            )
            continue

        if requested >= br.MAX_TIINGO_REQUESTS_PER_RUN:
            reason = f"request cap reached ({br.MAX_TIINGO_REQUESTS_PER_RUN})"
            if is_forced_refresh:
                forced_refresh_deferred += 1
                status = "forced_refresh_deferred_request_cap"
            else:
                status = None
            use_cache_or_defer(
                ticker,
                existing,
                price_map,
                warnings,
                errors,
                coverage_gaps,
                update_log,
                reason,
                status,
                expected_date=expected_date,
            )
            continue

        if is_new_full_download and new_full_downloads >= br.MAX_NEW_FULL_DOWNLOADS_PER_RUN:
            reason = f"new full download cap reached ({br.MAX_NEW_FULL_DOWNLOADS_PER_RUN})"
            use_cache_or_defer(
                ticker,
                existing,
                price_map,
                warnings,
                errors,
                coverage_gaps,
                update_log,
                reason,
                expected_date=expected_date,
            )
            continue

        try:
            fetch_start = get_incremental_fetch_start(existing)
            requested += 1
            if is_forced_refresh:
                forced_refresh_requested += 1
            if is_new_full_download:
                new_full_downloads += 1
            new_df = br.fetch_tiingo(ticker, fetch_start)

            downloaded_latest = pd.to_datetime(new_df["date"]).max().date()
            if downloaded_latest > expected_date:
                future_quarantined += 1
                if is_forced_refresh:
                    forced_refresh_deferred += 1
                reason = (
                    f"provider returned future-dated bar {downloaded_latest} beyond expected market date "
                    f"{expected_date}; response quarantined without saving"
                )
                status = (
                    "forced_refresh_future_data_quarantine"
                    if is_forced_refresh
                    else "provider_future_data_quarantine"
                )
                use_cache_or_defer(
                    ticker,
                    existing,
                    price_map,
                    warnings,
                    errors,
                    coverage_gaps,
                    update_log,
                    reason,
                    status,
                    expected_date=expected_date,
                )
                update_log[ticker]["provider_latest_date"] = downloaded_latest.isoformat()
                update_log[ticker]["provider_response_quarantined"] = True
                update_log[ticker]["tiingo_request_spent"] = True
                time.sleep(br.REQUEST_SLEEP_SECONDS)
                continue

            merged, downloaded_rows, appended_rows = save_append_only(ticker, existing, new_df)
            rows_appended_total += appended_rows
            price_map[ticker] = merged
            merged_latest = pd.to_datetime(merged["date"]).max().date()
            exact_date_fresh = merged_latest == expected_date
            lag_business_days = business_day_lag(merged_latest, expected_date) if merged_latest < expected_date else 0
            if exact_date_fresh:
                successful_fresh_requests += 1
                status = "fresh_from_tiingo_append_only" if appended_rows else "fresh_checked_expected_date_append_only"
                if is_forced_refresh:
                    forced_refresh_succeeded += 1
            else:
                status = (
                    "forced_refresh_incomplete_still_lagging"
                    if is_forced_refresh
                    else "updated_but_still_lagging"
                )
                if is_forced_refresh:
                    forced_refresh_deferred += 1
                warnings[ticker] = (
                    f"fetch completed but cache remains {lag_business_days} business days behind "
                    f"expected market date {expected_date}"
                )
            update_log[ticker] = {
                "status": status,
                "refresh_tier": refresh_tier(ticker),
                "coverage_gaps_first": COVERAGE_GAPS_FIRST,
                "fetch_start": fetch_start,
                "downloaded_rows": downloaded_rows,
                "new_rows_appended": appended_rows,
                "historical_rows_overwritten": 0,
                "append_only_mode": APPEND_ONLY_MODE,
                "total_rows_saved": int(len(merged)),
                "latest_date": merged_latest.isoformat(),
                "expected_latest_market_date": expected_date.isoformat(),
                "lag_business_days": lag_business_days,
                "exact_date_fresh": exact_date_fresh,
                "decision_eligible": exact_date_fresh,
                "forced_refresh": is_forced_refresh,
                "tiingo_request_spent": True,
            }
            print(
                f"[OK_APPEND_ONLY] {ticker}: fetched {downloaded_rows} rows from {fetch_start}; "
                f"appended {appended_rows}; saved {len(merged)} total"
            )

        except Exception as e:
            error_message = br.redact_sensitive_text(e)
            rate_limited = is_tiingo_rate_limit_error(RuntimeError(error_message))
            if is_forced_refresh:
                forced_refresh_deferred += 1
            if rate_limited:
                tiingo_circuit_open = True
                tiingo_circuit_reason = f"Tiingo 429 circuit breaker opened after {ticker}: {error_message}"
                print(f"[CIRCUIT_OPEN] {tiingo_circuit_reason}")

            if has_cache:
                price_map[ticker] = existing
                latest = pd.to_datetime(existing["date"]).max().date()
                lag_business_days = business_day_lag(latest, expected_date) if latest < expected_date else 0
                update_log[ticker] = {
                    "status": (
                        "forced_refresh_deferred_rate_limit"
                        if is_forced_refresh and rate_limited
                        else "forced_refresh_deferred_fetch_error"
                        if is_forced_refresh
                        else "cache_after_fetch_error_append_only"
                    ),
                    "refresh_tier": refresh_tier(ticker),
                    "coverage_gaps_first": COVERAGE_GAPS_FIRST,
                    "fetch_error": error_message,
                    "append_only_mode": APPEND_ONLY_MODE,
                    "historical_rows_overwritten": 0,
                    "latest_date": latest.isoformat(),
                    "expected_latest_market_date": expected_date.isoformat(),
                    "lag_business_days": lag_business_days,
                    "exact_date_fresh": latest == expected_date,
                    "decision_eligible": latest == expected_date,
                    "forced_refresh": is_forced_refresh,
                    "total_rows_loaded": int(len(existing)),
                    "tiingo_request_spent": True,
                }
                warnings[ticker] = f"using cached data after fetch error: {error_message}"
                print(f"[CACHE_AFTER_FAIL] {ticker}: {error_message}")
            else:
                status = "fetch_failed_no_cache_rate_limit" if rate_limited else "fetch_failed_no_cache"
                update_log[ticker] = {
                    "status": status,
                    "refresh_tier": refresh_tier(ticker),
                    "coverage_gaps_first": COVERAGE_GAPS_FIRST,
                    "fetch_error": error_message,
                    "expected_latest_market_date": expected_date.isoformat(),
                    "exact_date_fresh": False,
                    "decision_eligible": False,
                    "forced_refresh": False,
                    "tiingo_request_spent": True,
                }
                if rate_limited:
                    coverage_gaps[ticker] = error_message
                else:
                    errors[ticker] = error_message
                print(f"[FAIL] {ticker}: {error_message}")

        time.sleep(br.REQUEST_SLEEP_SECONDS)

    benchmarks = br.prepare_benchmarks(price_map)

    backtests = {}
    all_rows = []
    technicals = {}

    for ticker, df in price_map.items():
        if df.empty or len(df) < 30:
            errors[ticker] = errors.get(ticker, "not enough rows for indicators")
            continue
        try:
            nested, rows = br.summarize_all_rules(ticker, df, price_map, benchmarks)
            backtests[ticker] = nested
            all_rows.extend(rows)
            technicals[ticker] = br.latest_technical(df, price_map, ticker)
        except Exception as e:
            errors[ticker] = f"analysis_error: {br.redact_sensitive_text(e)}"

    exact_fresh_tickers: list[str] = []
    lagging_tickers: dict[str, int] = {}
    future_quarantined_tickers: list[str] = []
    missing_tickers: list[str] = []
    for ticker in br.TICKERS:
        item = update_log.get(ticker, {})
        status = str(item.get("status") or "")
        latest_value = item.get("latest_date")
        if "future" in status:
            future_quarantined_tickers.append(ticker)
            continue
        if not latest_value:
            missing_tickers.append(ticker)
            continue
        latest = pd.to_datetime(latest_value).date()
        if latest == expected_date and item.get("decision_eligible", True):
            exact_fresh_tickers.append(ticker)
        elif latest < expected_date:
            lagging_tickers[ticker] = business_day_lag(latest, expected_date)
        else:
            future_quarantined_tickers.append(ticker)

    freshness_contract = {
        "version": FRESHNESS_CONTRACT_VERSION,
        "date_basis": "weekday_business_days_not_exchange_sessions",
        "exact_expected_date_required_for_decisions": True,
        "expected_latest_market_date": expected_date.isoformat(),
        "exact_fresh_ticker_count": len(exact_fresh_tickers),
        "lagging_ticker_count": len(lagging_tickers),
        "future_quarantine_ticker_count": len(future_quarantined_tickers),
        "missing_ticker_count": len(missing_tickers),
        "exact_fresh_tickers": sorted(exact_fresh_tickers),
        "lagging_tickers_business_days": dict(sorted(lagging_tickers.items())),
        "future_quarantined_tickers": sorted(future_quarantined_tickers),
        "missing_tickers": sorted(missing_tickers),
    }

    market_data_fields = br.market_data_report_fields(
        price_map,
        "Tiingo Free API with local CSV cache fallback, tiered refresh, 429 circuit breaker, append-only cache updates",
    )
    report_generated_at_utc = datetime.now(timezone.utc).isoformat()

    report = {
        "generated_at_utc": report_generated_at_utc,
        "freshness_contract_version": FRESHNESS_CONTRACT_VERSION,
        **market_data_fields,
        "update_mode": (
            "large universe with coverage-gaps-first hydration: no-cache tickers are downloaded before tier rotation; "
            "critical anchors and caches beyond business-day lag limits bypass tier rotation; "
            "future-dated caches/provider responses are quarantined; "
            "existing historical CSV rows are not overwritten, only new dates are appended"
        ),
        "strategy_version": "Eason Master US Market Monitor Cloud Sync v5.1 shared-execution-contract",
        "strategy_contract_version": br.STRATEGY_CONTRACT_VERSION,
        "rule_fingerprint": br.RULE_FINGERPRINT,
        "strategy_fingerprint": br.STRATEGY_FINGERPRINT,
        "privacy_mode": "sanitized_public_report_no_cash_no_shares_no_account_value",
        "universe": {
            "configured_ticker_count": len(br.TICKERS),
            "loaded_ticker_count": len(price_map),
            "tiingo_request_attempt_count": requested,
            "fresh_request_count": successful_fresh_requests,
            "exact_fresh_ticker_count": len(exact_fresh_tickers),
            "lagging_ticker_count": len(lagging_tickers),
            "future_quarantine_ticker_count": len(future_quarantined_tickers),
            "missing_ticker_count": len(missing_tickers),
            "skipped_fresh_enough_no_request": skipped_fresh_enough,
            "skipped_by_tier_rotation": skipped_by_rotation,
            "forced_refresh_due_count": forced_refresh_due,
            "forced_refresh_requested_count": forced_refresh_requested,
            "forced_refresh_succeeded_count": forced_refresh_succeeded,
            "forced_refresh_deferred_count": forced_refresh_deferred,
            "future_quarantine_event_count": future_quarantined,
            "rows_appended_total": rows_appended_total,
            "append_only_mode": APPEND_ONLY_MODE,
            "coverage_gaps_first": COVERAGE_GAPS_FIRST,
            "historical_rows_overwritten": 0,
            "expected_latest_market_date": expected_date.isoformat(),
            "refresh_policy": {
                "core_daily": sorted(CORE_DAILY_REFRESH),
                "watch_every_3d": sorted(WATCH_EVERY_3D_REFRESH),
                "long_tail_weekly": "deterministic weekday rotation; no-cache tickers bypass rotation",
                "lag_basis": "weekday_business_days_not_exchange_sessions",
                "force_refresh_watch_lag_business_days": WATCH_FORCE_REFRESH_LAG_BUSINESS_DAYS,
                "force_refresh_long_tail_lag_business_days": LONG_TAIL_FORCE_REFRESH_LAG_BUSINESS_DAYS,
            },
            "max_tiingo_requests_per_run": br.MAX_TIINGO_REQUESTS_PER_RUN,
            "max_new_full_downloads_per_run": br.MAX_NEW_FULL_DOWNLOADS_PER_RUN,
            "tiingo_circuit_open": tiingo_circuit_open,
            "tiingo_circuit_reason": tiingo_circuit_reason,
            "warnings_count": len(warnings),
            "coverage_gaps_count": len(coverage_gaps),
            "errors_count": len(errors),
            "note": "coverage_gaps are tickers without usable cache yet; errors are true processing/fetch failures not classified as rotation/rate-limit coverage gaps.",
        },
        "freshness_contract": freshness_contract,
        "new_listing_policy": {
            "rule": "Short-history stocks are not rejected automatically.",
            "how_to_use": (
                "If a stock has less than 252 trading days, do not treat backtest evidence as reliable. "
                "It can still be recommended only as watchlist or tiny satellite when price action, relative strength, liquidity, "
                "fundamentals/news, and market regime confirm."
            ),
            "position_limit": "Usually 0% to 2% max unless user explicitly accepts high risk and evidence improves.",
        },
        "rule_engine": {
            "rules": [
                "pullback_reclaim_5dma",
                "rsi_oversold_reclaim_40",
                "ma20_reclaim_bullish",
                "ma50_reclaim_bullish",
                "relative_strength_rebound",
                "momentum_leader",
                "failed_rebound_risk",
            ],
            "horizons_days": br.HORIZONS,
            "minimum_valid_samples": br.MIN_SAMPLE,
            "sample_independence": "chronological non-overlapping signal windows",
            "execution": {
                "signal_time": "end_of_day_close",
                "entry_time": "next_trading_bar_close",
                "execution_shift_bars": br.EXECUTION_SHIFT_BARS,
            },
            "cost_assumptions": br.execution_cost_assumptions(),
            "strategy_contract_version": br.STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": br.RULE_FINGERPRINT,
            "strategy_fingerprint": br.STRATEGY_FINGERPRINT,
            "note": "Samples below minimum are reported but should not be used as primary buy evidence.",
        },
        "backtests": backtests,
        "rule_evidence_ranking": br.rank_rule_evidence(backtests),
        "technicals": technicals,
        "rolling_90d_correlation": br.corr_90d(price_map),
        "update_log": update_log,
        "warnings": warnings,
        "coverage_gaps": coverage_gaps,
        "errors": errors,
    }

    br.OUT.mkdir(exist_ok=True)
    report_json = json.dumps(report, indent=2, allow_nan=False)
    atomic_write_text(br.OUT / "market_report.json", report_json + "\n")

    atomic_write_csv(br.OUT / "backtest_summary.csv", br.report_dataframe(all_rows, report))

    ranking_rows = []
    for ticker, rows in report["rule_evidence_ranking"].items():
        for row in rows:
            ranking_rows.append({"ticker": ticker, **row})
    atomic_write_csv(
        br.OUT / "rule_evidence_ranking.csv",
        br.report_dataframe(ranking_rows, report),
    )

    atomic_write_text(
        br.OUT / "index.html",
        "<h1>Eason Quant Cloud Sync</h1>"
        "<p>Sanitized public report. Open artifact_manifest.json and decision_packet.json first; "
        "market_report.json is the large evidence source.</p>",
    )

    print("Saved coverage-gaps-first docs/market_report.json, docs/backtest_summary.csv, docs/rule_evidence_ranking.csv")


if __name__ == "__main__":
    main()
