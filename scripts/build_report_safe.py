from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from scripts import build_report as br


APPEND_ONLY_MODE = True
COVERAGE_GAPS_FIRST = True

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
    """Best-effort latest U.S. daily bar date without using an exchange calendar."""
    now_utc = now_utc or datetime.now(timezone.utc)
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    candidate = ny.date()

    if ny.weekday() >= 5 or ny.hour < 18:
        candidate = candidate - timedelta(days=1)

    while candidate.weekday() >= 5:
        candidate = candidate - timedelta(days=1)
    return candidate


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


def request_order_key(ticker: str) -> tuple:
    """Close coverage gaps first: no-cache tickers outrank tier rotation."""
    ticker = ticker.upper()
    last = br.cached_latest_date(ticker)
    no_cache_rank = 0 if last is None else 1
    tier_rank = {"core_daily": 0, "watch_every_3d": 1, "long_tail_weekly": 2}[refresh_tier(ticker)]
    last_for_sort = last if last is not None else datetime(1900, 1, 1).date()
    return (no_cache_rank, tier_rank, last_for_sort, ticker)


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
) -> None:
    if existing is not None and not existing.empty:
        latest_date = pd.to_datetime(existing["date"]).max().date().isoformat()
        effective_status = status or ("cache_only_after_tiingo_circuit_breaker" if "429" in reason.lower() else "cache_only")
        price_map[ticker] = existing
        update_log[ticker] = {
            "status": effective_status,
            "reason": reason,
            "refresh_tier": refresh_tier(ticker),
            "append_only_mode": APPEND_ONLY_MODE,
            "coverage_gaps_first": COVERAGE_GAPS_FIRST,
            "latest_date": latest_date,
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
    if latest >= expected_date:
        return False, f"cache already covers expected latest market date {expected_date}", "cache_fresh_enough_no_request"

    if not tier_due_today(ticker, now_utc):
        return False, f"not due today by {refresh_tier(ticker)} rotation", "cache_only_tier_rotation"

    return True, f"due today by {refresh_tier(ticker)} rotation and cache is stale", "fetch"


def main() -> None:
    price_map: dict[str, pd.DataFrame] = {}
    warnings: dict[str, str] = {}
    errors: dict[str, str] = {}
    coverage_gaps: dict[str, str] = {}
    update_log: dict[str, dict] = {}

    requested = 0
    skipped_fresh_enough = 0
    skipped_by_rotation = 0
    new_full_downloads = 0
    rows_appended_total = 0
    tiingo_circuit_open = False
    tiingo_circuit_reason = ""
    now_utc = datetime.now(timezone.utc)
    expected_date = expected_latest_market_date(now_utc)

    ordered_tickers = sorted(br.TICKERS, key=request_order_key)

    for ticker in ordered_tickers:
        existing = br.load_existing(ticker)
        has_cache = not existing.empty
        is_new_full_download = not has_cache

        should_fetch, skip_reason, skip_status = should_fetch_today(ticker, existing, expected_date, now_utc)
        if not should_fetch:
            if skip_status == "cache_fresh_enough_no_request":
                skipped_fresh_enough += 1
            else:
                skipped_by_rotation += 1
            use_cache_or_defer(ticker, existing, price_map, warnings, errors, coverage_gaps, update_log, skip_reason, skip_status)
            continue

        if tiingo_circuit_open:
            status = "fetch_failed_no_cache_rate_limit" if not has_cache else "cache_only_after_tiingo_circuit_breaker"
            use_cache_or_defer(ticker, existing, price_map, warnings, errors, coverage_gaps, update_log, tiingo_circuit_reason, status)
            continue

        if requested >= br.MAX_TIINGO_REQUESTS_PER_RUN:
            reason = f"request cap reached ({br.MAX_TIINGO_REQUESTS_PER_RUN})"
            use_cache_or_defer(ticker, existing, price_map, warnings, errors, coverage_gaps, update_log, reason)
            continue

        if is_new_full_download and new_full_downloads >= br.MAX_NEW_FULL_DOWNLOADS_PER_RUN:
            reason = f"new full download cap reached ({br.MAX_NEW_FULL_DOWNLOADS_PER_RUN})"
            use_cache_or_defer(ticker, existing, price_map, warnings, errors, coverage_gaps, update_log, reason)
            continue

        try:
            fetch_start = get_incremental_fetch_start(existing)
            new_df = br.fetch_tiingo(ticker, fetch_start)
            requested += 1
            if is_new_full_download:
                new_full_downloads += 1

            merged, downloaded_rows, appended_rows = save_append_only(ticker, existing, new_df)
            rows_appended_total += appended_rows
            price_map[ticker] = merged
            update_log[ticker] = {
                "status": "fresh_from_tiingo_append_only" if appended_rows else "fresh_checked_no_new_rows_append_only",
                "refresh_tier": refresh_tier(ticker),
                "coverage_gaps_first": COVERAGE_GAPS_FIRST,
                "fetch_start": fetch_start,
                "downloaded_rows": downloaded_rows,
                "new_rows_appended": appended_rows,
                "historical_rows_overwritten": 0,
                "append_only_mode": APPEND_ONLY_MODE,
                "total_rows_saved": int(len(merged)),
                "latest_date": pd.to_datetime(merged["date"]).max().date().isoformat(),
                "expected_latest_market_date": expected_date.isoformat(),
                "tiingo_request_spent": True,
            }
            print(
                f"[OK_APPEND_ONLY] {ticker}: fetched {downloaded_rows} rows from {fetch_start}; "
                f"appended {appended_rows}; saved {len(merged)} total"
            )

        except Exception as e:
            requested += 1
            rate_limited = is_tiingo_rate_limit_error(e)
            if rate_limited:
                tiingo_circuit_open = True
                tiingo_circuit_reason = f"Tiingo 429 circuit breaker opened after {ticker}: {e}"
                print(f"[CIRCUIT_OPEN] {tiingo_circuit_reason}")

            if has_cache:
                price_map[ticker] = existing
                update_log[ticker] = {
                    "status": "cache_after_fetch_error_append_only",
                    "refresh_tier": refresh_tier(ticker),
                    "coverage_gaps_first": COVERAGE_GAPS_FIRST,
                    "fetch_error": str(e),
                    "append_only_mode": APPEND_ONLY_MODE,
                    "historical_rows_overwritten": 0,
                    "latest_date": pd.to_datetime(existing["date"]).max().date().isoformat(),
                    "expected_latest_market_date": expected_date.isoformat(),
                    "total_rows_loaded": int(len(existing)),
                    "tiingo_request_spent": True,
                }
                warnings[ticker] = f"using cached data after fetch error: {e}"
                print(f"[CACHE_AFTER_FAIL] {ticker}: {e}")
            else:
                status = "fetch_failed_no_cache_rate_limit" if rate_limited else "fetch_failed_no_cache"
                update_log[ticker] = {
                    "status": status,
                    "refresh_tier": refresh_tier(ticker),
                    "coverage_gaps_first": COVERAGE_GAPS_FIRST,
                    "fetch_error": str(e),
                    "expected_latest_market_date": expected_date.isoformat(),
                    "tiingo_request_spent": True,
                }
                if rate_limited:
                    coverage_gaps[ticker] = str(e)
                else:
                    errors[ticker] = str(e)
                print(f"[FAIL] {ticker}: {e}")

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
            errors[ticker] = f"analysis_error: {e}"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Tiingo Free API with local CSV cache fallback, tiered refresh, 429 circuit breaker, append-only cache updates",
        "update_mode": (
            "large universe with coverage-gaps-first hydration: no-cache tickers are downloaded before tier rotation; "
            "cached data is always loaded; stale cached data still follows tier rotation; "
            "existing historical CSV rows are not overwritten, only new dates are appended"
        ),
        "strategy_version": "Eason Master US Market Monitor Cloud Sync v4.5.3 coverage-gaps-first",
        "privacy_mode": "sanitized_public_report_no_cash_no_shares_no_account_value",
        "universe": {
            "configured_ticker_count": len(br.TICKERS),
            "loaded_ticker_count": len(price_map),
            "fresh_request_count": requested,
            "skipped_fresh_enough_no_request": skipped_fresh_enough,
            "skipped_by_tier_rotation": skipped_by_rotation,
            "rows_appended_total": rows_appended_total,
            "append_only_mode": APPEND_ONLY_MODE,
            "coverage_gaps_first": COVERAGE_GAPS_FIRST,
            "historical_rows_overwritten": 0,
            "expected_latest_market_date": expected_date.isoformat(),
            "refresh_policy": {
                "core_daily": sorted(CORE_DAILY_REFRESH),
                "watch_every_3d": sorted(WATCH_EVERY_3D_REFRESH),
                "long_tail_weekly": "cached stale tickers still use deterministic weekday rotation; no-cache tickers bypass rotation",
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
    with open(br.OUT / "market_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    pd.DataFrame(all_rows).to_csv(br.OUT / "backtest_summary.csv", index=False)

    ranking_rows = []
    for ticker, rows in report["rule_evidence_ranking"].items():
        for row in rows:
            ranking_rows.append({"ticker": ticker, **row})
    pd.DataFrame(ranking_rows).to_csv(br.OUT / "rule_evidence_ranking.csv", index=False)

    with open(br.OUT / "index.html", "w", encoding="utf-8") as f:
        f.write(
            "<h1>Eason Quant Cloud Sync</h1>"
            "<p>Sanitized public report. Open action_board.json, market_report.json, "
            "latest_market_summary.json, latest_decision_summary.json, eason_signal.json, "
            "backtest_summary.csv, or rule_evidence_ranking.csv.</p>"
        )

    print("Saved coverage-gaps-first docs/market_report.json, docs/backtest_summary.csv, docs/rule_evidence_ranking.csv")


if __name__ == "__main__":
    main()
