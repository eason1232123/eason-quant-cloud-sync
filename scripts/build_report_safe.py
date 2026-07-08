from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from scripts import build_report as br


def is_tiingo_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return "http 429" in text or "hourly request allocation" in text or "run over your hourly" in text


def use_cache_or_defer(ticker: str, existing: pd.DataFrame, price_map: dict, errors: dict, update_log: dict, reason: str) -> None:
    if existing is not None and not existing.empty:
        price_map[ticker] = existing
        update_log[ticker] = {
            "status": "cache_only_after_tiingo_circuit_breaker" if "429" in reason.lower() else "cache_only",
            "reason": reason,
            "latest_date": pd.to_datetime(existing["date"]).max().date().isoformat(),
            "total_rows_loaded": int(len(existing)),
        }
        errors[ticker] = f"using cached data: {reason}"
        print(f"[CACHE] {ticker}: {reason}")
    else:
        errors[ticker] = f"deferred_no_cache: {reason}"
        print(f"[DEFER] {ticker}: {reason}")


def main() -> None:
    price_map: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    update_log: dict[str, dict] = {}

    requested = 0
    new_full_downloads = 0
    tiingo_circuit_open = False
    tiingo_circuit_reason = ""

    ordered_tickers = sorted(br.TICKERS, key=br.request_priority)

    for ticker in ordered_tickers:
        existing = br.load_existing(ticker)
        has_cache = not existing.empty
        is_new_full_download = not has_cache

        if tiingo_circuit_open:
            use_cache_or_defer(ticker, existing, price_map, errors, update_log, tiingo_circuit_reason)
            continue

        if requested >= br.MAX_TIINGO_REQUESTS_PER_RUN:
            reason = f"request cap reached ({br.MAX_TIINGO_REQUESTS_PER_RUN})"
            use_cache_or_defer(ticker, existing, price_map, errors, update_log, reason)
            continue

        if is_new_full_download and new_full_downloads >= br.MAX_NEW_FULL_DOWNLOADS_PER_RUN:
            reason = f"new full download cap reached ({br.MAX_NEW_FULL_DOWNLOADS_PER_RUN})"
            use_cache_or_defer(ticker, existing, price_map, errors, update_log, reason)
            continue

        try:
            fetch_start = br.get_fetch_start_date(ticker)
            new_df = br.fetch_tiingo(ticker, fetch_start)
            requested += 1
            if is_new_full_download:
                new_full_downloads += 1

            merged = br.merge_and_save(ticker, new_df)
            price_map[ticker] = merged
            update_log[ticker] = {
                "status": "fresh_from_tiingo",
                "fetch_start": fetch_start,
                "new_rows_downloaded": int(len(new_df)),
                "total_rows_saved": int(len(merged)),
                "latest_date": pd.to_datetime(merged["date"]).max().date().isoformat(),
            }
            print(f"[OK] {ticker}: fetched {len(new_df)} rows from {fetch_start}; saved {len(merged)} total")

        except Exception as e:
            requested += 1
            if is_tiingo_rate_limit_error(e):
                tiingo_circuit_open = True
                tiingo_circuit_reason = f"Tiingo 429 circuit breaker opened after {ticker}: {e}"
                print(f"[CIRCUIT_OPEN] {tiingo_circuit_reason}")

            if has_cache:
                price_map[ticker] = existing
                update_log[ticker] = {
                    "status": "cache_after_fetch_error",
                    "fetch_error": str(e),
                    "latest_date": pd.to_datetime(existing["date"]).max().date().isoformat(),
                    "total_rows_loaded": int(len(existing)),
                }
                errors[ticker] = f"using cached data after fetch error: {e}"
                print(f"[CACHE_AFTER_FAIL] {ticker}: {e}")
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
        "data_source": "Tiingo Free API with local CSV cache fallback and 429 circuit breaker",
        "update_mode": (
            "large universe, capped Tiingo requests per run, capped new full-history downloads, "
            "cached data used when request cap or Tiingo 429 occurs; first 429 opens circuit breaker"
        ),
        "strategy_version": "Eason Master US Market Monitor Cloud Sync v4.1 large-universe-cache-safe-circuit-breaker",
        "privacy_mode": "sanitized_public_report_no_cash_no_shares_no_account_value",
        "universe": {
            "configured_ticker_count": len(br.TICKERS),
            "loaded_ticker_count": len(price_map),
            "fresh_request_count": requested,
            "max_tiingo_requests_per_run": br.MAX_TIINGO_REQUESTS_PER_RUN,
            "max_new_full_downloads_per_run": br.MAX_NEW_FULL_DOWNLOADS_PER_RUN,
            "tiingo_circuit_open": tiingo_circuit_open,
            "tiingo_circuit_reason": tiingo_circuit_reason,
            "note": "A ticker can remain in the universe even if not refreshed today. ChatGPT should check update_log status and latest_date.",
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
            "<p>Sanitized public report. Open market_report.json, latest_summary.json, eason_signal.json, "
            "backtest_summary.csv, or rule_evidence_ranking.csv.</p>"
        )

    print("Saved sanitized docs/market_report.json, docs/backtest_summary.csv, docs/rule_evidence_ranking.csv")


if __name__ == "__main__":
    main()
