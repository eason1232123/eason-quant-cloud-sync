from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DOCS_DIR = Path("docs")
REPORT_PATH = DOCS_DIR / "market_report.json"
SIGNAL_JSON_PATH = DOCS_DIR / "eason_signal.json"
SIGNAL_TXT_PATH = DOCS_DIR / "eason_signal.txt"


def safe_float(value: Any):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def ticker_tech(report: dict, ticker: str) -> dict:
    obj = report.get("technicals", {})
    if not isinstance(obj, dict):
        return {}
    return obj.get(ticker, {}) if isinstance(obj.get(ticker, {}), dict) else {}


def above(tech: dict, key: str) -> bool:
    return tech.get(key) is True


def market_regime(report: dict) -> dict:
    spy = ticker_tech(report, "SPY")
    qqq = ticker_tech(report, "QQQ")
    smh = ticker_tech(report, "SMH")

    risk_points = 0
    notes = []

    for name, tech in [("SPY", spy), ("QQQ", qqq), ("SMH", smh)]:
        if not tech:
            risk_points += 1
            notes.append(f"{name} missing")
            continue
        if not above(tech, "above_ma50"):
            risk_points += 1
            notes.append(f"{name} below MA50")
        if not above(tech, "above_ma200"):
            risk_points += 1
            notes.append(f"{name} below MA200")

    qqq_rsi = safe_float(qqq.get("rsi14"))
    if qqq_rsi is not None and qqq_rsi > 75:
        risk_points += 1
        notes.append("QQQ RSI overbought")

    if risk_points <= 1:
        regime = "RISK_ON"
        buy_permission = "BUY_ALLOWED_WITH_POSITION_LIMITS"
    elif risk_points <= 3:
        regime = "CAUTION"
        buy_permission = "SMALL_BUY_OR_WAIT"
    else:
        regime = "RISK_OFF"
        buy_permission = "WAIT_OR_REDUCE"

    return {
        "market_regime": regime,
        "risk_points": risk_points,
        "buy_permission": buy_permission,
        "notes": notes,
    }


def top_candidates(report: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rankings = report.get("rule_evidence_ranking", {})
    techs = report.get("technicals", {})
    buys = []
    risks = []
    new_listing = []

    if not isinstance(techs, dict):
        techs = {}

    for ticker, tech in techs.items():
        if not isinstance(tech, dict):
            continue

        active = tech.get("active_signals", {})
        if not isinstance(active, dict):
            active = {}

        trading_days = int(tech.get("trading_days") or 0)
        short_history = bool(tech.get("short_history"))

        momentum_ok = (
            tech.get("above_ma20") is True
            and (tech.get("above_ma50") is True or trading_days < 80)
            and (safe_float(tech.get("ret_20d")) or 0) > 0
            and (safe_float(tech.get("relative_20d")) or 0) > 0
        )

        if short_history and momentum_ok:
            new_listing.append({
                "ticker": ticker,
                "status": "SHORT_HISTORY_MOMENTUM_OK",
                "trading_days": trading_days,
                "policy": "watchlist_or_tiny_satellite_only_until_backtest_sample_is_sufficient",
                "latest_price": tech.get("latest_price"),
                "ret_20d": tech.get("ret_20d"),
                "relative_20d": tech.get("relative_20d"),
            })

        if active.get("failed_rebound_risk") is True:
            risks.append({
                "ticker": ticker,
                "risk": "failed_rebound_risk",
                "latest_price": tech.get("latest_price"),
                "ret_5d": tech.get("ret_5d"),
                "above_ma50": tech.get("above_ma50"),
            })

        ticker_rows = rankings.get(ticker, []) if isinstance(rankings, dict) else []
        best = ticker_rows[0] if ticker_rows else {}
        best_score = safe_float(best.get("evidence_score_0_100")) or 0

        if (
            active.get("momentum_leader")
            or active.get("ma20_reclaim_bullish")
            or active.get("relative_strength_rebound")
            or active.get("pullback_reclaim_5dma")
        ):
            buys.append({
                "ticker": ticker,
                "latest_price": tech.get("latest_price"),
                "active_signals": {k: v for k, v in active.items() if v is True},
                "best_evidence_score_0_100": best_score,
                "best_rule": best.get("rule"),
                "valid_backtest": bool(best.get("valid")),
                "short_history": short_history,
                "trading_days": trading_days,
                "note": (
                    "short-history; cannot rely on backtest, tiny satellite/watch only"
                    if short_history else "normal evidence candidate"
                ),
            })

    buys.sort(key=lambda x: (not x["short_history"], x["valid_backtest"], x["best_evidence_score_0_100"]), reverse=True)
    new_listing.sort(key=lambda x: (x.get("relative_20d") or 0, x.get("ret_20d") or 0), reverse=True)
    return buys[:25], risks[:25], new_listing[:25]


def decide_action(regime: dict, buys: list[dict], risks: list[dict]) -> str:
    if regime["market_regime"] == "RISK_OFF":
        return "WAIT_OR_REDUCE"
    if len(risks) >= 6 and regime["market_regime"] != "RISK_ON":
        return "WAIT"
    strong_buys = [b for b in buys if (b.get("valid_backtest") and (b.get("best_evidence_score_0_100") or 0) >= 55)]
    if regime["market_regime"] == "RISK_ON" and strong_buys:
        return "BUY_ALLOWED"
    if buys:
        return "SMALL_BUY_OR_WAIT"
    return "WAIT"


def main() -> None:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing {REPORT_PATH}. Run scripts/build_report.py first.")

    with REPORT_PATH.open("r", encoding="utf-8") as f:
        report = json.load(f)

    regime = market_regime(report)
    buys, risks, new_listing = top_candidates(report)
    action = decide_action(regime, buys, risks)

    errors = report.get("errors", {})
    universe = report.get("universe", {})
    stale_or_partial = bool(errors)

    signal = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": "market_report.json",
        "signal_version": "v1-large-universe-cache-safe",
        "today_action": action,
        **regime,
        "data_status": "partial_check_errors" if stale_or_partial else "ok",
        "universe": universe,
        "buy_candidates": buys,
        "risk_candidates": risks,
        "new_listing_candidates": new_listing,
        "new_listing_policy": {
            "will_chatgpt_recommend_new_good_stocks_without_full_backtest": True,
            "rule": (
                "Yes, but not as a normal high-conviction backtest pick. If a stock is newly listed or has short history, "
                "ChatGPT may recommend it only as watchlist or tiny satellite after checking price action, relative strength, "
                "liquidity, fundamentals/news, valuation/context, and overall market regime."
            ),
            "default_position_limit": "0% to 2% max before sufficient backtest sample; usually observe first.",
        },
        "errors": errors,
        "note": (
            "This is a public sanitized signal file. Final trading decisions should still check live price, bid/ask, news, "
            "earnings date, valuation, portfolio concentration, and the user's position size limits."
        ),
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with SIGNAL_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    lines = [
        "Eason Quant Final Signal",
        "=" * 30,
        f"generated_at_utc: {signal['generated_at_utc']}",
        f"today_action: {signal['today_action']}",
        f"market_regime: {signal['market_regime']}",
        f"buy_permission: {signal['buy_permission']}",
        f"data_status: {signal['data_status']}",
        "",
        "Top buy candidates:",
    ]
    for row in buys[:10]:
        lines.append("- " + json.dumps(row, ensure_ascii=False, sort_keys=True))
    lines.append("")
    lines.append("New listing / short-history candidates:")
    for row in new_listing[:10]:
        lines.append("- " + json.dumps(row, ensure_ascii=False, sort_keys=True))
    lines.append("")
    lines.append("Risk candidates:")
    for row in risks[:10]:
        lines.append("- " + json.dumps(row, ensure_ascii=False, sort_keys=True))

    with SIGNAL_TXT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {SIGNAL_JSON_PATH}")
    print(f"Wrote {SIGNAL_TXT_PATH}")


if __name__ == "__main__":
    main()
