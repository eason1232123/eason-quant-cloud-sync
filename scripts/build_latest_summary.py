"""
Build a small ChatGPT-friendly summary from docs/market_report.json.

This script does not fetch data and does not change the main quant logic.
It only reads the full report and writes:
  - docs/latest_summary.json
  - docs/latest_summary.txt
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DOCS_DIR = Path("docs")
FULL_REPORT_PATH = DOCS_DIR / "market_report.json"
SUMMARY_JSON_PATH = DOCS_DIR / "latest_summary.json"
SUMMARY_TXT_PATH = DOCS_DIR / "latest_summary.txt"

MAX_TICKERS_IN_SUMMARY = 120
MAX_EVIDENCE_ROWS = 50
MAX_ACTIVE_SIGNALS = 80

TECHNICAL_KEEP_KEYS = {
    "ticker",
    "date",
    "latest_date",
    "price",
    "close",
    "adjClose",
    "last_price",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "rsi14",
    "ma5",
    "ma10",
    "ma20",
    "ma50",
    "ma100",
    "ma200",
    "ma20_slope_10d",
    "ma50_slope_20d",
    "above_ma5",
    "above_ma20",
    "above_ma50",
    "above_ma200",
    "drawdown",
    "drawdown_from_high",
    "drawdown_from_52w_high",
    "volatility_20d",
    "volatility_60d",
    "relative_strength_20d",
    "relative_strength_60d",
    "benchmark",
    "active_rules",
    "active_signals",
    "signals",
    "triggers",
}

EVIDENCE_KEEP_KEYS = {
    "ticker",
    "rule",
    "horizon_days",
    "horizon",
    "samples",
    "valid",
    "min_sample_required",
    "win_rate",
    "avg_return",
    "median_return",
    "worst_return",
    "best_return",
    "avg_mae",
    "worst_mae",
    "avg_alpha_vs_QQQ",
    "avg_alpha_vs_SPY",
    "avg_alpha_vs_SMH",
    "lookback_start",
    "lookback_end",
}


def clean_value(value: Any) -> Any:
    """Convert NaN/inf to None so JSON is standards-compliant."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    return value


def truthy_signal(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "active", "triggered", "buy", "sell", "risk"}
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    return False


def extract_tickers(report: dict[str, Any]) -> list[str]:
    tickers: set[str] = set()
    for key in ("technicals", "backtests"):
        obj = report.get(key)
        if isinstance(obj, dict):
            tickers.update(str(k).upper() for k in obj.keys())
    ranking = report.get("rule_evidence_ranking")
    if isinstance(ranking, list):
        for row in ranking:
            if isinstance(row, dict) and row.get("ticker"):
                tickers.add(str(row["ticker"]).upper())
    return sorted(tickers)[:MAX_TICKERS_IN_SUMMARY]


def compact_technicals(report: dict[str, Any]) -> dict[str, Any]:
    technicals = report.get("technicals", {})
    if not isinstance(technicals, dict):
        return {}

    compact: dict[str, Any] = {}
    for ticker, payload in technicals.items():
        if not isinstance(payload, dict):
            compact[str(ticker).upper()] = clean_value(payload)
            continue

        row: dict[str, Any] = {}
        for k, v in payload.items():
            key = str(k)
            lower_key = key.lower()
            if key in TECHNICAL_KEEP_KEYS or lower_key in TECHNICAL_KEEP_KEYS:
                row[key] = v
            elif any(token in lower_key for token in ("signal", "trigger", "rule", "rsi", "ma", "drawdown", "return", "vol")):
                # Keep small scalar signal/indicator-like fields without copying big nested histories.
                if not isinstance(v, (list, dict)):
                    row[key] = v
        compact[str(ticker).upper()] = clean_value(row)
    return compact


def flatten_rule_evidence_from_backtests(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    backtests = report.get("backtests", {})
    if not isinstance(backtests, dict):
        return rows

    for ticker, rules in backtests.items():
        if not isinstance(rules, dict):
            continue
        for rule, horizons in rules.items():
            if not isinstance(horizons, dict):
                continue
            for horizon_key, stats in horizons.items():
                if not isinstance(stats, dict):
                    continue
                row = {k: stats.get(k) for k in EVIDENCE_KEEP_KEYS if k in stats}
                row.setdefault("ticker", ticker)
                row.setdefault("rule", rule)
                if "horizon_days" not in row:
                    try:
                        row["horizon_days"] = int(str(horizon_key).replace("d", ""))
                    except Exception:
                        row["horizon"] = horizon_key
                rows.append(clean_value(row))
    return rows


def compact_rule_evidence(report: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = report.get("rule_evidence_ranking")
    rows: list[dict[str, Any]] = []

    if isinstance(ranking, list):
        for item in ranking:
            if isinstance(item, dict):
                rows.append({k: item.get(k) for k in EVIDENCE_KEEP_KEYS if k in item})
            else:
                rows.append({"value": item})
    elif isinstance(ranking, dict):
        # Some future report versions may store ranking in a dict.
        inner = ranking.get("rows") or ranking.get("items") or ranking.get("data")
        if isinstance(inner, list):
            for item in inner:
                if isinstance(item, dict):
                    rows.append({k: item.get(k) for k in EVIDENCE_KEEP_KEYS if k in item})

    if not rows:
        rows = flatten_rule_evidence_from_backtests(report)

    def score(row: dict[str, Any]) -> tuple[int, float, float, int]:
        valid = 1 if row.get("valid") is True else 0
        horizon = row.get("horizon_days")
        horizon_bonus = 1 if horizon in (20, 60) else 0
        avg = row.get("avg_return")
        alpha = row.get("avg_alpha_vs_QQQ")
        return (
            valid,
            horizon_bonus,
            float(alpha) if isinstance(alpha, (int, float)) else -999.0,
            int(row.get("samples") or 0),
        )

    rows = [clean_value(r) for r in rows]
    rows.sort(key=score, reverse=True)
    return rows[:MAX_EVIDENCE_ROWS]


def extract_active_signals(report: dict[str, Any], compact_tech: dict[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    # Preserve existing top-level active signal structures if present.
    for key in ("active_signals", "latest_signals", "current_triggers", "triggers"):
        obj = report.get(key)
        if isinstance(obj, list):
            for item in obj:
                signals.append(clean_value(item if isinstance(item, dict) else {"signal": item}))
        elif isinstance(obj, dict):
            for ticker, item in obj.items():
                if item:
                    signals.append(clean_value({"ticker": ticker, "signals": item}))

    # Infer ticker-level signals from technical fields.
    for ticker, payload in compact_tech.items():
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            lower_key = str(key).lower()
            if any(token in lower_key for token in ("signal", "trigger", "active_rule", "active_rules")) and truthy_signal(value):
                signals.append(clean_value({"ticker": ticker, "field": key, "value": value}))

    # De-duplicate by JSON representation.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in signals:
        sig = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if sig not in seen:
            seen.add(sig)
            deduped.append(item)
    return deduped[:MAX_ACTIVE_SIGNALS]


def build_text_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Eason Quant Latest Summary")
    lines.append("=" * 32)
    lines.append(f"generated_at_utc: {summary.get('generated_at_utc')}")
    lines.append(f"strategy_version: {summary.get('strategy_version')}")
    lines.append(f"privacy_mode: {summary.get('privacy_mode')}")
    lines.append(f"ticker_count: {summary.get('ticker_count')}")
    lines.append(f"errors: {json.dumps(summary.get('errors', {}), ensure_ascii=False)}")
    lines.append("")

    tickers = summary.get("tickers", [])
    lines.append("Tickers:")
    lines.append(", ".join(tickers) if tickers else "None")
    lines.append("")

    active = summary.get("active_signals", [])
    lines.append(f"Active signals ({len(active)}):")
    if active:
        for item in active[:20]:
            lines.append("- " + json.dumps(item, ensure_ascii=False, sort_keys=True))
    else:
        lines.append("None")
    lines.append("")

    evidence = summary.get("top_rule_evidence", [])
    lines.append(f"Top rule evidence ({len(evidence)} shown):")
    for row in evidence[:20]:
        ticker = row.get("ticker", "")
        rule = row.get("rule", "")
        horizon = row.get("horizon_days", row.get("horizon", ""))
        valid = row.get("valid", "")
        samples = row.get("samples", "")
        win_rate = row.get("win_rate", "")
        avg_return = row.get("avg_return", "")
        alpha = row.get("avg_alpha_vs_QQQ", "")
        lines.append(
            f"- {ticker} | {rule} | {horizon}d | valid={valid} | samples={samples} | "
            f"win_rate={win_rate} | avg_return={avg_return} | alpha_vs_QQQ={alpha}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    if not FULL_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing {FULL_REPORT_PATH}. Run scripts/build_report.py first.")

    with FULL_REPORT_PATH.open("r", encoding="utf-8") as f:
        report = json.load(f)

    compact_tech = compact_technicals(report)
    summary: dict[str, Any] = {
        "generated_at_utc": report.get("generated_at_utc"),
        "source_file": "market_report.json",
        "summary_file_version": "v3.1-chatgpt-light-summary",
        "data_source": report.get("data_source"),
        "update_mode": report.get("update_mode"),
        "strategy_version": report.get("strategy_version"),
        "privacy_mode": report.get("privacy_mode"),
        "rule_engine": report.get("rule_engine"),
        "errors": report.get("errors", {}),
        "tickers": extract_tickers(report),
        "ticker_count": len(extract_tickers(report)),
        "latest_technicals": compact_tech,
        "active_signals": extract_active_signals(report, compact_tech),
        "top_rule_evidence": compact_rule_evidence(report),
        "note": "Small public summary for ChatGPT reading. Full evidence remains in market_report.json and CSV files.",
    }
    summary = clean_value(summary)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    with SUMMARY_TXT_PATH.open("w", encoding="utf-8") as f:
        f.write(build_text_summary(summary))

    print(f"Wrote {SUMMARY_JSON_PATH}")
    print(f"Wrote {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()
