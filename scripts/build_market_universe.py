from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

EXCLUDED_NAME_TERMS = (
    " warrant", " warrants", " right", " rights", " unit", " units",
    " preferred", " preference", " depositary shares", " notes due",
    " bond", " debenture",
)


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "eason-quant/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def clean_symbol(value: str) -> str:
    return value.strip().upper().replace("$", "-")


def is_eligible(symbol: str, name: str, test_issue: str, etf: str) -> tuple[bool, str]:
    if not symbol or symbol.startswith("File Creation Time"):
        return False, "invalid_symbol"
    if test_issue.strip().upper() == "Y":
        return False, "test_issue"
    if etf.strip().upper() == "Y":
        return False, "etf"
    lower_name = f" {name.strip().lower()}"
    if any(term in lower_name for term in EXCLUDED_NAME_TERMS):
        return False, "non_common_security"
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol):
        return False, "unsupported_symbol"
    return True, "eligible"


def parse_pipe_file(text: str, exchange_group: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    for raw in reader:
        if not raw:
            continue
        if exchange_group == "NASDAQ":
            symbol = clean_symbol(raw.get("Symbol", ""))
            name = raw.get("Security Name", "").strip()
            exchange = "NASDAQ"
            test_issue = raw.get("Test Issue", "")
            etf = raw.get("ETF", "")
        else:
            symbol = clean_symbol(raw.get("ACT Symbol", ""))
            name = raw.get("Security Name", "").strip()
            exchange_code = raw.get("Exchange", "").strip().upper()
            exchange = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe"}.get(
                exchange_code, exchange_code or "OTHER"
            )
            test_issue = raw.get("Test Issue", "")
            etf = raw.get("ETF", "")

        eligible, reason = is_eligible(symbol, name, test_issue, etf)
        if not eligible:
            continue
        rows.append(
            {
                "ticker": symbol,
                "security_name": name,
                "exchange": exchange,
                "source": exchange_group,
                "eligibility": reason,
            }
        )
    return rows


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    sources: dict[str, str] = {}
    errors: list[str] = []
    all_rows: list[dict[str, str]] = []

    for label, url in (("NASDAQ", NASDAQ_URL), ("OTHER", OTHER_URL)):
        try:
            text = fetch_text(url)
            sources[label] = url
            all_rows.extend(parse_pipe_file(text, label))
        except Exception as exc:  # keep prior universe available when an upstream file is temporarily down
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    by_ticker: dict[str, dict[str, str]] = {}
    for row in all_rows:
        by_ticker.setdefault(row["ticker"], row)

    rows = sorted(by_ticker.values(), key=lambda item: item["ticker"])
    output_csv = DOCS / "market_universe.csv"
    output_json = DOCS / "market_universe_summary.json"

    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["ticker", "security_name", "exchange", "source", "eligibility"],
            )
            writer.writeheader()
            writer.writerows(rows)
    elif not output_csv.exists():
        raise RuntimeError("Unable to build market universe and no prior universe cache exists")

    exchange_counts: dict[str, int] = {}
    for row in rows:
        exchange_counts[row["exchange"]] = exchange_counts.get(row["exchange"], 0) + 1

    summary = {
        "generated_at_utc": generated_at,
        "version": "market-universe-v1",
        "purpose": "Broad US-listed common-stock discovery universe. ETFs, test issues and obvious non-common securities are excluded.",
        "eligible_ticker_count": len(rows),
        "exchange_counts": dict(sorted(exchange_counts.items())),
        "sources": sources,
        "errors": errors,
        "limitations": [
            "This is a listing universe, not a buy list.",
            "Eligibility does not confirm liquidity, market cap, profitability or data history.",
            "Downstream ranking must apply price, liquidity, quality and risk filters before promotion.",
        ],
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
