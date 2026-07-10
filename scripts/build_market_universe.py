from __future__ import annotations

import csv, io, json, re, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCLUDED_NAME_TERMS = (" warrant", " warrants", " right", " rights", " unit", " units", " preferred", " preference", " depositary shares", " notes due", " bond", " debenture")

def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "eason-quant/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")

def clean_symbol(value: str) -> str:
    return value.strip().upper().replace("$", "-")

def is_eligible(symbol: str, name: str, test_issue: str, etf: str) -> bool:
    if not symbol or symbol.startswith("File Creation Time") or test_issue.strip().upper() == "Y" or etf.strip().upper() == "Y":
        return False
    if any(term in f" {name.strip().lower()}" for term in EXCLUDED_NAME_TERMS):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol))

def parse_pipe_file(text: str, group: str) -> list[dict[str, str]]:
    rows = []
    for raw in csv.DictReader(io.StringIO(text), delimiter="|"):
        if group == "NASDAQ":
            symbol = clean_symbol(raw.get("Symbol", "")); name = raw.get("Security Name", "").strip(); exchange = "NASDAQ"
        else:
            symbol = clean_symbol(raw.get("ACT Symbol", "")); name = raw.get("Security Name", "").strip(); code = raw.get("Exchange", "").strip().upper(); exchange = {"A":"NYSE American","N":"NYSE","P":"NYSE Arca","Z":"Cboe"}.get(code, code or "OTHER")
        if is_eligible(symbol, name, raw.get("Test Issue", ""), raw.get("ETF", "")):
            rows.append({"ticker":symbol,"security_name":name,"exchange":exchange,"source":group,"eligibility":"eligible"})
    return rows

def main() -> None:
    errors, sources, all_rows = [], {}, []
    for label, url in (("NASDAQ", NASDAQ_URL), ("OTHER", OTHER_URL)):
        try:
            all_rows.extend(parse_pipe_file(fetch_text(url), label)); sources[label] = url
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    rows = sorted({r["ticker"]: r for r in all_rows}.values(), key=lambda r: r["ticker"])
    csv_path, json_path = DOCS / "market_universe.csv", DOCS / "market_universe_summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ticker","security_name","exchange","source","eligibility"]); w.writeheader(); w.writerows(rows)
    elif not csv_path.exists():
        raise RuntimeError("Unable to build market universe and no prior cache exists")
    counts = {}
    for row in rows: counts[row["exchange"]] = counts.get(row["exchange"], 0) + 1
    payload = {"generated_at_utc":datetime.now(timezone.utc).isoformat(),"version":"market-universe-v1","eligible_ticker_count":len(rows),"exchange_counts":dict(sorted(counts.items())),"sources":sources,"errors":errors}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))

if __name__ == "__main__": main()
