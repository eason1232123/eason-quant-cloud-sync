from __future__ import annotations

import csv, json, math
from datetime import datetime, timezone
from pathlib import Path
from config import TICKERS

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

def f(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def pct(values, n):
    return None if len(values) <= n or values[-n-1] == 0 else (values[-1] / values[-n-1] - 1) * 100

def vol(values, n=60):
    sample = values[-(n+1):]
    returns = [sample[i] / sample[i-1] - 1 for i in range(1, len(sample)) if sample[i-1] != 0]
    if len(returns) < 2: return None
    mean = sum(returns) / len(returns)
    return math.sqrt(sum((x-mean)**2 for x in returns)/(len(returns)-1)) * math.sqrt(252) * 100

def read_prices(path: Path):
    with path.open("r", newline="", encoding="utf-8") as h: rows = list(csv.DictReader(h))
    closes, volumes, latest = [], [], ""
    for row in rows:
        close = next((f(row.get(k)) for k in ("adjClose","adj_close","close","Close") if f(row.get(k)) is not None), None)
        if close is None or close <= 0: continue
        closes.append(close); latest = str(row.get("date") or row.get("Date") or latest)
        volume = next((f(row.get(k)) for k in ("adjVolume","adj_volume","volume","Volume") if f(row.get(k)) is not None), None)
        if volume is not None and volume >= 0: volumes.append(volume)
    if len(closes) < 21: return None
    adv = None
    if len(volumes) >= 20:
        pairs = list(zip(closes[-20:], volumes[-20:])); adv = sum(p*v for p,v in pairs)/len(pairs)
    high = max(closes[-252:])
    return {"latest_date":latest,"latest_price":closes[-1],"history_rows":len(closes),"ret_20d_pct":pct(closes,20),"ret_60d_pct":pct(closes,60),"ret_120d_pct":pct(closes,120),"drawdown_52w_pct":(closes[-1]/high-1)*100,"volatility_60d_pct":vol(closes),"avg_dollar_volume_20":adv}

def ranks(values, reverse=False):
    ordered = sorted(values.items(), key=lambda x:x[1], reverse=reverse)
    if len(ordered) <= 1: return {k:50.0 for k,_ in ordered}
    return {k:i/(len(ordered)-1)*100 for i,(k,_) in enumerate(ordered)}

def main():
    records = {}
    for path in sorted(DOCS.glob("*_daily.csv")):
        ticker = path.name.removesuffix("_daily.csv").upper(); metrics = read_prices(path)
        if metrics: records[ticker] = {"ticker":ticker, **metrics}
    fields = {
        "r20": {t:r["ret_20d_pct"] for t,r in records.items() if r["ret_20d_pct"] is not None},
        "r60": {t:r["ret_60d_pct"] for t,r in records.items() if r["ret_60d_pct"] is not None},
        "r120": {t:r["ret_120d_pct"] for t,r in records.items() if r["ret_120d_pct"] is not None},
        "dd": {t:r["drawdown_52w_pct"] for t,r in records.items() if r["drawdown_52w_pct"] is not None},
        "liq": {t:math.log10(max(r["avg_dollar_volume_20"],1)) for t,r in records.items() if r["avg_dollar_volume_20"] is not None},
        "v": {t:r["volatility_60d_pct"] for t,r in records.items() if r["volatility_60d_pct"] is not None},
    }
    rr = {"r20":ranks(fields["r20"],True),"r60":ranks(fields["r60"],True),"r120":ranks(fields["r120"],True),"dd":ranks(fields["dd"],True),"liq":ranks(fields["liq"],True),"v":ranks(fields["v"],False)}
    core = {x.upper() for x in TICKERS}; ranked = []
    for t,r in records.items():
        if any(t not in rr[k] for k in rr): continue
        score = .15*rr["r20"][t]+.25*rr["r60"][t]+.25*rr["r120"][t]+.15*rr["dd"][t]+.15*rr["liq"][t]+.05*rr["v"][t]
        ranked.append({**r,"discovery_score_0_100":round(score,2),"is_core_94":t in core})
    ranked.sort(key=lambda x:x["discovery_score_0_100"], reverse=True)
    for i,r in enumerate(ranked,1): r["rank"] = i
    candidates = ranked[:300]; promoted = [r for r in ranked if not r["is_core_94"]][:max(0,150-len(core))]; deep = sorted(core|{r["ticker"] for r in promoted})
    payload = {"generated_at_utc":datetime.now(timezone.utc).isoformat(),"version":"dynamic-candidates-v1","coverage":{"cached_tickers_scored":len(ranked),"configured_core_count":len(core),"dynamic_candidate_count":len(candidates),"deep_pool_count":len(deep)},"deep_pool":deep,"top_candidates":candidates[:50]}
    (DOCS/"dynamic_candidates.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    with (DOCS/"dynamic_candidates.csv").open("w",newline="",encoding="utf-8") as h:
        names=["rank","ticker","discovery_score_0_100","is_core_94","latest_date","latest_price","history_rows","ret_20d_pct","ret_60d_pct","ret_120d_pct","drawdown_52w_pct","volatility_60d_pct","avg_dollar_volume_20"]
        w=csv.DictWriter(h,fieldnames=names,extrasaction="ignore"); w.writeheader(); w.writerows(candidates)
    print(json.dumps(payload["coverage"],ensure_ascii=False))

if __name__ == "__main__": main()
