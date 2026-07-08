import os
import time
import json
from pathlib import Path
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd

from config import TICKERS, START_DATE, PORTFOLIO

API_KEY = os.getenv("TIINGO_API_KEY")
if not API_KEY:
    raise SystemExit("Missing TIINGO_API_KEY GitHub Secret.")

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

def fetch_tiingo(ticker: str) -> pd.DataFrame:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": START_DATE,
        "resampleFreq": "daily",
        "token": API_KEY
    }
    r = requests.get(url, params=params, timeout=40)
    if r.status_code != 200:
        raise RuntimeError(f"{ticker} HTTP {r.status_code}: {r.text[:200]}")
    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError(f"{ticker}: empty data")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker
    price_col = "adjClose" if "adjClose" in df.columns else "close"
    df["price"] = df[price_col]
    return df.sort_values("date").reset_index(drop=True)

def backtest_pullback_stabilization(df, drop_n=0.04, drop_window=5, ma_n=5, horizon=20):
    df = df.copy()
    df["drop_ret"] = df["price"] / df["price"].shift(drop_window) - 1
    df["ma"] = df["price"].rolling(ma_n).mean()
    df["signal"] = (df["drop_ret"] <= -drop_n) & (df["price"] > df["ma"])
    df["fwd"] = df["price"].shift(-horizon) / df["price"] - 1

    low_col = "adjLow" if "adjLow" in df.columns else "low"
    maes = []
    for i in range(len(df)):
        if i + horizon >= len(df):
            maes.append(np.nan)
        else:
            future_min = df.loc[i+1:i+horizon, low_col].min()
            maes.append(future_min / df.loc[i, "price"] - 1)
    df["mae"] = maes

    sig = df[df["signal"]].dropna(subset=["fwd", "mae"])
    if len(sig) == 0:
        return {
            "samples": 0,
            "valid": False,
            "reason": "no signal samples"
        }

    bh = df["price"].iloc[-1] / df["price"].iloc[0] - 1
    return {
        "rule": f"5-day return <= -{drop_n:.0%} and close > {ma_n}DMA; forward horizon {horizon} trading days",
        "lookback_start": df["date"].min().date().isoformat(),
        "lookback_end": df["date"].max().date().isoformat(),
        "samples": int(len(sig)),
        "valid": bool(len(sig) >= 20),
        "win_rate": round(float((sig["fwd"] > 0).mean()), 4),
        "avg_20d_return": round(float(sig["fwd"].mean()), 4),
        "median_20d_return": round(float(sig["fwd"].median()), 4),
        "worst_20d_return": round(float(sig["fwd"].min()), 4),
        "best_20d_return": round(float(sig["fwd"].max()), 4),
        "avg_mae": round(float(sig["mae"].mean()), 4),
        "worst_mae": round(float(sig["mae"].min()), 4),
        "buy_hold_total_return": round(float(bh), 4),
        "latest_price": round(float(df["price"].iloc[-1]), 4),
        "latest_date": df["date"].max().date().isoformat()
    }

def latest_technical(df):
    df = df.copy()
    df["ma5"] = df["price"].rolling(5).mean()
    df["ma20"] = df["price"].rolling(20).mean()
    df["ma50"] = df["price"].rolling(50).mean()
    df["ret_1d"] = df["price"].pct_change(1)
    df["ret_5d"] = df["price"].pct_change(5)
    row = df.iloc[-1]
    return {
        "latest_date": row["date"].date().isoformat(),
        "latest_price": round(float(row["price"]), 4),
        "ret_1d": round(float(row["ret_1d"]), 4) if pd.notna(row["ret_1d"]) else None,
        "ret_5d": round(float(row["ret_5d"]), 4) if pd.notna(row["ret_5d"]) else None,
        "above_ma5": bool(row["price"] > row["ma5"]) if pd.notna(row["ma5"]) else None,
        "above_ma20": bool(row["price"] > row["ma20"]) if pd.notna(row["ma20"]) else None,
        "above_ma50": bool(row["price"] > row["ma50"]) if pd.notna(row["ma50"]) else None,
    }

def portfolio_health(latest_prices):
    cash = PORTFOLIO["cash"]
    rows = []
    pos_value = 0.0
    for ticker, shares in PORTFOLIO["positions"].items():
        price = latest_prices.get(ticker)
        if price is None:
            continue
        value = shares * price
        pos_value += value
        rows.append({"ticker": ticker, "shares": shares, "price": round(price, 2), "value": round(value, 2)})
    total = cash + pos_value
    for r in rows:
        r["weight"] = round(r["value"] / total, 4)
    weights = {r["ticker"]: r["weight"] for r in rows}
    return {
        "account_value_est": round(total, 2),
        "cash": cash,
        "cash_weight": round(cash / total, 4),
        "positions": rows,
        "qqq_smh_msft_weight": round(sum(weights.get(x, 0) for x in ["QQQ", "SMH", "MSFT"]), 4),
        "semiconductor_weight": round(sum(weights.get(x, 0) for x in ["SMH", "SOXX", "SOXQ", "NVDA", "AVGO", "AMD", "ASML", "TSM", "MU"]), 4),
        "msft_weight": weights.get("MSFT", 0),
        "guardrails": {
            "cash_min_15_ok": cash / total >= 0.15,
            "defense_cash_floor_3000_ok": cash >= 3000,
            "semis_max_30_ok": sum(weights.get(x, 0) for x in ["SMH", "SOXX", "SOXQ", "NVDA", "AVGO", "AMD", "ASML", "TSM", "MU"]) <= 0.30,
            "msft_max_25_ok": weights.get("MSFT", 0) <= 0.25,
            "tech_core_under_80_ok": sum(weights.get(x, 0) for x in ["QQQ", "SMH", "MSFT"]) <= 0.80
        }
    }

def corr_90d(price_map):
    core = ["QQQ", "SMH", "SPY", "MSFT"]
    merged = None
    for t in core:
        df = price_map[t][["date", "price"]].rename(columns={"price": t})
        merged = df if merged is None else merged.merge(df, on="date", how="inner")
    rets = merged.set_index("date").pct_change().dropna().tail(90)
    corr = rets.corr()
    return corr.round(4).to_dict()

def main():
    price_map = {}
    errors = {}
    for ticker in TICKERS:
        try:
            df = fetch_tiingo(ticker)
            price_map[ticker] = df
            df.to_csv(OUT / f"{ticker}_daily.csv", index=False)
            print(f"[OK] {ticker}: {len(df)}")
        except Exception as e:
            errors[ticker] = str(e)
            print(f"[FAIL] {ticker}: {e}")
        time.sleep(0.8)

    backtests = {}
    technicals = {}
    latest_prices = {}
    for ticker, df in price_map.items():
        backtests[ticker] = backtest_pullback_stabilization(df)
        technicals[ticker] = latest_technical(df)
        latest_prices[ticker] = float(df["price"].iloc[-1])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Tiingo Free API, adjusted daily prices when available",
        "strategy_version": "Eason Master US Market Monitor 9.6/10 Cloud Sync v1",
        "portfolio_health": portfolio_health(latest_prices),
        "backtests": backtests,
        "technicals": technicals,
        "rolling_90d_correlation": corr_90d(price_map) if all(x in price_map for x in ["QQQ", "SMH", "SPY", "MSFT"]) else None,
        "errors": errors
    }

    with open(OUT / "market_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame([
        {"ticker": t, **v} for t, v in backtests.items()
    ]).to_csv(OUT / "backtest_summary.csv", index=False)

    with open(OUT / "index.html", "w", encoding="utf-8") as f:
        f.write("<h1>Eason Quant Cloud Sync</h1><p>Open market_report.json or backtest_summary.csv</p>")

    print("Saved docs/market_report.json and docs/backtest_summary.csv")

if __name__ == "__main__":
    main()
