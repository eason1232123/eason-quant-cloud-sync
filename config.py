# Eason Quant large universe, cache-safe mode.
# Keep many stocks in the universe, but scripts/build_report.py only refreshes a capped number per run
# so the free Tiingo API is less likely to hit HTTP 429.
# Workflow trigger: 2026-07-08T15:00:00-04:00

TICKERS = [
    # Market / regime anchors
    "SPY", "QQQ", "SMH", "SOXX", "VGT", "XLK", "IYW", "IGV", "IGM", "SGOV",

    # Magnificent 7 / AI concentration
    "MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA",

    # Semiconductor / AI hardware
    "AMD", "ASML", "TSM", "ARM", "MU", "LRCX", "AMAT", "KLAC", "INTC",
    "MRVL", "QCOM", "ON", "ADI", "TXN", "NXPI", "MCHP", "MPWR",

    # AI software / cloud / cybersecurity
    "ORCL", "PLTR", "CRM", "ADBE", "SNOW", "DDOG", "NET", "NOW", "PANW",
    "CRWD", "ZS", "OKTA", "MDB", "TEAM", "SHOP", "CFLT", "ESTC", "GTLB", "PATH",

    # AI infrastructure / power / data-center
    "VRT", "ETN", "PWR", "CEG", "NRG", "GRID", "SMR", "GE", "VST", "DLR",
    "EQIX", "ANET", "CSCO",

    # Defensive / low-correlation / macro
    "GLD", "TLT", "IEF", "SHY", "XLV", "XLP", "XLU", "USMV", "VYM",

    # High-volatility satellite / innovation
    "COIN", "MSTR", "ARKK", "SOFI", "RBLX", "UBER", "ABNB", "HOOD", "RDDT",
    "RKLB", "ASTS", "HIMS", "APP",

    # Quality / opportunity-cost comparisons
    "COST", "LLY", "JPM", "XLF", "XLE",
]

START_DATE = "2005-01-01"
