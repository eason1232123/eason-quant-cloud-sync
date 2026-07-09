# Eason Quant focused decision universe, cache-safe mode.
# Goal: improve GitHub coverage quality for ChatGPT trade decisions under free Tiingo limits.
# Broad market ideas can still be checked by ChatGPT with live public data, but this
# quant/backtest universe stays focused on liquid, decision-relevant tickers.
# Workflow trigger: 2026-07-09T10:00:00-04:00

TICKERS = [
    # Market / regime anchors
    "SPY", "QQQ", "SMH", "SOXX", "VGT", "XLK", "SGOV", "GLD", "TLT", "IEF",

    # Magnificent 7 / AI concentration
    "MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA",

    # Semiconductor / AI hardware
    "AMD", "ASML", "TSM", "ARM", "MU", "LRCX", "AMAT", "KLAC", "QCOM",

    # AI software / cloud / cybersecurity
    "ORCL", "PLTR", "CRM", "NET", "PANW", "CRWD", "NOW",

    # AI infrastructure / power / data-center
    "VRT", "ETN", "PWR", "CEG", "ANET",

    # Defensive / quality / opportunity-cost comparisons
    "XLV", "XLP", "XLU", "COST", "LLY", "JPM",
]

START_DATE = "2005-01-01"
