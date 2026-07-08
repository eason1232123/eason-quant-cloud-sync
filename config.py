# Eason Data Hub 48
# GitHub only acts as the data pipe / evidence warehouse.
# Final trading decisions remain outside GitHub and should be made using the full Eason 9.6/10 framework.

TICKERS = [
    # Core / existing strategy anchors
    "QQQ", "SMH", "SOXX", "SPY", "MSFT", "SGOV",

    # Semiconductor / AI hardware tactical layer
    "NVDA", "AVGO", "AMD", "ASML", "TSM", "MU", "LRCX", "AMAT", "KLAC", "ARM",

    # Software / AI application layer
    "IGV", "CRWD", "PLTR", "SNOW", "DDOG", "NET", "NOW", "PANW", "MDB", "ORCL",

    # AI infrastructure / power / data-center layer
    "VRT", "ETN", "PWR", "CEG", "NRG", "GRID",

    # Defensive / low-correlation layer
    "GLD", "TLT", "IEF", "XLV", "XLP", "XLU", "USMV",

    # High-volatility satellite layer
    "TSLA", "COIN", "ARKK", "XBI", "IBB",

    # Large-cap quality / opportunity-cost comparison layer
    "AAPL", "GOOGL", "AMZN", "META",
]

START_DATE = "2005-01-01"
