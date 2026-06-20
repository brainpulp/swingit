"""The trading universe: 40 S&P 500 large-cap tickers.

The list is intentionally fixed for reproducible backtests. SPY (the regime
filter) is kept separate from the tradable universe.
"""

from __future__ import annotations

UNIVERSE: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC",
    "DIS", "ADBE", "CRM", "NFLX", "XOM", "CVX", "PFE", "KO",
    "PEP", "CSCO", "INTC", "WMT", "MRK", "ABT", "T", "VZ",
    "CMCSA", "NKE", "ORCL", "COST", "MCD", "QCOM", "TXN", "AMD",
]

assert len(UNIVERSE) == 40, "universe should contain exactly 40 tickers"
