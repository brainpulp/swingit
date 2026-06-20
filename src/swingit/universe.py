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

# Extended ~150-name S&P 500 universe, used by the experiments harness to test
# whether more breadth (more signals) materially improves capital deployment.
_EXTRA: list[str] = [
    "GOOG", "AVGO", "ACN", "LLY", "ABBV", "TMO", "DHR", "NEE", "PM", "IBM",
    "GE", "CAT", "HON", "UNP", "LOW", "LIN", "AMGN", "SBUX", "GS", "MS",
    "BLK", "AXP", "SPGI", "BKNG", "GILD", "MDT", "ISRG", "ADP", "AMT", "C",
    "DE", "MMM", "CB", "MO", "SO", "DUK", "BMY", "CI", "ZTS", "MDLZ",
    "USB", "PNC", "TGT", "CL", "ITW", "BDX", "NSC", "EW", "AON", "APD",
    "ICE", "FCX", "EMR", "REGN", "ETN", "FDX", "HUM", "GD", "NOC", "LMT",
    "MMC", "SCHW", "PGR", "ELV", "COP", "SLB", "EOG", "PSX", "MPC", "VLO",
    "KMI", "WMB", "OXY", "HCA", "MET", "AIG", "PRU", "AFL", "ALL", "TRV",
    "BK", "COF", "DOW", "DD", "PPG", "SHW", "NEM", "ECL", "ROP", "PH",
    "CMI", "ROK", "ADI", "MU", "LRCX", "KLAC", "AMAT", "NXPI", "MCHP",
    "SNPS", "CDNS", "INTU", "NOW", "WFC", "ADSK", "CSX", "PCAR", "MAR",
    "HLT", "YUM", "CMG", "ROST", "TJX", "DG", "DLTR", "KMB", "GIS", "KHC",
    "HSY", "STZ", "KDP", "MNST", "KR", "SYY", "F", "GM",
]

# De-duplicated union, preserving order.
UNIVERSE_EXTENDED: list[str] = list(dict.fromkeys([*UNIVERSE, *_EXTRA]))

assert len(UNIVERSE_EXTENDED) == len(set(UNIVERSE_EXTENDED)), "duplicate tickers"
assert len(UNIVERSE_EXTENDED) >= 150, "extended universe should hold >= 150 tickers"
