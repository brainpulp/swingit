"""Technical indicators used by the strategy.

All functions operate on pandas Series indexed by date and return Series of the
same length (with leading NaNs where there is insufficient history). None of
them look into the future.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Uses Wilder smoothing (an EWMA with ``alpha = 1 / period``), which is the
    canonical RSI definition.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 the stock only rose: RSI is 100 by definition.
    out = out.where(avg_loss != 0, 100.0)
    out[avg_gain == 0] = out[avg_gain == 0].where(avg_loss[avg_gain == 0] == 0, 0.0)
    return out


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def n_day_return(close: pd.Series, days: int) -> pd.Series:
    """Trailing return over ``days`` trading days: close[t] / close[t-days] - 1."""
    return close.pct_change(periods=days)
