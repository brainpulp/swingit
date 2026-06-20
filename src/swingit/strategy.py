"""Signal generation for the mean-reversion strategy.

The entry signal is evaluated using only information available at the *close* of
the signal day, so there is no lookahead. The engine is responsible for
executing the resulting entry at the next day's open.
"""

from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .indicators import n_day_return, rsi, sma


def compute_regime(close: pd.Series, period: int) -> pd.Series:
    """Boolean Series: True when ``close`` is above its ``period``-day SMA."""
    ma = sma(close, period)
    return (close > ma).fillna(False)


def add_signals(
    df: pd.DataFrame, regime: pd.Series, cfg: StrategyConfig
) -> pd.DataFrame:
    """Return a copy of ``df`` with indicator and ``signal`` columns added.

    ``regime`` is the boolean regime Series (indexed by date) from
    :func:`compute_regime`; it is aligned onto ``df``'s index.
    """
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["drop"] = n_day_return(out["close"], cfg.drop_lookback_days)
    out["regime"] = regime.reindex(out.index).fillna(False)

    out["signal"] = (
        (out["rsi"] < cfg.rsi_entry)
        & (out["drop"] <= cfg.drop_lookback_return)
        & out["regime"]
    )
    # Rows without enough history to compute RSI / drop cannot signal.
    out.loc[out["rsi"].isna() | out["drop"].isna(), "signal"] = False
    return out
