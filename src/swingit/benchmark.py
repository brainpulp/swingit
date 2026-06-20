"""Buy-and-hold benchmark (default: SPY) for context.

A strategy that makes money in isolation can still be a poor use of capital if
simply holding the index would have done better. This module builds an
apples-to-apples buy-and-hold equity curve over the same period and cost model.
"""

from __future__ import annotations

import pandas as pd

from .config import BacktestConfig
from .metrics import equity_stats


def buy_and_hold_curve(
    close: pd.Series, initial_capital: float, cost_per_leg: float
) -> pd.Series:
    """Equity curve for investing everything at the first close and holding.

    A single entry cost (commission + slippage) is charged up front; the
    position is marked to market at each close thereafter.
    """
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    entry = close.iloc[0]
    shares = initial_capital * (1.0 - cost_per_leg) / entry
    curve = shares * close
    curve.name = "benchmark"
    return curve


def benchmark(
    close: pd.Series, config: BacktestConfig
) -> tuple[pd.Series, dict]:
    """Return the benchmark equity curve and its curve-level stats."""
    cost_per_leg = config.commission_rate + config.slippage_rate
    curve = buy_and_hold_curve(close, config.initial_capital, cost_per_leg)
    stats = equity_stats(curve) if not curve.empty else {}
    return curve, stats
