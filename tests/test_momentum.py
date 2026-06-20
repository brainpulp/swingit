import numpy as np
import pandas as pd

from swingit.momentum import MomentumParams, backtest_momentum


def _prices(spec, n=300, start="2020-01-01"):
    """Build a wide close-price frame; spec maps ticker -> daily drift."""
    idx = pd.bdate_range(start, periods=n)
    data = {}
    for t, drift in spec.items():
        data[t] = 100 * np.exp(np.cumsum(np.full(n, drift)))
    return pd.DataFrame(data, index=idx)


def test_momentum_picks_the_uptrend():
    prices = _prices({"UP": 0.002, "FLAT": 0.0, "DOWN": -0.002})
    params = MomentumParams(lookback=40, skip=0, top_n=1, absolute_filter=True)
    curve, info = backtest_momentum(prices, cost_per_leg=0.0, params=params)
    # It should ride the uptrend and finish well above the $100k start.
    assert curve.iloc[-1] > 130_000
    assert info["n_rebalances"] > 0


def test_momentum_goes_to_cash_when_all_falling():
    prices = _prices({"A": -0.002, "B": -0.001, "C": -0.003})
    params = MomentumParams(lookback=40, skip=0, top_n=2, absolute_filter=True)
    curve, _ = backtest_momentum(prices, cost_per_leg=0.0, params=params)
    # Absolute-momentum filter -> mostly cash -> capital preserved (no big loss).
    assert curve.iloc[-1] > 95_000


def test_absolute_filter_off_stays_invested():
    prices = _prices({"A": -0.002, "B": -0.001, "C": -0.003})
    params = MomentumParams(lookback=40, skip=0, top_n=1, absolute_filter=False)
    curve, _ = backtest_momentum(prices, cost_per_leg=0.0, params=params)
    # Forced to hold the "least bad" downtrend -> loses money.
    assert curve.iloc[-1] < 100_000
