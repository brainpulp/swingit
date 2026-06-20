import numpy as np
import pandas as pd

from swingit.allocations import static_mix_curve, trend_curve


def _series(drift, n=300, vol=0.0, seed=0, start="2020-01-01"):
    r = np.random.default_rng(seed)
    steps = np.full(n, drift) + (r.normal(0, vol, n) if vol else 0.0)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=pd.bdate_range(start, periods=n))


def test_static_mix_blends_returns():
    # Stocks up strongly, bonds flat -> a 60/40 mix lands between the two.
    spy = _series(0.002)
    agg = _series(0.0)
    rets = pd.DataFrame({"SPY": spy.pct_change(), "AGG": agg.pct_change()}).fillna(0.0)
    allstock = static_mix_curve(rets, {"SPY": 1.0}, 0.0, 100_000)
    mix = static_mix_curve(rets, {"SPY": 0.6, "AGG": 0.4}, 0.0, 100_000)
    assert mix.iloc[-1] < allstock.iloc[-1]      # less upside
    assert mix.iloc[-1] > 100_000                # but still grew


def test_trend_rule_sidesteps_a_downtrend():
    # Long uptrend (so the 200-day average forms and we're invested), then a
    # sustained crash. The trend rule should exit and suffer a smaller drawdown.
    up = _series(0.0015, n=450)
    down = pd.Series(
        up.iloc[-1] * np.exp(np.cumsum(np.full(300, -0.003))),
        index=pd.bdate_range(up.index[-1] + pd.offsets.BDay(1), periods=300),
    )
    close = pd.concat([up, down])
    held = 100_000 * (1 + close.pct_change().fillna(0)).cumprod()
    trend = trend_curve(close, cost_per_leg=0.0, initial=100_000, ma=200)

    def max_dd(c):
        return float((c / c.cummax() - 1).min())

    # Far smaller drawdown, and more money left at the end.
    assert max_dd(trend) > max_dd(held)
    assert trend.iloc[-1] > held.iloc[-1]


def test_trend_rule_costs_reduce_curve():
    close = _series(0.001, n=400, vol=0.02, seed=3)
    cheap = trend_curve(close, cost_per_leg=0.0, initial=100_000, ma=200)
    pricey = trend_curve(close, cost_per_leg=0.01, initial=100_000, ma=200)
    assert pricey.iloc[-1] <= cheap.iloc[-1]
