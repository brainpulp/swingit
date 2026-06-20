import numpy as np
import pandas as pd

from swingit.indicators import n_day_return, rsi, sma


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 30, dtype=float))
    out = rsi(s, 14)
    assert out.dropna().iloc[-1] == 100.0


def test_rsi_bounds_and_midrange():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    out = rsi(s, 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_rsi_drop_pushes_low():
    # A long steady uptrend then a sharp multi-day drop -> RSI should fall.
    up = list(np.arange(100, 130, 1.0))
    down = [129, 125, 120, 114, 108, 101]
    s = pd.Series(up + down)
    out = rsi(s, 14)
    assert out.iloc[-1] < 40


def test_sma():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert np.isnan(out.iloc[0])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_n_day_return():
    s = pd.Series([100, 101, 102, 96], dtype=float)
    out = n_day_return(s, 3)
    assert out.iloc[3] == (96 / 100 - 1)
