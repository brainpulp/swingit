import numpy as np
import pandas as pd

from swingit.value import compare


def _series(drift, n=260, start="2020-01-01"):
    return pd.Series(
        100 * np.exp(np.cumsum(np.full(n, drift))),
        index=pd.bdate_range(start, periods=n),
    )


def test_compare_labels_and_ranking():
    closes = {"SPY": _series(0.0008), "RPV": _series(0.0003)}
    labels = {"SPY": "S&P 500", "RPV": "Pure Value"}
    curves, stats = compare(closes, cost_per_leg=0.0, initial=100_000, labels=labels)
    assert set(curves) == {"S&P 500", "Pure Value"}
    # The stronger trend ends with more money.
    assert stats["S&P 500"]["final_equity"] > stats["Pure Value"]["final_equity"]
    assert stats["S&P 500"]["final_equity"] > 100_000


def test_compare_applies_entry_cost():
    closes = {"SPY": _series(0.0)}
    _, stats = compare(closes, cost_per_leg=0.01, initial=100_000)
    # Flat prices + 1% entry cost -> ends slightly below start.
    assert stats["SPY"]["final_equity"] < 100_000
