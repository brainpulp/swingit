"""Shared test helpers."""

from __future__ import annotations

import pandas as pd
import pytest


def make_ohlcv(opens, highs, lows, closes, signals=None, start="2020-01-01"):
    """Build an OHLCV DataFrame on consecutive business days.

    ``signals`` (optional) sets the boolean ``signal`` column; defaults to all
    False.
    """
    n = len(opens)
    idx = pd.bdate_range(start=start, periods=n)
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=idx,
    )
    df.index.name = "date"
    if signals is None:
        signals = [False] * n
    df["signal"] = signals
    return df


@pytest.fixture
def make_ohlcv_fixture():
    return make_ohlcv
