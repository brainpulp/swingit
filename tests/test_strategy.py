import numpy as np
import pandas as pd

from swingit.config import StrategyConfig
from swingit.strategy import add_signals, compute_regime
from tests.conftest import make_ohlcv


def _falling_frame(n=40):
    # Long flat-ish history then a sharp drop to push RSI down and 3-day return
    # below -4%.
    closes = list(np.linspace(100, 100, n - 4)) + [100, 97, 94, 90]
    closes = [float(c) for c in closes]
    opens = closes
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return make_ohlcv(opens, highs, lows, closes)


def test_signal_fires_in_uptrend_regime():
    df = _falling_frame()
    cfg = StrategyConfig()
    regime = pd.Series(True, index=df.index)  # SPY above MA everywhere
    out = add_signals(df, regime, cfg)
    # The final, sharply-down bar should signal.
    assert bool(out["signal"].iloc[-1]) is True


def test_regime_filter_blocks_signal():
    df = _falling_frame()
    cfg = StrategyConfig()
    regime = pd.Series(False, index=df.index)  # SPY below MA -> no entries
    out = add_signals(df, regime, cfg)
    assert out["signal"].sum() == 0


def test_no_signal_without_drop():
    # Low RSI possible but no >=4% 3-day drop -> no signal.
    closes = [float(c) for c in np.linspace(100, 100, 40)]
    df = make_ohlcv(closes, [c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)
    regime = pd.Series(True, index=df.index)
    out = add_signals(df, regime, StrategyConfig())
    assert out["signal"].sum() == 0


def test_regime_filter_toggle_off_ignores_regime():
    df = _falling_frame()
    # Regime says "never trade", but the filter is disabled -> signal still fires.
    regime = pd.Series(False, index=df.index)
    cfg = StrategyConfig(use_regime_filter=False)
    out = add_signals(df, regime, cfg)
    assert bool(out["signal"].iloc[-1]) is True


def test_compute_regime():
    closes = pd.Series([float(c) for c in range(1, 11)])
    reg = compute_regime(closes, period=3)
    # Rising series -> price above its own trailing MA once defined.
    assert reg.iloc[-1] == True  # noqa: E712
    assert reg.iloc[0] == False  # noqa: E712  (not enough history)
