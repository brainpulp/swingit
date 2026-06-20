import numpy as np
import pandas as pd

from swingit.config import BacktestConfig, StrategyConfig
from swingit.strategy import compute_regime
from swingit.sweep import run_sweep
from tests.conftest import make_ohlcv


def _market(seed, n=320):
    r = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(r.normal(0.0002, 0.02, n)))
    close = pd.Series(px, index=pd.bdate_range("2020-01-01", periods=n))
    op = close.shift(1).fillna(close.iloc[0])
    hi = np.maximum(op, close) * (1 + abs(r.normal(0, 0.01, n)))
    lo = np.minimum(op, close) * (1 - abs(r.normal(0, 0.01, n)))
    df = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": 1e6})
    df.index.name = "date"
    return df


def test_sweep_shape_and_baseline():
    raw = {f"T{i}": _market(i + 1) for i in range(3)}
    spy = _market(0)
    regime = compute_regime(spy["close"], 50)  # short MA so regime is usable
    cfg = BacktestConfig(strategy=StrategyConfig(regime_ma_period=50))
    grid = {"rsi_entry": [28.0, 31.0], "stop_loss": [-0.03], "profit_target": [0.04]}

    out = run_sweep(raw, regime, cfg, grid)
    assert len(out) == 2  # 2 x 1 x 1 combos
    assert {"rsi_entry", "cagr", "sharpe", "is_baseline"}.issubset(out.columns)
    # Exactly one baseline row (matches the config's RSI of 28).
    assert out["is_baseline"].sum() == 1
    # Sorted by CAGR descending.
    assert list(out["cagr"]) == sorted(out["cagr"], reverse=True)


def test_sweep_uses_provided_grid_size():
    raw = {"T1": _market(5)}
    regime = compute_regime(_market(0)["close"], 50)
    cfg = BacktestConfig(strategy=StrategyConfig(regime_ma_period=50))
    grid = {"rsi_entry": [25.0, 28.0, 31.0], "stop_loss": [-0.02, -0.03],
            "profit_target": [0.04]}
    out = run_sweep(raw, regime, cfg, grid)
    assert len(out) == 6  # 3 x 2 x 1
