import numpy as np
import pandas as pd

from swingit.config import BacktestConfig, StrategyConfig
from swingit.experiments import run_scenarios
from swingit.strategy import compute_regime


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


def test_run_scenarios_shape_and_columns():
    raw = {f"T{i}": _market(i + 1) for i in range(6)}
    regime = compute_regime(_market(0)["close"], 50)
    cfg = BacktestConfig(strategy=StrategyConfig(regime_ma_period=50))
    scenarios = [
        ("small regime on", [f"T{i}" for i in range(3)], True),
        ("all regime off", [f"T{i}" for i in range(6)], False),
    ]
    out = run_scenarios(raw, regime, cfg, scenarios)
    assert list(out["scenario"]) == ["small regime on", "all regime off"]
    assert {"n_trades", "avg_exposure", "cagr", "sharpe", "max_drawdown"}.issubset(out.columns)
    assert out.loc[0, "n_tickers"] == 3
    assert out.loc[1, "n_tickers"] == 6


def test_disabling_regime_does_not_reduce_trades():
    raw = {f"T{i}": _market(i + 10) for i in range(5)}
    regime = compute_regime(_market(0)["close"], 50)
    cfg = BacktestConfig(strategy=StrategyConfig(regime_ma_period=50))
    tickers = list(raw)
    out = run_scenarios(
        raw, regime, cfg,
        [("on", tickers, True), ("off", tickers, False)],
    )
    on_trades = out.loc[out["scenario"] == "on", "n_trades"].iloc[0]
    off_trades = out.loc[out["scenario"] == "off", "n_trades"].iloc[0]
    # Removing a gating filter can only keep or increase the number of signals.
    assert off_trades >= on_trades
