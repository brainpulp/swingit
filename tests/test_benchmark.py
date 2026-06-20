import pandas as pd
import pytest

from swingit.benchmark import benchmark, buy_and_hold_curve
from swingit.config import BacktestConfig


def test_buy_and_hold_curve_grows_with_price():
    close = pd.Series([100.0, 110.0, 121.0], index=pd.bdate_range("2020-01-01", periods=3))
    curve = buy_and_hold_curve(close, 1000.0, cost_per_leg=0.0)
    # No costs: final/initial equals price ratio.
    assert curve.iloc[0] == 1000.0
    assert curve.iloc[-1] == pytest.approx(1210.0)


def test_buy_and_hold_applies_entry_cost():
    close = pd.Series([100.0, 100.0], index=pd.bdate_range("2020-01-01", periods=2))
    curve = buy_and_hold_curve(close, 1000.0, cost_per_leg=0.01)
    assert curve.iloc[0] == pytest.approx(990.0)  # 1% entry cost


def test_benchmark_returns_stats():
    close = pd.Series(
        [100.0, 101.0, 103.0, 102.0, 105.0],
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    cfg = BacktestConfig()
    curve, stats = benchmark(close, cfg)
    assert "cagr" in stats and "sharpe" in stats and "max_drawdown" in stats
    assert stats["final_equity"] > 0
