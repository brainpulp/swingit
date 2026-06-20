import pandas as pd

from swingit.config import BacktestConfig, StrategyConfig
from swingit.engine import run_backtest
from swingit.metrics import compute_metrics, drawdown_series
from tests.conftest import make_ohlcv


def _winning_result():
    df = make_ohlcv(
        opens=[50, 100, 100],
        highs=[51, 105, 105],
        lows=[49, 99, 99],
        closes=[50, 101, 101],
        signals=[True, False, False],
    )
    cfg = BacktestConfig(
        commission_rate=0.0, slippage_rate=0.0, strategy=StrategyConfig()
    )
    return run_backtest({"AAA": df}, cfg)


def test_drawdown_series_non_positive():
    eq = pd.Series([100, 110, 105, 120, 90], dtype=float)
    dd = drawdown_series(eq)
    assert (dd <= 0).all()
    assert dd.iloc[-1] == (90 / 120 - 1)


def test_metrics_winning_trade():
    res = _winning_result()
    m = compute_metrics(res)
    assert m.n_trades == 1
    assert m.win_rate == 1.0
    assert m.expectancy > 0
    assert m.profit_factor == float("inf")  # no losses
    assert m.final_equity > 100_000.0


def test_metrics_no_trades():
    df = make_ohlcv([100, 101], [102, 103], [99, 100], [101, 102], [False, False])
    res = run_backtest({"AAA": df}, BacktestConfig())
    m = compute_metrics(res)
    assert m.n_trades == 0
    assert m.win_rate == 0
    assert m.expectancy == 0
