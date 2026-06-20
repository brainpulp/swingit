import pytest

from swingit.config import BacktestConfig, StrategyConfig
from swingit.engine import run_backtest
from tests.conftest import make_ohlcv


def zero_cost_config(**kw):
    kw.setdefault("strategy", StrategyConfig())
    return BacktestConfig(
        initial_capital=100_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        **kw,
    )


def test_entry_uses_next_open_no_lookahead():
    # Signal on day 0 (close). Entry must fill at day 1 OPEN (=100), not day 0.
    df = make_ohlcv(
        opens=[50, 100, 104],
        highs=[51, 106, 105],
        lows=[49, 99, 103],
        closes=[50, 101, 104],
        signals=[True, False, False],
    )
    res = run_backtest({"AAA": df}, zero_cost_config())
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_price == pytest.approx(100.0)  # day 1 open, not day 0
    assert t.entry_date == df.index[1]


def test_target_exit():
    df = make_ohlcv(
        opens=[50, 100, 100],
        highs=[51, 105, 105],  # day1 high 105 >= target 104
        lows=[49, 99, 99],
        closes=[50, 101, 101],
        signals=[True, False, False],
    )
    res = run_backtest({"AAA": df}, zero_cost_config())
    t = res.trades[0]
    assert t.exit_reason == "target"
    assert t.exit_price == pytest.approx(104.0)
    assert t.net_return == pytest.approx(0.04)
    assert t.bars_held == 1


def test_stop_exit():
    df = make_ohlcv(
        opens=[50, 100, 100],
        highs=[51, 101, 101],
        lows=[49, 96, 96],  # day1 low 96 <= stop 97
        closes=[50, 98, 98],
        signals=[True, False, False],
    )
    res = run_backtest({"AAA": df}, zero_cost_config())
    t = res.trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(97.0)
    assert t.net_return == pytest.approx(-0.03)


def test_stop_checked_first_when_bar_hits_both():
    # Same bar touches both stop and target -> stop wins (conservative).
    df = make_ohlcv(
        opens=[50, 100, 100],
        highs=[51, 106, 106],  # would hit target
        lows=[49, 96, 96],     # also hits stop
        closes=[50, 101, 101],
        signals=[True, False, False],
    )
    res = run_backtest({"AAA": df}, zero_cost_config())
    assert res.trades[0].exit_reason == "stop"


def test_gap_down_through_stop_fills_at_open():
    # Open gaps below the stop -> fill at the (worse) open price.
    df = make_ohlcv(
        opens=[50, 100, 90],
        highs=[51, 101, 91],
        lows=[49, 101, 89],  # day1 stays flat, day2 gaps down
        closes=[50, 100, 90],
        signals=[True, False, False],
    )
    cfg = zero_cost_config(strategy=StrategyConfig(max_hold_days=10))
    res = run_backtest({"AAA": df}, cfg)
    t = res.trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(90.0)  # open, not 97 stop level


def test_time_exit():
    # Never hits target/stop; exits at close after max_hold_days bars.
    opens = [50] + [100] * 6
    highs = [51] + [102] * 6
    lows = [49] + [98] * 6
    closes = [50] + [99, 100, 101, 100, 99, 100]
    signals = [True] + [False] * 6
    df = make_ohlcv(opens, highs, lows, closes, signals)
    res = run_backtest({"AAA": df}, zero_cost_config())
    t = res.trades[0]
    assert t.exit_reason == "time"
    assert t.bars_held == 5
    # 5th bar after entry is index 5 (entry at index 1).
    assert t.exit_date == df.index[5]


def test_max_positions_cap_drops_signals():
    # Three tickers all signal on day 0; cap of 1 means only one trade.
    frames = {}
    for name in ["AAA", "BBB", "CCC"]:
        frames[name] = make_ohlcv(
            opens=[50, 100, 100],
            highs=[51, 105, 105],
            lows=[49, 99, 99],
            closes=[50, 101, 101],
            signals=[True, False, False],
        )
    cfg = zero_cost_config(max_positions=1)
    res = run_backtest(frames, cfg)
    assert len(res.trades) == 1


def test_costs_reduce_returns():
    df = make_ohlcv(
        opens=[50, 100, 100],
        highs=[51, 105, 105],
        lows=[49, 99, 99],
        closes=[50, 101, 101],
        signals=[True, False, False],
    )
    cfg = BacktestConfig(
        commission_rate=0.001, slippage_rate=0.001, strategy=StrategyConfig()
    )
    res = run_backtest({"AAA": df}, cfg)
    t = res.trades[0]
    # Gross would be +4%; net is reduced by ~0.4% round-trip costs.
    assert t.gross_return > t.net_return
    assert t.net_return < 0.04
    assert t.net_return > 0.03


def test_equity_curve_tracks_cash_after_close():
    df = make_ohlcv(
        opens=[50, 100, 100],
        highs=[51, 105, 105],
        lows=[49, 99, 99],
        closes=[50, 101, 101],
        signals=[True, False, False],
    )
    res = run_backtest({"AAA": df}, zero_cost_config())
    # After the winning trade, final equity should exceed initial capital.
    assert res.equity_curve.iloc[-1] > 100_000.0
    assert len(res.equity_curve) == len(df)
