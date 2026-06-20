from swingit.config import BacktestConfig, StrategyConfig
from swingit.engine import run_backtest
from swingit.metrics import compute_metrics
from swingit.report import build_html, write_report
from tests.conftest import make_ohlcv


def _result():
    df = make_ohlcv(
        opens=[50, 100, 100],
        highs=[51, 105, 105],
        lows=[49, 99, 99],
        closes=[50, 101, 101],
        signals=[True, False, False],
    )
    cfg = BacktestConfig(commission_rate=0.0, slippage_rate=0.0,
                         strategy=StrategyConfig())
    return run_backtest({"AAA": df}, cfg)


def test_html_is_self_contained():
    res = _result()
    html = build_html(res, compute_metrics(res))
    assert "data:image/png;base64," in html  # charts embedded inline
    # No external network assets referenced.
    assert "src=\"http" not in html
    assert "<table" in html  # trades table rendered


def test_write_report_creates_files(tmp_path):
    res = _result()
    html_path, csv_path = write_report(res, compute_metrics(res), tmp_path)
    assert html_path.exists() and html_path.stat().st_size > 1000
    assert csv_path.exists()
    assert csv_path.read_text().splitlines()[0].startswith("ticker,")
