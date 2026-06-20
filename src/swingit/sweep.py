"""Parameter-robustness sweep.

Re-runs the backtest across a grid of strategy parameters so we can see whether
the edge is stable or an artifact of one lucky parameter set. A strategy whose
results collapse when RSI<28 becomes RSI<26 is not one to trade.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

from .config import BacktestConfig
from .engine import run_backtest
from .metrics import compute_metrics
from .strategy import add_signals

# Default grid: the spec values sit in the middle of each axis.
DEFAULT_GRID: dict[str, list] = {
    "rsi_entry": [25.0, 28.0, 31.0],
    "stop_loss": [-0.02, -0.03, -0.04],
    "profit_target": [0.03, 0.04, 0.05],
}


def run_sweep(
    raw_data: dict[str, pd.DataFrame],
    regime: pd.Series,
    base_config: BacktestConfig,
    grid: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Run one backtest per parameter combination.

    ``raw_data`` is ticker -> OHLCV *without* a signal column (signals are
    recomputed for each combination). Returns a DataFrame with one row per
    combo, sorted by CAGR descending, with an ``is_baseline`` flag marking the
    combination that matches ``base_config.strategy``.
    """
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    base = base_config.strategy

    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        scfg = replace(base, **params)
        signal_data = {
            t: add_signals(df, regime, scfg) for t, df in raw_data.items()
        }
        cfg = replace(base_config, strategy=scfg)
        result = run_backtest(signal_data, cfg)
        m = compute_metrics(result)
        row = dict(params)
        row.update(
            n_trades=m.n_trades,
            win_rate=round(m.win_rate, 4),
            cagr=round(m.cagr, 4),
            sharpe=round(m.sharpe, 3),
            max_drawdown=round(m.max_drawdown, 4),
            total_return=round(m.total_return, 4),
            profit_factor=round(m.profit_factor, 3)
            if m.profit_factor != float("inf")
            else float("inf"),
            is_baseline=all(
                getattr(base, k) == v for k, v in params.items()
            ),
        )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True)
    return df
