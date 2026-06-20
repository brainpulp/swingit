"""Performance metrics derived from a :class:`BacktestResult`."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .engine import BacktestResult

TRADING_DAYS = 252


@dataclass
class Metrics:
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float  # mean net return per trade
    profit_factor: float
    sharpe: float
    max_drawdown: float
    cagr: float
    total_return: float
    final_equity: float

    def as_dict(self) -> dict:
        return asdict(self)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown (<= 0) at each point relative to the running peak."""
    peak = equity.cummax()
    return equity / peak - 1.0


def _sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.empty or rets.std(ddof=0) == 0:
        return 0.0
    return float(rets.mean() / rets.std(ddof=0) * np.sqrt(TRADING_DAYS))


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def compute_metrics(result: BacktestResult) -> Metrics:
    eq = result.equity_curve
    df = result.trades_df

    if df.empty:
        return Metrics(0, 0, 0, 0, 0, 0, _sharpe(eq),
                       float(drawdown_series(eq).min()) if len(eq) else 0.0,
                       _cagr(eq), 0.0, float(eq.iloc[-1]) if len(eq) else 0.0)

    rets = df["net_return"]
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    pnl = df["pnl"]
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return Metrics(
        n_trades=int(len(df)),
        win_rate=float(len(wins) / len(df)),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        expectancy=float(rets.mean()),
        profit_factor=profit_factor,
        sharpe=_sharpe(eq),
        max_drawdown=float(drawdown_series(eq).min()),
        cagr=_cagr(eq),
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        final_equity=float(eq.iloc[-1]),
    )
