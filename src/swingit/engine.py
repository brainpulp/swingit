"""Event-driven backtest engine.

Design notes / invariants:
  * **No lookahead.** Entry signals are produced from a day's *close* and the
    order is filled at the *next* day's open.
  * **Intraday exits.** Stop / target are checked against the bar's high & low.
    The stop is checked *first* (the conservative assumption when a single bar
    touches both levels). A gap through a level fills at the (worse) open price.
  * **Equal-weight portfolio** with a hard cap on concurrent positions. New
    signals are dropped when no slot is free.
  * Costs: ``slippage_rate`` is applied to the fill price and ``commission_rate``
    to the notional, on *each* leg.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float  # actual fill (incl. slippage)
    exit_price: float  # actual fill (incl. slippage)
    shares: float
    bars_held: int
    exit_reason: str  # "target" | "stop" | "time" | "eod"
    pnl: float  # net dollars after commissions
    net_return: float  # net of all costs
    gross_return: float  # raw exit/entry open-to-fill, before costs

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "entry_date": self.entry_date.date().isoformat(),
            "exit_date": self.exit_date.date().isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "shares": round(self.shares, 4),
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason,
            "pnl": round(self.pnl, 2),
            "net_return": round(self.net_return, 6),
            "gross_return": round(self.gross_return, 6),
        }


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series  # indexed by date
    config: BacktestConfig
    exposure: pd.Series  # daily count of open positions, indexed by date
    trades_df: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        rows = [t.as_row() for t in self.trades]
        self.trades_df = pd.DataFrame(rows)


@dataclass
class _OpenPosition:
    ticker: str
    entry_i: int
    entry_date: pd.Timestamp
    entry_fill: float
    entry_open: float
    shares: float
    stop_price: float
    target_price: float


class _Series:
    """Fast per-ticker array view keyed by date position."""

    __slots__ = ("dates", "open", "high", "low", "close", "signal", "pos")

    def __init__(self, df: pd.DataFrame):
        self.dates = df.index
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.signal = df["signal"].to_numpy(dtype=bool)
        self.pos = {d: i for i, d in enumerate(df.index)}


def run_backtest(
    data: dict[str, pd.DataFrame], config: BacktestConfig
) -> BacktestResult:
    """Run the backtest.

    ``data`` maps ticker -> OHLCV DataFrame that already contains a boolean
    ``signal`` column (see :func:`swingit.strategy.add_signals`).
    """
    cfg = config
    scfg = cfg.strategy
    series = {t: _Series(df) for t, df in data.items()}

    # Global, ordered set of trading dates across all tickers.
    all_dates = sorted(set().union(*[set(s.dates) for s in series.values()]))

    # pending_entries[date] -> list of tickers to enter at that date's open.
    pending: dict[pd.Timestamp, list[str]] = {}
    for ticker, s in series.items():
        sig_idx = np.flatnonzero(s.signal)
        for i in sig_idx:
            nxt = i + 1
            if nxt < len(s.dates):
                pending.setdefault(s.dates[nxt], []).append(ticker)

    cash = cfg.initial_capital
    open_positions: list[_OpenPosition] = []
    trades: list[Trade] = []
    last_price: dict[str, float] = {}
    equity_dates: list[pd.Timestamp] = []
    equity_values: list[float] = []
    exposure_counts: list[int] = []

    slip = cfg.slippage_rate
    comm = cfg.commission_rate

    def close_position(
        pos: _OpenPosition, exit_price: float, exit_date, bars_held, reason
    ) -> None:
        nonlocal cash
        exit_fill = exit_price * (1.0 - slip)
        proceeds = pos.shares * exit_fill
        cash += proceeds - proceeds * comm
        cost_basis = pos.shares * pos.entry_fill * (1.0 + comm)
        net_value = proceeds * (1.0 - comm)
        trades.append(
            Trade(
                ticker=pos.ticker,
                entry_date=pos.entry_date,
                exit_date=exit_date,
                entry_price=pos.entry_fill,
                exit_price=exit_fill,
                shares=pos.shares,
                bars_held=bars_held,
                exit_reason=reason,
                pnl=net_value - cost_basis,
                net_return=net_value / cost_basis - 1.0,
                gross_return=exit_price / pos.entry_open - 1.0,
            )
        )

    for d in all_dates:
        # --- 1. Execute pending entries at today's open -------------------
        to_enter = pending.get(d, [])
        if to_enter:
            # Mark current equity at today's open for equal-weight sizing.
            equity_open = cash
            for p in open_positions:
                s = series[p.ticker]
                i = s.pos.get(d)
                px = s.open[i] if i is not None else last_price.get(p.ticker, p.entry_fill)
                equity_open += p.shares * px
            alloc_target = equity_open / cfg.max_positions

            for ticker in to_enter:
                if len(open_positions) >= cfg.max_positions:
                    break  # no free slot — signal dropped
                s = series[ticker]
                i = s.pos.get(d)
                if i is None:
                    continue
                open_px = s.open[i]
                if not np.isfinite(open_px) or open_px <= 0:
                    continue
                entry_fill = open_px * (1.0 + slip)
                alloc = min(alloc_target, cash)
                shares = alloc / (entry_fill * (1.0 + comm))
                if shares <= 0:
                    continue
                cost = shares * entry_fill
                cash -= cost + cost * comm
                open_positions.append(
                    _OpenPosition(
                        ticker=ticker,
                        entry_i=i,
                        entry_date=d,
                        entry_fill=entry_fill,
                        entry_open=open_px,
                        shares=shares,
                        stop_price=entry_fill * (1.0 + scfg.stop_loss),
                        target_price=entry_fill * (1.0 + scfg.profit_target),
                    )
                )

        # --- 2. Check intraday exits for all open positions --------------
        still_open: list[_OpenPosition] = []
        for p in open_positions:
            s = series[p.ticker]
            i = s.pos.get(d)
            if i is None:
                still_open.append(p)  # ticker didn't trade today
                continue
            bars_held = i - p.entry_i + 1
            low, high, open_px, close_px = s.low[i], s.high[i], s.open[i], s.close[i]

            if low <= p.stop_price:
                # Gap through the stop fills at the (worse) open.
                exit_px = min(p.stop_price, open_px)
                close_position(p, exit_px, d, bars_held, "stop")
            elif high >= p.target_price:
                exit_px = max(p.target_price, open_px)
                close_position(p, exit_px, d, bars_held, "target")
            elif bars_held >= scfg.max_hold_days:
                close_position(p, close_px, d, bars_held, "time")
            else:
                still_open.append(p)
        open_positions = still_open

        # --- 3. Mark to market at the close ------------------------------
        equity = cash
        for p in open_positions:
            s = series[p.ticker]
            i = s.pos.get(d)
            if i is not None:
                last_price[p.ticker] = s.close[i]
            equity += p.shares * last_price.get(p.ticker, p.entry_fill)
        equity_dates.append(d)
        equity_values.append(equity)
        exposure_counts.append(len(open_positions))

    # --- Force-close anything still open at the final bar ----------------
    if open_positions and all_dates:
        last_d = all_dates[-1]
        for p in open_positions:
            s = series[p.ticker]
            i = s.pos.get(last_d, len(s.dates) - 1)
            close_position(p, s.close[i], last_d, i - p.entry_i + 1, "eod")
        # Recompute the final equity point (now all cash).
        equity_values[-1] = cash

    idx = pd.DatetimeIndex(equity_dates)
    equity_curve = pd.Series(equity_values, index=idx, name="equity")
    exposure = pd.Series(exposure_counts, index=idx, name="open_positions")
    return BacktestResult(
        trades=trades, equity_curve=equity_curve, config=cfg, exposure=exposure
    )
