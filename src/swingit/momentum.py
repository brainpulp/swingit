"""A momentum (trend-following) strategy — a structurally different bet.

Where the mean-reversion strategy buys falling stocks and sits in cash most of
the time, this one does the opposite: each month it holds the handful of
strongest-trending names and rotates as trends change. It is fully invested
whenever things are going up, and (optionally) steps to cash when nothing has
positive momentum — so it naturally deploys capital instead of starving it.

This is the classic, well-documented "cross-sectional + absolute momentum"
(a.k.a. dual momentum) approach. It is implemented as its own monthly-rebalance
portfolio backtest because its structure differs from the swing engine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import benchmark
from .config import BacktestConfig
from .data import load_universe
from .metrics import equity_stats
from .universe import UNIVERSE_EXTENDED

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MomentumParams:
    lookback: int = 252  # ~12 months of trailing history
    skip: int = 21       # skip most-recent ~1 month (the "12-1" convention)
    top_n: int = 10      # number of names held, equal weight
    absolute_filter: bool = True  # hold cash for names without positive momentum


def backtest_momentum(
    prices: pd.DataFrame,
    cost_per_leg: float,
    params: MomentumParams,
    initial_capital: float = 100_000.0,
) -> tuple[pd.Series, dict]:
    """Monthly-rebalance momentum backtest.

    ``prices`` is a wide DataFrame of close prices (dates x tickers). Returns the
    daily equity curve and an info dict (avg turnover, number of rebalances).
    """
    prices = prices.sort_index().ffill()
    rets = prices.pct_change().fillna(0.0)
    # Momentum score at date d: return from d-lookback to d-skip.
    mom = prices.shift(params.skip) / prices.shift(params.lookback) - 1.0

    dates = prices.index
    periods = pd.PeriodIndex(dates, freq="M")
    is_rebal = np.r_[True, periods[1:] != periods[:-1]]

    tickers = prices.columns
    weights = pd.Series(0.0, index=tickers)
    val = 1.0
    equity = np.empty(len(dates))
    turnovers: list[float] = []

    for i, d in enumerate(dates):
        if i > 0:
            r = rets.iloc[i]
            val *= 1.0 + float((weights * r).sum())
            # Let weights drift with returns, then renormalise if invested.
            w = weights * (1.0 + r)
            s = w.sum()
            weights = w / s if s > 0 else w

        if is_rebal[i] and i >= params.lookback:
            score = mom.iloc[i].dropna()
            if params.absolute_filter:
                score = score[score > 0]
            target = pd.Series(0.0, index=tickers)
            if len(score) > 0:
                top = score.nlargest(params.top_n).index
                target[top] = 1.0 / len(top)
            turnovers.append(float((target - weights).abs().sum()))
            val *= 1.0 - cost_per_leg * turnovers[-1]
            weights = target

        equity[i] = val

    curve = pd.Series(equity * initial_capital, index=dates, name="momentum")
    info = {
        "n_rebalances": len(turnovers),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }
    return curve, info


def run(config: BacktestConfig, params: MomentumParams | None = None,
        *, use_cache: bool = True):
    """Load the extended universe, run momentum, and return (curve, info, stats,
    benchmark_stats)."""
    params = params or MomentumParams()
    needed = list(dict.fromkeys([config.regime_symbol, *UNIVERSE_EXTENDED]))
    logger.info("loading %d symbols for momentum", len(needed))
    data = load_universe(needed, config.start, config.end, use_cache=use_cache)

    spy_close = data[config.regime_symbol]["close"]
    prices = pd.DataFrame(
        {t: df["close"] for t, df in data.items() if t != config.regime_symbol}
    )
    cost_per_leg = config.commission_rate + config.slippage_rate
    curve, info = backtest_momentum(
        prices, cost_per_leg, params, config.initial_capital
    )
    stats = equity_stats(curve)
    _, bench_stats = benchmark(spy_close, config)
    return curve, info, stats, bench_stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="swingit momentum strategy")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", default="dist")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = BacktestConfig(start=args.start, end=args.end)
    params = MomentumParams(top_n=args.top_n)
    curve, info, stats, bench = run(config, params, use_cache=not args.no_cache)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "momentum_equity.csv")

    def pct(x):
        return f"{x * 100:.1f}%"

    print("\n=== Momentum vs Buy & Hold SPY (2019-2024) ===")
    hdr = f"{'metric':<16}{'momentum':>12}{'SPY':>12}"
    print(hdr)
    print("-" * len(hdr))
    for k, label in [("total_return", "Total return"), ("cagr", "CAGR"),
                     ("sharpe", "Sharpe"), ("max_drawdown", "Max drawdown"),
                     ("final_equity", "Final $")]:
        mv = stats[k]
        bv = bench[k]
        if k in ("total_return", "cagr", "max_drawdown"):
            mv, bv = pct(mv), pct(bv)
        elif k == "final_equity":
            mv, bv = f"${mv:,.0f}", f"${bv:,.0f}"
        else:
            mv, bv = f"{mv:.2f}", f"{bv:.2f}"
        print(f"{label:<16}{mv:>12}{bv:>12}")
    beats = stats["cagr"] > bench["cagr"]
    print(f"\n  -> momentum {'BEATS' if beats else 'DOES NOT BEAT'} buy & hold on CAGR")
    print(f"  (rebalances: {info['n_rebalances']}, avg turnover: {pct(info['avg_turnover'])})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
