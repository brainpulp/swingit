"""Defensive allocation comparison.

Answers a practical question without any market-timing guesswork: over a window
that includes the 2020 crash and the 2022 bear market, how do simple, rules-based
defensive postures trade return for smaller drawdowns versus 100% stocks?

Strategies (all mechanical, decided in advance — no predicting):
  * Buy & Hold              — 100% stocks (SPY)
  * 80/20 and 60/40        — stocks/bonds (SPY/AGG), rebalanced monthly
  * 200-day trend (to cash) — hold SPY while above its 200-day average, else cash
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import load_universe
from .metrics import equity_stats

logger = logging.getLogger(__name__)


def _rebal_flags(dates: pd.DatetimeIndex, freq: str = "M") -> np.ndarray:
    p = pd.PeriodIndex(dates, freq=freq)
    return np.r_[True, p[1:] != p[:-1]]


def static_mix_curve(
    rets: pd.DataFrame,
    weights: dict[str, float],
    cost_per_leg: float,
    initial: float,
    freq: str = "M",
) -> pd.Series:
    """Equity curve for a fixed stock/bond mix, rebalanced every ``freq``."""
    dates = rets.index
    flags = _rebal_flags(dates, freq)
    cols = list(weights)
    target = pd.Series([weights[c] for c in cols], index=cols, dtype=float)
    w = pd.Series(0.0, index=cols)
    val = initial
    out = np.empty(len(dates))
    for i in range(len(dates)):
        if i > 0:
            r = rets[cols].iloc[i]
            val *= 1.0 + float((w * r).sum())
            w = w * (1.0 + r)
            s = w.sum()
            if s > 0:
                w = w / s
        if flags[i]:
            val *= 1.0 - cost_per_leg * float((target - w).abs().sum())
            w = target.copy()
        out[i] = val
    return pd.Series(out, index=dates)


def trend_curve(
    close: pd.Series, cost_per_leg: float, initial: float, ma: int = 200
) -> pd.Series:
    """Hold the asset while above its ``ma``-day average, else go to cash.

    The signal is taken at the close and acted on the next day (no lookahead).
    """
    ma_series = close.rolling(ma, min_periods=ma).mean()
    in_market = (close > ma_series).shift(1).fillna(False)
    r = close.pct_change().fillna(0.0)
    dates = close.index
    val = initial
    w = 0.0
    out = np.empty(len(dates))
    for i in range(len(dates)):
        if i > 0 and w > 0:
            val *= 1.0 + float(r.iloc[i])
        target = 1.0 if bool(in_market.iloc[i]) else 0.0
        if target != w:
            val *= 1.0 - cost_per_leg * abs(target - w)
            w = target
        out[i] = val
    return pd.Series(out, index=dates)


def run(config: BacktestConfig, *, use_cache: bool = True):
    """Load SPY (+AGG bonds) and build all allocation curves and their stats."""
    data = load_universe(["SPY", "AGG"], config.start, config.end, use_cache=use_cache)
    spy = data["SPY"]["close"]
    cost = config.commission_rate + config.slippage_rate
    cap = config.initial_capital

    curves: dict[str, pd.Series] = {}
    if "AGG" in data:
        rets = pd.DataFrame(
            {"SPY": spy.pct_change(), "AGG": data["AGG"]["close"].pct_change()}
        ).fillna(0.0)
        curves["Buy & Hold (100% stocks)"] = static_mix_curve(rets, {"SPY": 1.0}, cost, cap)
        curves["80/20 stocks/bonds"] = static_mix_curve(rets, {"SPY": 0.8, "AGG": 0.2}, cost, cap)
        curves["60/40 stocks/bonds"] = static_mix_curve(rets, {"SPY": 0.6, "AGG": 0.4}, cost, cap)
    else:  # bonds unavailable — still show stock-only curves
        rets = pd.DataFrame({"SPY": spy.pct_change()}).fillna(0.0)
        curves["Buy & Hold (100% stocks)"] = static_mix_curve(rets, {"SPY": 1.0}, cost, cap)
    curves["200-day trend (to cash)"] = trend_curve(spy, cost, cap)

    stats = {name: equity_stats(curve) for name, curve in curves.items()}
    return curves, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="swingit defensive allocation comparison")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", default="dist")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = BacktestConfig(start=args.start, end=args.end)
    curves, stats = run(config, use_cache=not args.no_cache)

    rows = []
    for name, s in stats.items():
        rows.append(
            {
                "strategy": name,
                "total_return": round(s["total_return"], 4),
                "cagr": round(s["cagr"], 4),
                "sharpe": round(s["sharpe"], 3),
                "max_drawdown": round(s["max_drawdown"], 4),
                "final_equity": round(s["final_equity"], 0),
            }
        )
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "allocations.csv", index=False)

    hdr = f"{'strategy':<28}{'tot.ret':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'final $':>12}"
    print("\n=== Defensive allocations (2019-2024) ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['strategy']:<28}{r['total_return'] * 100:>8.1f}%"
            f"{r['cagr'] * 100:>7.1f}%{r['sharpe']:>8.2f}"
            f"{r['max_drawdown'] * 100:>7.1f}%{'$' + format(int(r['final_equity']), ','):>12}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
