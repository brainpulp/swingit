"""Value-investing comparison — done honestly.

Backtesting stock-picking-by-valuation needs *historical* (point-in-time)
fundamentals, which free price data does not provide; using today's P/E ratios
to pick trades in the past is hindsight bias. So instead we test value the
honest way: buy-and-hold real value-index ETFs that actually implemented the
strategy live, versus the plain S&P 500.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .allocations import static_mix_curve
from .config import BacktestConfig
from .data import load_universe
from .metrics import equity_stats

logger = logging.getLogger(__name__)

# symbol -> human label
SYMBOLS = {
    "SPY": "S&P 500 (SPY)",
    "IVE": "S&P 500 Value (IVE)",
    "VTV": "Large-cap Value (VTV)",
    "RPV": "Pure Value (RPV)",
}


def compare(
    closes: dict[str, pd.Series],
    cost_per_leg: float,
    initial: float,
    labels: dict[str, str] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, dict]]:
    """Buy-and-hold equity curve + stats for each symbol's close series."""
    labels = labels or {}
    curves: dict[str, pd.Series] = {}
    for sym, close in closes.items():
        rets = pd.DataFrame({sym: close.pct_change()}).fillna(0.0)
        curves[labels.get(sym, sym)] = static_mix_curve(
            rets, {sym: 1.0}, cost_per_leg, initial
        )
    stats = {name: equity_stats(curve) for name, curve in curves.items()}
    return curves, stats


def run(config: BacktestConfig, *, use_cache: bool = True):
    syms = list(SYMBOLS)
    data = load_universe(syms, config.start, config.end, use_cache=use_cache)
    closes = {s: data[s]["close"] for s in syms if s in data}
    cost = config.commission_rate + config.slippage_rate
    return compare(closes, cost, config.initial_capital, SYMBOLS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="swingit value-investing comparison")
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
    _, stats = run(config, use_cache=not args.no_cache)

    rows = [{"strategy": name, **s} for name, s in stats.items()]
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "value.csv", index=False)

    hdr = f"{'strategy':<24}{'tot.ret':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'final $':>12}"
    print("\n=== Value investing vs the S&P 500 (2019-2024) ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['strategy']:<24}{r['total_return'] * 100:>8.1f}%"
            f"{r['cagr'] * 100:>7.1f}%{r['sharpe']:>8.2f}"
            f"{r['max_drawdown'] * 100:>7.1f}%{'$' + format(int(r['final_equity']), ','):>12}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
