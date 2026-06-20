"""Command-line entry point: run the Phase 1 backtest end to end.

    python -m swingit.cli --output dist

Downloads data (cached), generates signals, runs the event-driven engine,
computes metrics, and writes a self-contained HTML report + per-trade CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import BacktestConfig, StrategyConfig
from .data import load_universe
from .engine import run_backtest
from .metrics import compute_metrics
from .report import write_report
from .strategy import add_signals, compute_regime
from .universe import UNIVERSE


def build_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        start=args.start,
        end=args.end,
        strategy=StrategyConfig(),
    )


def run(config: BacktestConfig, symbols: list[str], *, use_cache: bool = True):
    """Load data, attach signals, and run the backtest. Returns a BacktestResult."""
    logger = logging.getLogger("swingit")

    needed = list(dict.fromkeys([config.regime_symbol, *symbols]))
    logger.info("loading %d symbols", len(needed))
    data = load_universe(needed, config.start, config.end, use_cache=use_cache)

    if config.regime_symbol not in data:
        raise RuntimeError(f"regime symbol {config.regime_symbol} failed to load")
    regime = compute_regime(
        data[config.regime_symbol]["close"], config.strategy.regime_ma_period
    )

    signal_data = {
        t: add_signals(df, regime, config.strategy)
        for t, df in data.items()
        if t != config.regime_symbol
    }
    return run_backtest(signal_data, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="swingit Phase 1 backtest")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--output", default="dist", help="output directory")
    parser.add_argument("--no-cache", action="store_true", help="ignore data cache")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = build_config(args)
    result = run(config, UNIVERSE, use_cache=not args.no_cache)
    metrics = compute_metrics(result)

    html_path, csv_path = write_report(result, metrics, args.output)

    summary = metrics.as_dict()
    (Path(args.output) / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Backtest summary ===")
    for k, v in summary.items():
        print(f"  {k:>16}: {v}")
    print(f"\nReport: {html_path}\nTrades: {csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
