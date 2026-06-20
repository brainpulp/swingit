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

from .benchmark import benchmark
from .config import BacktestConfig, StrategyConfig
from .data import load_universe
from .engine import run_backtest
from .metrics import compute_metrics
from .report import write_report
from .strategy import add_signals, compute_regime
from .sweep import run_sweep
from .universe import UNIVERSE


def build_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        start=args.start,
        end=args.end,
        strategy=StrategyConfig(),
    )


def prepare(config: BacktestConfig, symbols: list[str], *, use_cache: bool = True):
    """Load data and compute the regime. Returns (raw_data, regime, spy_close).

    ``raw_data`` excludes the regime symbol and carries *no* signal column, so it
    can be reused by the parameter sweep.
    """
    logger = logging.getLogger("swingit")
    needed = list(dict.fromkeys([config.regime_symbol, *symbols]))
    logger.info("loading %d symbols", len(needed))
    data = load_universe(needed, config.start, config.end, use_cache=use_cache)

    if config.regime_symbol not in data:
        raise RuntimeError(f"regime symbol {config.regime_symbol} failed to load")
    spy_close = data[config.regime_symbol]["close"]
    regime = compute_regime(spy_close, config.strategy.regime_ma_period)
    raw_data = {t: df for t, df in data.items() if t != config.regime_symbol}
    return raw_data, regime, spy_close


def run(config: BacktestConfig, symbols: list[str], *, use_cache: bool = True):
    """Load data, attach signals, and run the backtest. Returns a BacktestResult."""
    raw_data, regime, _ = prepare(config, symbols, use_cache=use_cache)
    signal_data = {t: add_signals(df, regime, config.strategy) for t, df in raw_data.items()}
    return run_backtest(signal_data, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="swingit Phase 1 backtest")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--output", default="dist", help="output directory")
    parser.add_argument("--no-cache", action="store_true", help="ignore data cache")
    parser.add_argument("--no-sweep", action="store_true", help="skip the parameter sweep")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("swingit")

    config = build_config(args)
    raw_data, regime, spy_close = prepare(config, UNIVERSE, use_cache=not args.no_cache)

    signal_data = {t: add_signals(df, regime, config.strategy) for t, df in raw_data.items()}
    result = run_backtest(signal_data, config)
    metrics = compute_metrics(result)

    benchmark_curve, benchmark_stats = benchmark(spy_close, config)

    sweep_df = None
    if not args.no_sweep:
        logger.info("running parameter sweep")
        sweep_df = run_sweep(raw_data, regime, config)
        (Path(args.output)).mkdir(parents=True, exist_ok=True)

    html_path, csv_path = write_report(
        result, metrics, args.output,
        benchmark_curve=benchmark_curve,
        benchmark_stats=benchmark_stats,
        sweep_df=sweep_df,
    )

    summary = metrics.as_dict()
    summary["benchmark"] = benchmark_stats
    (Path(args.output) / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Backtest summary ===")
    for k, v in metrics.as_dict().items():
        print(f"  {k:>16}: {v}")
    if benchmark_stats:
        print(f"\n=== Buy & Hold {config.regime_symbol} ===")
        for k, v in benchmark_stats.items():
            print(f"  {k:>16}: {v}")
        verdict = "OUTPERFORMS" if metrics.cagr > benchmark_stats["cagr"] else "UNDERPERFORMS"
        print(f"\n  -> strategy {verdict} buy & hold on CAGR")
    print(f"\nReport: {html_path}\nTrades: {csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
