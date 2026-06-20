"""Edge-validation experiments.

Runs the strategy under a few structural variations to answer one question:
*can the real-but-small per-trade edge be deployed into competitive returns?*

Scenarios:
  * baseline          — 40 names, regime filter on (the spec)
  * wide universe     — ~150 names, regime filter on
  * no regime         — 40 names, regime filter off
  * wide + no regime  — ~150 names, regime filter off

Each is compared against buy-and-hold SPY over the same window.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .benchmark import benchmark
from .config import BacktestConfig, StrategyConfig
from .data import load_universe
from .engine import run_backtest
from .metrics import compute_metrics
from .strategy import add_signals, compute_regime
from .universe import UNIVERSE, UNIVERSE_EXTENDED

# (label, tickers, use_regime_filter)
DEFAULT_SCENARIOS = [
    ("baseline (40, regime on)", UNIVERSE, True),
    ("wide universe (~150, regime on)", UNIVERSE_EXTENDED, True),
    ("no regime (40)", UNIVERSE, False),
    ("wide + no regime (~150)", UNIVERSE_EXTENDED, False),
]


def run_scenarios(
    raw_data: dict[str, pd.DataFrame],
    regime: pd.Series,
    base_config: BacktestConfig,
    scenarios,
) -> pd.DataFrame:
    """Pure core: run each scenario over already-loaded ``raw_data``.

    ``raw_data`` is ticker -> OHLCV without a signal column.
    """
    rows = []
    for label, tickers, use_regime in scenarios:
        subset = {t: raw_data[t] for t in tickers if t in raw_data}
        scfg = replace(base_config.strategy, use_regime_filter=use_regime)
        cfg = replace(base_config, strategy=scfg)
        signal_data = {t: add_signals(df, regime, scfg) for t, df in subset.items()}
        result = run_backtest(signal_data, cfg)
        m = compute_metrics(result)
        rows.append(
            {
                "scenario": label,
                "n_tickers": len(subset),
                "n_trades": m.n_trades,
                "avg_exposure": round(m.avg_exposure, 4),
                "total_return": round(m.total_return, 4),
                "cagr": round(m.cagr, 4),
                "sharpe": round(m.sharpe, 3),
                "max_drawdown": round(m.max_drawdown, 4),
            }
        )
    return pd.DataFrame(rows)


def run(config: BacktestConfig, *, use_cache: bool = True):
    """Load the extended universe once and run all default scenarios."""
    logger = logging.getLogger("swingit")
    needed = list(dict.fromkeys([config.regime_symbol, *UNIVERSE_EXTENDED]))
    logger.info("loading %d symbols for experiments", len(needed))
    data = load_universe(needed, config.start, config.end, use_cache=use_cache)

    spy_close = data[config.regime_symbol]["close"]
    regime = compute_regime(spy_close, config.strategy.regime_ma_period)
    raw_data = {t: df for t, df in data.items() if t != config.regime_symbol}

    df = run_scenarios(raw_data, regime, config, DEFAULT_SCENARIOS)
    _, bench_stats = benchmark(spy_close, config)
    return df, bench_stats


def _format_table(df: pd.DataFrame, bench_stats: dict, symbol: str) -> str:
    lines = []
    header = f"{'scenario':<32}{'trades':>8}{'expo':>8}{'tot.ret':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in df.iterrows():
        lines.append(
            f"{r['scenario']:<32}{r['n_trades']:>8}"
            f"{r['avg_exposure'] * 100:>7.1f}%"
            f"{r['total_return'] * 100:>8.1f}%"
            f"{r['cagr'] * 100:>7.1f}%"
            f"{r['sharpe']:>8.2f}"
            f"{r['max_drawdown'] * 100:>7.1f}%"
        )
    if bench_stats:
        lines.append("-" * len(header))
        lines.append(
            f"{'buy & hold ' + symbol:<32}{'-':>8}{'100.0%':>8}"
            f"{bench_stats['total_return'] * 100:>8.1f}%"
            f"{bench_stats['cagr'] * 100:>7.1f}%"
            f"{bench_stats['sharpe']:>8.2f}"
            f"{bench_stats['max_drawdown'] * 100:>7.1f}%"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="swingit edge-validation experiments")
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

    config = BacktestConfig(start=args.start, end=args.end, strategy=StrategyConfig())
    df, bench_stats = run(config, use_cache=not args.no_cache)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "experiments.csv", index=False)

    print("\n=== Edge-validation experiments ===")
    print(_format_table(df, bench_stats, config.regime_symbol))
    print(f"\nWrote {out / 'experiments.csv'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
