# swingit

Automated **swing trading** system for Interactive Brokers (IBKR).

The strategy is a **mean-reversion** play on S&P 500 large caps:

> Enter **long** when **RSI(14) < 28** *and* price dropped **≥ 4% over 3 trading
> days**, but only while **SPY is above its 200-day moving average** (regime
> filter). Exit on a **+4% profit target**, a **−3% stop loss**, or a **5
> trading-day time limit**. Costs modelled at **0.1% commission + 0.1% slippage
> per leg**.

## Roadmap

The project is built in four phases. **Only Phase 1 is implemented today.**

1. **Backtest** ✅ — event-driven historical simulation + HTML report. *(this repo)*
2. **Paper trading** with manual confirmation — *planned*
3. **Live trading** with manual confirmation — *planned*
4. **Full automation** — *planned*

## Phase 1 — Backtest

* **Universe:** 40 S&P 500 large caps (`src/swingit/universe.py`); SPY drives the
  regime filter.
* **Period:** 2019–2024, daily bars from [yfinance] (cached locally).
* **Engine:** event-driven, **no lookahead** — signals are evaluated at the
  *close* and orders fill at the *next open*. Intraday high/low drive the
  target/stop, with the **stop checked first** (a gap through a level fills at
  the worse open price). **Equal-weight** portfolio with a **max-concurrent-
  positions** cap.
* **Output:** a **self-contained HTML report** (`dist/index.html`) and a
  **per-trade CSV** (`dist/trades.csv`).

### Metrics reported

Win rate · expectancy · profit factor · Sharpe · max drawdown · CAGR · total
return — plus charts for the equity curve, drawdown, trade-return distribution,
per-ticker P&L, and exit-reason breakdown.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the backtest -> writes dist/index.html, dist/trades.csv, dist/metrics.json
python -m swingit.cli --output dist

# Open the report
open dist/index.html        # macOS  (xdg-open on Linux)
```

Useful flags: `--start`, `--end`, `--capital`, `--max-positions`, `--no-cache`,
`-v`.

## Tests

```bash
pytest -q
```

Tests use synthetic OHLCV fixtures (no network) and cover the indicators,
signal logic, every engine exit path (target / stop / stop-checked-first / gap /
time / position cap / costs), the metrics, and the report.

## Continuous backtest on GitHub

`.github/workflows/backtest.yml` runs the test suite, executes the backtest, and
**publishes the HTML report to GitHub Pages** on every push (and weekly). Phase 1
therefore lives and runs entirely on GitHub.

> **Enable Pages once:** repo **Settings → Pages → Build and deployment →
> Source: GitHub Actions.** The report is then served at the Pages URL shown in
> the workflow's `deploy` job.

## Project layout

```
src/swingit/
  config.py      strategy & backtest parameters
  universe.py    the 40-ticker trading universe
  data.py        yfinance loading + on-disk cache
  indicators.py  RSI (Wilder), SMA, n-day return
  strategy.py    signal generation + regime filter
  engine.py      event-driven backtest engine
  metrics.py     performance metrics
  report.py      self-contained HTML report + CSV
  cli.py         end-to-end runner (python -m swingit.cli)
tests/           synthetic-data unit tests
```

## Disclaimer

For research and educational purposes only. Nothing here is investment advice.
Backtested results do not guarantee future performance.

[yfinance]: https://github.com/ranaroussi/yfinance
