"""Self-contained HTML report generation.

Charts are rendered with matplotlib (Agg backend) and embedded as base64 PNGs
so the resulting ``.html`` is a single, portable file with no external assets —
ideal for publishing to GitHub Pages.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .engine import BacktestResult  # noqa: E402
from .metrics import Metrics, drawdown_series  # noqa: E402


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_chart(result: BacktestResult) -> str:
    eq = result.equity_curve
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(eq.index, eq.values, color="#1f77b4", lw=1.4)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def _drawdown_chart(result: BacktestResult) -> str:
    dd = drawdown_series(result.equity_curve) * 100.0
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def _returns_hist(result: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if not result.trades_df.empty:
        rets = result.trades_df["net_return"] * 100.0
        ax.hist(rets, bins=30, color="#2ca02c", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
    ax.set_title("Trade Return Distribution")
    ax.set_xlabel("Net return (%)")
    ax.set_ylabel("Trades")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def _per_ticker_chart(result: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if not result.trades_df.empty:
        by = result.trades_df.groupby("ticker")["pnl"].sum().sort_values()
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in by.values]
        ax.barh(by.index, by.values, color=colors)
    ax.set_title("Net P&L by Ticker ($)")
    ax.grid(alpha=0.3, axis="x")
    return _fig_to_b64(fig)


def _exit_reason_chart(result: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if not result.trades_df.empty:
        counts = result.trades_df["exit_reason"].value_counts()
        ax.bar(counts.index, counts.values, color="#9467bd", alpha=0.8)
        for x, v in enumerate(counts.values):
            ax.text(x, v, str(v), ha="center", va="bottom")
    ax.set_title("Exits by Reason")
    ax.set_ylabel("Trades")
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_b64(fig)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _metric_cards(m: Metrics, result: BacktestResult) -> str:
    cfg = result.config
    pf = "∞" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    cards = [
        ("Total Return", _pct(m.total_return)),
        ("CAGR", _pct(m.cagr)),
        ("Sharpe", f"{m.sharpe:.2f}"),
        ("Max Drawdown", _pct(m.max_drawdown)),
        ("Win Rate", _pct(m.win_rate)),
        ("Expectancy / trade", _pct(m.expectancy)),
        ("Profit Factor", pf),
        ("Trades", str(m.n_trades)),
        ("Avg Win", _pct(m.avg_win)),
        ("Avg Loss", _pct(m.avg_loss)),
        ("Final Equity", f"${m.final_equity:,.0f}"),
        ("Initial Capital", f"${cfg.initial_capital:,.0f}"),
    ]
    return "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in cards
    )


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}" />'


def build_html(result: BacktestResult, metrics: Metrics) -> str:
    cfg = result.config
    scfg = cfg.strategy
    eq = result.equity_curve
    period = (
        f"{eq.index[0].date()} → {eq.index[-1].date()}" if len(eq) else "n/a"
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rules = (
        f"Enter long when RSI({scfg.rsi_period}) &lt; {scfg.rsi_entry:g} and "
        f"price dropped ≥ {abs(scfg.drop_lookback_return) * 100:g}% over "
        f"{scfg.drop_lookback_days} days, only while {cfg.regime_symbol} is above "
        f"its {scfg.regime_ma_period}-day MA. "
        f"Exit at +{scfg.profit_target * 100:g}% target, "
        f"{scfg.stop_loss * 100:g}% stop (checked first), or after "
        f"{scfg.max_hold_days} trading days. Costs: "
        f"{cfg.commission_rate * 100:g}% commission + "
        f"{cfg.slippage_rate * 100:g}% slippage per leg. "
        f"Equal weight, max {cfg.max_positions} concurrent positions."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>swingit — Backtest Report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f5f6f8; color: #1a1a1a; }}
  header {{ background: #11253f; color: #fff; padding: 24px 32px; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .sub {{ opacity: 0.8; font-size: 13px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 24px 32px 64px; }}
  .rules {{ background: #fff; border-left: 4px solid #11253f; padding: 14px 18px;
            border-radius: 6px; font-size: 14px; line-height: 1.5; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; margin-bottom: 28px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 14px 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase;
                  letter-spacing: 0.04em; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .panel {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  img {{ max-width: 100%; height: auto; display: block; }}
  h2 {{ font-size: 16px; margin: 0 0 12px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ position: sticky; top: 0; background: #fafafa; }}
  .scroll {{ max-height: 420px; overflow: auto; }}
  footer {{ text-align: center; color: #9ca3af; font-size: 12px; padding: 24px; }}
  @media (max-width: 720px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>swingit — Phase 1 Backtest</h1>
  <div class="sub">Period {period} &nbsp;·&nbsp; {len(result.trades)} trades &nbsp;·&nbsp; generated {generated}</div>
</header>
<main>
  <div class="rules"><strong>Strategy:</strong> {rules}</div>

  <div class="cards">{_metric_cards(metrics, result)}</div>

  <div class="panel">
    <h2>Equity Curve</h2>
    {_img(_equity_chart(result), "equity curve")}
  </div>
  <div class="panel">
    <h2>Drawdown</h2>
    {_img(_drawdown_chart(result), "drawdown")}
  </div>

  <div class="grid2">
    <div class="panel">{_img(_returns_hist(result), "return distribution")}</div>
    <div class="panel">{_img(_exit_reason_chart(result), "exits by reason")}</div>
  </div>
  <div class="panel">{_img(_per_ticker_chart(result), "per ticker pnl")}</div>

  <div class="panel">
    <h2>Trades</h2>
    <div class="scroll">{_trades_table(result)}</div>
  </div>
</main>
<footer>Generated by swingit · for research only, not investment advice.</footer>
</body>
</html>
"""


def _trades_table(result: BacktestResult) -> str:
    df = result.trades_df
    if df.empty:
        return "<p>No trades generated.</p>"
    cols = [
        "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
        "bars_held", "exit_reason", "pnl", "net_return",
    ]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body_rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "net_return":
                v = f"{v * 100:.2f}%"
            elif c == "pnl":
                v = f"${v:,.0f}"
            cells.append(f"<td>{v}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_report(
    result: BacktestResult, metrics: Metrics, out_dir: str | Path
) -> tuple[Path, Path]:
    """Write ``index.html`` and ``trades.csv`` into ``out_dir``.

    Returns the paths to the HTML report and the CSV.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    html_path = out / "index.html"
    html_path.write_text(build_html(result, metrics), encoding="utf-8")

    csv_path = out / "trades.csv"
    result.trades_df.to_csv(csv_path, index=False)

    return html_path, csv_path
