"""Market-data loading via yfinance, with a small on-disk cache.

Each symbol is cached as a parquet (or CSV fallback) file so repeated backtest
runs — and CI — don't re-download the same history.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("SWINGIT_CACHE_DIR", ".cache/data"))

# Canonical OHLCV column names used throughout the package.
OHLCV = ["open", "high", "low", "close", "volume"]


def _cache_path(symbol: str, start: str, end: str) -> Path:
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe}_{start}_{end}.parquet"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten/rename a yfinance frame to lowercase OHLCV columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # yfinance uses "adj close"; we trade on raw OHLC but keep close consistent.
    cols = [c for c in OHLCV if c in df.columns]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.dropna(how="all")


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # pragma: no cover - parquet engine may be missing
        csv = path.with_suffix(".csv")
        if csv.exists():
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception:  # pragma: no cover - fall back to CSV if no parquet engine
        df.to_csv(path.with_suffix(".csv"))


def load_symbol(
    symbol: str, start: str, end: str, *, use_cache: bool = True
) -> pd.DataFrame:
    """Load daily OHLCV for a single symbol, using the cache when possible."""
    path = _cache_path(symbol, start, end)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None and not cached.empty:
            logger.debug("cache hit for %s", symbol)
            return cached

    import yfinance as yf  # imported lazily so tests don't need the network

    logger.info("downloading %s (%s -> %s)", symbol, start, end)
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    df = _normalise(raw)
    if use_cache:
        _write_cache(path, df)
    return df


def load_universe(
    symbols: list[str], start: str, end: str, *, use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for many symbols; symbols that fail are skipped with a warning."""
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            out[symbol] = load_symbol(symbol, start, end, use_cache=use_cache)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("skipping %s: %s", symbol, exc)
    if not out:
        raise RuntimeError("failed to load any symbols")
    return out
