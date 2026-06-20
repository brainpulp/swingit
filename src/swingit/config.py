"""Configuration objects for the strategy and backtest engine.

Keeping the strategy parameters in one place means the same definition can be
reused by later phases (paper / live trading) without restating the rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StrategyConfig:
    """Mean-reversion entry / exit rules.

    Entry (signal evaluated at the *close* of day T, executed at the *open* of
    day T+1):
      * RSI(``rsi_period``) < ``rsi_entry``
      * Price dropped at least ``drop_lookback_return`` over
        ``drop_lookback_days`` trading days.
      * Regime filter: SPY close is above its ``regime_ma_period``-day moving
        average.

    Exit (checked intraday on every day the position is held, starting with the
    entry day):
      * Stop loss at ``stop_loss`` (checked first, the conservative assumption).
      * Profit target at ``profit_target``.
      * Time limit: force exit at the close once the position has been held for
        ``max_hold_days`` trading days.
    """

    rsi_period: int = 14
    rsi_entry: float = 28.0
    drop_lookback_days: int = 3
    drop_lookback_return: float = -0.04  # -4% over the lookback window
    regime_ma_period: int = 200

    profit_target: float = 0.04  # +4%
    stop_loss: float = -0.03  # -3%
    max_hold_days: int = 5  # trading days, entry day counts as day 1

    def __post_init__(self) -> None:
        if self.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        if self.stop_loss >= 0:
            raise ValueError("stop_loss must be negative")
        if self.max_hold_days < 1:
            raise ValueError("max_hold_days must be >= 1")


@dataclass(frozen=True)
class BacktestConfig:
    """Portfolio-level and cost assumptions for the backtest engine."""

    initial_capital: float = 100_000.0
    max_positions: int = 5  # max concurrent positions (equal weight)

    commission_rate: float = 0.001  # 0.1% per leg
    slippage_rate: float = 0.001  # 0.1% per leg

    start: str = "2019-01-01"
    end: str = "2024-12-31"

    regime_symbol: str = "SPY"

    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @property
    def round_trip_cost(self) -> float:
        """Approximate total cost of a round trip as a fraction of notional."""
        return 2 * (self.commission_rate + self.slippage_rate)
