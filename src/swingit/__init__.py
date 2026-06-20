"""swingit — automated swing trading system for Interactive Brokers.

Phase 1 (this package) implements the backtesting layer for a mean-reversion
strategy on S&P 500 large caps. Later phases (paper trading, live trading with
manual confirmation, and full automation) build on top of the same strategy
definition.
"""

__version__ = "0.1.0"

from .config import BacktestConfig, StrategyConfig

__all__ = ["BacktestConfig", "StrategyConfig", "__version__"]
