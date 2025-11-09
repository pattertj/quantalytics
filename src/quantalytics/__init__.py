"""Top-level package for Quantalytics."""

from .metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    downside_deviation,
    information_ratio,
    max_drawdown,
    rolling_beta,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
)
from .charts import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_rolling_metric,
    plot_return_distribution,
)
from .reports import generate_tearsheet

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "downside_deviation",
    "information_ratio",
    "max_drawdown",
    "rolling_beta",
    "sharpe_ratio",
    "sortino_ratio",
    "tail_ratio",
    "plot_cumulative_returns",
    "plot_drawdowns",
    "plot_rolling_metric",
    "plot_return_distribution",
    "generate_tearsheet",
]

__version__ = "0.1.0"
