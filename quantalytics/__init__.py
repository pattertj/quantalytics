"""Quantalytics: fast modern quantitative analytics library."""

from .metrics.performance import (
    performance_summary,
    PerformanceMetrics,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    downside_deviation,
)
from .charts.timeseries import (
    cumulative_returns_chart,
    rolling_volatility_chart,
    drawdown_chart,
)
from .reporting.tearsheet import (
    Tearsheet,
    TearsheetSection,
    TearsheetConfig,
    render_basic_tearsheet,
)

__all__ = [
    "PerformanceMetrics",
    "performance_summary",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "downside_deviation",
    "cumulative_returns_chart",
    "rolling_volatility_chart",
    "drawdown_chart",
    "Tearsheet",
    "TearsheetSection",
    "TearsheetConfig",
    "render_basic_tearsheet",
]
