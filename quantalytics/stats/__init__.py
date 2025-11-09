"""Statistical primitives for Quantalytics."""

from .core import (
    best_period_return,
    cagr,
    cagr_percent,
    kurtosis,
    skew,
    skewness,
    total_return,
    volatility,
    win_rate,
    worst_period_return,
)

__all__ = [
    "best_period_return",
    "skewness",
    "skew",
    "kurtosis",
    "total_return",
    "volatility",
    "cagr",
    "cagr_percent",
    "win_rate",
    "worst_period_return",
]
