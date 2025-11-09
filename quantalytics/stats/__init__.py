"""Statistical primitives for Quantalytics."""

from .core import cagr, cagr_percent, kurtosis, skew, skewness, total_return, volatility

__all__ = [
    "skewness",
    "skew",
    "kurtosis",
    "total_return",
    "volatility",
    "cagr",
    "cagr_percent",
]
