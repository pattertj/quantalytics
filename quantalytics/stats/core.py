"""Descriptive statistics utilities."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..utils.timeseries import ensure_datetime_index


def _to_series(values: Iterable[float] | pd.Series) -> pd.Series:
    series = pd.Series(values).dropna()
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError("Input data must be numeric")
    return series


def _annualization_factor(freq: str | int | None, fallback: int = 252) -> int:
    if isinstance(freq, str):
        freq_map = {
            "D": 252,
            "B": 252,
            "W": 52,
            "M": 12,
            "Q": 4,
            "A": 1,
        }
        return freq_map.get(freq.upper(), fallback)
    if isinstance(freq, int):
        return freq
    return fallback


def skewness(returns: Iterable[float] | pd.Series) -> float:
    """Sample skewness of the return distribution."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return series.skew()


def skew(returns: Iterable[float] | pd.Series) -> float:
    """Alias for ``skewness`` to match abbreviated naming."""

    return skewness(returns)


def kurtosis(
    returns: Iterable[float] | pd.Series,
    fisher: bool = True,
) -> float:
    """Excess kurtosis (Pearson if ``fisher=False``)."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    value = float(series.kurtosis())
    if not fisher and not math.isnan(value):
        return value + 3.0
    return value


def total_return(returns: Iterable[float] | pd.Series) -> float:
    """Compound return over the full series."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return float(np.prod(1 + series) - 1)


def volatility(returns: Iterable[float] | pd.Series, ddof: int = 1) -> float:
    """Realized (non-annualized) volatility expressed as standard deviation."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return float(series.std(ddof=ddof))


def cagr(
    returns: Iterable[float] | pd.Series,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Compound annual growth rate."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    ann_factor = _annualization_factor(periods_per_year)
    gross_return = float(np.prod(1 + series))
    if gross_return <= 0:
        return float("nan")

    years = len(series) / ann_factor
    if years <= 0:
        return float("nan")

    return math.pow(gross_return, 1 / years) - 1


def cagr_percent(
    returns: Iterable[float] | pd.Series,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Compound annual growth rate expressed in percent."""

    value = cagr(returns, periods_per_year=periods_per_year)
    if math.isnan(value):
        return value
    return value * 100.0


def _period_returns(
    returns: Iterable[float] | pd.Series,
    period: str,
) -> pd.Series:
    series = ensure_datetime_index(_to_series(returns))
    period_map = {
        "day": "D",
        "daily": "D",
        "week": "W",
        "weekly": "W",
        "month": "M",
        "monthly": "M",
        "quarter": "Q",
        "quarterly": "Q",
        "year": "A",
        "annual": "A",
        "yearly": "A",
    }
    freq = period_map.get(period.lower())
    if freq is None:
        raise ValueError("period must be one of day/week/month/quarter/year")
    grouped = (1 + series).groupby(pd.Grouper(freq=freq)).prod() - 1
    return grouped.dropna()


def best_period_return(
    returns: Iterable[float] | pd.Series,
    period: str = "day",
) -> float:
    """Best compounded return for the supplied period, expressed as a percentage."""

    grouped = _period_returns(returns, period)
    if grouped.empty:
        return float("nan")
    return float(grouped.max() * 100.0)


def worst_period_return(
    returns: Iterable[float] | pd.Series,
    period: str = "day",
) -> float:
    """Worst compounded return for the supplied period, expressed as a percentage."""

    grouped = _period_returns(returns, period)
    if grouped.empty:
        return float("nan")
    return float(grouped.min() * 100.0)


__all__ = [
    "skewness",
    "skew",
    "kurtosis",
    "total_return",
    "volatility",
    "cagr",
    "cagr_percent",
    "best_period_return",
    "worst_period_return",
]
