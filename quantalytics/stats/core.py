"""Descriptive statistics utilities."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd


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


def kurtosis(
    returns: Iterable[float] | pd.Series,
    fisher: bool = True,
) -> float:
    """Excess kurtosis (Pearson if ``fisher=False``)."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return series.kurtosis(fisher=fisher)


def total_return(returns: Iterable[float] | pd.Series) -> float:
    """Compound return over the full series."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return float(np.prod(1 + series) - 1)


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


__all__ = ["skewness", "kurtosis", "total_return", "cagr"]
