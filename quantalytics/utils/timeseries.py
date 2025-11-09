"""Timeseries utilities used across Quantalytics."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def ensure_datetime_index(series: Iterable[float] | pd.Series) -> pd.Series:
    """Ensure the series has a DatetimeIndex."""

    series = pd.Series(series)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    return series.sort_index()


def rolling_statistic(series: pd.Series, window: int, function: str = "mean") -> pd.Series:
    """Compute a rolling statistic with sensible defaults."""

    if window <= 0:
        raise ValueError("window must be positive")
    if function not in {"mean", "std", "median"}:
        raise ValueError("Unsupported rolling function")

    rolling = series.rolling(window=window, min_periods=max(2, window // 2))
    if function == "mean":
        return rolling.mean()
    if function == "std":
        return rolling.std(ddof=1)
    return rolling.median()


__all__ = ["ensure_datetime_index", "rolling_statistic"]
