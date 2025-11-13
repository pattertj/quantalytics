"""Utility helpers for Quantalytics."""

from .timeseries import (
    _count_consecutive,
    aggregate_returns,
    ensure_datetime_index,
    log_returns,
    multi_shift,
    normalize_returns,
    rolling_statistic,
)

__all__: list[str] = [
    "ensure_datetime_index",
    "rolling_statistic",
    "normalize_returns",
    "aggregate_returns",
    "_count_consecutive",
    "log_returns",
    "multi_shift",
]
