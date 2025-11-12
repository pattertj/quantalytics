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


def skew(returns: Iterable[float] | pd.Series) -> float:
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


def volatility(
    returns: Iterable[float] | pd.Series,
    *,
    periods: int = 365,
    annualize: bool = True,
) -> float:
    """Realized volatility as standard deviation, optionally annualized."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    std = float(series.std(ddof=1))
    if annualize:
        return std * math.sqrt(periods)
    return std


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
    period: Optional[str] = None,
) -> pd.Series:
    series = ensure_datetime_index(_to_series(returns))
    if period is None:
        return series
    period_map = {
        "day": "D",
        "daily": "D",
        "week": "W",
        "weekly": "W",
        "month": "ME",
        "monthly": "ME",
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
    period: str | None = None,
) -> float:
    """Best compounded return for the supplied period, expressed as a percentage."""

    grouped = _period_returns(returns, period)
    if grouped.empty:
        return float("nan")
    return float(grouped.max() * 100.0)


def worst_period_return(
    returns: Iterable[float] | pd.Series,
    period: str | None = None,
) -> float:
    """Worst compounded return for the supplied period, expressed as a percentage."""

    grouped = _period_returns(returns, period)
    if grouped.empty:
        return float("nan")
    return float(grouped.min() * 100.0)


def win_rate(
    returns: Iterable[float] | pd.Series,
    period: str | None = None,
) -> float:
    """Win rate (percentage of positive periods) for the given frequency."""

    grouped = _period_returns(returns, period)
    if grouped.empty:
        return float("nan")
    wins = (grouped > 0).sum()
    return float(wins / len(grouped))


def avg_loss(returns: Iterable[float] | pd.Series) -> float:
    """Average loss magnitude across periods."""

    series = _to_series(returns)
    losses = series[series < 0]
    if losses.empty:
        return 0.0
    return float(losses.mean())


def avg_win(returns: Iterable[float] | pd.Series) -> float:
    """Average win magnitude across periods."""

    series = _to_series(returns)
    wins = series[series > 0]
    if wins.empty:
        return 0.0
    return float(wins.mean())


def payoff_ratio(returns: Iterable[float] | pd.Series) -> float:
    """Measures the payoff ratio (average win / average loss)."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    wins = series[series > 0]
    losses = series[series < 0]
    if losses.empty:
        return float("nan") if wins.empty else float("inf")
    avg_win_value = 0.0 if wins.empty else float(wins.mean())
    avg_loss_value = float(abs(losses).mean())
    if avg_loss_value == 0:
        return float("inf") if avg_win_value > 0 else float("nan")
    return avg_win_value / avg_loss_value


def win_loss_ratio(returns: Iterable[float] | pd.Series) -> float:
    """Shorthand for payoff_ratio function."""
    return payoff_ratio(returns)


def profit_ratio(returns: Iterable[float] | pd.Series) -> float:
    """Measures the profit ratio (win ratio / loss ratio)."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    total = len(series)
    win_ratio = series.gt(0).sum() / total
    loss_ratio = series.lt(0).sum() / total
    if loss_ratio == 0:
        return float("inf") if win_ratio > 0 else float("nan")
    return float(win_ratio / loss_ratio)


def max_consecutive_losses(returns: Iterable[float] | pd.Series) -> int:
    """Maximum number of back-to-back losing periods."""

    series = _to_series(returns)
    max_run = current = 0
    for value in series:
        if value < 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def max_consecutive_wins(returns: Iterable[float] | pd.Series) -> int:
    """Maximum number of back-to-back winning periods."""

    series = _to_series(returns)
    max_run = current = 0
    for value in series:
        if value > 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def gain_to_pain_ratio(returns: Iterable[float] | pd.Series) -> float:
    """Net return relative to the magnitude of losses."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    total_gain = float(series.sum())
    loss_sum = float(abs(series[series < 0].sum()))
    if loss_sum == 0:
        return float("inf") if total_gain > 0 else float("nan")
    return total_gain / loss_sum


def kelly_criterion(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Fraction of capital to allocate per the Kelly criterion."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    ann_factor = _annualization_factor(periods_per_year)
    excess = series - risk_free_rate / ann_factor
    if excess.empty:
        return float("nan")
    avg = excess.mean()
    variance = excess.var(ddof=1)
    if variance == 0 or math.isnan(variance):
        return float("inf") if avg > 0 else float("nan")
    return float(avg / variance)


def information_ratio(
    returns: Iterable[float] | pd.Series,
    benchmark: Iterable[float] | pd.Series,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Annualized information ratio relative to a benchmark."""

    strata = _to_series(returns)
    bench = _to_series(benchmark)
    if strata.empty or bench.empty:
        return float("nan")
    joined = strata.align(bench, join="inner")
    aligned_strategy, aligned_benchmark = joined
    diff = aligned_strategy - aligned_benchmark
    diff = diff.dropna()
    if diff.empty:
        return float("nan")
    tracking_error = diff.std(ddof=1)
    if tracking_error == 0 or math.isnan(tracking_error):
        return float("nan")
    ann_factor = _annualization_factor(periods_per_year)
    return (diff.mean() / tracking_error) * math.sqrt(ann_factor)


def r_squared(
    returns: Iterable[float] | pd.Series,
    benchmark: Iterable[float] | pd.Series,
) -> float:
    """Coefficient of determination between strategy returns and a benchmark."""

    strata = _to_series(returns)
    bench = _to_series(benchmark)
    if strata.empty or bench.empty:
        return float("nan")
    aligned_strategy, aligned_benchmark = strata.align(bench, join="inner")
    aligned_strategy = aligned_strategy.dropna()
    aligned_benchmark = aligned_benchmark.dropna()
    if aligned_strategy.empty or aligned_benchmark.empty:
        return float("nan")
    mean = aligned_strategy.mean()
    total_sum_squares = ((aligned_strategy - mean) ** 2).sum()
    residual_sum_squares = ((aligned_strategy - aligned_benchmark) ** 2).sum()
    if total_sum_squares == 0:
        return 1.0 if residual_sum_squares == 0 else float("nan")
    return float(1 - residual_sum_squares / total_sum_squares)


def risk_of_ruin(returns: Iterable[float] | pd.Series) -> float:
    """
    Estimate risk-of-ruin via a simplified formula based on win percentage.
    """

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    wins = win_rate(series)
    if wins <= -1 or wins >= 1:
        return 0.0 if wins >= 1 else 1.0
    return ((1 - wins) / (1 + wins)) ** len(series)


def profit_factor(returns: Iterable[float] | pd.Series):
    """Measures the profit ratio (wins/loss)"""
    returns_series = _to_series(returns)
    return abs(
        returns_series[returns_series >= 0].sum()
        / returns_series[returns_series < 0].sum()
    )


def cpc_index(returns: Iterable[float] | pd.Series):
    """
    Measures the cpc ratio
    (profit factor * win % * win loss ratio)
    """
    return profit_factor(returns) * win_rate(returns) * win_loss_ratio(returns)


def tail_ratio(returns: Iterable[float] | pd.Series, cutoff=0.95):
    """
    Measures the ratio between the right
    (95%) and left tail (5%).
    """
    returns_series = _to_series(returns)
    return abs(returns_series.quantile(cutoff) / returns_series.quantile(1 - cutoff))


def common_sense_ratio(returns: Iterable[float] | pd.Series):
    """Measures the common sense ratio (profit factor * tail ratio)"""
    return profit_factor(returns) * tail_ratio(returns)


def omega_ratio(
    returns: Iterable[float] | pd.Series,
    threshold: float = 0.0,
) -> float:
    """Omega ratio comparing threshold-exceeding gains to threshold-breaching losses."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    gains = series[series > threshold] - threshold
    losses = threshold - series[series < threshold]
    gain_sum = float(gains.sum())
    loss_sum = float(losses.sum())
    if loss_sum == 0:
        return float("inf") if gain_sum > 0 else float("nan")
    return gain_sum / loss_sum


__all__ = [
    "skew",
    "kurtosis",
    "total_return",
    "volatility",
    "cagr",
    "cagr_percent",
    "best_period_return",
    "worst_period_return",
    "win_rate",
    "avg_loss",
    "avg_win",
    "max_consecutive_losses",
    "max_consecutive_wins",
    "gain_to_pain_ratio",
    "kelly_criterion",
    "information_ratio",
    "r_squared",
    "risk_of_ruin",
    "win_loss_ratio",
    "profit_factor",
    "cpc_index",
    "tail_ratio",
    "common_sense_ratio",
    "omega_ratio",
    "payoff_ratio",
    "profit_ratio",
]
