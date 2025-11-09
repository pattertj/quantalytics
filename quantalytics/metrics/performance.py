"""Performance metrics for portfolios and strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for common performance statistics."""

    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    downside_deviation: float
    cumulative_return: float

    def as_dict(self) -> dict[str, float]:
        """Return a dictionary representation suitable for DataFrames."""

        return {
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "downside_deviation": self.downside_deviation,
            "cumulative_return": self.cumulative_return,
        }


def _to_series(returns: Iterable[float] | pd.Series) -> pd.Series:
    series = pd.Series(returns)
    series = series.dropna()
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError("Returns must be numeric")
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


def sharpe(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Calculate the annualized Sharpe ratio."""

    series = _to_series(returns)
    excess_returns = series - risk_free_rate / _annualization_factor(periods_per_year)
    std = excess_returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    ann_factor = math.sqrt(_annualization_factor(periods_per_year))
    return excess_returns.mean() / std * ann_factor


def downside_deviation(
    returns: Iterable[float] | pd.Series,
    target: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Calculate annualized downside deviation from a target return."""

    series = _to_series(returns)
    ann_factor = _annualization_factor(periods_per_year)
    downside = np.clip(target / ann_factor - series, a_min=0, a_max=None)
    variance = np.mean(downside**2)
    return math.sqrt(variance) * math.sqrt(ann_factor)


def sortino_ratio(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Calculate the annualized Sortino ratio."""

    series = _to_series(returns)
    ann_factor = math.sqrt(_annualization_factor(periods_per_year))
    excess = series - risk_free_rate / _annualization_factor(periods_per_year)
    dd = downside_deviation(series, target=target_return, periods_per_year=periods_per_year)
    if dd == 0 or math.isnan(dd):
        return float("nan")
    return excess.mean() / dd * ann_factor


def cumulative_returns(returns: Iterable[float] | pd.Series) -> pd.Series:
    series = _to_series(returns)
    return (1 + series).cumprod() - 1


def max_drawdown(returns: Iterable[float] | pd.Series) -> float:
    """Compute the maximum drawdown from a series of returns."""

    cum_returns = cumulative_returns(returns)
    running_max = (1 + cum_returns).cummax()
    drawdowns = (1 + cum_returns) / running_max - 1
    return drawdowns.min()


def calmar_ratio(
    returns: Iterable[float] | pd.Series,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Calculate the Calmar ratio using annualized return and max drawdown."""

    series = _to_series(returns)
    ann_factor = _annualization_factor(periods_per_year)
    ann_return = (1 + series.mean()) ** ann_factor - 1
    mdd = abs(max_drawdown(series))
    if mdd == 0 or math.isnan(mdd):
        return float("nan")
    return ann_return / mdd


def annualized_volatility(
    returns: Iterable[float] | pd.Series,
    periods_per_year: Optional[int | str] = None,
) -> float:
    series = _to_series(returns)
    ann_factor = _annualization_factor(periods_per_year)
    return series.std(ddof=1) * math.sqrt(ann_factor)


def annualized_return(
    returns: Iterable[float] | pd.Series,
    periods_per_year: Optional[int | str] = None,
) -> float:
    series = _to_series(returns)
    ann_factor = _annualization_factor(periods_per_year)
    return (1 + series.mean()) ** ann_factor - 1


def performance_summary(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> PerformanceMetrics:
    """Generate a collection of core performance metrics."""

    series = _to_series(returns)
    cum_return = cumulative_returns(series).iloc[-1] if not series.empty else float("nan")
    ann_factor = _annualization_factor(periods_per_year)
    ann_ret = annualized_return(series, periods_per_year=ann_factor)
    ann_vol = annualized_volatility(series, periods_per_year=ann_factor)
    sharpe = sharpe(series, risk_free_rate=risk_free_rate, periods_per_year=ann_factor)
    sortino = sortino_ratio(
        series,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        periods_per_year=ann_factor,
    )
    calmar = calmar_ratio(series, periods_per_year=ann_factor)
    mdd = max_drawdown(series)
    dd = downside_deviation(series, target=target_return, periods_per_year=ann_factor)

    return PerformanceMetrics(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        downside_deviation=dd,
        cumulative_return=cum_return,
    )


__all__ = [
    "PerformanceMetrics",
    "performance_summary",
    "sharpe",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "downside_deviation",
    "cumulative_returns",
    "annualized_return",
    "annualized_volatility",
]
