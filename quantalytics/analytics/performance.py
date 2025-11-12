"""Performance metrics for portfolios and strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SQRT_TWO = math.sqrt(2.0)


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
    return freq if isinstance(freq, int) else fallback


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
    dd = downside_deviation(
        series, target=target_return, periods_per_year=periods_per_year
    )
    if dd == 0 or math.isnan(dd):
        return float("nan")
    return excess.mean() / dd * ann_factor


def cumulative_returns(returns: Iterable[float] | pd.Series) -> pd.Series:
    series = _to_series(returns)
    return (1 + series).cumprod() - 1


def _drawdown_path(series: pd.Series) -> pd.Series:
    """Return drawdown series from cumulative returns."""

    cum_returns = cumulative_returns(series)
    running_max = (1 + cum_returns).cummax()
    return (1 + cum_returns) / running_max - 1


def _drawdown_segments(series: pd.Series) -> tuple[pd.Series, list[pd.Series]]:
    drawdowns = _drawdown_path(series)
    underwater = drawdowns < 0
    if not bool(underwater.any()):
        return drawdowns, []

    segment_ids = (underwater != underwater.shift(fill_value=False)).cumsum()
    segments = [
        segment for _, segment in drawdowns[underwater].groupby(segment_ids[underwater])
    ]
    return drawdowns, segments


def max_drawdown(returns: Iterable[float] | pd.Series) -> float:
    """Compute the maximum drawdown from a series of returns."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")
    drawdowns, _ = _drawdown_segments(series)
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
    return float("nan") if mdd == 0 or math.isnan(mdd) else ann_return / mdd


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
    cum_return = float("nan") if series.empty else cumulative_returns(series).iloc[-1]
    ann_factor = _annualization_factor(periods_per_year)
    ann_ret = annualized_return(series, periods_per_year=ann_factor)
    ann_vol = annualized_volatility(series, periods_per_year=ann_factor)
    sharpe_ratio = sharpe(
        series, risk_free_rate=risk_free_rate, periods_per_year=ann_factor
    )
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
        sharpe=sharpe_ratio,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        downside_deviation=dd,
        cumulative_return=cum_return,
    )


def max_drawdown_percent(returns: Iterable[float] | pd.Series) -> float:
    """Maximum drawdown expressed as a positive percentage."""

    value = max_drawdown(returns)
    if math.isnan(value):
        return value
    return abs(value) * 100.0


def longest_drawdown_days(returns: Iterable[float] | pd.Series) -> float:
    """Longest continuous drawdown stretch measured in days (or periods if index lacks dates)."""

    series = _to_series(returns)
    if series.empty:
        return 0.0

    _, segments = _drawdown_segments(series)
    if not segments:
        return 0.0

    max_days = 0.0
    for segment in segments:
        index = segment.index
        if isinstance(index, pd.DatetimeIndex):
            delta = (index[-1] - index[0]).days + 1
            days = max(1, delta)
        else:
            days = len(segment)
        max_days = max(max_days, float(days))
    return max_days


def underwater_percent(returns: Iterable[float] | pd.Series) -> float:
    """Current drawdown (underwater) value expressed as a percentage."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    drawdowns, _ = _drawdown_segments(series)
    current = drawdowns.iloc[-1]
    if math.isnan(current):
        return current
    return abs(min(0.0, current)) * 100.0


def average_drawdown(returns: Iterable[float] | pd.Series) -> float:
    """Average drawdown depth (positive percentage)."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    _, segments = _drawdown_segments(series)
    if not segments:
        return 0.0

    depths = [abs(segment.min()) for segment in segments if not segment.empty]
    if not depths:
        return 0.0
    return float(np.mean(depths) * 100.0)


def average_drawdown_days(returns: Iterable[float] | pd.Series) -> float:
    """Average duration (in days) of drawdown periods."""

    series = _to_series(returns)
    if series.empty:
        return 0.0

    _, segments = _drawdown_segments(series)
    if not segments:
        return 0.0

    durations: list[float] = []
    for segment in segments:
        index = segment.index
        if isinstance(index, pd.DatetimeIndex):
            delta = (index[-1] - index[0]).days + 1
            days = max(1, delta)
        else:
            days = len(segment)
        durations.append(float(days))

    return float(np.mean(durations)) if durations else 0.0


def recovery_factor(returns: Iterable[float] | pd.Series) -> float:
    """Recovery factor = cumulative return / |max drawdown|."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    cum = cumulative_returns(series)
    if cum.empty:
        return float("nan")
    total = cum.iloc[-1]
    mdd = abs(max_drawdown(series))
    if mdd == 0 or math.isnan(mdd):
        return float("nan")
    return total / mdd


def ulcer_index(returns: Iterable[float] | pd.Series) -> float:
    """Ulcer index calculated from squared drawdowns."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    drawdowns, _ = _drawdown_segments(series)
    squared = np.square(np.clip(drawdowns, a_max=0.0, a_min=None) * 100.0)
    if len(squared) == 0:
        return 0.0
    return float(math.sqrt(np.mean(squared)))


def serenity_index(
    returns: Iterable[float] | pd.Series,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Serenity index approximated as annualized return (%) divided by Ulcer index."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    ann_factor = _annualization_factor(periods_per_year)
    ann_return = annualized_return(series, periods_per_year=ann_factor) * 100.0
    ulcer = ulcer_index(series)
    if math.isnan(ulcer):
        return float("nan")
    if ulcer == 0:
        return float("inf") if ann_return > 0 else 0.0
    return ann_return / ulcer


def romad(returns: Iterable[float] | pd.Series) -> float:
    """Return over maximum drawdown."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    total = cumulative_returns(series).iloc[-1]
    mdd = abs(max_drawdown(series))
    return float("nan") if mdd == 0 or math.isnan(mdd) else total / mdd


def _probability_inputs(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float,
    periods_per_year: Optional[int | str],
) -> tuple[pd.Series, float, float, float]:
    series = _to_series(returns)
    ann_factor = _annualization_factor(periods_per_year)
    excess = series - risk_free_rate / ann_factor
    std = excess.std(ddof=1)
    sharpe_like = float("nan") if std == 0 or math.isnan(std) else excess.mean() / std
    return excess, sharpe_like, excess.skew(), excess.kurtosis()


def prob_sharpe_ratio(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    target_sharpe: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Probability that the sample Sharpe ratio exceeds ``target_sharpe``."""

    excess, sr, skew, kurt = _probability_inputs(
        returns, risk_free_rate, periods_per_year
    )
    if math.isnan(sr) or len(excess) < 3:
        return float("nan")

    denominator = math.sqrt(max(1e-12, 1 - skew * sr + ((kurt - 1) / 4.0) * (sr**2)))
    z_score = (sr - target_sharpe) * math.sqrt(len(excess) - 1) / denominator
    return 0.5 * (1 + math.erf(z_score / SQRT_TWO))


def smart_sharpe_ratio(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Sharpe ratio adjusted for higher moments."""

    series = _to_series(returns)
    base = sharpe(
        series, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    if math.isnan(base):
        return base

    _, _, skew, kurt = _probability_inputs(series, risk_free_rate, periods_per_year)
    adjustment = 1 + (skew / 6.0) * base - (kurt / 24.0) * (base**2)
    return base * adjustment


def smart_sortino_ratio(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Sortino ratio adjusted for higher moments."""

    base = sortino_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        periods_per_year=periods_per_year,
    )
    if math.isnan(base):
        return base

    series = _to_series(returns)
    skew = series.skew()
    kurt = series.kurtosis()
    adjustment = 1 + (skew / 6.0) * base - (kurt / 24.0) * (base**2)
    return base * adjustment


def sortino_over_sqrt_two(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Sortino ratio scaled by sqrt(2)."""

    value = sortino_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        periods_per_year=periods_per_year,
    )
    return value if math.isnan(value) else value / SQRT_TWO


def smart_sortino_over_sqrt_two(
    returns: Iterable[float] | pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: Optional[int | str] = None,
) -> float:
    """Smart Sortino ratio scaled by sqrt(2)."""

    value = smart_sortino_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        periods_per_year=periods_per_year,
    )
    return value if math.isnan(value) else value / SQRT_TWO


def omega_ratio(
    returns: Iterable[float] | pd.Series,
    target_return: float = 0.0,
) -> float:
    """Omega ratio (upside partial moment divided by downside partial moment)."""

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    gains = np.clip(series - target_return, a_min=0, a_max=None)
    losses = np.clip(target_return - series, a_min=0, a_max=None)
    loss_sum = float(losses.sum())
    return float("inf") if loss_sum == 0 else float(gains.sum() / loss_sum)


def value_at_risk(
    returns: Iterable[float] | pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical value-at-risk (positive loss) at the given confidence level."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    quantile = float(series.quantile(1 - confidence))
    return max(0.0, -quantile)


def conditional_value_at_risk(
    returns: Iterable[float] | pd.Series,
    confidence: float = 0.95,
) -> float:
    """Conditional VaR (expected shortfall) beyond the VaR threshold."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    series = _to_series(returns)
    if series.empty:
        return float("nan")

    var_threshold = series.quantile(1 - confidence)
    tail = series[series <= var_threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


__all__ = [
    "PerformanceMetrics",
    "performance_summary",
    "sharpe",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "max_drawdown_percent",
    "longest_drawdown_days",
    "underwater_percent",
    "downside_deviation",
    "cumulative_returns",
    "annualized_return",
    "annualized_volatility",
    "romad",
    "prob_sharpe_ratio",
    "smart_sharpe_ratio",
    "smart_sortino_ratio",
    "sortino_over_sqrt_two",
    "smart_sortino_over_sqrt_two",
    "omega_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
    "average_drawdown",
    "average_drawdown_days",
    "recovery_factor",
    "ulcer_index",
    "serenity_index",
]
