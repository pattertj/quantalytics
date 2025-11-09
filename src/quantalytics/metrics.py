"""Performance and risk metrics for quantitative analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(returns: pd.Series | pd.DataFrame | np.ndarray) -> pd.Series:
    """Validate and convert input returns to a :class:`pandas.Series`."""

    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("DataFrame inputs must have a single column of returns")
        series = returns.iloc[:, 0]
    elif isinstance(returns, pd.Series):
        series = returns
    else:
        series = pd.Series(np.asarray(returns))

    if series.isna().all():
        raise ValueError("Return series contains only NaN values")

    series = series.dropna()
    if series.empty:
        raise ValueError("Return series must contain at least one non-NaN value")
    return series


def annualized_return(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized return of a returns series."""

    series = _to_series(returns)
    compounded_growth = (1 + series).prod()
    n_periods = series.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualized_volatility(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Annualized standard deviation of returns."""

    series = _to_series(returns)
    return series.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""

    series = _to_series(returns)
    excess = series - risk_free_rate / periods_per_year
    vol = annualized_volatility(excess, periods_per_year=periods_per_year)
    if vol == 0:
        return np.nan
    return annualized_return(excess, periods_per_year=periods_per_year) / vol


def downside_deviation(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    mar: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized downside deviation relative to a minimum acceptable return."""

    series = _to_series(returns)
    downside = np.minimum(0.0, series - mar / periods_per_year)
    return downside.std(ddof=1) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    mar: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio."""

    downside = downside_deviation(returns, mar=mar, periods_per_year=periods_per_year)
    if downside == 0:
        return np.nan
    excess_return = annualized_return(returns, periods_per_year=periods_per_year) - mar
    return excess_return / downside


def max_drawdown(returns: pd.Series | pd.DataFrame | np.ndarray) -> float:
    """Compute the maximum drawdown for a series of returns."""

    series = _to_series(returns)
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1.0
    return drawdowns.min()


def calmar_ratio(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio calculated as CAGR divided by max drawdown magnitude."""

    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns, periods_per_year=periods_per_year) / abs(mdd)


def tail_ratio(returns: pd.Series | pd.DataFrame | np.ndarray) -> float:
    """Ratio of the 95th percentile gain to the 5th percentile loss."""

    series = _to_series(returns)
    gains = np.percentile(series, 95)
    losses = np.percentile(series, 5)
    if losses == 0:
        return np.nan
    return gains / abs(losses)


def information_ratio(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    benchmark: pd.Series | pd.DataFrame | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Annualized information ratio of returns relative to a benchmark."""

    active = _to_series(returns) - _to_series(benchmark)
    tracking_error = annualized_volatility(active, periods_per_year=periods_per_year)
    if tracking_error == 0:
        return np.nan
    return annualized_return(active, periods_per_year=periods_per_year) / tracking_error


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling beta of returns relative to a benchmark."""

    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        raise TypeError("returns and benchmark must be pandas Series for rolling beta")

    if len(returns) != len(benchmark):
        raise ValueError("returns and benchmark must have the same length")

    covariance = returns.rolling(window).cov(benchmark)
    benchmark_var = benchmark.rolling(window).var()
    beta = covariance / benchmark_var
    return beta.dropna()
