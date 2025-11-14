from __future__ import annotations

from math import ceil as _ceil
from typing import Optional, overload
from warnings import warn

import numpy as _np
from numpy._core.fromnumeric import prod
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import Series

from quantalytics import utils as _utils

# ======== STATS ========


@overload
def pct_rank(prices: Series, window=60) -> Series: ...
@overload
def pct_rank(prices: DataFrame, window=60) -> DataFrame: ...
def pct_rank(prices: Series | DataFrame, window=60) -> Series | DataFrame:
    """Rank prices by window"""
    rank: DataFrame = _utils.multi_shift(df=prices, shift=window).T.rank(pct=True).T
    return rank.iloc[:, 0] * 100.0


@overload
def compsum(returns: Series) -> Series: ...
@overload
def compsum(returns: DataFrame) -> DataFrame: ...
def compsum(returns: Series | DataFrame) -> Series | DataFrame:
    """Calculates rolling compounded returns"""
    return returns.add(other=1).cumprod() - 1


@overload
def comp(returns: Series) -> float: ...
@overload
def comp(returns: DataFrame) -> Series: ...
def comp(returns: Series | DataFrame) -> float | Series:
    """Calculates total compounded returns"""
    return returns.add(other=1).prod() - 1


def distribution(
    returns: Series | DataFrame, compounded=True, prepare_returns=True
) -> dict:
    """Returns the distribution of returns
    Args:
        * returns (Series, DataFrame): Input return series
        * compounded (bool): Calculate compounded returns?
    """

    def get_outliers(data):
        """Returns outliers"""
        # https://datascience.stackexchange.com/a/57199
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.
        filtered = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)
        return {
            "values": data.loc[filtered].tolist(),
            "outliers": data.loc[~filtered].tolist(),
        }

    if isinstance(returns, DataFrame):
        warn(
            "Pandas DataFrame was passed (Series expected). Only first column will be used."
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    apply_fnc = comp if compounded else "sum"
    daily = returns.dropna()

    if prepare_returns:
        daily = _utils.normalize_returns(daily)

    return {
        "Daily": get_outliers(daily),
        "Weekly": get_outliers(daily.resample("W-MON").apply(apply_fnc)),
        "Monthly": get_outliers(daily.resample("ME").apply(apply_fnc)),
        "Quarterly": get_outliers(daily.resample("QE").apply(apply_fnc)),
        "Yearly": get_outliers(daily.resample("YE").apply(apply_fnc)),
    }


@overload
def expected_return(
    returns: Series,
    aggregate=None,
    compounded=True,
    prepare_returns=True,
) -> float: ...
@overload
def expected_return(
    returns: DataFrame,
    aggregate=None,
    compounded=True,
    prepare_returns=True,
) -> Series: ...
def expected_return(
    returns: Series | DataFrame,
    aggregate=None,
    compounded=True,
    prepare_returns=True,
) -> float | Series:
    """
    Returns the expected return for a given period
    by calculating the geometric holding period return
    """
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return prod(1 + returns, axis=0) ** (1 / len(returns)) - 1


@overload
def geometric_mean(returns: Series, aggregate=None, compounded=True) -> float: ...
@overload
def geometric_mean(returns: DataFrame, aggregate=None, compounded=True) -> Series: ...
def geometric_mean(
    returns: Series | DataFrame, aggregate=None, compounded=True
) -> float | Series:
    """Shorthand for expected_return()"""
    return expected_return(returns=returns, aggregate=aggregate, compounded=compounded)


@overload
def ghpr(returns: Series, aggregate=None, compounded=True) -> float: ...
@overload
def ghpr(returns: DataFrame, aggregate=None, compounded=True) -> Series: ...
def ghpr(
    returns: Series | DataFrame, aggregate=None, compounded=True
) -> float | Series:
    """Shorthand for expected_return()"""
    return expected_return(returns=returns, aggregate=aggregate, compounded=compounded)


@overload
def outliers(returns: Series, quantile=0.95) -> Series: ...
@overload
def outliers(returns: DataFrame, quantile=0.95) -> DataFrame: ...
def outliers(returns: Series | DataFrame, quantile=0.95) -> Series | DataFrame:
    """Returns series of outliers"""
    return returns[returns > returns.quantile(quantile)].dropna(how="all")


@overload
def remove_outliers(returns: Series, quantile=0.95) -> Series: ...
@overload
def remove_outliers(returns: DataFrame, quantile=0.95) -> DataFrame: ...
def remove_outliers(returns: Series | DataFrame, quantile=0.95) -> Series | DataFrame:
    """Returns series of returns without the outliers"""
    return returns[returns < returns.quantile(quantile)]


@overload
def best(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> float: ...
@overload
def best(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def best(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> float | Series:
    """Returns the best day/month/week/quarter/year's return"""
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    return _utils.aggregate_returns(returns, aggregate, compounded).max()


@overload
def worst(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> float: ...
@overload
def worst(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def worst(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> float | Series:
    """Returns the worst day/month/week/quarter/year's return"""
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    return _utils.aggregate_returns(returns, aggregate, compounded).min()


@overload
def consecutive_wins(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> int: ...
@overload
def consecutive_wins(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def consecutive_wins(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> int | Series:
    """Returns the maximum consecutive wins by day/month/week/quarter/year"""
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) > 0
    return _utils._count_consecutive(returns).max()


@overload
def consecutive_losses(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> int: ...
@overload
def consecutive_losses(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def consecutive_losses(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> int | Series:
    """
    Returns the maximum consecutive losses by
    day/month/week/quarter/year
    """
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) < 0
    return _utils._count_consecutive(returns).max()


@overload
def exposure(returns: Series, prepare_returns=True) -> float: ...
@overload
def exposure(returns: DataFrame, prepare_returns=True) -> Series: ...
def exposure(returns: Series | DataFrame, prepare_returns=True) -> float | Series:
    """Returns the market exposure time (returns != 0)"""
    if prepare_returns:
        returns = _utils.normalize_returns(returns)

    def _exposure(ret):
        """Returns the market exposure time (returns != 0)"""
        ex = len(ret[(~_np.isnan(ret)) & (ret != 0)]) / len(ret)
        return _ceil(ex * 100) / 100

    if isinstance(returns, DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _exposure(returns[col])
        return Series(_df)
    return _exposure(returns)


@overload
def win_rate(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> float: ...
@overload
def win_rate(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def win_rate(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> float | Series:
    """Calculates the win ratio for a period"""

    def _win_rate(series):
        try:
            return len(series[series > 0]) / len(series[series != 0])
        except Exception:
            return 0.0

    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    if isinstance(returns, DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])

        return Series(_df)

    return _win_rate(returns)


@overload
def avg_return(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> float: ...
@overload
def avg_return(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def avg_return(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> float | Series:
    """Calculates the average return/trade return for a period"""
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns != 0].dropna().mean()


@overload
def avg_win(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> float: ...
@overload
def avg_win(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def avg_win(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> float | Series:
    """
    Calculates the average winning
    return/trade return for a period
    """
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns > 0].dropna().mean()


@overload
def avg_loss(
    returns: Series, aggregate=None, compounded=True, prepare_returns=True
) -> float: ...
@overload
def avg_loss(
    returns: DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> Series: ...
def avg_loss(
    returns: Series | DataFrame, aggregate=None, compounded=True, prepare_returns=True
) -> float | Series:
    """
    Calculates the average low if
    return/trade return for a period
    """
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns < 0].dropna().mean()


@overload
def volatility(
    returns: Series, periods=365, annualize=True, prepare_returns=True
) -> float: ...
@overload
def volatility(
    returns: DataFrame, periods=365, annualize=True, prepare_returns=True
) -> Series: ...
def volatility(
    returns: Series | DataFrame, periods=365, annualize=True, prepare_returns=True
) -> float | Series:
    """Calculates the volatility of returns for a period"""
    if prepare_returns:
        returns = _utils.normalize_returns(returns)
    std = returns.std()
    return std * _np.sqrt(periods) if annualize else std


@overload
def rolling_volatility(
    returns: Series, rolling_period=126, periods=365, prepare_returns=True
) -> Series: ...
@overload
def rolling_volatility(
    returns: DataFrame, rolling_period=126, periods=365, prepare_returns=True
) -> DataFrame: ...
def rolling_volatility(
    returns: Series | DataFrame, rolling_period=126, periods=365, prepare_returns=True
) -> Series | DataFrame:
    """Calculates the rolling volatility of returns for a period
    Args:
        * returns (Series, DataFrame): Input return series
        * rolling_period (int): Rolling period
        * periods: periods per year
    """
    if prepare_returns:
        returns = _utils.normalize_returns(returns, rolling_period)

    return returns.rolling(rolling_period).std() * _np.sqrt(periods)


@overload
def implied_volatility(returns: Series, periods=365, annualize=True) -> Series: ...
@overload
def implied_volatility(
    returns: DataFrame, periods=365, annualize=True
) -> DataFrame: ...
def implied_volatility(
    returns: Series | DataFrame, periods=365, annualize=True
) -> Series | DataFrame:
    """Calculates the implied volatility of returns for a period"""
    logret = _utils.log_returns(returns)
    if annualize:
        return logret.rolling(periods).std() * _np.sqrt(periods)
    return logret.std()


@overload
def max_drawdown(returns: Series) -> float: ...
@overload
def max_drawdown(returns: DataFrame) -> Series: ...
def max_drawdown(returns: Series | DataFrame) -> float | Series:
    """Compute the maximum drawdown from a series of returns."""

    series: Series | DataFrame = _utils.normalize_returns(data=returns)

    cum_returns = (1 + series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1

    return drawdown.min()


@overload
def cagr(returns: Series, rf: float = 0.0, periods: Optional[int] = None) -> float: ...
@overload
def cagr(
    returns: DataFrame, rf: float = 0.0, periods: Optional[int] = None
) -> Series: ...
def cagr(
    returns: Series | DataFrame, rf: float = 0.0, periods: Optional[int] = None
) -> float | Series:
    """
    Calculate Compound Annual Growth Rate (CAGR) from returns.

    Parameters
    ----------
    returns : Series or DataFrame
        Return series with DatetimeIndex or numeric index
    rf : float, default 0.0
        Annual risk-free rate as decimal. If provided, returns excess CAGR (CAGR - rf)
    periods : int, optional
        Number of periods per year (e.g., 252 for daily, 12 for monthly)
        If None, attempts to infer from DatetimeIndex

    Returns
    -------
    float or Series
        CAGR as decimal (e.g., 0.132 for 13.2% annual growth)
        If rf > 0, returns excess CAGR (CAGR - rf)
        Returns float for Series input, Series for DataFrame input

    Examples
    --------
    >>> returns = Series([0.1, 0.05, -0.03, 0.08])
    >>> cagr(returns, periods=252)
    0.0482  # 4.82% annualized

    >>> cagr(returns, rf=0.02, periods=252)
    0.0282  # 2.82% excess return over 2% risk-free rate

    Notes
    -----
    CAGR = (Total Return)^(1/years) - 1
    where Total Return = product(1 + returns)

    When rf is provided, returns: CAGR - rf
    """
    # Use the date range from the dataset, otherwise override.
    if not isinstance(returns.index, DatetimeIndex):
        periods: int = periods or 252

    if returns.empty:
        return Series(dtype=float) if isinstance(returns, DataFrame) else _np.nan
    # Calculate total return (compound)
    if isinstance(returns, DataFrame):
        # Handle each column
        total_return = (1 + returns).prod()

        # Calculate years
        years = _calculate_years(returns, periods)

        # CAGR for each column
        result = _np.power(total_return, 1 / years) - 1

        # Subtract risk-free rate if provided
        if rf != 0:
            result = result - rf

        # Replace inf/-inf/invalid with NaN
        result = result.replace([_np.inf, -_np.inf], _np.nan)

        return result

    else:  # Series
        # Clean data
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return _np.nan

        # Calculate total return
        total_return = (1 + clean_returns).prod()

        if total_return <= 0:
            return _np.nan  # Can't calculate CAGR if total return is negative

        # Calculate years
        years = _calculate_years(clean_returns, periods)

        if years <= 0:
            return _np.nan

        # CAGR calculation
        annual_return = float(_np.power(total_return, 1 / years) - 1)

        # Return excess CAGR if rf provided
        return annual_return - rf if rf != 0 else annual_return


def _calculate_years(
    returns: Series | DataFrame, periods: Optional[int] = None
) -> float:
    """
    Calculate number of years in the return series.

    Parameters
    ----------
    returns : Series or DataFrame
        Return data with index
    periods : int, optional
        If provided, uses len(returns) / periods
        If None, attempts to calculate from DatetimeIndex

    Returns
    -------
    float
        Number of years
    """
    # Try to infer from DatetimeIndex
    if isinstance(returns.index, DatetimeIndex):
        if isinstance(returns, DataFrame):
            idx = returns.dropna(how="all").index
        else:
            idx = returns.dropna().index

        if len(idx) < 2:
            return 0

        # Calculate actual time span
        time_delta = idx[-1] - idx[0]
        return time_delta.total_seconds() / (365.25 * 24 * 60 * 60)

    if periods is not None:
        # Simple calculation based on number of periods
        if isinstance(returns, DataFrame):
            return len(returns) / periods
        else:
            return len(returns.dropna()) / periods

    # Can't determine years without periods or DatetimeIndex
    raise ValueError(
        "Cannot determine time period. Either provide periods or ensure returns has a DatetimeIndex"
    )
