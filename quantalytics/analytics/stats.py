from __future__ import annotations

from math import ceil as _ceil
from typing import Optional, overload
from warnings import warn

import numpy as _np
import pandas as _pd
from numpy._core.fromnumeric import prod
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import Series

from quantalytics import utils as _utils
from quantalytics.utils.timeseries import _infer_periods

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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
    returns = _utils.aggregate_returns(returns, aggregate, compounded) < 0
    return _utils._count_consecutive(returns).max()


@overload
def exposure(returns: Series, prepare_returns=True) -> float: ...
@overload
def exposure(returns: DataFrame, prepare_returns=True) -> Series: ...
def exposure(returns: Series | DataFrame, prepare_returns=True) -> float | Series:
    """Returns the market exposure time (returns != 0)"""
    returns = _utils.normalize_returns(returns) if prepare_returns else returns

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

    returns = _utils.normalize_returns(returns) if prepare_returns else returns
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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
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
    returns = _utils.normalize_returns(returns) if prepare_returns else returns
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns < 0].dropna().mean()


@overload
def volatility(
    returns: Series, periods: float | None = None, annualize=True, prepare_returns=True
) -> float: ...
@overload
def volatility(
    returns: DataFrame,
    periods: float | None = None,
    annualize=True,
    prepare_returns=True,
) -> Series: ...
def volatility(
    returns: Series | DataFrame,
    periods: float | None = None,
    annualize=True,
    prepare_returns=True,
) -> float | Series:
    """Calculates the volatility of returns for a period"""
    normalized = _utils.normalize_returns(returns) if prepare_returns else returns

    std = normalized.std()

    if not annualize:
        return std

    periods = periods or _infer_periods(normalized)

    return std * _np.sqrt(periods)


@overload
def rolling_volatility(
    returns: Series,
    rolling_period=126,
    periods: float | None = None,
    prepare_returns=True,
) -> Series: ...
@overload
def rolling_volatility(
    returns: DataFrame,
    rolling_period=126,
    periods: float | None = None,
    prepare_returns=True,
) -> DataFrame: ...
def rolling_volatility(
    returns: Series | DataFrame,
    rolling_period=126,
    periods: float | None = None,
    prepare_returns=True,
) -> Series | DataFrame:
    """Calculates the rolling volatility of returns for a period
    Args:
        * returns (Series, DataFrame): Input return series
        * rolling_period (int): Rolling period
        * periods: periods per year
    """
    returns = (
        _utils.normalize_returns(returns, nperiods=rolling_period)
        if prepare_returns
        else returns
    )

    periods = periods or _infer_periods(returns)

    return returns.rolling(rolling_period).std() * _np.sqrt(periods)


@overload
def implied_volatility(
    returns: Series, periods: float | None = None, annualize=True
) -> Series: ...
@overload
def implied_volatility(
    returns: DataFrame, periods: float | None = None, annualize=True
) -> DataFrame: ...
def implied_volatility(
    returns: Series | DataFrame, periods: float | None = None, annualize=True
) -> Series | DataFrame:
    """Calculates the implied volatility of returns for a period"""
    logret = _utils.log_returns(returns)

    if not annualize:
        return logret.std()

    periods = periods or _infer_periods(returns)

    return logret.rolling(periods).std() * _np.sqrt(periods)


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


@overload
def drawdown_details(drawdown: Series) -> DataFrame: ...
@overload
def drawdown_details(drawdown: DataFrame) -> DataFrame: ...
def drawdown_details(drawdown: Series | DataFrame) -> DataFrame:
    """
    Calculate detailed statistics for each individual drawdown period.

    Analyzes a drawdown series to identify and characterize each distinct drawdown period,
    providing comprehensive statistics including start/end dates, duration, valley (maximum
    drawdown point), maximum drawdown percentage, and 99th percentile drawdown (excluding outliers).

    Parameters
    ----------
    drawdown : Series or DataFrame
        Drawdown series (typically output from `to_drawdown_series`).
        Values should be <= 0, where 0 indicates no drawdown and negative values
        indicate the depth of drawdown from the running maximum.
        If DataFrame, processes each column independently and concatenates results.

    Returns
    -------
    DataFrame
        For Series input: DataFrame with one row per drawdown period and columns:
            - start: Timestamp when drawdown period began
            - valley: Timestamp of maximum drawdown point
            - end: Timestamp when drawdown period ended (recovered to 0)
            - days: Duration of drawdown period in days
            - max drawdown: Maximum drawdown percentage (as positive number, e.g., 15.2 for -15.2%)
            - 99% max drawdown: 99th percentile drawdown excluding outliers

        For DataFrame input: Multi-level column DataFrame where first level is original
        column names and second level contains the statistics above.

    Examples
    --------
    >>> from quantalytics.analytics import to_drawdown_series, drawdown_details
    >>> import pandas as pd
    >>> returns = pd.Series([0.01, -0.02, -0.01, 0.03, 0.02, -0.05, 0.01])
    >>> dd = to_drawdown_series(returns)
    >>> details = drawdown_details(dd)
    >>> print(details.columns)
    Index(['start', 'valley', 'end', 'days', 'max drawdown', '99% max drawdown'], dtype='object')

    Notes
    -----
    - A drawdown period begins when drawdown becomes non-zero and ends when it returns to zero
    - If the series starts in a drawdown, the first period's start is set to the series start
    - If the series ends in a drawdown, the last period's end is set to the series end
    - The 99% max drawdown uses `remove_outliers` to exclude extreme values, providing
      a more robust measure of typical drawdown severity
    - All drawdown percentages are returned as positive values for easier interpretation

    See Also
    --------
    max_drawdown : Calculate maximum drawdown from returns
    to_drawdown_series : Convert returns to drawdown series (in metrics module)
    """

    def _drawdown_details(drawdown_series: Series) -> DataFrame:
        """Calculate drawdown details for a single drawdown series."""
        columns = (
            "start",
            "valley",
            "end",
            "days",
            "max drawdown",
            "99% max drawdown",
        )

        if drawdown_series.empty:
            return _pd.DataFrame(index=[], columns=columns)
        # Mark periods with no drawdown (drawdown = 0)
        no_dd = drawdown_series == 0

        # Extract drawdown start dates (transition from 0 to non-zero)
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts.values].index)

        # Extract drawdown end dates (transition from non-zero to 0)
        ends = no_dd & (~no_dd).shift(1)
        ends = list(ends[ends.values].index)

        # Handle edge cases: series starting or ending in drawdown
        if ends and (not starts or starts[0] > ends[0]):
            # Series starts in drawdown
            starts.insert(0, drawdown_series.index[0])
        if not ends or (starts and starts[-1] > ends[-1]):
            # Series ends in drawdown
            ends.append(drawdown_series.index[-1])

        # Return empty DataFrame if no drawdowns found
        if not starts:
            return _pd.DataFrame(
                index=[],
                columns=(
                    "start",
                    "valley",
                    "end",
                    "days",
                    "max drawdown",
                    "99% max drawdown",
                ),
            )

        # Build detailed statistics for each drawdown period
        data = []
        for i in range(len(starts)):
            # Check if this drawdown has recovered (ends[i] has drawdown == 0)
            # or if series ends in drawdown (ends[i] is last index)
            if drawdown_series[ends[i]] == 0:
                # Drawdown recovered: exclude the recovery day (ends[i])
                last_dd_day = ends[i] - _pd.Timedelta(days=1)
                dd_period = drawdown_series[starts[i] : last_dd_day]
                days_in_dd = (last_dd_day - starts[i]).days + 1
            else:
                # Series ends in drawdown: include the last day
                dd_period = drawdown_series[starts[i] : ends[i]]
                days_in_dd = (ends[i] - starts[i]).days + 1

            # Calculate 99th percentile drawdown (excluding outliers)
            clean_dd = -remove_outliers(-dd_period, 0.99)

            # Collect statistics
            data.append(
                (
                    starts[i],
                    dd_period.idxmin(),  # valley = point of max drawdown
                    ends[i],  # end = recovery date or last date
                    days_in_dd,
                    dd_period.min() * 100,  # Convert to percentage (as negative)
                    clean_dd.min() * 100,  # 99% drawdown as percentage
                )
            )

        # Create DataFrame with results
        df = _pd.DataFrame(
            data=data,
            columns=(
                "start",
                "valley",
                "end",
                "days",
                "max drawdown",
                "99% max drawdown",
            ),
        )

        # Format date columns as date strings (without time) for better display
        for col in ["start", "valley", "end"]:
            df[col] = df[col].apply(
                lambda x: x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)
            )

        # Convert drawdown percentages to positive values (easier interpretation)
        df["max drawdown"] = -df["max drawdown"]
        df["99% max drawdown"] = -df["99% max drawdown"]

        return df

    # Handle DataFrame input by processing each column
    if isinstance(drawdown, DataFrame):
        dfs = {}
        for col in drawdown.columns:
            dfs[col] = _drawdown_details(drawdown[col])
        return _pd.concat(dfs, axis=1)

    return _drawdown_details(drawdown)
