"""Benchmark comparison utilities for portfolio analysis."""

from __future__ import annotations

from typing import Optional, overload

from numpy import ndarray
from numpy.ma.core import array, where
from numpy.ma.extras import cov, unique
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import Series
from scipy.stats import linregress as _linregress

from quantalytics.analytics.stats import comp
from quantalytics.utils import timeseries as _utils


@overload
def treynor_ratio(
    returns: Series,
    benchmark: Series | DataFrame,
    periods: float = 365.0,
    rf: float = 0.0,
    prepare_returns: bool = True,
) -> float: ...
@overload
def treynor_ratio(
    returns: DataFrame,
    benchmark: Series | DataFrame,
    periods: float = 365.0,
    rf: float = 0.0,
    prepare_returns: bool = True,
) -> float: ...
def treynor_ratio(
    returns: Series | DataFrame,
    benchmark: Series | DataFrame,
    periods: float = 365.0,
    rf: float = 0.0,
    prepare_returns: bool = True,
) -> float:
    """Calculate the Treynor ratio of returns relative to a benchmark.

    The Treynor ratio measures risk-adjusted returns per unit of systematic risk (beta).
    It shows how much excess return is generated for each unit of market risk taken.
    Higher values indicate better risk-adjusted performance.

    Args:
        returns (Series | DataFrame): Portfolio returns data. If DataFrame, uses first column.
        benchmark (Series | DataFrame): Benchmark returns data for comparison.
        periods (float, optional): Number of periods per year for annualization.
            Defaults to 365.0 (daily data).
        rf (float, optional): Risk-free rate to subtract from returns.
            Defaults to 0.0.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float: Treynor ratio value. Returns 0 if beta is 0 (no systematic risk).
            Calculated as: (total_return - rf) / beta.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        >>> benchmark = pd.Series([0.008, 0.015, -0.005, 0.025, 0.008])
        >>> treynor_ratio(returns, benchmark, periods=252)
        0.15  # Excess return per unit of beta

    Notes:
        - Unlike Sharpe ratio which uses total volatility, Treynor uses beta (systematic risk)
        - Returns 0 if beta is 0 (portfolio has no systematic risk)
        - Higher values indicate better compensation for systematic risk
        - Useful for well-diversified portfolios where systematic risk dominates
        - Not suitable for comparing portfolios with different betas to the same benchmark

    See Also:
        sharpe: Risk-adjusted return using total volatility
        information_ratio: Risk-adjusted return relative to tracking error
    """
    # Handle DataFrame input by using first column
    if isinstance(returns, DataFrame):
        returns = returns[returns.columns[0]]

    # Calculate beta using greeks function
    beta_value = greeks(
        returns, benchmark, periods=periods, prepare_returns=prepare_returns
    )["beta"]

    # Return 0 if beta is 0 to avoid division by zero
    if beta_value == 0:
        return 0.0

    # Calculate total compounded return
    total_return = comp(returns)

    # Calculate Treynor ratio
    return float((total_return - rf) / beta_value)


@overload
def r_squared(
    returns: Series, benchmark: Series | DataFrame, prepare_returns: bool = True
) -> float: ...
@overload
def r_squared(
    returns: DataFrame, benchmark: Series | DataFrame, prepare_returns: bool = True
) -> float: ...
def r_squared(
    returns: Series | DataFrame,
    benchmark: Series | DataFrame,
    prepare_returns: bool = True,
) -> float:
    """Calculate the R-squared (coefficient of determination) between returns and benchmark.

    R-squared measures how well the returns fit a linear regression line with the benchmark,
    indicating the proportion of variance in returns explained by the benchmark.

    Args:
        returns (Series | DataFrame): Portfolio returns data. If DataFrame, uses first column.
        benchmark (Series | DataFrame): Benchmark returns data for comparison.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float: R-squared value between 0 and 1, where 1 indicates perfect fit.
            Returns 0 if all values are identical (to avoid regression errors).

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> benchmark = pd.Series([0.008, 0.015, -0.005, 0.025])
        >>> r_squared(returns, benchmark)
        0.95
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    # if all values are identical, return 0 to avoid errors
    if len(unique(array(normalized))) == 1:
        return 0.0
    # Check if all returns index x values are identical to prevent the error ValueError: Cannot calculate a linear regression if all x values are identical
    if len(unique(array(normalized.index))) == 1:
        return 0.0

    # Check if all benchmark values are identical
    if len(unique(array(benchmark))) == 1:
        return 0.0

    period: DatetimeIndex = (
        returns.index
        if isinstance(returns.index, DatetimeIndex)
        else DatetimeIndex(returns.index)
    )
    _, _, r_val, _, _ = _linregress(
        normalized, _utils._prepare_benchmark(benchmark, period)
    )
    return float(r_val**2)


@overload
def r2(returns: Series, benchmark: Series | DataFrame) -> float: ...
@overload
def r2(returns: DataFrame, benchmark: Series | DataFrame) -> float: ...
def r2(returns: Series | DataFrame, benchmark: Series | DataFrame) -> float:
    """Calculate R-squared between returns and benchmark (shorthand for r_squared).

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        benchmark (Series | DataFrame): Benchmark returns data for comparison.

    Returns:
        float: R-squared value between 0 and 1.

    See Also:
        r_squared: Full function with additional options.
    """
    return r_squared(returns, benchmark)


@overload
def information_ratio(
    returns: Series, benchmark: Series | DataFrame, prepare_returns: bool = True
) -> float: ...
@overload
def information_ratio(
    returns: DataFrame, benchmark: Series | DataFrame, prepare_returns: bool = True
) -> Series: ...
def information_ratio(
    returns: Series | DataFrame,
    benchmark: Series | DataFrame,
    prepare_returns: bool = True,
) -> float | Series:
    """Calculate the information ratio of returns relative to a benchmark.

    The information ratio measures risk-adjusted returns relative to a benchmark,
    calculated as the mean of excess returns divided by the tracking error (standard
    deviation of excess returns). Higher values indicate better risk-adjusted
    outperformance.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        benchmark (Series | DataFrame): Benchmark returns data for comparison.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Information ratio value. Returns float for Series input,
            Series for DataFrame input (one value per column).

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> benchmark = pd.Series([0.008, 0.015, -0.005, 0.025])
        >>> information_ratio(returns, benchmark)
        0.42
    """
    normalized: Series | DataFrame = (
        _utils.normalize_returns(data=returns) if prepare_returns else returns
    )

    period: DatetimeIndex = (
        normalized.index
        if isinstance(normalized.index, DatetimeIndex)
        else DatetimeIndex(normalized.index)
    )
    diff_rets = normalized - _utils._prepare_benchmark(
        benchmark, period, prepare_returns=prepare_returns
    )

    return diff_rets.mean() / diff_rets.std()


@overload
def greeks(
    returns: Series,
    benchmark: Series | DataFrame,
    periods: float = 365.0,
    prepare_returns: bool = True,
) -> Series: ...
@overload
def greeks(
    returns: DataFrame,
    benchmark: Series | DataFrame,
    periods: float = 365.0,
    prepare_returns: bool = True,
) -> Series: ...
def greeks(
    returns: Series | DataFrame,
    benchmark: Series | DataFrame,
    periods: float = 365.0,
    prepare_returns: bool = True,
) -> Series:
    """Calculate alpha and beta (the 'greeks') of the portfolio relative to a benchmark.

    Alpha measures the excess return of the portfolio over what would be predicted
    by the beta and benchmark returns (annualized). Beta measures the portfolio's
    sensitivity to benchmark movements.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        benchmark (Series | DataFrame): Benchmark returns data for comparison.
        periods (float, optional): Number of periods per year for annualization.
            Defaults to 365.0 (daily data).
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        Series: Series with 'alpha' and 'beta' values. Alpha is annualized based on
            the periods parameter.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> benchmark = pd.Series([0.008, 0.015, -0.005, 0.025])
        >>> greeks(returns, benchmark, periods=252)
        beta     1.15
        alpha    0.12
        dtype: float64
    """
    # ----------------------------
    # data cleanup
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    period: DatetimeIndex = (
        normalized.index
        if isinstance(normalized.index, DatetimeIndex)
        else DatetimeIndex(normalized.index)
    )
    benchmark = _utils._prepare_benchmark(benchmark, period)
    # ----------------------------

    # find covariance
    matrix: ndarray = cov(returns, benchmark)
    beta = matrix[0, 1] / matrix[1, 1]

    # calculates measures now
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods

    return Series(
        {
            "beta": beta,
            "alpha": alpha,
            # "vol": sqrt(matrix[0, 0]) * sqrt(periods)
        }
    ).fillna(0)


@overload
def rolling_greeks(
    returns: Series,
    benchmark: Series | DataFrame,
    periods: int = 365,
    prepare_returns: bool = True,
) -> DataFrame: ...
@overload
def rolling_greeks(
    returns: DataFrame,
    benchmark: Series | DataFrame,
    periods: int = 365,
    prepare_returns: bool = True,
) -> DataFrame: ...
def rolling_greeks(
    returns: Series | DataFrame,
    benchmark: Series | DataFrame,
    periods: int = 365,
    prepare_returns: bool = True,
) -> DataFrame:
    """Calculate rolling alpha and beta of the portfolio over time.

    Computes alpha and beta values using a rolling window, allowing you to see
    how the portfolio's relationship with the benchmark changes over time.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        benchmark (Series | DataFrame): Benchmark returns data for comparison.
        periods (int, optional): Rolling window size in number of periods.
            Defaults to 365 (approximately 1 year for daily data).
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        DataFrame: DataFrame with 'alpha' and 'beta' columns showing rolling values
            over time, indexed by the same dates as the input returns.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03] * 100)
        >>> benchmark = pd.Series([0.008, 0.015, -0.005, 0.025] * 100)
        >>> rolling_greeks(returns, benchmark, periods=30)
                    beta     alpha
        2020-01-01  1.12     0.0003
        2020-01-02  1.15     0.0002
        ...
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    period: DatetimeIndex = (
        normalized.index
        if isinstance(normalized.index, DatetimeIndex)
        else DatetimeIndex(normalized.index)
    )
    df = DataFrame(
        data={
            "returns": returns,
            "benchmark": _utils._prepare_benchmark(benchmark, period),
        }
    )
    df = df.fillna(0)
    corr = df.rolling(periods).corr().unstack()["returns"]["benchmark"]
    std = df.rolling(periods).std()
    beta = corr * std["returns"] / std["benchmark"]

    alpha = df["returns"].mean() - beta * df["benchmark"].mean()

    return DataFrame(index=returns.index, data={"beta": beta, "alpha": alpha})


@overload
def compare(
    returns: Series,
    benchmark: Series | DataFrame,
    aggregate: Optional[str | DatetimeIndex] = None,
    compounded: bool = True,
    round_vals: Optional[int] = None,
    prepare_returns: bool = True,
) -> DataFrame: ...
@overload
def compare(
    returns: DataFrame,
    benchmark: Series | DataFrame,
    aggregate: Optional[str | DatetimeIndex] = None,
    compounded: bool = True,
    round_vals: Optional[int] = None,
    prepare_returns: bool = True,
) -> DataFrame: ...
def compare(
    returns: Series | DataFrame,
    benchmark: Series | DataFrame,
    aggregate: Optional[str | DatetimeIndex] = None,
    compounded: bool = True,
    round_vals: Optional[int] = None,
    prepare_returns: bool = True,
) -> DataFrame:
    """Compare portfolio returns to benchmark on various time period bases.

    Creates a comparison table showing returns vs benchmark performance, aggregated
    by the specified time period (day/week/month/quarter/year). For Series input,
    includes additional columns showing the multiplier and whether each period won.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        benchmark (Series | DataFrame): Benchmark returns data for comparison.
        aggregate (str | DatetimeIndex, optional): Time period for aggregation.
            Options: "day", "week", "month", "quarter", "year", or custom DatetimeIndex.
            If None, uses raw returns. Defaults to None.
        compounded (bool, optional): Whether to compound returns when aggregating.
            Defaults to True.
        round_vals (int, optional): Number of decimal places to round results.
            If None, no rounding is applied. Defaults to None.
        prepare_returns (bool, optional): Whether to normalize returns before comparison.
            Defaults to True.

    Returns:
        DataFrame: Comparison table with benchmark and returns columns (as percentages).
            For Series input, includes 'Multiplier' (Returns/Benchmark ratio) and
            'Won' (+/-) columns. For DataFrame input, includes one return column per
            strategy column.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=pd.date_range('2020-01-01', periods=4))
        >>> benchmark = pd.Series([0.008, 0.015, -0.005, 0.025], index=pd.date_range('2020-01-01', periods=4))
        >>> compare(returns, benchmark, aggregate='month', round_vals=2)
                    Benchmark  Returns  Multiplier  Won
        2020-01     5.2        5.5      1.06        +
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    period: DatetimeIndex = (
        normalized.index
        if isinstance(normalized.index, DatetimeIndex)
        else DatetimeIndex(normalized.index)
    )
    benchmark = _utils._prepare_benchmark(benchmark, period)

    if isinstance(returns, Series):
        data = DataFrame(
            data={
                "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded)
                * 100,
                "Returns": _utils.aggregate_returns(returns, aggregate, compounded)
                * 100,
            }
        )

        data["Multiplier"] = data["Returns"] / data["Benchmark"]
        data["Won"] = where(data["Returns"] >= data["Benchmark"], "+", "-")
    elif isinstance(returns, DataFrame):
        bench = {
            "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded)
            * 100
        }
        strategy = {
            f"Returns_{str(i)}": _utils.aggregate_returns(
                returns[col], aggregate, compounded
            )
            * 100
            for i, col in enumerate(returns.columns)
        }
        data = DataFrame(data=bench | strategy)

    return round(data, round_vals) if round_vals is not None else data


__all__: list[str] = [
    "treynor_ratio",
    "r_squared",
    "r2",
    "information_ratio",
    "greeks",
    "rolling_greeks",
    "compare",
]
