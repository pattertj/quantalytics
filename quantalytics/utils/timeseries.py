"""Timeseries utilities used across Quantalytics."""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, overload

from numpy import arange, inf, log, nan
from pandas.core.dtypes.missing import isna
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex, date_range
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.tools.datetimes import to_datetime

from quantalytics.analytics import stats as _stats


def _infer_periods(returns: Series | DataFrame) -> int:
    """Infer the number of periods per year from returns data

    Attempts to determine the frequency of the returns data based on its index
    and maps it to the appropriate number of periods per year.

    Parameters
    ----------
    returns : Series or DataFrame
        Return data with DateTimeIndex for frequency inference

    Returns
    -------
    int
        Number of periods per year based on inferred frequency:
        - 365 for daily (D)
        - 252 for business days (B)
        - 52 for weekly (W)
        - 12 for monthly (M)
        - 4 for quarterly (Q)
        Defaults to 252 if unable to infer frequency
    """
    mapping = None

    # Try to calculate from DatetimeIndex
    if hasattr(returns.index, "inferred_freq"):
        freq_map = {
            "D": 252,  # Daily
            "B": 252,  # Business day
            "W": 52,  # Weekly
            "M": 12,  # Monthly
            "Q": 4,  # Quarterly
        }
        freq_code = (
            returns.index.inferred_freq[0] if returns.index.inferred_freq else "D"
        )
        mapping = freq_map.get(freq_code, 252)

    return mapping or 252


def _prepare_benchmark(
    benchmark: Series | DataFrame,
    period: DatetimeIndex | str = "max",
    rf: float = 0.0,
    prepare_returns: bool = True,
) -> Series | DataFrame:
    """Prepare benchmark data for comparison with strategy returns.

    Args:
        benchmark (Series | DataFrame): Benchmark return or price data. If DataFrame,
            uses first column only.
        period (DatetimeIndex | str, optional): Time period for benchmark alignment.
            If DatetimeIndex, resamples benchmark to match the provided dates.
            If "max" or other string, uses benchmark as-is. Defaults to "max".
        rf (float, optional): Risk-free rate for excess return calculation.
            Defaults to 0.0.
        prepare_returns (bool, optional): Whether to normalize returns and apply
            risk-free rate adjustment. If False, returns raw data. Defaults to True.

    Raises:
        ValueError: If benchmark is None.

    Returns:
        Series | DataFrame: Cleaned and aligned benchmark data with timezone-naive
            DatetimeIndex, optionally normalized to excess returns.
    """
    if benchmark is None:
        raise ValueError("")

    if isinstance(benchmark, DataFrame):
        benchmark = benchmark[benchmark.columns[0]].copy()

    if isinstance(period, DatetimeIndex) and set(period) != set(benchmark.index):
        # Adjust Benchmark to Strategy frequency
        benchmark_prices = to_prices(benchmark, base=1)
        new_index = date_range(start=period[0], end=period[-1], freq="D")
        benchmark = (
            benchmark_prices.reindex(new_index, method="bfill")
            .reindex(period)
            .pct_change()
            .fillna(0)
        )
        benchmark = benchmark[benchmark.index.isin(period)]

    if isinstance(benchmark.index, DatetimeIndex):
        benchmark.index = benchmark.index.tz_localize(None)

    if prepare_returns:
        return normalize_returns(data=benchmark.dropna(), rf=rf)

    return benchmark.dropna()


@overload
def to_prices(returns: DataFrame, base: float = 1e5) -> DataFrame: ...
@overload
def to_prices(returns: Series, base: float = 1e5) -> Series: ...
def to_prices(returns: Series | DataFrame, base: float = 1e5) -> Series | DataFrame:
    """Converts returns series to price data"""
    returns = returns.copy().fillna(0).replace([inf, -inf], float("NaN"))

    return base + base * _stats.compsum(returns)


def multi_shift(df: Series | DataFrame, shift: int = 3) -> DataFrame:
    """Get last N rows relative to another row in pandas"""
    if isinstance(df, Series):
        df = DataFrame(data=df)

    dfs = [df.shift(i) for i in arange(shift)]
    for ix, dfi in enumerate(dfs[1:]):
        cols = [str(col) for col in dfi.columns]
        dfs[ix + 1].columns = [f"{col}{ix + 1}" for col in cols]
    return concat(objs=dfs, axis=1, sort=True)


def _count_consecutive(data) -> Series | DataFrame:
    """
    Count consecutive occurrences in data (like cumsum() with reset on zeroes)

    Parameters
    ----------
    data : Series or DataFrame
        Input data to count consecutive occurrences

    Returns
    -------
    Series or DataFrame
        Data with consecutive counts
    """

    def _count(data):
        # Group by consecutive values and count occurrences
        return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

    # Handle DataFrame by processing each column
    if isinstance(data, DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def group_returns(
    returns: Series | DataFrame, groupby: str, compounded: bool = False
) -> Series | DataFrame:
    """
    Summarize returns by grouping criteria

    Parameters
    ----------
    returns : Series or DataFrame
        Returns data
    groupby : grouper object
        Pandas groupby object or criteria
    compounded : bool, default False
        Whether to compound returns or use simple sum

    Returns
    -------
    Series or DataFrame
        Grouped returns

    Examples
    --------
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        # Use compounded returns calculation
        return returns.groupby(groupby).apply(_stats.comp)
    # Use simple sum for non-compounded returns
    return returns.groupby(groupby).sum()


def aggregate_returns(
    returns: Series | DataFrame, period=None, compounded: bool = True
) -> Series | DataFrame:
    """Aggregate returns based on specified time periods"""

    # Validate inputs
    index = returns.index
    if not isinstance(index, DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")

    if returns.empty or period is None or period == "day":
        return returns

    # Normalize timezone
    if index.tz is not None:
        returns = returns.copy()
        returns.index = index.tz_localize(None)

    # Define period mappings
    period_mappings = {
        "month": lambda idx: [idx.year, idx.month],
        "quarter": lambda idx: [idx.year, idx.quarter],
        "year": lambda idx: idx.year,
        "week": lambda idx: [idx.year, idx.isocalendar().week],
        "ME": lambda idx: [idx.year, idx.month],
        "QE": lambda idx: [idx.year, idx.quarter],
        "YE": lambda idx: idx.year,
        "W": lambda idx: [idx.year, idx.isocalendar().week],
    }

    # Handle string periods
    if isinstance(period, str):
        period_lower = period.lower()

        # Check exact matches first
        if period in period_mappings:
            grouper = period_mappings[period](returns.index)
        # Then check partial matches
        elif "month" in period_lower:
            grouper = period_mappings["month"](returns.index)
        elif "quarter" in period_lower:
            grouper = period_mappings["quarter"](returns.index)
        elif "year" in period_lower or "eoy" in period_lower:
            grouper = period_mappings["year"](returns.index)
        elif "week" in period_lower or "eow" in period_lower:
            grouper = period_mappings["week"](returns.index)
        else:
            warnings.warn(f"Unrecognized period '{period}'")
            return returns
    else:
        # Custom period grouping
        grouper = period

    return group_returns(returns, grouper, compounded=compounded)


def ensure_datetime_index(series: Iterable[float] | Series) -> Series:
    """Ensure the series has a DatetimeIndex."""

    series = Series(series)
    if not isinstance(series.index, DatetimeIndex):
        series.index = to_datetime(series.index, errors="coerce")
    return series.sort_index()


def rolling_statistic(series: Series, window: int, function: str = "mean") -> Series:
    """Compute a rolling statistic with sensible defaults."""

    if window <= 0:
        raise ValueError("window must be positive")
    if function not in {"mean", "std", "median"}:
        raise ValueError("Unsupported rolling function")

    rolling = series.rolling(window=window, min_periods=max(2, window // 2))
    if function == "mean":
        return rolling.mean()
    return rolling.std(ddof=1) if function == "std" else rolling.median()


def infer_periods(series: Series, fallback_periods: int = 365) -> tuple[float, float]:
    """Infer the span (years) and periods per year from the series."""

    length = len(series)
    if length == 0:
        return 0.0, fallback_periods
    if isinstance(series.index, DatetimeIndex) and length > 1:
        days = max((series.index[-1] - series.index[0]).days, 1)
        years = days / 365.0
        periods_per_year = len(series) / years if years > 0 else fallback_periods
        return years, periods_per_year
    years = length / fallback_periods if fallback_periods else 0.0
    return years, fallback_periods


def normalize_returns(
    data: Iterable[float] | Series | DataFrame,
    rf: float = 0.0,
    data_type: str = "auto",
    fill_method: str = "zero",
    apply_excess_returns: bool = True,
    nperiods: Optional[int] = None,
) -> Series | DataFrame:
    """
    Converts price or return data into cleaned returns (optionally excess).
    """

    if not isinstance(data, (Series, DataFrame)) and isinstance(data, Iterable):
        data = Series(list(data))
    if not isinstance(data, (Series, DataFrame)):
        raise TypeError(
            "data must be a pandas Series, DataFrame, or iterable convertible to Series"
        )

    result = data.copy()
    result = _convert_to_returns(result, data_type)
    result = _clean_data(result, fill_method)

    if rf > 0 and apply_excess_returns:
        result = _to_excess_returns(result, rf, nperiods)

    return result


def _convert_to_returns(
    data: Series | DataFrame,
    data_type: str,
) -> Series | DataFrame:
    data_type = data_type.lower()
    if data_type == "returns":
        return data
    if data_type == "prices":
        return _calculate_returns(data)
    if isinstance(data, DataFrame):
        result = data.copy()
        for col in data.columns:
            column = data[col]
            if _is_likely_prices(column):
                result[col] = column.pct_change()
        return result
    return data.pct_change() if _is_likely_prices(data) else data


def _is_likely_prices(series: Series) -> bool:
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return False
    has_negative = (clean_series < 0).any()
    in_return_range = (clean_series >= -0.5) & (clean_series <= 0.5)
    mostly_returns = in_return_range.mean() > 0.95
    has_high_values = (clean_series > 2).any()

    if len(clean_series) > 10:
        autocorr = clean_series.autocorr(lag=1)
        high_autocorr = False if isna(autocorr) else autocorr > 0.95
    else:
        high_autocorr = False

    if has_high_values and not has_negative:
        return True
    if mostly_returns:
        return False
    if high_autocorr and not has_negative and clean_series.max() > 1:
        return True
    return clean_series.min() >= 0 and clean_series.max() > 1


def _calculate_returns(data: Series | DataFrame) -> Series | DataFrame:
    return data.pct_change()


def _clean_data(
    data: Series | DataFrame,
    fill_method: str,
) -> Series | DataFrame:
    result = data.replace([inf, -inf], nan)
    if fill_method == "zero":
        result = result.fillna(0)
    elif fill_method == "drop":
        result = result.dropna()
    elif fill_method != "none":
        raise ValueError(f"Invalid fill_method: {fill_method}")
    return result


def _to_excess_returns(
    returns: Series | DataFrame,
    rf: float,
    nperiods: Optional[int] = None,
) -> Series | DataFrame:
    period_rf = (1 + rf) ** (1 / nperiods) - 1 if nperiods is not None else rf
    return returns - period_rf


def log_returns(
    returns: Series | DataFrame, rf: float = 0.0, nperiods: Optional[int] = None
):
    """
    Shorthand for to_log_returns function

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    rf : float, default 0.0
        Risk-free rate
    nperiods : int, optional
        Number of periods for risk-free rate conversion

    Returns
    -------
    pd.Series or pd.DataFrame
        Log returns
    """
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(returns: Series | DataFrame, rf=0.0, nperiods=None):
    """
    Convert returns series to log returns

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data
    rf : float, default 0.0
        Risk-free rate
    nperiods : int, optional
        Number of periods for risk-free rate conversion

    Returns
    -------
    pd.Series or pd.DataFrame
        Log returns calculated as ln(1 + returns)
    """
    returns: Series | DataFrame = normalize_returns(
        data=returns, rf=rf, nperiods=nperiods
    )
    try:
        # Calculate log returns: ln(1 + returns)
        return log(returns + 1).replace([inf, -inf], float("NaN"))  # type: ignore
    except (ValueError, TypeError, AttributeError, OverflowError) as e:
        from warnings import warn

        warn(f"Error converting to log returns: {type(e).__name__}: {e}, returning 0.0")
        return 0.0


__all__: list[str] = [
    "ensure_datetime_index",
    "rolling_statistic",
    "infer_periods",
    "normalize_returns",
]
__all__: list[str] = [
    "ensure_datetime_index",
    "rolling_statistic",
    "infer_periods",
    "normalize_returns",
]
