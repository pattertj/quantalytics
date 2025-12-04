from __future__ import annotations

from typing import Any, cast

import narwhals as nw
from narwhals._native import IntoDataFrame, IntoSeries

from quantalytics.analytics import (
    benchmark_correlation,
    calmar,
    conditional_value_at_risk,
    greeks,
    information_ratio,
    kurtosis,
    max_drawdown,
    omega,
    probabilistic_sharpe_ratio,
    r_squared,
    recovery_factor,
    romad,
    serenity_index,
    sharpe,
    skew,
    smart_sharpe,
    smart_sortino,
    sortino,
    treynor_ratio,
    ulcer_index,
    value_at_risk,
)
from quantalytics.analytics.metrics import to_drawdown_series
from quantalytics.analytics.stats import (
    avg_loss,
    avg_win,
    best,
    cagr,
    comp,
    drawdown_details,
    expected_return,
    volatility,
    win_rate,
    worst,
)
from quantalytics.utils import timeseries as _utils
from quantalytics.utils.timeseries import normalize_returns


def _get_period_return(returns_series, months_back: int | None = None) -> float:
    """Calculate return for a trailing period in months.

    Args:
        returns_series: pandas Series with DatetimeIndex
        months_back: Number of months to look back (None = all time)

    Returns:
        Cumulative return for the period, or 0.0 if insufficient data
    """
    import pandas as pd
    from dateutil.relativedelta import relativedelta

    if len(returns_series) == 0:
        return 0.0

    # Ensure we have a DatetimeIndex
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        return 0.0

    end_date = returns_series.index[-1]

    if months_back is None:
        # All-time return
        period_data = returns_series
    else:
        # Calculate start date
        start_date = end_date - relativedelta(months=months_back)
        period_data = returns_series[returns_series.index >= start_date]

    if len(period_data) == 0:
        return 0.0

    # Calculate cumulative return
    return float((1 + period_data).prod() - 1)


def _get_mtd_return(returns_series) -> float:
    """Calculate month-to-date return.

    Args:
        returns_series: pandas Series with DatetimeIndex

    Returns:
        MTD cumulative return, or 0.0 if no data
    """
    import pandas as pd

    if len(returns_series) == 0:
        return 0.0

    if not isinstance(returns_series.index, pd.DatetimeIndex):
        return 0.0

    end_date = returns_series.index[-1]
    # First day of current month
    month_start = pd.Timestamp(year=end_date.year, month=end_date.month, day=1)

    mtd_data = returns_series[returns_series.index >= month_start]

    if len(mtd_data) == 0:
        return 0.0

    return float((1 + mtd_data).prod() - 1)


def _get_qtd_return(returns_series) -> float:
    """Calculate quarter-to-date return.

    Args:
        returns_series: pandas Series with DatetimeIndex

    Returns:
        QTD cumulative return, or 0.0 if no data
    """
    import pandas as pd

    if len(returns_series) == 0:
        return 0.0

    if not isinstance(returns_series.index, pd.DatetimeIndex):
        return 0.0

    end_date = returns_series.index[-1]
    # Calculate quarter start month (1, 4, 7, or 10)
    quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
    quarter_start = pd.Timestamp(year=end_date.year, month=quarter_start_month, day=1)

    qtd_data = returns_series[returns_series.index >= quarter_start]

    if len(qtd_data) == 0:
        return 0.0

    return float((1 + qtd_data).prod() - 1)


def _get_ytd_return(returns_series) -> float:
    """Calculate year-to-date return.

    Args:
        returns_series: pandas Series with DatetimeIndex

    Returns:
        YTD cumulative return, or 0.0 if no data
    """
    import pandas as pd

    if len(returns_series) == 0:
        return 0.0

    if not isinstance(returns_series.index, pd.DatetimeIndex):
        return 0.0

    end_date = returns_series.index[-1]
    # First day of current year
    year_start = pd.Timestamp(year=end_date.year, month=1, day=1)

    ytd_data = returns_series[returns_series.index >= year_start]

    if len(ytd_data) == 0:
        return 0.0

    return float((1 + ytd_data).prod() - 1)


def _annualize_return(total_return: float, years: float) -> float:
    """Annualize a total return given the number of years.

    Args:
        total_return: Total cumulative return
        years: Number of years

    Returns:
        Annualized return (CAGR)
    """
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def _round_metric(value: Any, decimals: int = 4) -> float:
    """Round a metric value to the specified number of decimal places.

    Args:
        value: The metric value to round (int, float, or Series)
        decimals: Number of decimal places (default: 4)

    Returns:
        Rounded float value
    """
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        # If conversion fails, return 0.0
        return 0.0


@nw.narwhalify(eager_only=True)
def monthly_returns(
    returns: IntoSeries,
    eoy: bool = True,
    compounded: bool = True,
    prepare_returns: bool = True,
):
    """Calculate monthly returns organized in a pivot table format.

    Aggregates return data by month and year, creating a table where rows represent
    years and columns represent months. Optionally includes end-of-year (EOY) returns
    as an additional column. This format is commonly used in performance tearsheets
    and allows for easy visualization of seasonality and year-over-year comparisons.

    Args:
        returns (IntoSeriesT): Time series of portfolio returns with DatetimeIndex.
            Can be daily, weekly, or any frequency that will be aggregated to monthly.
        eoy (bool, optional): Whether to include end-of-year returns as a separate column.
            If True, adds an 'EOY' column showing annual returns for each year.
            Defaults to True.
        compounded (bool, optional): Whether to compound returns when aggregating.
            If True, uses geometric compounding: (1+r1)*(1+r2)-1.
            If False, uses arithmetic sum: r1+r2.
            Defaults to True.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Normalization handles various input formats and cleans the data.
            Defaults to True.

    Returns:
        DataFrame: Pivot table with years as rows and months as columns.
            Columns are ordered chronologically (JAN through DEC, plus EOY if eoy=True).
            All column names are uppercase. Missing months are filled with 0.
            Index name is set to None for cleaner display.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        >>> returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        >>> monthly = monthly_returns(returns)
        >>> monthly.columns
        Index(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'EOY'], dtype='object')
        >>> monthly.loc['2023', 'JAN']  # January 2023 returns
        0.0234

        >>> # Without EOY column
        >>> monthly_no_eoy = monthly_returns(returns, eoy=False)
        >>> monthly_no_eoy.columns
        Index(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], dtype='object')

        >>> # Arithmetic aggregation instead of compounding
        >>> monthly_simple = monthly_returns(returns, compounded=False)

    Notes:
        - Returns are grouped by first day of month (YYYY-MM-01) for aggregation
        - Missing months in the data are filled with 0, not NaN
        - All 12 month columns are always present, even if there's no data
        - Month abbreviations follow standard format: Jan, Feb, Mar, etc.
        - Years are represented as strings in the index
        - EOY returns represent the full year's performance when eoy=True
        - Compounded returns are typically more accurate for multi-period performance
        - The function uses pandas operations for pivot table creation
        - Input must have a DatetimeIndex for proper month/year extraction

    See Also:
        cagr: Compound Annual Growth Rate calculation
        group_returns: Underlying function for aggregating returns by period
        normalize_returns: Return normalization and cleaning
    """
    # Convert narwhals series to pandas for compatibility with existing utils
    import pandas as pd

    returns_pd = returns.to_pandas()

    if prepare_returns:
        returns_pd = normalize_returns(data=returns_pd)
    returns_pd.index = pd.to_datetime(returns_pd.index, errors="coerce")
    returns_pd = returns_pd[~returns_pd.index.isna()]
    original_returns = returns_pd.copy()

    # Group returns by month (first day of each month)
    monthly_grouped = _utils.group_returns(
        returns_pd, returns_pd.index.strftime("%Y-%m-01"), compounded
    )

    # Create DataFrame with proper structure
    returns_df = pd.DataFrame(monthly_grouped)
    returns_df.columns = ["Returns"]
    datetime_index = pd.DatetimeIndex(pd.to_datetime(returns_df.index, errors="coerce"))
    returns_df.index = datetime_index

    # Extract year and month for pivot table
    index_for_format = cast(pd.DatetimeIndex, returns_df.index)
    returns_df["Year"] = index_for_format.strftime("%Y")
    returns_df["Month"] = index_for_format.strftime("%b")

    # Create pivot table: rows=years, columns=months, values=returns
    pivot = returns_df.pivot(index="Year", columns="Month", values="Returns").fillna(0)

    # Ensure all 12 months are present as columns
    all_months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for month in all_months:
        if month not in pivot.columns:
            pivot.loc[:, month] = 0

    # Order columns chronologically
    pivot = pivot[all_months]

    # Add end-of-year column if requested
    if eoy:
        pivot["eoy"] = _utils.group_returns(
            original_returns, original_returns.index.year, compounded=compounded
        ).values

    # Convert all column names to uppercase for consistency
    pivot.columns = map(lambda x: str(x).upper(), pivot.columns)
    pivot.index.name = None

    # Convert back to narwhals DataFrame to let decorator handle native type conversion
    return nw.from_native(pivot)


@nw.narwhalify(eager_only=True)
def metrics(
    returns: IntoSeries | IntoDataFrame,
    risk_free_rate: float = 0.0,
    periods: int | None = None,
    benchmark: IntoSeries | IntoDataFrame | None = None,
) -> IntoSeries:
    """Generate a collection of core performance metrics matching quantstats_lumi API.

    Args:
        returns: Portfolio returns series or dataframe
        risk_free_rate: Risk-free rate (annualized)
        periods: Number of periods per year (e.g., 252 for daily)
        benchmark: Optional benchmark returns for benchmark-dependent metrics

    Returns:
        IntoSeries: Narwhals Series with all metrics as rows (index=metric names, values=metric values).
            The decorator automatically converts this to the parent backend type (pandas/Polars/etc).
            Metric names and order match quantstats_lumi for compatibility.
    """

    # Convert narwhals series/dataframe to pandas for compatibility with existing utils
    returns_pd = returns.to_pandas()
    series = normalize_returns(data=returns_pd)
    # Check if series is a DataFrame by checking for ndim attribute or shape
    base_series = (
        series.iloc[:, 0] if hasattr(series, "ndim") and series.ndim == 2 else series
    )

    if base_series.empty:
        zero = 0.0
        metrics_dict = {
            "annualized_return": zero,
            "annualized_volatility": zero,
            "sharpe": zero,
            "sortino": zero,
            "calmar": zero,
            "max_drawdown": zero,
            "cumulative_return": zero,
            "smart_sharpe": zero,
            "smart_sortino": zero,
            "omega": zero,
            "romad": zero,
            "longest_drawdown_days": 0.0,
            "average_drawdown": 0.0,
            "average_drawdown_days": 0.0,
            "underwater_pct": 0.0,
            "recovery_factor": 0.0,
            "ulcer_index": 0.0,
            "skewness": zero,
            "kurtosis": zero,
            "value_at_risk": zero,
            "expected_shortfall": zero,
            "serenity_index": zero,
            "time_in_market": 0.0,
            "avg_up_month": 0.0,
            "avg_down_month": 0.0,
            "avg_up_day": 0.0,
            "avg_up_week": 0.0,
            "avg_up_quarter": 0.0,
            "avg_up_year": 0.0,
            "avg_down_day": 0.0,
            "avg_down_week": 0.0,
            "avg_down_quarter": 0.0,
            "avg_down_year": 0.0,
            "winning_days": 0,
            "losing_days": 0,
            "expected_daily": 0.0,
            "expected_weekly": 0.0,
            "expected_monthly": 0.0,
            "expected_quarterly": 0.0,
            "expected_yearly": 0.0,
            "best_day": 0.0,
            "best_week": 0.0,
            "worst_day": 0.0,
            "worst_week": 0.0,
            "best_month": 0.0,
            "best_quarter": 0.0,
            "worst_month": 0.0,
            "worst_quarter": 0.0,
            "best_year": 0.0,
            "worst_year": 0.0,
        }
        # Create native series from the backend that was passed in
        # Get the native module from the input (pandas, polars, etc.)
        native_namespace = nw.get_native_namespace(returns)
        if hasattr(native_namespace, "Series"):
            # For pandas and pandas-like backends
            metrics_series = native_namespace.Series(metrics_dict)
        else:
            # Fallback: use nw.new_series with the values and index separately
            import pandas as pd

            metrics_series = pd.Series(metrics_dict)
        return nw.from_native(metrics_series, series_only=True)

    cum_return = _round_metric(comp(returns=series))
    ann_ret = _round_metric(cagr(returns=series, periods=periods))
    ann_vol = _round_metric(volatility(returns=series, periods=periods))
    sharpe_ratio = _round_metric(
        sharpe(returns=series, rf=risk_free_rate, periods=periods)
    )
    sortino_ratio = _round_metric(
        sortino(returns=series, rf=risk_free_rate, periods=periods)
    )
    calmar_ratio = _round_metric(calmar(returns=series, periods=periods))
    mdd = _round_metric(max_drawdown(returns=series))
    smart_sharpe_ratio = _round_metric(
        smart_sharpe(returns=series, rf=risk_free_rate, periods=periods)
    )
    smart_sortino_ratio = _round_metric(
        smart_sortino(returns=series, rf=risk_free_rate, periods=periods)
    )
    omega_ratio = _round_metric(omega(returns=series))
    romad_ratio = _round_metric(romad(returns=series, periods=periods))
    drawdown_series = to_drawdown_series(returns=base_series, prepare_returns=False)
    details = drawdown_details(drawdown_series)
    longest_dd_days = 0.0 if details.empty else _round_metric(details["days"].max())
    average_drawdown = (
        0.0 if details.empty else _round_metric(details["max drawdown"].mean())
    )
    average_dd_days = 0.0 if details.empty else _round_metric(details["days"].mean())
    underwater_pct = (
        _round_metric(((drawdown_series < 0).sum() / len(drawdown_series)) * 100)
        if len(drawdown_series)
        else 0.0
    )
    recovery = _round_metric(
        recovery_factor(returns=base_series, rf=risk_free_rate, prepare_returns=False)
    )
    ulcer_idx = _round_metric(ulcer_index(returns=base_series, prepare_returns=False))
    skewness = _round_metric(skew(returns=series))
    kurt = _round_metric(kurtosis(returns=series))
    var_value = _round_metric(value_at_risk(returns=base_series, confidence=0.95))
    es_value = _round_metric(
        conditional_value_at_risk(returns=base_series, confidence=0.95)
    )
    serenity = _round_metric(serenity_index(returns=series, rf=risk_free_rate))
    time_in_market = (
        _round_metric((base_series.abs() > 0).sum() / len(base_series))
        if len(base_series)
        else 0.0
    )
    avg_up_day = _round_metric(
        avg_win(
            returns=base_series,
            aggregate=None,
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_day = _round_metric(
        avg_loss(
            returns=base_series,
            aggregate=None,
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_week = _round_metric(
        avg_win(
            returns=base_series,
            aggregate="W",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_week = _round_metric(
        avg_loss(
            returns=base_series,
            aggregate="W",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_month = _round_metric(
        avg_win(
            returns=base_series,
            aggregate="ME",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_month = _round_metric(
        avg_loss(
            returns=base_series,
            aggregate="ME",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_quarter = _round_metric(
        avg_win(
            returns=base_series,
            aggregate="QE",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_quarter = _round_metric(
        avg_loss(
            returns=base_series,
            aggregate="QE",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_year = _round_metric(
        avg_win(
            returns=base_series,
            aggregate="YE",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_year = _round_metric(
        avg_loss(
            returns=base_series,
            aggregate="YE",
            compounded=True,
            prepare_returns=False,
        )
    )
    nonzero_days = int((base_series != 0).sum())
    daily_win_ratio = _round_metric(
        win_rate(
            returns=base_series, aggregate=None, compounded=True, prepare_returns=False
        )
    )
    winning_days = int(round(daily_win_ratio * nonzero_days))
    losing_days = max(0, nonzero_days - winning_days)
    expected_daily = _round_metric(
        expected_return(returns=base_series, compounded=True, prepare_returns=False)
    )
    expected_weekly = _round_metric(
        expected_return(
            returns=base_series, aggregate="W", compounded=True, prepare_returns=False
        )
    )
    expected_monthly = _round_metric(
        expected_return(
            returns=base_series, aggregate="ME", compounded=True, prepare_returns=False
        )
    )
    expected_quarterly = _round_metric(
        expected_return(
            returns=base_series, aggregate="QE", compounded=True, prepare_returns=False
        )
    )
    expected_yearly = _round_metric(
        expected_return(
            returns=base_series, aggregate="YE", compounded=True, prepare_returns=False
        )
    )
    best_day = _round_metric(best(base_series))
    worst_day = _round_metric(worst(base_series))
    best_week = _round_metric(best(base_series, aggregate="W"))
    worst_week = _round_metric(worst(base_series, aggregate="W"))
    best_month = _round_metric(best(base_series, aggregate="ME"))
    worst_month = _round_metric(worst(base_series, aggregate="ME"))
    best_quarter = _round_metric(best(base_series, aggregate="QE"))
    worst_quarter = _round_metric(worst(base_series, aggregate="QE"))
    best_year = _round_metric(best(base_series, aggregate="YE"))
    worst_year = _round_metric(worst(base_series, aggregate="YE"))

    # Calculate new metrics for quantstats_lumi compatibility
    import pandas as pd

    # Convert benchmark if provided
    benchmark_pd = None
    if benchmark is not None:
        benchmark_pd = (
            benchmark.to_pandas() if hasattr(benchmark, "to_pandas") else benchmark
        )
        benchmark_pd = normalize_returns(data=benchmark_pd)
        if hasattr(benchmark_pd, "ndim") and benchmark_pd.ndim == 2:
            benchmark_pd = benchmark_pd.iloc[:, 0]

    # Start and End Period
    start_period = (
        str(base_series.index[0].date())
        if isinstance(base_series.index, pd.DatetimeIndex) and len(base_series) > 0
        else ""
    )
    end_period = (
        str(base_series.index[-1].date())
        if isinstance(base_series.index, pd.DatetimeIndex) and len(base_series) > 0
        else ""
    )

    # Probabilistic Sharpe Ratio
    psr = _round_metric(
        probabilistic_sharpe_ratio(series, rf=risk_free_rate, periods=periods)
    )

    # Sortino / sqrt(2) ratios
    from math import sqrt

    sortino_sqrt2 = _round_metric(sortino_ratio / sqrt(2))
    smart_sortino_sqrt2 = _round_metric(smart_sortino_ratio / sqrt(2))

    # Period returns
    mtd_return = _round_metric(_get_mtd_return(base_series))
    ytd_return = _round_metric(_get_ytd_return(base_series))
    return_3m = _round_metric(_get_period_return(base_series, months_back=3))
    return_6m = _round_metric(_get_period_return(base_series, months_back=6))
    return_1y = _round_metric(_get_period_return(base_series, months_back=12))
    return_3y = _round_metric(_get_period_return(base_series, months_back=36))
    return_5y = _round_metric(_get_period_return(base_series, months_back=60))
    return_10y = _round_metric(_get_period_return(base_series, months_back=120))

    # Ann ualized multi-year returns
    return_3y_ann = _round_metric(_annualize_return(return_3y, 3.0))
    return_5y_ann = _round_metric(_annualize_return(return_5y, 5.0))
    return_10y_ann = _round_metric(_annualize_return(return_10y, 10.0))
    # All-time annualized is just CAGR
    return_alltime_ann = ann_ret

    # Win rates by period
    monthly_win_ratio = _round_metric(
        win_rate(base_series, aggregate="ME", compounded=True, prepare_returns=False)
    )
    quarterly_win_ratio = _round_metric(
        win_rate(base_series, aggregate="QE", compounded=True, prepare_returns=False)
    )
    yearly_win_ratio = _round_metric(
        win_rate(base_series, aggregate="YE", compounded=True, prepare_returns=False)
    )

    # Benchmark-dependent metrics (only if benchmark provided)
    correlation = 0.0
    r2_value = 0.0
    info_ratio = 0.0
    beta = 0.0
    alpha = 0.0
    treynor = 0.0

    if benchmark_pd is not None and len(benchmark_pd) > 0:
        try:
            # Note: benchmark_pd is already normalized, but these functions will
            # normalize again with prepare_returns=True (default). This is safe.
            correlation = _round_metric(
                benchmark_correlation(base_series, benchmark_pd)
            )
            r2_value = _round_metric(r_squared(base_series, benchmark_pd))
            info_ratio = _round_metric(information_ratio(base_series, benchmark_pd))
            greeks_dict = greeks(
                base_series,
                benchmark_pd,
                periods=float(periods) if periods is not None else 365.0,
            )
            beta = _round_metric(greeks_dict.get("beta", 0.0))
            alpha = _round_metric(greeks_dict.get("alpha", 0.0))
            treynor = _round_metric(
                treynor_ratio(
                    base_series,
                    benchmark_pd,
                    rf=risk_free_rate,
                    periods=float(periods) if periods is not None else 365.0,
                )
            )
        except Exception:  # nosec B110
            # If benchmark calculations fail, keep zeros
            pass

    # Create metrics dictionary with quantstats_lumi ordering
    metrics_dict = {
        # Metadata
        "Start Period": start_period,
        "End Period": end_period,
        "Risk-Free Rate %": risk_free_rate * 100,
        "Time in Market %": time_in_market * 100,
        # Core Returns
        "Cumulative Return %": cum_return * 100,
        "CAGR﹪": ann_ret * 100,
        # Risk-Adjusted Ratios
        "Sharpe": sharpe_ratio,
        "Prob. Sharpe Ratio %": psr * 100,
        "Smart Sharpe": smart_sharpe_ratio,
        "Sortino": sortino_ratio,
        "Smart Sortino": smart_sortino_ratio,
        "Sortino/√2": sortino_sqrt2,
        "Smart Sortino/√2": smart_sortino_sqrt2,
        "Omega": omega_ratio,
        "Max Drawdown %": mdd * 100,
        "Longest DD Days": longest_dd_days,
        "Calmar": calmar_ratio,
        "RoMaD": romad_ratio,
        "Volatility (ann.) %": ann_vol * 100,
        "Skew": skewness,
        "Kurtosis": kurt,
    }

    # Add benchmark metrics if benchmark provided
    if benchmark_pd is not None:
        metrics_dict["R^2"] = r2_value
        metrics_dict["Information Ratio"] = info_ratio

    # Continue with remaining metrics
    metrics_dict.update(
        {
            # Expected Returns
            "Expected Daily %": expected_daily * 100,
            "Expected Weekly %": expected_weekly * 100,
            "Expected Monthly %": expected_monthly * 100,
            "Expected Quarterly %": expected_quarterly * 100,
            "Expected Yearly %": expected_yearly * 100,
            # Risk Metrics
            "Daily Value-at-Risk %": var_value * 100,
            "Expected Shortfall (cVaR) %": es_value * 100,
            # Period Returns
            "MTD %": mtd_return * 100,
            "3M %": return_3m * 100,
            "6M %": return_6m * 100,
            "YTD %": ytd_return * 100,
            "1Y %": return_1y * 100,
            "3Y (ann.) %": return_3y_ann * 100,
            "5Y (ann.) %": return_5y_ann * 100,
            "10Y (ann.) %": return_10y_ann * 100,
            "All-time (ann.) %": return_alltime_ann * 100,
            # Best/Worst Performance
            "Best Day %": best_day * 100,
            "Worst Day %": worst_day * 100,
            "Best Week %": best_week * 100,
            "Worst Week %": worst_week * 100,
            "Best Month %": best_month * 100,
            "Worst Month %": worst_month * 100,
            "Best Quarter %": best_quarter * 100,
            "Worst Quarter %": worst_quarter * 100,
            "Best Year %": best_year * 100,
            "Worst Year %": worst_year * 100,
            # Drawdown Details
            "Avg. Drawdown %": average_drawdown,  # Already as percentage
            "Avg. Drawdown Days": average_dd_days,
            "Underwater %": underwater_pct,  # Already as percentage
            # Recovery Metrics
            "Recovery Factor": recovery,
            "Ulcer Index": ulcer_idx,
            "Serenity Index": serenity,
            # Win/Loss Analysis
            "Avg. Up Day %": avg_up_day * 100,
            "Avg. Up Week %": avg_up_week * 100,
            "Avg. Up Month %": avg_up_month * 100,
            "Avg. Up Quarter %": avg_up_quarter * 100,
            "Avg. Up Year %": avg_up_year * 100,
            "Avg. Down Day %": avg_down_day * 100,
            "Avg. Down Week %": avg_down_week * 100,
            "Avg. Down Month %": avg_down_month * 100,
            "Avg. Down Quarter %": avg_down_quarter * 100,
            "Avg. Down Year %": avg_down_year * 100,
            "Win Days %": daily_win_ratio * 100,
            "Win Month %": monthly_win_ratio * 100,
            "Win Quarter %": quarterly_win_ratio * 100,
            "Win Year %": yearly_win_ratio * 100,
            "Winning Days": winning_days,
            "Losing Days": losing_days,
        }
    )

    # Add Greek metrics if benchmark provided
    if benchmark_pd is not None:
        metrics_dict["Beta"] = beta
        metrics_dict["Alpha (ann.) %"] = alpha * 100
        metrics_dict["Correlation"] = correlation
        metrics_dict["Treynor Ratio (ann.) %"] = treynor * 100

    # Create native series from the backend that was passed in
    # Get the native module from the input (pandas, polars, etc.)
    native_namespace = nw.get_native_namespace(returns)
    if hasattr(native_namespace, "Series"):
        # For pandas and pandas-like backends
        metrics_series = native_namespace.Series(metrics_dict)
    else:
        # Fallback: use nw.new_series with the values and index separately
        import pandas as pd

        metrics_series = pd.Series(metrics_dict)
    return nw.from_native(metrics_series, series_only=True)


# Backward compatibility alias
performance_summary = metrics


# Helper functions for formatting metrics (used by tearsheet)
def _coerce_numeric(value: float) -> float:
    """Convert a metric value to a numeric float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_metric(value: float, scale: float = 1.0, suffix: str = "") -> str:
    """Format a metric value with optional scaling and suffix."""
    from math import isnan

    numeric = _coerce_numeric(value)
    numeric *= scale
    if isnan(numeric):
        return "N/A"
    formatted = f"{numeric:.2f}"
    return f"{formatted}{suffix}" if suffix else formatted


def _format_days(value: float) -> str:
    """Format a metric value as days."""
    from math import isnan

    numeric = _coerce_numeric(value)
    if isnan(numeric):
        return "0 days"
    return f"{int(round(numeric))} days"


def format_risk_adjusted_rows(metrics_series: Any) -> list[dict[str, str]]:
    """Format risk-adjusted metrics for display.

    Args:
        metrics_series: Series containing metric values (can be pandas or other backend)

    Returns:
        List of dicts with 'label' and 'value' keys formatted for display
    """
    # Convert to dict for easy access
    if hasattr(metrics_series, "to_dict"):
        m = metrics_series.to_dict()
    else:
        m = dict(metrics_series)

    # Use new metric names (with backward compatibility for old names)
    return [
        {
            "label": "Sharpe Ratio",
            "value": _format_metric(m.get("Sharpe", m.get("sharpe", 0.0))),
        },
        {
            "label": "Prob. Sharpe Ratio %",
            "value": _format_metric(m.get("Prob. Sharpe Ratio %", 0.0)),
        },
        {
            "label": "Sortino Ratio",
            "value": _format_metric(m.get("Sortino", m.get("sortino", 0.0))),
        },
        {
            "label": "Smart Sharpe",
            "value": _format_metric(m.get("Smart Sharpe", m.get("smart_sharpe", 0.0))),
        },
        {
            "label": "Smart Sortino",
            "value": _format_metric(
                m.get("Smart Sortino", m.get("smart_sortino", 0.0))
            ),
        },
        {"label": "Sortino/√2", "value": _format_metric(m.get("Sortino/√2", 0.0))},
        {
            "label": "Smart Sortino/√2",
            "value": _format_metric(m.get("Smart Sortino/√2", 0.0)),
        },
        {
            "label": "Calmar Ratio",
            "value": _format_metric(m.get("Calmar", m.get("calmar", 0.0))),
        },
        {
            "label": "Omega Ratio",
            "value": _format_metric(m.get("Omega", m.get("omega", 0.0))),
        },
        {
            "label": "RoMaD",
            "value": _format_metric(m.get("RoMaD", m.get("romad", 0.0))),
        },
    ]


def format_volatility_rows(metrics_series: Any) -> list[dict[str, str]]:
    """Format volatility metrics for display.

    Args:
        metrics_series: Series containing metric values (can be pandas or other backend)

    Returns:
        List of dicts with 'label' and 'value' keys formatted for display
    """
    # Convert to dict for easy access
    if hasattr(metrics_series, "to_dict"):
        m = metrics_series.to_dict()
    else:
        m = dict(metrics_series)

    # Use new metric names with backward compatibility
    # New names already have % scaling applied, so check if value > 1 to determine if it's already scaled
    def get_vol(key_new, key_old):
        val = m.get(key_new, m.get(key_old, 0.0))
        # If using new key and value looks like it's already scaled (> 1), return as-is with suffix
        if key_new in m and abs(val) > 1:
            return _format_metric(val, scale=1, suffix="%")
        # Otherwise scale it
        return _format_metric(val, scale=100, suffix="%")

    return [
        {
            "label": "Annualized Vol",
            "value": get_vol("Volatility (ann.) %", "annualized_volatility"),
        },
        {
            "label": "Max Drawdown",
            "value": get_vol("Max Drawdown %", "max_drawdown"),
        },
        {
            "label": "Longest DD Days",
            "value": _format_days(
                m.get("Longest DD Days", m.get("longest_drawdown_days", 0.0))
            ),
        },
        {
            "label": "Average Drawdown",
            "value": _format_metric(
                m.get("Avg. Drawdown %", m.get("average_drawdown", 0.0)), suffix="%"
            ),
        },
        {
            "label": "Average DD Days",
            "value": _format_days(
                m.get("Avg. Drawdown Days", m.get("average_drawdown_days", 0.0))
            ),
        },
        {
            "label": "Underwater %",
            "value": _format_metric(
                m.get("Underwater %", m.get("underwater_pct", 0.0)), suffix="%"
            ),
        },
        {
            "label": "Recovery Factor",
            "value": _format_metric(
                m.get("Recovery Factor", m.get("recovery_factor", 0.0))
            ),
        },
        {
            "label": "Ulcer Index",
            "value": _format_metric(m.get("Ulcer Index", m.get("ulcer_index", 0.0))),
        },
    ]


def format_tail_rows(metrics_series: Any) -> list[dict[str, str]]:
    """Format tail risk metrics for display.

    Args:
        metrics_series: Series containing metric values (can be pandas or other backend)

    Returns:
        List of dicts with 'label' and 'value' keys formatted for display
    """
    # Convert to dict for easy access
    if hasattr(metrics_series, "to_dict"):
        m = metrics_series.to_dict()
    else:
        m = dict(metrics_series)

    # Helper to handle potentially already-scaled values
    def get_scaled(key_new, key_old):
        val = m.get(key_new, m.get(key_old, 0.0))
        if key_new in m and abs(val) > 1:
            return _format_metric(val, scale=1)
        return _format_metric(val, scale=100)

    return [
        {
            "label": "Skewness",
            "value": _format_metric(m.get("Skew", m.get("skewness", 0.0))),
        },
        {
            "label": "Kurtosis",
            "value": _format_metric(m.get("Kurtosis", m.get("kurtosis", 0.0))),
        },
        {
            "label": "Daily VaR",
            "value": get_scaled("Daily Value-at-Risk %", "value_at_risk"),
        },
        {
            "label": "Expected Shortfall",
            "value": get_scaled("Expected Shortfall (cVaR) %", "expected_shortfall"),
        },
        {
            "label": "Serenity Index",
            "value": _format_metric(
                m.get("Serenity Index", m.get("serenity_index", 0.0))
            ),
        },
    ]


def format_consistency_rows(metrics_series: Any) -> list[dict[str, str]]:
    """Format consistency metrics for display.

    Args:
        metrics_series: Series containing metric values (can be pandas or other backend)

    Returns:
        List of dicts with 'label' and 'value' keys formatted for display
    """
    # Convert to dict for easy access
    if hasattr(metrics_series, "to_dict"):
        m = metrics_series.to_dict()
    else:
        m = dict(metrics_series)

    return [
        {
            "label": "Time in Market",
            "value": _format_metric(
                m.get("time_in_market", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Best Day",
            "value": _format_metric(m.get("best_day", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Best Week",
            "value": _format_metric(m.get("best_week", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Best Month",
            "value": _format_metric(m.get("best_month", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Best Quarter",
            "value": _format_metric(m.get("best_quarter", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Best Year",
            "value": _format_metric(m.get("best_year", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Avg Up Day",
            "value": _format_metric(m.get("avg_up_day", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Avg Up Week",
            "value": _format_metric(m.get("avg_up_week", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Avg Up Month",
            "value": _format_metric(m.get("avg_up_month", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Avg Up Quarter",
            "value": _format_metric(
                m.get("avg_up_quarter", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Avg Up Year",
            "value": _format_metric(m.get("avg_up_year", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Expected Daily%",
            "value": _format_metric(
                m.get("expected_daily", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Expected Weekly%",
            "value": _format_metric(
                m.get("expected_weekly", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Expected Monthly%",
            "value": _format_metric(
                m.get("expected_monthly", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Expected Quarterly%",
            "value": _format_metric(
                m.get("expected_quarterly", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Expected Yearly%",
            "value": _format_metric(
                m.get("expected_yearly", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Avg Down Day",
            "value": _format_metric(m.get("avg_down_day", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Avg Down Week",
            "value": _format_metric(m.get("avg_down_week", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Avg Down Month",
            "value": _format_metric(
                m.get("avg_down_month", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Avg Down Quarter",
            "value": _format_metric(
                m.get("avg_down_quarter", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Avg Down Year",
            "value": _format_metric(m.get("avg_down_year", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Worst Day",
            "value": _format_metric(m.get("worst_day", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Worst Week",
            "value": _format_metric(m.get("worst_week", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Worst Month",
            "value": _format_metric(m.get("worst_month", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Worst Quarter",
            "value": _format_metric(m.get("worst_quarter", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Worst Year",
            "value": _format_metric(m.get("worst_year", 0.0), scale=100, suffix="%"),
        },
        {"label": "Winning Days", "value": f"{int(m.get('winning_days', 0))}"},
        {"label": "Losing Days", "value": f"{int(m.get('losing_days', 0))}"},
    ]
