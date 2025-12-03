from __future__ import annotations

from typing import Any, cast

import narwhals as nw
from narwhals._native import IntoDataFrame, IntoSeries

from quantalytics.analytics import (
    calmar,
    conditional_value_at_risk,
    kurtosis,
    max_drawdown,
    omega,
    recovery_factor,
    romad,
    serenity_index,
    sharpe,
    skew,
    smart_sharpe,
    smart_sortino,
    sortino,
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
) -> IntoSeries:
    """Generate a collection of core performance metrics.

    Returns:
        IntoSeries: Narwhals Series with all metrics as rows (index=metric names, values=metric values).
            The decorator automatically converts this to the parent backend type (pandas/Polars/etc).
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

    cum_return = comp(returns=series)
    ann_ret = cagr(returns=series, periods=periods)
    ann_vol = volatility(returns=series, periods=periods)
    sharpe_ratio = sharpe(returns=series, rf=risk_free_rate, periods=periods)
    sortino_ratio = sortino(returns=series, rf=risk_free_rate, periods=periods)
    calmar_ratio = calmar(returns=series, periods=periods)
    mdd = max_drawdown(returns=series)
    smart_sharpe_ratio = smart_sharpe(
        returns=series, rf=risk_free_rate, periods=periods
    )
    smart_sortino_ratio = smart_sortino(
        returns=series, rf=risk_free_rate, periods=periods
    )
    omega_ratio = omega(returns=series)
    romad_ratio = romad(returns=series, periods=periods)
    drawdown_series = to_drawdown_series(returns=base_series, prepare_returns=False)
    details = drawdown_details(drawdown_series)
    longest_dd_days = 0.0 if details.empty else float(details["days"].max())
    average_drawdown = 0.0 if details.empty else float(details["max drawdown"].mean())
    average_dd_days = 0.0 if details.empty else float(details["days"].mean())
    underwater_pct = (
        float(((drawdown_series < 0).sum() / len(drawdown_series)) * 100)
        if len(drawdown_series)
        else 0.0
    )
    recovery = recovery_factor(
        returns=base_series, rf=risk_free_rate, prepare_returns=False
    )
    ulcer_idx = ulcer_index(returns=base_series, prepare_returns=False)
    skewness = skew(returns=series)
    kurt = kurtosis(returns=series)
    var_value = value_at_risk(returns=base_series, confidence=0.95)
    es_value = conditional_value_at_risk(returns=base_series, confidence=0.95)
    serenity = serenity_index(returns=series, rf=risk_free_rate)
    time_in_market = (
        float((base_series.abs() > 0).sum() / len(base_series))
        if len(base_series)
        else 0.0
    )
    avg_up_day = float(
        avg_win(
            returns=base_series,
            aggregate=None,
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_day = float(
        avg_loss(
            returns=base_series,
            aggregate=None,
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_week = float(
        avg_win(
            returns=base_series,
            aggregate="W",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_week = float(
        avg_loss(
            returns=base_series,
            aggregate="W",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_month = float(
        avg_win(
            returns=base_series,
            aggregate="ME",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_month = float(
        avg_loss(
            returns=base_series,
            aggregate="ME",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_quarter = float(
        avg_win(
            returns=base_series,
            aggregate="QE",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_quarter = float(
        avg_loss(
            returns=base_series,
            aggregate="QE",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_up_year = float(
        avg_win(
            returns=base_series,
            aggregate="YE",
            compounded=True,
            prepare_returns=False,
        )
    )
    avg_down_year = float(
        avg_loss(
            returns=base_series,
            aggregate="YE",
            compounded=True,
            prepare_returns=False,
        )
    )
    nonzero_days = int((base_series != 0).sum())
    daily_win_ratio = win_rate(
        returns=base_series, aggregate=None, compounded=True, prepare_returns=False
    )
    winning_days = int(round(daily_win_ratio * nonzero_days))
    losing_days = max(0, nonzero_days - winning_days)
    expected_daily = float(
        expected_return(returns=base_series, compounded=True, prepare_returns=False)
    )
    expected_weekly = float(
        expected_return(
            returns=base_series, aggregate="W", compounded=True, prepare_returns=False
        )
    )
    expected_monthly = float(
        expected_return(
            returns=base_series, aggregate="ME", compounded=True, prepare_returns=False
        )
    )
    expected_quarterly = float(
        expected_return(
            returns=base_series, aggregate="QE", compounded=True, prepare_returns=False
        )
    )
    expected_yearly = float(
        expected_return(
            returns=base_series, aggregate="YE", compounded=True, prepare_returns=False
        )
    )
    best_day = float(best(base_series))
    worst_day = float(worst(base_series))
    best_week = float(best(base_series, aggregate="W"))
    worst_week = float(worst(base_series, aggregate="W"))
    best_month = float(best(base_series, aggregate="ME"))
    worst_month = float(worst(base_series, aggregate="ME"))
    best_quarter = float(best(base_series, aggregate="QE"))
    worst_quarter = float(worst(base_series, aggregate="QE"))
    best_year = float(best(base_series, aggregate="YE"))
    worst_year = float(worst(base_series, aggregate="YE"))

    # Create metrics dictionary
    metrics_dict = {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe_ratio,
        "sortino": sortino_ratio,
        "calmar": calmar_ratio,
        "max_drawdown": mdd,
        "cumulative_return": cum_return,
        "smart_sharpe": smart_sharpe_ratio,
        "smart_sortino": smart_sortino_ratio,
        "omega": omega_ratio,
        "romad": romad_ratio,
        "longest_drawdown_days": longest_dd_days,
        "average_drawdown": average_drawdown,
        "average_drawdown_days": average_dd_days,
        "underwater_pct": underwater_pct,
        "recovery_factor": recovery,
        "ulcer_index": ulcer_idx,
        "skewness": skewness,
        "kurtosis": kurt,
        "value_at_risk": var_value,
        "expected_shortfall": es_value,
        "serenity_index": serenity,
        "time_in_market": time_in_market,
        "avg_up_month": avg_up_month,
        "avg_down_month": avg_down_month,
        "avg_up_day": avg_up_day,
        "avg_up_week": avg_up_week,
        "avg_up_quarter": avg_up_quarter,
        "avg_up_year": avg_up_year,
        "avg_down_day": avg_down_day,
        "avg_down_week": avg_down_week,
        "avg_down_quarter": avg_down_quarter,
        "avg_down_year": avg_down_year,
        "winning_days": winning_days,
        "losing_days": losing_days,
        "expected_daily": expected_daily,
        "expected_weekly": expected_weekly,
        "expected_monthly": expected_monthly,
        "expected_quarterly": expected_quarterly,
        "expected_yearly": expected_yearly,
        "best_day": best_day,
        "best_week": best_week,
        "worst_day": worst_day,
        "worst_week": worst_week,
        "best_month": best_month,
        "worst_month": worst_month,
        "best_quarter": best_quarter,
        "worst_quarter": worst_quarter,
        "best_year": best_year,
        "worst_year": worst_year,
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
    from math import sqrt

    # Convert to dict for easy access
    if hasattr(metrics_series, "to_dict"):
        m = metrics_series.to_dict()
    else:
        m = dict(metrics_series)

    sortino_per_sqrt2 = _format_metric(m.get("sortino", 0.0), scale=1 / sqrt(2))
    smart_sortino_per_sqrt2 = _format_metric(
        m.get("smart_sortino", 0.0), scale=1 / sqrt(2)
    )

    return [
        {"label": "Sharpe Ratio", "value": _format_metric(m.get("sharpe", 0.0))},
        {"label": "Sortino Ratio", "value": _format_metric(m.get("sortino", 0.0))},
        {"label": "Smart Sharpe", "value": _format_metric(m.get("smart_sharpe", 0.0))},
        {
            "label": "Smart Sortino",
            "value": _format_metric(m.get("smart_sortino", 0.0)),
        },
        {"label": "Sortino/√2", "value": sortino_per_sqrt2},
        {"label": "Smart Sortino/√2", "value": smart_sortino_per_sqrt2},
        {"label": "Calmar Ratio", "value": _format_metric(m.get("calmar", 0.0))},
        {"label": "Omega Ratio", "value": _format_metric(m.get("omega", 0.0))},
        {"label": "RoMaD", "value": _format_metric(m.get("romad", 0.0))},
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

    return [
        {
            "label": "Annualized Vol",
            "value": _format_metric(
                m.get("annualized_volatility", 0.0), scale=100, suffix="%"
            ),
        },
        {
            "label": "Max Drawdown",
            "value": _format_metric(m.get("max_drawdown", 0.0), scale=100, suffix="%"),
        },
        {
            "label": "Longest DD Days",
            "value": _format_days(m.get("longest_drawdown_days", 0.0)),
        },
        {
            "label": "Average Drawdown",
            "value": _format_metric(m.get("average_drawdown", 0.0), suffix="%"),
        },
        {
            "label": "Average DD Days",
            "value": _format_days(m.get("average_drawdown_days", 0.0)),
        },
        {
            "label": "Underwater %",
            "value": _format_metric(m.get("underwater_pct", 0.0), suffix="%"),
        },
        {
            "label": "Recovery Factor",
            "value": _format_metric(m.get("recovery_factor", 0.0)),
        },
        {"label": "Ulcer Index", "value": _format_metric(m.get("ulcer_index", 0.0))},
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

    return [
        {"label": "Skewness", "value": _format_metric(m.get("skewness", 0.0))},
        {"label": "Kurtosis", "value": _format_metric(m.get("kurtosis", 0.0))},
        {
            "label": "Daily VaR",
            "value": _format_metric(m.get("value_at_risk", 0.0), scale=100),
        },
        {
            "label": "Expected Shortfall",
            "value": _format_metric(m.get("expected_shortfall", 0.0), scale=100),
        },
        {
            "label": "Serenity Index",
            "value": _format_metric(m.get("serenity_index", 0.0)),
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
