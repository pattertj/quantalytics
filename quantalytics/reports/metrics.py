from __future__ import annotations

from dataclasses import dataclass
from math import isnan, sqrt
from typing import cast

import narwhals as nw
import pandas as _pd
from narwhals._native import IntoSeries
from pandas.core.frame import DataFrame
from pandas.core.series import Series

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


@dataclass
class PerformanceMetrics:
    """Container for common performance statistics."""

    annualized_return: float | Series
    annualized_volatility: float | Series
    sharpe: float | Series
    sortino: float | Series
    calmar: float | Series
    max_drawdown: float | Series
    cumulative_return: float | Series
    smart_sharpe: float | Series
    smart_sortino: float | Series
    omega: float | Series
    romad: float | Series
    longest_drawdown_days: float
    average_drawdown: float
    average_drawdown_days: float
    underwater_pct: float
    recovery_factor: float
    ulcer_index: float
    skewness: float | Series
    kurtosis: float | Series
    value_at_risk: float | Series
    expected_shortfall: float | Series
    serenity_index: float | Series
    time_in_market: float
    avg_up_month: float
    avg_down_month: float
    avg_up_day: float
    avg_up_week: float
    avg_up_quarter: float
    avg_up_year: float
    avg_down_day: float
    avg_down_week: float
    avg_down_quarter: float
    avg_down_year: float
    winning_days: int
    losing_days: int
    expected_daily: float
    expected_weekly: float
    expected_monthly: float
    expected_quarterly: float
    expected_yearly: float
    best_day: float
    best_week: float
    worst_day: float
    worst_week: float
    best_month: float
    best_quarter: float
    worst_month: float
    worst_quarter: float
    best_year: float
    worst_year: float

    def as_dict(self) -> dict[str, float | Series]:
        """Return a dictionary representation suitable for DataFrames."""

        return {
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "cumulative_return": self.cumulative_return,
            "smart_sharpe": self.smart_sharpe,
            "smart_sortino": self.smart_sortino,
            "omega": self.omega,
            "romad": self.romad,
            "longest_drawdown_days": self.longest_drawdown_days,
            "average_drawdown": self.average_drawdown,
            "average_drawdown_days": self.average_drawdown_days,
            "underwater_pct": self.underwater_pct,
            "recovery_factor": self.recovery_factor,
            "ulcer_index": self.ulcer_index,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "value_at_risk": self.value_at_risk,
            "expected_shortfall": self.expected_shortfall,
            "serenity_index": self.serenity_index,
            "time_in_market": self.time_in_market,
            "avg_up_month": self.avg_up_month,
            "avg_down_month": self.avg_down_month,
            "avg_up_day": self.avg_up_day,
            "avg_up_week": self.avg_up_week,
            "avg_up_quarter": self.avg_up_quarter,
            "avg_up_year": self.avg_up_year,
            "avg_down_day": self.avg_down_day,
            "avg_down_week": self.avg_down_week,
            "avg_down_quarter": self.avg_down_quarter,
            "avg_down_year": self.avg_down_year,
            "winning_days": self.winning_days,
            "losing_days": self.losing_days,
            "expected_daily": self.expected_daily,
            "expected_weekly": self.expected_weekly,
            "expected_monthly": self.expected_monthly,
            "expected_quarterly": self.expected_quarterly,
            "expected_yearly": self.expected_yearly,
            "best_day": self.best_day,
            "best_week": self.best_week,
            "worst_day": self.worst_day,
            "worst_week": self.worst_week,
            "best_month": self.best_month,
            "best_quarter": self.best_quarter,
            "worst_month": self.worst_month,
            "worst_quarter": self.worst_quarter,
            "best_year": self.best_year,
            "worst_year": self.worst_year,
        }

    def _coerce_numeric(self, value: float | Series) -> float:
        if isinstance(value, Series):
            if value.empty:
                return float("nan")
            value = value.iloc[-1]
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _format_metric(
        self, value: float | Series, scale: float = 1.0, suffix: str = ""
    ) -> str:
        numeric = self._coerce_numeric(value)
        numeric *= scale
        if isnan(numeric):
            return "N/A"
        formatted = f"{numeric:.2f}"
        return f"{formatted}{suffix}" if suffix else formatted

    def _format_days(self, value: float | Series) -> str:
        numeric = self._coerce_numeric(value)
        if isnan(numeric):
            return "0 days"
        return f"{int(round(numeric))} days"

    def risk_adjusted_rows(self) -> list[dict[str, str]]:
        sortino_per_sqrt2 = self._format_metric(self.sortino, scale=1 / sqrt(2))
        smart_sortino_per_sqrt2 = self._format_metric(
            self.smart_sortino, scale=1 / sqrt(2)
        )
        return [
            {"label": "Sharpe Ratio", "value": self._format_metric(self.sharpe)},
            {"label": "Sortino Ratio", "value": self._format_metric(self.sortino)},
            {"label": "Smart Sharpe", "value": self._format_metric(self.smart_sharpe)},
            {
                "label": "Smart Sortino",
                "value": self._format_metric(self.smart_sortino),
            },
            {"label": "Sortino/√2", "value": sortino_per_sqrt2},
            {"label": "Smart Sortino/√2", "value": smart_sortino_per_sqrt2},
            {"label": "Calmar Ratio", "value": self._format_metric(self.calmar)},
            {"label": "Omega Ratio", "value": self._format_metric(self.omega)},
            {"label": "RoMaD", "value": self._format_metric(self.romad)},
        ]

    def volatility_rows(self) -> list[dict[str, str]]:
        return [
            {
                "label": "Annualized Vol",
                "value": self._format_metric(
                    self.annualized_volatility, scale=100, suffix="%"
                ),
            },
            {
                "label": "Max Drawdown",
                "value": self._format_metric(self.max_drawdown, scale=100, suffix="%"),
            },
            {
                "label": "Longest DD Days",
                "value": self._format_days(self.longest_drawdown_days),
            },
            {
                "label": "Average Drawdown",
                "value": self._format_metric(self.average_drawdown, suffix="%"),
            },
            {
                "label": "Average DD Days",
                "value": self._format_days(self.average_drawdown_days),
            },
            {
                "label": "Underwater %",
                "value": self._format_metric(self.underwater_pct, suffix="%"),
            },
            {
                "label": "Recovery Factor",
                "value": self._format_metric(self.recovery_factor),
            },
            {"label": "Ulcer Index", "value": self._format_metric(self.ulcer_index)},
        ]

    def tail_rows(self) -> list[dict[str, str]]:
        return [
            {"label": "Skewness", "value": self._format_metric(self.skewness)},
            {"label": "Kurtosis", "value": self._format_metric(self.kurtosis)},
            {
                "label": "Daily VaR",
                "value": self._format_metric(self.value_at_risk, scale=100),
            },
            {
                "label": "Expected Shortfall",
                "value": self._format_metric(self.expected_shortfall, scale=100),
            },
            {
                "label": "Serenity Index",
                "value": self._format_metric(self.serenity_index),
            },
        ]

    def consistency_rows(self) -> list[dict[str, str]]:
        return [
            {
                "label": "Time in Market",
                "value": self._format_metric(
                    self.time_in_market, scale=100, suffix="%"
                ),
            },
            {
                "label": "Best Day",
                "value": self._format_metric(self.best_day, scale=100, suffix="%"),
            },
            {
                "label": "Best Week",
                "value": self._format_metric(self.best_week, scale=100, suffix="%"),
            },
            {
                "label": "Best Month",
                "value": self._format_metric(self.best_month, scale=100, suffix="%"),
            },
            {
                "label": "Best Quarter",
                "value": self._format_metric(self.best_quarter, scale=100, suffix="%"),
            },
            {
                "label": "Best Year",
                "value": self._format_metric(self.best_year, scale=100, suffix="%"),
            },
            {
                "label": "Avg Up Day",
                "value": self._format_metric(self.avg_up_day, scale=100, suffix="%"),
            },
            {
                "label": "Avg Up Week",
                "value": self._format_metric(self.avg_up_week, scale=100, suffix="%"),
            },
            {
                "label": "Avg Up Month",
                "value": self._format_metric(self.avg_up_month, scale=100, suffix="%"),
            },
            {
                "label": "Avg Up Quarter",
                "value": self._format_metric(
                    self.avg_up_quarter, scale=100, suffix="%"
                ),
            },
            {
                "label": "Avg Up Year",
                "value": self._format_metric(self.avg_up_year, scale=100, suffix="%"),
            },
            {
                "label": "Expected Daily%",
                "value": self._format_metric(
                    self.expected_daily, scale=100, suffix="%"
                ),
            },
            {
                "label": "Expected Weekly%",
                "value": self._format_metric(
                    self.expected_weekly, scale=100, suffix="%"
                ),
            },
            {
                "label": "Expected Monthly%",
                "value": self._format_metric(
                    self.expected_monthly, scale=100, suffix="%"
                ),
            },
            {
                "label": "Expected Quarterly%",
                "value": self._format_metric(
                    self.expected_quarterly, scale=100, suffix="%"
                ),
            },
            {
                "label": "Expected Yearly%",
                "value": self._format_metric(
                    self.expected_yearly, scale=100, suffix="%"
                ),
            },
            {
                "label": "Avg Down Day",
                "value": self._format_metric(self.avg_down_day, scale=100, suffix="%"),
            },
            {
                "label": "Avg Down Week",
                "value": self._format_metric(self.avg_down_week, scale=100, suffix="%"),
            },
            {
                "label": "Avg Down Month",
                "value": self._format_metric(
                    self.avg_down_month, scale=100, suffix="%"
                ),
            },
            {
                "label": "Avg Down Quarter",
                "value": self._format_metric(
                    self.avg_down_quarter, scale=100, suffix="%"
                ),
            },
            {
                "label": "Avg Down Year",
                "value": self._format_metric(self.avg_down_year, scale=100, suffix="%"),
            },
            {
                "label": "Worst Day",
                "value": self._format_metric(self.worst_day, scale=100, suffix="%"),
            },
            {
                "label": "Worst Week",
                "value": self._format_metric(self.worst_week, scale=100, suffix="%"),
            },
            {
                "label": "Worst Month",
                "value": self._format_metric(self.worst_month, scale=100, suffix="%"),
            },
            {
                "label": "Worst Quarter",
                "value": self._format_metric(self.worst_quarter, scale=100, suffix="%"),
            },
            {
                "label": "Worst Year",
                "value": self._format_metric(self.worst_year, scale=100, suffix="%"),
            },
            {"label": "Winning Days", "value": f"{int(self.winning_days)}"},
            {"label": "Losing Days", "value": f"{int(self.losing_days)}"},
        ]


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
    returns_pd = returns.to_pandas()

    if prepare_returns:
        returns_pd = normalize_returns(data=returns_pd)
    returns_pd.index = _pd.to_datetime(returns_pd.index, errors="coerce")
    returns_pd = returns_pd[~returns_pd.index.isna()]
    original_returns = returns_pd.copy()

    # Group returns by month (first day of each month)
    monthly_grouped = _utils.group_returns(
        returns_pd, returns_pd.index.strftime("%Y-%m-01"), compounded
    )

    # Create DataFrame with proper structure
    returns_df = _pd.DataFrame(monthly_grouped)
    returns_df.columns = ["Returns"]
    datetime_index = _pd.DatetimeIndex(
        _pd.to_datetime(returns_df.index, errors="coerce")
    )
    returns_df.index = datetime_index

    # Extract year and month for pivot table
    index_for_format = cast(_pd.DatetimeIndex, returns_df.index)
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


def performance_summary(
    returns: Series | DataFrame,
    risk_free_rate: float = 0.0,
    periods: int | None = None,
) -> PerformanceMetrics:
    """Generate a collection of core performance metrics."""

    series = normalize_returns(data=returns)
    base_series = series.iloc[:, 0] if isinstance(series, DataFrame) else series

    if base_series.empty:
        zero = 0.0
        return PerformanceMetrics(
            annualized_return=zero,
            annualized_volatility=zero,
            sharpe=zero,
            sortino=zero,
            calmar=zero,
            max_drawdown=zero,
            cumulative_return=zero,
            smart_sharpe=zero,
            smart_sortino=zero,
            omega=zero,
            romad=zero,
            longest_drawdown_days=0.0,
            average_drawdown=0.0,
            average_drawdown_days=0.0,
            underwater_pct=0.0,
            recovery_factor=0.0,
            ulcer_index=0.0,
            skewness=zero,
            kurtosis=zero,
            value_at_risk=zero,
            expected_shortfall=zero,
            serenity_index=zero,
            time_in_market=0.0,
            avg_up_month=0.0,
            avg_down_month=0.0,
            avg_up_day=0.0,
            avg_up_week=0.0,
            avg_up_quarter=0.0,
            avg_up_year=0.0,
            avg_down_day=0.0,
            avg_down_week=0.0,
            avg_down_quarter=0.0,
            avg_down_year=0.0,
            winning_days=0,
            losing_days=0,
            expected_daily=0.0,
            expected_weekly=0.0,
            expected_monthly=0.0,
            expected_quarterly=0.0,
            expected_yearly=0.0,
            best_day=0.0,
            best_week=0.0,
            worst_day=0.0,
            worst_week=0.0,
            best_month=0.0,
            best_quarter=0.0,
            worst_month=0.0,
            worst_quarter=0.0,
            best_year=0.0,
            worst_year=0.0,
        )

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
    longest_dd_days = float(details["days"].max()) if not details.empty else 0.0
    average_drawdown = (
        float(details["max drawdown"].mean()) if not details.empty else 0.0
    )
    average_dd_days = float(details["days"].mean()) if not details.empty else 0.0
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
    return PerformanceMetrics(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe=sharpe_ratio,
        sortino=sortino_ratio,
        calmar=calmar_ratio,
        max_drawdown=mdd,
        cumulative_return=cum_return,
        smart_sharpe=smart_sharpe_ratio,
        smart_sortino=smart_sortino_ratio,
        omega=omega_ratio,
        romad=romad_ratio,
        longest_drawdown_days=longest_dd_days,
        average_drawdown=average_drawdown,
        average_drawdown_days=average_dd_days,
        underwater_pct=underwater_pct,
        recovery_factor=recovery,
        ulcer_index=ulcer_idx,
        skewness=skewness,
        kurtosis=kurt,
        value_at_risk=var_value,
        expected_shortfall=es_value,
        serenity_index=serenity,
        time_in_market=time_in_market,
        avg_up_month=avg_up_month,
        avg_down_month=avg_down_month,
        avg_up_day=avg_up_day,
        avg_up_week=avg_up_week,
        avg_up_quarter=avg_up_quarter,
        avg_up_year=avg_up_year,
        avg_down_day=avg_down_day,
        avg_down_week=avg_down_week,
        avg_down_quarter=avg_down_quarter,
        avg_down_year=avg_down_year,
        winning_days=winning_days,
        losing_days=losing_days,
        expected_daily=expected_daily,
        expected_weekly=expected_weekly,
        expected_monthly=expected_monthly,
        expected_quarterly=expected_quarterly,
        expected_yearly=expected_yearly,
        best_day=best_day,
        best_week=best_week,
        worst_day=worst_day,
        worst_week=worst_week,
        best_month=best_month,
        worst_month=worst_month,
        best_quarter=best_quarter,
        worst_quarter=worst_quarter,
        best_year=best_year,
        worst_year=worst_year,
    )
