"""HTML tear sheet generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import narwhals as nw
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.environment import Template
from narwhals._native import IntoSeries

from quantalytics.analytics import rolling_sharpe, rolling_sortino
from quantalytics.analytics.metrics import to_drawdown_series
from quantalytics.analytics.stats import (
    avg_loss,
    avg_win,
    best,
    comp,
    compsum,
    drawdown_details,
    expected_return,
    rolling_volatility,
    win_rate,
    worst,
)

from ..charts.timeseries import (
    cumulative_returns_chart,
    drawdown_chart,
    rolling_volatility_chart,
)
from .metric_registry import resolve_summary_specs
from .metrics import monthly_returns, performance_summary

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)


@dataclass
class TearsheetSection:
    """Describes a section of the tear sheet."""

    title: str
    description: str
    figure_html: Optional[str] = None


@dataclass
class Tearsheet:
    """Represents a rendered tear sheet."""

    html: str

    def to_html(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.html, encoding="utf-8")


def _package_version(name: str = "quantalytics") -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "0.0.0"


def _scalar_value(value: float | pd.Series) -> float:
    if isinstance(value, pd.Series):
        if value.empty:
            return float("nan")
        return float(value.iloc[-1])
    return float(value)


def _format_date_iso(value) -> str:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            raise ValueError
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _format_date_readable(value) -> str:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            raise ValueError
        return f"{ts.day} {ts:%b %Y}"
    except Exception:
        return str(value)


def _format_index_dates(index) -> list[str]:
    def _as_index(value):
        if isinstance(value, pd.Timestamp):
            return pd.DatetimeIndex([value])
        return pd.DatetimeIndex(value)

    try:
        dates = _as_index(index)
        dates = dates[~dates.isna()]
        return dates.strftime("%Y-%m-%d").tolist()
    except Exception:
        converted = pd.to_datetime(index, errors="coerce")
        dates = _as_index(converted)
        dates = dates[~dates.isna()]
        return dates.strftime("%Y-%m-%d").tolist()


def _period_return(series: pd.Series, start: pd.Timestamp) -> float:
    subset = series[series.index >= start]
    if subset.empty:
        return 0.0
    return float(comp(returns=subset))


@dataclass(frozen=True)
class _SummarySpec:
    key: str
    label: str
    tooltip: str | None = None
    scale: float = 1.0
    suffix: str = ""
    decimals: int = 2


SUMMARY_METRIC_CONFIG: dict[str, _SummarySpec] = {
    "annualized_return": _SummarySpec(
        key="annualized_return", label="CAGR", scale=100, suffix="%", decimals=2
    ),
    "sharpe": _SummarySpec(key="sharpe", label="Sharpe Ratio", decimals=2),
    "max_drawdown": _SummarySpec(
        key="max_drawdown", label="Max Drawdown", scale=100, suffix="%", decimals=2
    ),
    "win_rate": _SummarySpec(
        key="win_rate", label="Win Rate", scale=100, suffix="%", decimals=2
    ),
    "romad": _SummarySpec(key="romad", label="RoMaD", decimals=2),
    "sortino": _SummarySpec(key="sortino", label="Sortino", decimals=2),
}
DEFAULT_SUMMARY_KEYS: list[str] = list(SUMMARY_METRIC_CONFIG.keys())


def _resolve_summary_specs(keys: Iterable[str] | None) -> list[_SummarySpec]:
    requested = list(keys) if keys is not None else DEFAULT_SUMMARY_KEYS
    specs: list[_SummarySpec] = []
    for key in requested:
        spec = SUMMARY_METRIC_CONFIG.get(key)
        if spec is None:
            raise ValueError(f"Unsupported summary stat '{key}'.")
        specs.append(spec)
    return specs


def _format_summary_metric(value: Any, scale: float, decimals: int, suffix: str) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = _scalar_value(value)
    except Exception:
        return "N/A"
    numeric *= scale
    if math.isnan(numeric):
        return "N/A"
    formatted = f"{numeric:.{decimals}f}"
    return f"{formatted}{suffix}"


@nw.narwhalify(eager_only=True)
def html(
    returns: IntoSeries,
    title: str = "Strategy Tearsheet",
    benchmark: Optional[IntoSeries] = None,
    risk_free_rate: float = 0.0,
    periods: int | None = None,
    log_scale: bool = False,
    sections: List[TearsheetSection] | None = None,
    header_logo: str | None = None,
    parameters: Mapping[str, str] | None = None,
    summary_stats: Iterable[str] | None = None,
) -> Tearsheet:
    """Render a high-level tear sheet from series of returns."""

    pandas_returns = returns.to_pandas()
    pandas_bench = benchmark.to_pandas() if benchmark else None

    if pandas_returns.empty:
        pandas_returns = pandas_returns.copy()
        pandas_returns.index = pd.DatetimeIndex([])

    coverage_text = "Data coverage unavailable"
    if not pandas_returns.empty:
        try:
            dates = pd.DatetimeIndex(
                pd.to_datetime(pandas_returns.index, errors="coerce")
            )
            dates = dates[~dates.isna()]
            if dates.size:
                coverage_text = f"{_format_date_readable(dates.min())} → {_format_date_readable(dates.max())}"
        except Exception:
            coverage_text = "Data coverage unavailable"

    metrics = performance_summary(
        returns=pandas_returns,
        risk_free_rate=risk_free_rate,
        periods=periods,
    )

    metrics_dict = metrics.as_dict()
    wins = int(metrics_dict.get("winning_days", 0))
    losses = int(metrics_dict.get("losing_days", 0))
    total = max(1, wins + losses)
    metrics_dict["win_rate"] = wins / total
    summary_specs = resolve_summary_specs(summary_stats)
    summary_stat_cards = [
        {
            "label": spec.label,
            "tooltip": spec.tooltip,
            "value": _format_summary_metric(
                metrics_dict.get(spec.value_key),
                scale=spec.scale,
                decimals=spec.decimals,
                suffix=spec.suffix,
            ),
        }
        for spec in summary_specs
    ]

    summary_series = pd.Series(dtype=float)
    summary_series.index = pd.DatetimeIndex([])

    default_sections: list[TearsheetSection] = [
        TearsheetSection(
            title="Cumulative Returns",
            description=r"Growth of $1 invested in the strategy versus benchmark.",
            figure_html=_safe_chart(
                cumulative_returns_chart,
                returns=pandas_returns,
                benchmark=pandas_bench,
                log_scale=log_scale,
            ),
        ),
        TearsheetSection(
            title="Rolling Volatility",
            description="Rolling measure of realized volatility.",
            figure_html=_safe_chart(
                rolling_volatility_chart,
                returns=pandas_returns,
            ),
        ),
        TearsheetSection(
            title="Drawdowns",
            description="Depth and duration of drawdowns over time.",
            figure_html=_safe_chart(
                drawdown_chart,
                returns=pandas_returns,
            ),
        ),
    ]
    sections = list(sections) if sections is not None else default_sections

    worst_drawdowns: list[dict] = []
    eoy_rows: list[dict[str, float | str]] = []
    eoy_years: list[str] = []
    eoy_returns: list[float] = []
    eoy_average: float = 0.0
    per_year_best: list[float] = []
    per_year_avg_up: list[float] = []
    per_year_expected: list[float] = []
    per_year_avg_down: list[float] = []
    per_year_worst: list[float] = []
    per_year_win_rate: list[float] = []
    per_year_time_in_market: list[float] = []
    rolling_vol_trimmed: list[float] = []
    rolling_vol_dates_trimmed: list[str] = []
    longest_drawdowns: list[dict[str, str]] = []
    monthly_df = None
    if not pandas_returns.empty:
        try:
            monthly_df = monthly_returns(
                returns=pandas_returns,
                eoy=False,
                compounded=True,
                prepare_returns=False,
            )
        except Exception:
            monthly_df = None

    if pandas_returns.empty or monthly_df is None:
        heatmap_months = []
        heatmap_years = []
        heatmap_values = []
        win_rate_periods = [
            ("Day", None),
            ("Week", "week"),
            ("Month", "month"),
            ("Quarter", "quarter"),
            ("Year", "year"),
        ]
        win_rate_buckets = [name for name, _ in win_rate_periods]
        win_rate_values = [0.0 for _ in win_rate_periods]
        period_rows = []
        daily_dates = []
        daily_returns = []
        eoy_rows = []
        eoy_years = []
        eoy_returns = []
        eoy_average = 0.0
        daily_dates: list[str] = []
        daily_returns: list[float] = []
        rolling_sharpe_values: list[float | None] = []
        rolling_sortino_values: list[float | None] = []
        rolling_sharpe_trimmed = []
        rolling_sharpe_dates_trimmed: list[str] = []
        rolling_sortino_trimmed = []
        rolling_sortino_dates_trimmed: list[str] = []
        underwater_series = []
        rolling_sortino_values: list[float | None] = []
    else:
        heatmap_months = [month.capitalize() for month in monthly_df.columns]
        heatmap_years = monthly_df.index.astype(str).tolist()
        heatmap_values = (
            (monthly_df.fillna(0) * 100).round(2).astype(float).values.tolist()
        )
        win_rate_periods = [
            ("Day", None),
            ("Week", "week"),
            ("Month", "month"),
            ("Quarter", "quarter"),
            ("Year", "year"),
        ]
        win_rate_buckets = [name for name, _ in win_rate_periods]
        win_rate_values = [
            round(
                win_rate(
                    returns=pandas_returns,
                    aggregate=agg,
                    compounded=True,
                    prepare_returns=False,
                )
                * 100,
                2,
            )
            for _, agg in win_rate_periods
        ]
        sorted_returns = pandas_returns.sort_index()
        first_date = sorted_returns.index[0]
        last_date = sorted_returns.index[-1]
        summary_series = compsum(sorted_returns)
        if log_scale:
            summary_series = (1 + summary_series).clip(lower=1e-6)
        period_map = [
            ("MTD", last_date - pd.DateOffset(months=1)),
            ("3M", last_date - pd.DateOffset(months=3)),
            ("6M", last_date - pd.DateOffset(months=6)),
            ("YTD", last_date.replace(month=1, day=1)),
            ("1Y", last_date - pd.DateOffset(years=1)),
            ("3Y", last_date - pd.DateOffset(years=3)),
            ("5Y", last_date - pd.DateOffset(years=5)),
            ("10Y", last_date - pd.DateOffset(years=10)),
            ("All-time", first_date),
        ]
        period_rows = []
        for label, start in period_map:
            start = max(start, first_date)
            value = _period_return(series=sorted_returns, start=start)
            period_rows.append({"label": label, "value": f"{value * 100:.2f}%"})
        daily_dates = sorted_returns.index.strftime("%Y-%m-%d").tolist()
        daily_returns = (sorted_returns * 100).round(2).tolist()
        eoy_rows = []
        per_year_best: list[float] = []
        per_year_avg_up: list[float] = []
        per_year_expected: list[float] = []
        per_year_avg_down: list[float] = []
        per_year_worst: list[float] = []
        per_year_win_rate: list[float] = []
        per_year_time_in_market: list[float] = []
        for year, group in sorted_returns.groupby(sorted_returns.index.year):
            year_return = comp(group)
            period_start = group.index.min()
            period_end = group.index.max()
            period_days = max(1, (period_end - period_start).days + 1)
            period_years = max(period_days / 365, 1 / 365)
            year_cagr = (1 + year_return) ** (1 / period_years) - 1
            per_year_best.append(
                float(
                    round(
                        best(
                            group,
                            aggregate=None,
                            compounded=True,
                            prepare_returns=False,
                        )
                        * 100,
                        2,
                    )
                )
            )
            per_year_avg_up.append(
                float(
                    round(
                        avg_win(
                            group,
                            aggregate=None,
                            compounded=True,
                            prepare_returns=False,
                        )
                        * 100,
                        2,
                    )
                )
            )
            per_year_expected.append(
                float(
                    round(
                        expected_return(group, compounded=True, prepare_returns=False)
                        * 100,
                        2,
                    )
                )
            )
            per_year_avg_down.append(
                float(
                    round(
                        avg_loss(
                            group,
                            aggregate=None,
                            compounded=True,
                            prepare_returns=False,
                        )
                        * 100,
                        2,
                    )
                )
            )
            per_year_worst.append(
                float(
                    round(
                        worst(
                            group,
                            aggregate=None,
                            compounded=True,
                            prepare_returns=False,
                        )
                        * 100,
                        2,
                    )
                )
            )
            year_win = win_rate(
                group, aggregate=None, compounded=True, prepare_returns=False
            )
            per_year_win_rate.append(float(round(year_win * 100, 2)))
            time_in_mkt = (
                float(((group.abs() > 0).sum() / len(group)) * 100)
                if len(group)
                else 0.0
            )
            per_year_time_in_market.append(float(round(time_in_mkt, 2)))
            eoy_rows.append(
                {
                    "year": str(year),
                    "annual_return": float(round(year_return * 100, 2)),
                    "cumulative_return": float(round(year_cagr * 100, 2)),
                }
            )
        eoy_years = [str(row["year"]) for row in eoy_rows]
        eoy_returns = [float(row["annual_return"]) for row in eoy_rows]
        eoy_average = (
            round(float(sum(eoy_returns)) / len(eoy_returns), 2) if eoy_returns else 0.0
        )
        window = min(len(pandas_returns), 126) if pandas_returns.size else 0

        def _trim_prefix(
            values: list[float | None], axis: list[str]
        ) -> tuple[list[float], list[str]]:
            start = 0
            for i, value in enumerate(values):
                if value is not None:
                    start = i
                    break
            else:
                return [], []
            trimmed_values = [value for value in values[start:] if value is not None]
            trimmed_dates = axis[start:]
            return trimmed_values, trimmed_dates

        if window >= 3:
            rolling_series = rolling_sharpe(
                returns=pandas_returns,
                rolling_period=window,
                prepare_returns=False,
            )
            rolling_sortino_series = rolling_sortino(
                returns=pandas_returns,
                rolling_period=window,
                prepare_returns=False,
            )
            rolling_sharpe_values = [
                None
                if val is None or (isinstance(val, float) and math.isnan(val))
                else float(val)
                for val in rolling_series
            ]
            rolling_sortino_values = [
                None
                if val is None or (isinstance(val, float) and math.isnan(val))
                else float(val)
                for val in rolling_sortino_series
            ]
        else:
            rolling_sharpe_values = []
            rolling_sortino_values = []

        rolling_sharpe_trimmed = []
        rolling_sharpe_dates_trimmed: list[str] = []
        rolling_sortino_trimmed = []
        rolling_sortino_dates_trimmed: list[str] = []
        rolling_vol_trimmed = []
        rolling_vol_dates_trimmed: list[str] = []
        if pandas_returns.size:
            axis_dates = sorted_returns.index.strftime("%Y-%m-%d").tolist()
            rolling_sharpe_trimmed, rolling_sharpe_dates_trimmed = _trim_prefix(
                rolling_sharpe_values, axis_dates
            )
            rolling_sortino_trimmed, rolling_sortino_dates_trimmed = _trim_prefix(
                rolling_sortino_values, axis_dates
            )
            if window >= 3:
                rolling_vol_series = rolling_volatility(
                    returns=pandas_returns,
                    rolling_period=window,
                    periods=periods,
                    prepare_returns=False,
                )
                rolling_vol_values = [
                    None
                    if val is None or (isinstance(val, float) and math.isnan(val))
                    else float(val)
                    for val in rolling_vol_series
                ]
                rolling_vol_trimmed, rolling_vol_dates_trimmed = _trim_prefix(
                    rolling_vol_values, axis_dates
                )
        drawdown_series = to_drawdown_series(
            returns=sorted_returns, prepare_returns=False
        )
        underwater_series = (drawdown_series * 100).round(2).astype(float).tolist()
        details = drawdown_details(drawdown_series)
        worst_drawdown_df = pd.DataFrame()
        if not details.empty:
            worst_drawdown_df = details.sort_values(
                "max drawdown", ascending=False
            ).head(10)
        worst_drawdowns = [
            {
                "start": _format_date_iso(row["start"]),
                "valley": _format_date_iso(row["valley"]),
                "end": _format_date_iso(row["end"]),
                "drawdown": f"{row['max drawdown']:.2f}%",
                "days": int(row["days"]),
            }
            for _, row in worst_drawdown_df.iterrows()
        ]
        longest = (
            details.sort_values("days", ascending=False).head(5)
            if not details.empty
            else pd.DataFrame()
        )
        longest_drawdowns = [
            {
                "start": _format_date_iso(row["start"]),
                "end": _format_date_iso(row["end"]),
            }
            for _, row in longest.iterrows()
        ]

    template: Template = _TEMPLATE_ENV.get_template("tearsheet.html")
    parameter_rows = list(parameters.items()) if parameters else []

    risk_adjusted_rows = metrics.risk_adjusted_rows()
    vol_rows = metrics.volatility_rows()
    tail_rows = metrics.tail_rows()
    consistency_rows = metrics.consistency_rows()

    sharpe_baseline = _scalar_value(metrics.sharpe)
    sortino_baseline = _scalar_value(metrics.sortino)
    vol_baseline = _scalar_value(metrics.annualized_volatility)
    summary_dates = _format_index_dates(summary_series.index)
    html = template.render(
        title=title,
        metrics=metrics.as_dict(),
        sections=sections,
        header_primary=coverage_text,
        header_secondary=f"Generated by Quantalytics (v{_package_version()})",
        header_logo=header_logo,
        parameter_rows=parameter_rows,
        risk_adjusted_rows=risk_adjusted_rows,
        vol_rows=vol_rows,
        tail_rows=tail_rows,
        consistency_rows=consistency_rows,
        heatmap_months=heatmap_months,
        heatmap_years=heatmap_years,
        heatmap_values=heatmap_values,
        win_rate_buckets=win_rate_buckets,
        win_rate_values=win_rate_values,
        period_rows=period_rows,
        daily_dates=daily_dates,
        daily_returns=daily_returns,
        eoy_rows=eoy_rows,
        eoy_years=eoy_years,
        eoy_returns=eoy_returns,
        eoy_average=eoy_average,
        rolling_sharpe_series=rolling_sharpe_trimmed,
        rolling_sharpe_dates=rolling_sharpe_dates_trimmed,
        rolling_sortino_series=rolling_sortino_trimmed,
        rolling_sortino_dates=rolling_sortino_dates_trimmed,
        rolling_vol_series=rolling_vol_trimmed,
        rolling_vol_dates=rolling_vol_dates_trimmed,
        sharpe_baseline=sharpe_baseline,
        sortino_baseline=sortino_baseline,
        vol_baseline=vol_baseline,
        underwater_series=underwater_series,
        summary_dates=summary_dates,
        summary_values=(summary_series * 100).round(2).tolist(),
        worst_drawdowns=worst_drawdowns,
        longest_drawdowns=longest_drawdowns,
        summary_stat_cards=summary_stat_cards,
        per_year_best=per_year_best,
        per_year_avg_up=per_year_avg_up,
        per_year_expected=per_year_expected,
        per_year_avg_down=per_year_avg_down,
        per_year_worst=per_year_worst,
        per_year_win_rate=per_year_win_rate,
        per_year_time_in_market=per_year_time_in_market,
    )
    return Tearsheet(html=html)


def _figure_to_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _safe_chart(chart_func, *args, **kwargs) -> str | None:
    try:
        return _figure_to_html(chart_func(*args, **kwargs))
    except ValueError:
        return None


__all__: list[str] = [
    "Tearsheet",
    "TearsheetSection",
    "html",
]
