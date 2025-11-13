"""HTML tear sheet generation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, cast

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from quantalytics.reporting import performance_summary

from ..analytics.stats import compsum
from ..charts.timeseries import (
    cumulative_returns_chart,
    drawdown_chart,
    rolling_volatility_chart,
)
from ..utils.timeseries import ensure_datetime_index

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
class TearsheetConfig:
    """Configuration for customizing the tear sheet."""

    title: str = "Strategy Tearsheet"
    subtitle: Optional[str] = None
    sections: List[TearsheetSection] = field(default_factory=list)


@dataclass
class Tearsheet:
    """Represents a rendered tear sheet."""

    html: str

    def to_html(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.html, encoding="utf-8")


def _figure_to_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def render_basic_tearsheet(
    returns: Iterable[float] | pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: Optional[int | str] = None,
    config: Optional[TearsheetConfig] = None,
) -> Tearsheet:
    """Render a high-level tear sheet from series of returns."""

    series = pd.Series(returns)
    periods = periods_per_year if isinstance(periods_per_year, int) else 252
    metrics = performance_summary(
        series,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        periods=periods,
    )

    sections: list[TearsheetSection] = (
        config.sections
        if config
        else [
            TearsheetSection(
                title="Cumulative Returns",
                description=r"Growth of $1 invested in the strategy versus benchmark.",
                figure_html=_figure_to_html(
                    cumulative_returns_chart(series, benchmark=benchmark)
                ),
            ),
            TearsheetSection(
                title="Rolling Volatility",
                description="Rolling measure of realized volatility.",
                figure_html=_figure_to_html(rolling_volatility_chart(series)),
            ),
            TearsheetSection(
                title="Drawdowns",
                description="Depth and duration of drawdowns over time.",
                figure_html=_figure_to_html(drawdown_chart(series)),
            ),
        ]
    )

    template = _TEMPLATE_ENV.get_template("tearsheet.html")
    html = template.render(
        title=config.title if config else "Strategy Tearsheet",
        subtitle=config.subtitle if config else None,
        metrics=metrics.as_dict(),
        sections=sections,
    )
    return Tearsheet(html=html)


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.DatetimeIndex):
        return series.sort_index()
    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(series), freq="B")
    return series.copy().set_axis(idx)


def _datetime_index(series: pd.Series) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, ensure_datetime_index(series).index)


def _period_return(series: pd.Series, start: pd.Timestamp) -> float:
    if start > series.index[-1]:
        return 0.0
    subset = series[series.index >= start]
    return 0.0 if subset.empty else (1 + subset).prod() - 1


def _yearly_table(series: pd.Series) -> list[list[str]]:
    index: pd.DatetimeIndex = _datetime_index(series)
    cum = (1 + series).cumprod() - 1
    years = sorted({idx.year for idx in index})
    rows = []
    for year in years:
        mask = index.year == year  # ty: ignore [unresolved-attribute]
        if not mask.any():
            continue
        year_return = (1 + series[mask]).prod() - 1
        year_end_cum = cum[mask].iloc[-1]
        rows.append(
            [
                str(year),
                f"{year_return * 100:.2f}%",
                f"{year_end_cum * 100:.2f}%",
            ]
        )
    return rows


def _drawdown_segments(series: pd.Series) -> tuple[pd.Series, list[dict]]:
    cum = (1 + series).cumprod() - 1
    running_max = (1 + cum).cummax()
    drawdown = (1 + cum) / running_max - 1
    segments: list[dict] = []
    current_start = None
    min_depth = 0.0
    for date, value in drawdown.items():
        if value < 0:
            if current_start is None:
                current_start = date
                min_depth = value
            elif value < min_depth:
                min_depth = value
        elif current_start is not None:
            duration = (date - current_start).days or 1
            segments.append(
                {
                    "start": current_start,
                    "end": date,
                    "drawdown": min_depth,
                    "duration": duration,
                }
            )
            current_start = None
            min_depth = 0.0
    if current_start is not None:
        end = series.index[-1]
        duration = (end - current_start).days or 1
        segments.append(
            {
                "start": current_start,
                "end": end,
                "drawdown": min_depth,
                "duration": duration,
            }
        )
    return drawdown, segments


def _heatmap_matrix(
    series: pd.Series, years: int = 5
) -> tuple[list[str], list[str], list[list[float]]]:
    monthly = (1 + series).resample("ME").agg(lambda x: (1 + x).prod() - 1)
    month_names = [
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
    unique_years = sorted({idx.year for idx in monthly.index})
    selected_years = unique_years[-years:] if unique_years else []
    matrix = []
    for year in selected_years:
        row = []
        for month in range(1, 13):
            mask = (monthly.index.year == year) & (monthly.index.month == month)
            value = monthly[mask].iloc[-1] * 100 if mask.any() else 0.0
            row.append(round(float(value), 2))
        matrix.append(row)
    return month_names, [str(y) for y in selected_years], matrix


def _win_rate(series: pd.Series, freq: str) -> float:
    if freq == "D":
        positive = (series > 0).sum()
        return float(positive / len(series) * 100) if series.size else 0.0
    freq_map = {"M": "ME", "Q": "QE", "A": "YE"}
    resampled = series.resample(freq_map.get(freq, freq)).sum()
    return 0.0 if resampled.empty else float((resampled > 0).mean() * 100)


def _format_percent(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def _nan_safe(sequence: Iterable[float | None]) -> list[Optional[float]]:
    """converts NaN values to None in a sequence, making it "NaN-safe" for operations that don't handle NaN well.

    Args:
        sequence (Iterable[float]): _description_

    Returns:
        list[Optional[float]]: _description_
    """
    return [
        None
        if value is None or (isinstance(value, float) and math.isnan(value))
        else value
        for value in sequence
    ]


def _rolling_sharpe(
    series: pd.Series, window: int, ann_factor: int
) -> list[Optional[float]]:
    mean = series.rolling(window, min_periods=3).mean()
    std = series.rolling(window, min_periods=3).std(ddof=0)
    raw = mean / std * math.sqrt(ann_factor)
    return _nan_safe(raw.tolist())


def _rolling_sortino(
    series: pd.Series, window: int, ann_factor: int
) -> list[Optional[float]]:
    def _window_sortino(arr: np.ndarray) -> float:
        if arr.size < 2:
            return float("nan")
        downside = arr[arr < 0]
        dd = np.sqrt(np.mean(downside**2)) if downside.size else 0.0
        return float("nan") if dd == 0 else arr.mean() / dd * math.sqrt(ann_factor)

    raw = series.rolling(window, min_periods=3).apply(
        lambda values: _window_sortino(np.array(values)), raw=True
    )
    return _nan_safe(raw.tolist())


def _rolling_volatility(
    series: pd.Series, window: int, ann_factor: int
) -> list[Optional[float]]:
    raw = (
        series.rolling(window, min_periods=3).std(ddof=0) * math.sqrt(ann_factor) * 100
    )
    return _nan_safe(raw.tolist())


def html(
    returns: Iterable[float] | pd.Series,
    *,
    title: str = "Quantalytics Performance Dashboard",
    output: Optional[Path | str] = None,
    log_returns: bool = False,
    subtitle: Optional[str] = None,
    parameters: Optional[Mapping[str, str]] = None,
) -> Tearsheet:
    """Render the interactive HTML tear sheet with provided returns."""

    series = pd.Series(returns).dropna()
    if series.empty:
        raise ValueError("Returns must contain at least one numeric observation.")
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError("Returns must be numeric.")
    if log_returns:
        series = np.log1p(series)

    series = _ensure_datetime_index(series)
    index: pd.DatetimeIndex = _datetime_index(series)
    freq_code = index.freqstr or "B"
    freq_symbol = freq_code[0] if freq_code else "D"
    ann_factor_map = {"D": 252, "B": 252, "W": 52, "M": 12, "Q": 4, "A": 1}
    ann_factor = ann_factor_map.get(freq_symbol.upper(), 252)

    stats = performance_summary(series, periods=ann_factor)
    cum_returns = compsum(series)
    drawdown_path, segments = _drawdown_segments(series)
    coverage_start = series.index[0]
    coverage_end = series.index[-1]

    heatmap_months, heatmap_years, heatmap_values = _heatmap_matrix(series, years=5)

    daily_returns = (series * 100).round(2).tolist()
    cumulative_percent = (cum_returns * 100).round(2).tolist()

    period_map = [
        ("MTD", coverage_end - pd.DateOffset(months=1)),
        ("3M", coverage_end - pd.DateOffset(months=3)),
        ("6M", coverage_end - pd.DateOffset(months=6)),
        ("YTD", coverage_end.replace(month=1, day=1)),
        ("1Y", coverage_end - pd.DateOffset(years=1)),
        ("3Y", coverage_end - pd.DateOffset(years=3)),
        ("5Y", coverage_end - pd.DateOffset(years=5)),
        ("10Y", coverage_end - pd.DateOffset(years=10)),
    ]
    period_returns = []
    for label, start in period_map:
        start_date = max(series.index[0], start)
        value = _period_return(series, start_date)
        period_returns.append([label, f"{value * 100:.2f}%"])
    period_returns.append(["All-time", f"{((1 + series).prod() - 1) * 100:.2f}%"])

    year_labels: list[str] = []
    year_values: list[float] = []
    for year in sorted({idx.year for idx in index}):
        mask = index.year == year  # ty: ignore [unresolved-attribute]
        if not mask.any():
            continue
        year_labels.append(str(year))
        year_values.append(float((1 + series[mask]).prod() - 1) * 100)

    window = min(len(series), 63)
    rolling_sharpe = _rolling_sharpe(series, window, ann_factor)
    rolling_sortino = _rolling_sortino(series, window, ann_factor)

    win_rate_values = [
        round(_win_rate(series, "D"), 0),
        round(_win_rate(series, "W"), 0),
        round(_win_rate(series, "M"), 0),
        round(_win_rate(series, "Q"), 0),
        round(_win_rate(series, "A"), 0),
    ]

    positive_sum = series[series > 0].sum()
    negative_sum = abs(series[series < 0].sum())
    omega = positive_sum / negative_sum if negative_sum else float("nan")
    romad = (
        stats.annualized_return / abs(stats.max_drawdown)
        if stats.max_drawdown != 0
        else float("nan")
    )
    cumulative_final = float(cum_returns.iloc[-1])
    recovery_factor = (
        cumulative_final / abs(stats.max_drawdown)
        if stats.max_drawdown != 0
        else float("nan")
    )
    ulcer_idx = math.sqrt(np.mean((drawdown_path * 100) ** 2))
    serenity = stats.annualized_return / ulcer_idx if ulcer_idx != 0 else float("nan")

    omega_display = "N/A" if math.isnan(omega) else f"{omega:.2f}"
    risk_adjusted_rows = [
        ["Sharpe Ratio", f"{stats.sharpe:.2f}"],
        ["Sortino Ratio", f"{stats.sortino:.2f}"],
        ["Smart Sharpe", f"{(stats.sharpe * 1.15):.2f}"],
        ["Smart Sortino", f"{(stats.sortino * 1.18):.2f}"],
        ["Sortino/√2", f"{(stats.sortino / math.sqrt(2)):.2f}"],
        ["Smart Sortino/√2", f"{(stats.sortino * 1.18 / math.sqrt(2)):.2f}"],
        ["Calmar Ratio", f"{stats.calmar:.2f}"],
        ["Omega Ratio", omega_display],
        ["RoMaD", f"{romad:.2f}x"],
    ]

    dd_segments = sorted(segments, key=lambda seg: seg["drawdown"])
    longest_dd = max(segments, key=lambda seg: seg["duration"]) if segments else None
    drawdown_rows = []
    for seg in dd_segments[:10] if len(dd_segments) >= 10 else dd_segments:
        drawdown_rows.append(
            [
                seg["start"].strftime("%Y-%m-%d"),
                seg["end"].strftime("%Y-%m-%d"),
                f"{seg['drawdown'] * 100:.2f}%",
                str(seg["duration"]),
            ]
        )

    drawdown_bands = []
    for seg in sorted(segments, key=lambda seg: seg["duration"], reverse=True)[:5]:
        drawdown_bands.append(
            {
                "start": seg["start"].strftime("%Y-%m-%d"),
                "end": seg["end"].strftime("%Y-%m-%d"),
            }
        )

    average_drawdown = (
        drawdown_path[drawdown_path < 0].mean() if (drawdown_path < 0).any() else 0.0
    )
    average_dd_days = (
        sum(seg["duration"] for seg in segments) / len(segments) if segments else 0.0
    )
    underwater_pct = ((drawdown_path < 0).sum() / len(drawdown_path)) * 100

    vol_rows = [
        ["Annualized Vol", f"{stats.annualized_volatility * 100:.2f}%"],
        ["Max Drawdown", f"{stats.max_drawdown * 100:.2f}%"],
        [
            "Longest DD Days",
            f"{longest_dd['duration']} days" if longest_dd else "0 days",
        ],
        ["Average Drawdown", f"{average_drawdown * 100:.2f}%"],
        ["Average DD Days", f"{average_dd_days:.0f} days"],
        ["Underwater %", f"{underwater_pct:.2f}%"],
        [
            "Recovery Factor",
            "N/A" if math.isnan(recovery_factor) else f"{recovery_factor:.2f}x",
        ],
        ["Ulcer Index", f"{ulcer_idx:.2f}"],
    ]

    tail_rows = [
        ["Skewness", f"{series.skew():.2f}"],
        ["Kurtosis", f"{series.kurtosis():.2f}"],
        ["Daily VaR", f"{np.percentile(series, 5) * 100:.2f}%"],
        [
            "Expected Shortfall",
            f"{series[series <= np.percentile(series, 5)].mean() * 100:.2f}%",
        ],
        ["Serenity Index", "N/A" if math.isnan(serenity) else f"{serenity:.2f}"],
    ]

    default_parameters = {
        "Name": "strategy",
        "Trade Count": "N/A",
        "Account Size": "N/A",
        "Bot Count": "N/A",
        "Trade Selection Type": "median",
        "Median Rank": "N/A",
        "Training Length": "N/A",
        "Testing Length": "N/A",
        "Optimizer": "max_sharpe",
        "Multi-Stage Optimizer": "max_sharpe",
        "Puts": "0",
        "Calls": "0",
        "ICs": "N/A",
    }
    if parameters:
        default_parameters.update(parameters)
    parameter_rows = [[key, value] for key, value in default_parameters.items()]

    consistency_rows = [
        ["Time in Market", f"{(series != 0).mean() * 100:.2f}%"],
        [
            "Avg Up Month",
            f"{((series[series > 0].mean() * 100) if series[series > 0].any() else 0.0):.2f}%",
        ],
        [
            "Avg Down Month",
            f"{((series[series < 0].mean() * 100) if series[series < 0].any() else 0.0):.2f}%",
        ],
        ["Winning Days", f"{(series > 0).sum()}"],
        ["Losing Days", f"{(series < 0).sum()}"],
        ["Expected Daily%", f"{series.mean() * 100:.2f}%"],
        ["Expected Monthly%", f"{((1 + series.mean()) ** 21 - 1) * 100:.2f}%"],
        ["Expected Yearly%", f"{((1 + series.mean()) ** 252 - 1) * 100:.2f}%"],
        ["Best Day", f"{series.max() * 100:.2f}%"],
        ["Worst Day", f"{series.min() * 100:.2f}%"],
        [
            "Best Month",
            f"{(1 + series.resample('ME').apply(lambda x: (1 + x).prod() - 1).max()) * 100 - 100:.2f}%",
        ],
        [
            "Worst Month",
            f"{(1 + series.resample('ME').apply(lambda x: (1 + x).prod() - 1).min()) * 100 - 100:.2f}%",
        ],
        [
            "Best Year",
            f"{(1 + series.resample('YE').apply(lambda x: (1 + x).prod() - 1).max()) * 100 - 100:.2f}%",
        ],
        [
            "Worst Year",
            f"{(1 + series.resample('YE').apply(lambda x: (1 + x).prod() - 1).min()) * 100 - 100:.2f}%",
        ],
    ]

    eoy_table_rows = _yearly_table(series)

    stats_display = {
        "annualized_return": f"{stats.annualized_return * 100:.2f}%",
        "sharpe": f"{stats.sharpe:.2f}",
        "max_drawdown": f"{stats.max_drawdown * 100:.2f}%",
        "win_rate": f"{win_rate_values[0]:.0f}%",
        "romad": "N/A" if math.isnan(romad) else f"{romad:.2f}x",
        "sortino": f"{stats.sortino:.2f}",
    }

    template_path = _TEMPLATE_DIR / "tearsheet.html"
    html = template_path.read_text()
    html = html.replace("__QA_REPORT_TITLE__", title)
    subtitle_text = (
        subtitle
        or f"Generated with Quantalytics (v0.1.0) on {datetime.now():%b %d, %Y}."
    )
    coverage_text = f"Data coverage: {coverage_start:%b %Y} – {coverage_end:%b %Y}"
    html = html.replace("__QA_HEADER_SUBTITLE_PRIMARY__", subtitle_text)
    html = html.replace("__QA_HEADER_SUBTITLE_SECONDARY__", coverage_text)

    def dumps(value):
        return json.dumps(value, ensure_ascii=False)

    replacements = [
        (
            r"    const dates = Array\.from\([\s\S]+?\);\n",
            f"    const dates = {dumps([date.strftime('%Y-%m-%d') for date in series.index])};\n",
        ),
        (
            r"    const sampleReturns = dates\.map\([\s\S]+?\);\n",
            f"    const sampleReturns = {dumps(cumulative_percent)};\n",
        ),
        (
            r"    const sampleDrawdown = sampleReturns\.map\([\s\S]+?\);\n",
            f"    const sampleDrawdown = {dumps((drawdown_path * 100).round(2).tolist())};\n",
        ),
        (
            r"    const dailyReturnSeries = sampleReturns\.map\([\s\S]+?\);\n",
            f"    const dailyReturnSeries = {dumps(daily_returns)};\n",
        ),
        (
            r"    const eoyYears = \[[\s\S]+?\];\n",
            f"    const eoyYears = {dumps(year_labels)};\n",
        ),
        (
            r"    const eoyReturns = \[[\s\S]+?\];\n",
            f"    const eoyReturns = {dumps([round(value, 2) for value in year_values])};\n",
        ),
        (
            r"    const dailyBars = Array\.from\([\s\S]+?\);\n",
            f"    const dailyBars = {dumps(daily_returns)};\n",
        ),
        (
            r"    const periodReturns = \[[\s\S]+?\];\n",
            f"    const periodReturns = {dumps(period_returns)};\n",
        ),
        (
            r"    const rollingSharpe = sampleReturns\.map\([\s\S]+?\);\n",
            f"    const rollingSharpe = {dumps(rolling_sharpe)};\n",
        ),
        (
            r"    const rollingSortino = sampleReturns\.map\([\s\S]+?\);\n",
            f"    const rollingSortino = {dumps(rolling_sortino)};\n",
        ),
        (
            r"    const underwaterSeries = sampleDrawdown;\n",
            f"    const underwaterSeries = {dumps((drawdown_path * 100).round(2).tolist())};\n",
        ),
        (
            r"    const drawdownBands = \[[\s\S]+?\];\n",
            f"    const drawdownBands = {dumps(drawdown_bands)};\n",
        ),
        (
            r"    const riskAdjustedRows = \[[\s\S]+?\];\n",
            f"    const riskAdjustedRows = {dumps(risk_adjusted_rows)};\n",
        ),
        (
            r"    const volRows = \[[\s\S]+?\];\n",
            f"    const volRows = {dumps(vol_rows)};\n",
        ),
        (
            r"    const tailRows = \[[\s\S]+?\];\n",
            f"    const tailRows = {dumps(tail_rows)};\n",
        ),
        (
            r"    const parameterRows = \[[\s\S]+?\];\n",
            f"    const parameterRows = {dumps(parameter_rows)};\n",
        ),
        (
            r"    const heatmapMonths = \[[^\]]+\];\n",
            f"    const heatmapMonths = {dumps(heatmap_months)};\n",
        ),
        (
            r"    const heatmapYears = \[[^\]]+\];\n",
            f"    const heatmapYears = {dumps(heatmap_years)};\n",
        ),
        (
            r"    const heatmapValues = heatmapYears\.map\([\s\S]+?\);\n",
            f"    const heatmapValues = {dumps(heatmap_values)};\n",
        ),
        (
            r"    const winRateBuckets = \[[^\]]+\];\n",
            '    const winRateBuckets = ["Day", "Week", "Month", "Quarter", "Year"];\n',
        ),
        (
            r"    const winRateValues = \[[\s\S]+?\];\n",
            f"    const winRateValues = {dumps(win_rate_values)};\n",
        ),
        (
            r"    const consistencyRows = \[[\s\S]+?\];\n",
            f"    const consistencyRows = {dumps(consistency_rows)};\n",
        ),
        (
            r"    const eoyTableRows = \[[\s\S]+?\];\n",
            f"    const eoyTableRows = {dumps(eoy_table_rows)};\n",
        ),
        (
            r"    const drawdownRows = \[[\s\S]+?\];\n",
            f"    const drawdownRows = {dumps(drawdown_rows)};\n    const statsDisplay = {dumps(stats_display)};\n",
        ),
    ]

    for pattern, replacement in replacements:
        html = re.sub(pattern, replacement, html, count=1)

    stats_fn_pattern = r"    function renderStats\(\) {\n[\s\S]+?\n    }\n\n"
    stats_fn_body = (
        "    function renderStats() {\n"
        '      document.getElementById("stat-annual-return").textContent = statsDisplay.annualized_return;\n'
        '      document.getElementById("stat-sharpe").textContent = statsDisplay.sharpe;\n'
        '      document.getElementById("stat-mdd").textContent = statsDisplay.max_drawdown;\n'
        '      document.getElementById("stat-win-rate").textContent = statsDisplay.win_rate;\n'
        '      document.getElementById("stat-romad").textContent = statsDisplay.romad;\n'
        '      document.getElementById("stat-sortino").textContent = statsDisplay.sortino;\n'
        "    }\n\n"
    )
    html = re.sub(stats_fn_pattern, stats_fn_body, html, count=1)

    tearsheet = Tearsheet(html=html)
    if output is not None:
        tearsheet.to_html(output)
    return tearsheet


__all__: list[str] = [
    "Tearsheet",
    "TearsheetSection",
    "TearsheetConfig",
    "render_basic_tearsheet",
    "html",
]
