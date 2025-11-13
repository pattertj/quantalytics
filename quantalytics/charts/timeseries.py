"""Interactive timeseries charts powered by Plotly."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from ..analytics.stats import compsum
from ..utils.timeseries import ensure_datetime_index, rolling_statistic


_COLOR_ACCENT = "#2A9D8F"
_COLOR_MUTED = "#264653"
_COLOR_WARNING = "#E76F51"


def _annualization_factor(periods_per_year: Optional[Union[int, str]]) -> int:
    if isinstance(periods_per_year, str):
        mapping = {"D": 252, "B": 252, "W": 52, "M": 12, "Q": 4, "A": 1}
        return mapping.get(periods_per_year.upper(), 252)
    if isinstance(periods_per_year, int):
        return periods_per_year
    return 252


def _validate_returns(returns: Iterable[float] | pd.Series) -> pd.Series:
    series = pd.Series(returns)
    if series.empty:
        raise ValueError("Returns series is empty")
    return series


def cumulative_returns_chart(
    returns: Iterable[float] | pd.Series,
    title: str = "Cumulative Returns",
    benchmark: Optional[pd.Series] = None,
) -> go.Figure:
    """Create a cumulative returns chart."""

    series = ensure_datetime_index(_validate_returns(returns))
    cum = compsum(series)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum,
            mode="lines",
            name="Strategy",
            line=dict(color=_COLOR_ACCENT),
        )
    )

    if benchmark is not None:
        benchmark_series = ensure_datetime_index(pd.Series(benchmark))
        benchmark_cum = compsum(benchmark_series)
        fig.add_trace(
            go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum,
                mode="lines",
                name="Benchmark",
                line=dict(color=_COLOR_MUTED, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def rolling_volatility_chart(
    returns: Iterable[float] | pd.Series,
    window: int = 21,
    periods_per_year: Optional[int | str] = 252,
    title: str = "Rolling Volatility",
) -> go.Figure:
    """Create a rolling volatility chart."""

    series = ensure_datetime_index(_validate_returns(returns))
    rolling_vol = rolling_statistic(series, window=window, function="std")
    factor = _annualization_factor(periods_per_year)
    annualized = rolling_vol * math.sqrt(factor)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=annualized.index,
            y=annualized,
            mode="lines",
            name=f"{window}-period Volatility",
            line=dict(color=_COLOR_ACCENT),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
        hovermode="x",
        yaxis=dict(tickformat=".0%"),
    )
    return fig


def drawdown_chart(
    returns: Iterable[float] | pd.Series,
    title: str = "Drawdowns",
) -> go.Figure:
    """Visualize drawdowns over time."""

    series = ensure_datetime_index(_validate_returns(returns))
    cum = compsum(series)
    running_max = (1 + cum).cummax()
    drawdown = (1 + cum) / running_max - 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color=_COLOR_WARNING),
            fill="tozeroy",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        yaxis=dict(tickformat=".0%"),
    )

    return fig


__all__ = ["cumulative_returns_chart", "rolling_volatility_chart", "drawdown_chart"]
