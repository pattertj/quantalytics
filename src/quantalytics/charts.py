"""Visualization utilities for the Quantalytics library."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


_DEFAULT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "axes.facecolor": "#f9f9f9",
    "axes.edgecolor": "#cccccc",
    "axes.titleweight": "bold",
}
class ChartContext:
    """Context manager applying Quantalytics default Matplotlib style."""

    def __enter__(self):
        self._old_params = {key: plt.rcParams.get(key) for key in _DEFAULT_STYLE}
        plt.rcParams.update(_DEFAULT_STYLE)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        plt.rcParams.update(self._old_params)
        plt.close("all")
        return False


def _prepare_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if series.empty:
        raise ValueError("Series must contain data")
    return series.dropna()


def _finalize_chart(
    figure: plt.Figure,
    title: str,
    output: Optional[Path | str],
    show: bool,
) -> plt.Figure:
    figure.suptitle(title)
    figure.tight_layout()
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
    if show:
        figure.show()
    return figure


def plot_cumulative_returns(
    returns: pd.Series,
    title: str = "Cumulative Returns",
    output: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot cumulative returns from a series of periodic returns."""

    series = _prepare_series(returns)
    cumulative = (1 + series).cumprod()

    with ChartContext():
        fig, ax = plt.subplots()
        ax.plot(cumulative.index, cumulative, label="Strategy")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        return _finalize_chart(fig, title, output, show)


def plot_drawdowns(
    returns: pd.Series,
    title: str = "Drawdowns",
    output: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot drawdowns for a returns series."""

    series = _prepare_series(returns)
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1

    with ChartContext():
        fig, ax = plt.subplots()
        ax.fill_between(drawdowns.index, drawdowns, 0, color="#d62728", alpha=0.4)
        ax.set_ylabel("Drawdown")
        return _finalize_chart(fig, title, output, show)


def plot_rolling_metric(
    returns: pd.Series,
    window: int,
    metric_fn,
    title: str,
    ylabel: str,
    output: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a rolling metric computed over the returns series."""

    series = _prepare_series(returns)
    rolling_values = series.rolling(window).apply(metric_fn, raw=False)

    with ChartContext():
        fig, ax = plt.subplots()
        ax.plot(rolling_values.index, rolling_values, label=ylabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return _finalize_chart(fig, title, output, show)


def plot_return_distribution(
    returns: pd.Series,
    bins: int = 50,
    title: str = "Return Distribution",
    output: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Figure:
    """Histogram of the returns distribution."""

    series = _prepare_series(returns)

    with ChartContext():
        fig, ax = plt.subplots()
        ax.hist(series, bins=bins, alpha=0.7, color="#1f77b4")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        return _finalize_chart(fig, title, output, show)
