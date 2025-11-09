"""HTML tear sheet generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..charts.timeseries import (
    cumulative_returns_chart,
    drawdown_chart,
    rolling_volatility_chart,
)
from ..metrics.performance import performance_summary

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
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
    metrics = performance_summary(
        series,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        periods_per_year=periods_per_year,
    )

    sections = config.sections if config else []
    if not sections:
        sections = [
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

    template = _TEMPLATE_ENV.get_template("tearsheet.html")
    html = template.render(
        title=config.title if config else "Strategy Tearsheet",
        subtitle=config.subtitle if config else None,
        metrics=metrics.as_dict(),
        sections=sections,
    )
    return Tearsheet(html=html)


__all__ = ["Tearsheet", "TearsheetSection", "TearsheetConfig", "render_basic_tearsheet"]
