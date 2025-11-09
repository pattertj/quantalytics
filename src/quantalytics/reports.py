"""Report generation utilities for Quantalytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from jinja2 import Environment, BaseLoader, select_autoescape

from . import charts, metrics


@dataclass
class TearsheetArtifact:
    """Container representing generated assets for a tear sheet."""

    html_path: Path
    figures: Dict[str, Path]
    pdf_path: Optional[Path] = None


_JINJA_ENV = Environment(
    loader=BaseLoader(),
    autoescape=select_autoescape(enabled_extensions=("html",)),
    trim_blocks=True,
    lstrip_blocks=True,
)

TEARSHEET_TEMPLATE = _JINJA_ENV.from_string(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ title }}</title>
  <style>
    body { font-family: 'Inter', Arial, sans-serif; margin: 0; padding: 2rem; background: #f4f5f7; }
    h1, h2 { color: #1f2933; }
    section { background: #ffffff; margin-bottom: 2rem; padding: 1.5rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 0.6rem 0.8rem; text-align: right; }
    th { text-align: left; color: #6b7280; }
    tr:nth-child(odd) { background-color: #f9fafb; }
    img { width: 100%; height: auto; border-radius: 10px; margin-top: 1rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <section>
    <h2>Overview</h2>
    <p>{{ description }}</p>
  </section>
  <section>
    <h2>Key Metrics</h2>
    <table>
      <tbody>
        {% for label, value in metrics %}
        <tr>
          <th>{{ label }}</th>
          <td>{{ value }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  <section>
    <h2>Charts</h2>
    <div class="grid">
      {% for chart in charts %}
      <figure>
        <img src="{{ chart.path.name }}" alt="{{ chart.label }}">
        <figcaption>{{ chart.label }}</figcaption>
      </figure>
      {% endfor %}
    </div>
  </section>
</body>
</html>
"""
)


def _format_metrics(metrics_dict: Dict[str, float]) -> list[tuple[str, str]]:
    formatted = []
    for label, value in metrics_dict.items():
        if isinstance(value, float):
            display = f"{value:0.4f}" if pd.notna(value) else "N/A"
        else:
            display = str(value)
        formatted.append((label, display))
    return formatted


def _save_figures(figures: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    image_paths: Dict[str, Path] = {}
    for label, fig in figures.items():
        output_path = output_dir / f"{label.replace(' ', '_').lower()}.png"
        fig.savefig(output_path, bbox_inches="tight")
        image_paths[label] = output_path
    return image_paths


def generate_tearsheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    title: str = "Strategy Tearsheet",
    description: str = "Automatically generated report by Quantalytics",
    output_dir: Path | str = "tearsheet",
    create_pdf: bool = False,
) -> TearsheetArtifact:
    """Generate an interactive tear sheet with metrics and charts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "Annualized Return": metrics.annualized_return(returns, periods_per_year),
        "Annualized Volatility": metrics.annualized_volatility(returns, periods_per_year),
        "Sharpe Ratio": metrics.sharpe_ratio(returns, periods_per_year=periods_per_year),
        "Sortino Ratio": metrics.sortino_ratio(returns, periods_per_year=periods_per_year),
        "Calmar Ratio": metrics.calmar_ratio(returns, periods_per_year=periods_per_year),
        "Max Drawdown": metrics.max_drawdown(returns),
        "Tail Ratio": metrics.tail_ratio(returns),
    }

    if benchmark is not None:
        metrics_payload["Information Ratio"] = metrics.information_ratio(
            returns, benchmark, periods_per_year=periods_per_year
        )

    figures = {
        "Cumulative Returns": charts.plot_cumulative_returns(returns),
        "Drawdowns": charts.plot_drawdowns(returns),
        "Return Distribution": charts.plot_return_distribution(returns),
    }

    figures["Rolling Volatility"] = charts.plot_rolling_metric(
        returns,
        window=min(63, len(returns)),
        metric_fn=lambda x: x.std(ddof=1),
        title="Rolling Volatility",
        ylabel="Volatility",
    )

    image_paths = _save_figures(figures, output_dir)

    html_content = TEARSHEET_TEMPLATE.render(
        title=title,
        description=description,
        metrics=_format_metrics(metrics_payload),
        charts=[{"label": label, "path": path} for label, path in image_paths.items()],
    )

    html_path = output_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_path = None
    if create_pdf:
        try:
            from weasyprint import HTML  # type: ignore

            pdf_path = output_dir / "tearsheet.pdf"
            HTML(string=html_content, base_url=str(output_dir)).write_pdf(pdf_path)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PDF generation requires the 'weasyprint' package to be installed"
            ) from exc

    return TearsheetArtifact(html_path=html_path, figures=image_paths, pdf_path=pdf_path)
