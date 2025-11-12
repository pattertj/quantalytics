# Quantalytics

Quantalytics is a fast, modern Python library for generating quantitative performance metrics, interactive charts, and publication-ready reports. It is designed for strategy researchers, portfolio managers, and data scientists who want an ergonomic toolchain without the overhead of large monolithic frameworks.

-## Features

- **Descriptive Stats** – Grab skew, kurtosis, total return, and CAGR via the lightweight `qa.stats` helpers.
- **Analytics Helpers** – Access payoff ratio, profit ratio, Kelly, omega, tail, and other advanced risk/efficiency diagnostics through `qa.analytics`.
- **Performance Metrics** – Compute Sharpe, Sortino, Calmar, max drawdown, annualized returns/volatility, and more in a single call.
- **Interactive Visuals** – Build Plotly-based charts for cumulative returns, rolling volatility, and drawdown analysis with sensible defaults.
- **Beautiful Reports** – Produce responsive HTML tear sheets with configurable sections, ready to export to PDF.
- **Composable API** – Small, well-typed functions that play nicely with pandas Series/DataFrames.
- **Production Ready Packaging** – Standards-based `pyproject.toml`, semantic versioning, and optional CLI hooks for release automation.

## Installation

```bash
pip install quantalytics
```

## Quickstart

```python
import pandas as pd
import quantalytics as qa

returns = pd.Series(
    [0.01, 0.02, -0.005, 0.015, -0.01, 0.03],
    index=pd.date_range("2024-01-01", periods=6, freq="B"),
)

summary = qa.metrics.performance_summary(returns)
print(summary.sharpe, summary.calmar)

fig = qa.charts.cumulative_returns_chart(returns)
fig.show()
```

## Documentation

Full tutorials and API references live on our Docusaurus site: [https://pattertj.github.io/quantalytics/](https://pattertj.github.io/quantalytics/). Start with the introduction, then dive into the stats, metrics, charts, or reports guides as needed.

## License

MIT License. See [LICENSE](LICENSE).
