# Quantalytics

Quantalytics is a fast, modern Python library for generating quantitative performance metrics, interactive charts, and publication-ready reports. It is designed for strategy researchers, portfolio managers, and data scientists who want an ergonomic toolchain without the overhead of large monolithic frameworks.

## Features

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
from quantalytics import performance_summary, cumulative_returns_chart, render_basic_tearsheet

returns = pd.Series([...], index=pd.date_range("2023-01-01", periods=252, freq="B"))

metrics = performance_summary(returns)
print(metrics)

fig = cumulative_returns_chart(returns)
fig.show()

tearsheet = render_basic_tearsheet(returns)
tearsheet.to_html("tearsheet.html")
```

## Documentation

Full documentation, tutorials, and API references are available at [https://quantalytics.dev/docs](https://quantalytics.dev/docs).

## Development

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. Run tests and linters:

   ```bash
   pytest
   ruff check .
   ```

3. Start the documentation site:

   ```bash
   cd docs
   npm install
   npm run start
   ```

## Publishing

1. Update the version in `pyproject.toml` following semantic versioning.
2. Build the distribution:

   ```bash
   rm -rf dist/
   python -m build
   ```

3. Upload to PyPI:

   ```bash
   twine upload dist/*
   ```

## License

MIT License. See [LICENSE](LICENSE).
