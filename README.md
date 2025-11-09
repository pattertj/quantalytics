# Quantalytics

Quantalytics is a fast, modern Python library for generating quantitative performance metrics, interactive charts, and publication-ready reports. It is designed for strategy researchers, portfolio managers, and data scientists who want an ergonomic toolchain without the overhead of large monolithic frameworks.

## Features

- **Descriptive Stats** – Grab skew, kurtosis, total return, and CAGR via the lightweight `qa.stats` helpers.
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

returns = pd.Series([...], index=pd.date_range("2023-01-01", periods=252, freq="B"))

metrics = qa.metrics.performance_summary(returns)
print(metrics)

print("Skew:", qa.stats.skewness(returns))
print("CAGR:", qa.stats.cagr(returns))

fig = qa.charts.cumulative_returns_chart(returns)
fig.show()

tearsheet = qa.reports.render_basic_tearsheet(returns)
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
   uv run pytest --cov=quantalytics --cov-report=term-missing --cov-fail-under=80
   ruff check .
   ```

3. Install and run the git hooks (powered by `pre-commit` so every commit runs formatting, linting, vuln scans, and coverage-enforced tests—builds fail if total coverage drops below 80%):

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

4. Start the documentation site:

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
