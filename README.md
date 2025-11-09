# Quantalytics

Quantalytics is a fast, modern Python library for building institutional-grade quantitative
analytics. It bundles high-performance metrics, polished Matplotlib visuals, and ready-to-share
tear sheets so you can go from raw return series to publication-ready insights in minutes.

## Features

- **Performance & risk metrics** – Sharpe, Sortino, max drawdown, Calmar, tail ratio, rolling beta,
  and more.
- **Publication-ready charts** – Consistent design system for cumulative returns, drawdowns,
  rolling statistics, and distributions.
- **Automated tear sheets** – Generate beautiful HTML reports (and optional PDF exports via
  WeasyPrint) with one function call.
- **Typing-first API** – Annotated functions backed by NumPy, pandas, and Matplotlib.
- **Distribution-ready** – `pyproject.toml` packaging, semantic versioning, and CI-friendly
  development tooling.

## Installation

```bash
pip install quantalytics
```

To enable PDF generation with [WeasyPrint](https://weasyprint.org/):

```bash
pip install "quantalytics[pdf]"
```

For contributors, install the development extras:

```bash
pip install "quantalytics[dev]"
```

## Quick start

```python
import pandas as pd
import quantalytics as qa

returns = pd.Series(...)

print("Sharpe", qa.sharpe_ratio(returns))
print("Max drawdown", qa.max_drawdown(returns))

fig = qa.plot_cumulative_returns(returns)
fig.show()

report = qa.generate_tearsheet(returns, title="Sample Strategy")
print("HTML report saved to", report.html_path)
```

## Documentation

Full documentation, tutorials, and API references live at
[docs.quantalytics.dev](https://docs.quantalytics.dev). This repository also contains a Docusaurus
site under `website/` that you can serve locally:

```bash
cd website
npm install
npm run start
```

## Releasing to PyPI

1. Update the version number in `pyproject.toml` and `src/quantalytics/__init__.py`.
2. Build artifacts with `hatch build`.
3. Publish with `twine upload dist/*`.

## Contributing

We welcome pull requests! Please run `pytest`, `ruff`, and `black` before submitting:

```bash
pytest
ruff check src tests
black src tests
```

See `CONTRIBUTING.md` for more details.
