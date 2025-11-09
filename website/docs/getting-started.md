---
sidebar_position: 2
---

# Getting Started

This guide helps you install Quantalytics, configure your environment, and generate your first
report.

## Install

```bash
pip install quantalytics
```

To enable PDF export via WeasyPrint:

```bash
pip install "quantalytics[pdf]"
```

## Load your returns

Quantalytics operates on pandas Series of periodic returns. Daily returns are assumed by default
(`periods_per_year=252`).

```python
import pandas as pd
import quantalytics as qa

returns = pd.read_csv("strategy_returns.csv", parse_dates=["date"], index_col="date")
returns = returns["strategy"]
```

## Compute metrics

```python
metrics = {
    "Sharpe": qa.sharpe_ratio(returns),
    "Sortino": qa.sortino_ratio(returns),
    "Max Drawdown": qa.max_drawdown(returns),
}
```

## Generate charts

```python
qa.plot_cumulative_returns(returns, show=True)
qa.plot_drawdowns(returns, show=True)
```

## Build a tear sheet

```python
report = qa.generate_tearsheet(returns, title="Momentum Strategy")
print(report.html_path)
```

Open the saved `index.html` file to explore your report.
