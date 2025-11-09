---
sidebar_position: 1
slug: /intro
---

# Introduction

Quantalytics is a batteries-included toolkit for generating quantitative analytics in Python. With a single install you get:

- **Metrics** for risk/return evaluation with sensible defaults and rich dataclasses.
- **Charts** powered by Plotly that reveal strategy behaviour and drawdowns at a glance.
- **Reports** that compile metrics and visuals into a responsive HTML tear sheet.

## Why Quantalytics?

- **Fast iteration** – Compose NumPy/pandas primitives with ergonomic helpers.
- **Beautiful outputs** – High contrast palettes and typography for presentation-ready visuals.
- **Production friendly** – Packaged with PyPI-ready metadata, documentation, and testing hooks.

Get started in minutes:

```bash
pip install quantalytics
```

```python
import quantalytics as qa

metrics = qa.metrics.performance_summary(strategy_returns)
print(metrics.as_dict())
```

Next, explore the [stats](./stats.md) and [metrics](./metrics.md) guides.
