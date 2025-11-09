---
sidebar_position: 3
---

# Charts

Plot strategy behaviour with a single function call. Quantalytics uses Plotly, so charts are interactive in notebooks and export cleanly to HTML.

## Cumulative Returns

```python
from quantalytics import cumulative_returns_chart

fig = cumulative_returns_chart(returns, benchmark=benchmark_returns)
fig.show()
```

## Rolling Volatility

```python
from quantalytics import rolling_volatility_chart

fig = rolling_volatility_chart(returns, window=63)
```

## Drawdowns

```python
from quantalytics import drawdown_chart

fig = drawdown_chart(returns)
```

All chart helpers return `plotly.graph_objects.Figure` objects, letting you further customize layout and styling.
