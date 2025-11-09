---
sidebar_position: 4
---

# Charts

Plot strategy behaviour with a single function call. Quantalytics uses Plotly, so charts are interactive in notebooks and export cleanly to HTML. Examples below assume:

```python
import quantalytics as qa
```

## Cumulative Returns

```python
fig = qa.charts.cumulative_returns_chart(returns, benchmark=benchmark_returns)
fig.show()
```

## Rolling Volatility

```python
fig = qa.charts.rolling_volatility_chart(returns, window=63)
```

## Drawdowns

```python
fig = qa.charts.drawdown_chart(returns)
```

All chart helpers return `plotly.graph_objects.Figure` objects, letting you further customize layout and styling.
