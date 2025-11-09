---
sidebar_position: 3
---

# Metrics

Quantalytics ships a curated set of risk/return statistics. All metric functions accept iterables or pandas Series and return floats. The `performance_summary` helper collects the most common measures into a single dataclass.

## Performance Summary

```python
import quantalytics as qa

summary = qa.metrics.performance_summary(returns, risk_free_rate=0.02)
summary.as_dict()
```

Fields available on `PerformanceMetrics`:

- `annualized_return`
- `annualized_volatility`
- `sharpe`
- `sortino`
- `calmar`
- `max_drawdown`
- `downside_deviation`
- `cumulative_return`

## Individual Metrics

```python
import quantalytics as qa

sharpe = qa.metrics.sharpe(returns, risk_free_rate=0.015)
sortino = qa.metrics.sortino_ratio(returns, risk_free_rate=0.015, target_return=0.0)
calmar = qa.metrics.calmar_ratio(returns)
drawdown = qa.metrics.max_drawdown(returns)
```

All metrics support customizing the `periods_per_year` argument to align with daily, weekly, or monthly data.
