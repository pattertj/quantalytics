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
romad = qa.metrics.romad(returns)
omega = qa.metrics.omega_ratio(returns)
max_dd_pct = qa.metrics.max_drawdown_percent(returns)
longest_dd_days = qa.metrics.longest_drawdown_days(returns)
underwater = qa.metrics.underwater_percent(returns)
avg_dd = qa.metrics.average_drawdown(returns)
avg_dd_days = qa.metrics.average_drawdown_days(returns)
recovery = qa.metrics.recovery_factor(returns)
ulcer = qa.metrics.ulcer_index(returns)
serenity = qa.metrics.serenity_index(returns)
```

All metrics support customizing the `periods_per_year` argument to align with daily, weekly, or monthly data.

## Tail Risk Metrics

```python
var = qa.analytics.value_at_risk(returns, confidence=0.95)
cvar = qa.analytics.conditional_value_at_risk(returns, confidence=0.95)
```

- `value_at_risk` returns the historical loss magnitude you should not exceed with the specified confidence.
- `conditional_value_at_risk` averages all losses beyond that VaR threshold (expected shortfall).

## Advanced Risk Metrics

```python
prob = qa.analytics.prob_sharpe_ratio(returns, risk_free_rate=0.0, target_sharpe=0.5)
smart_sharpe = qa.analytics.smart_sharpe_ratio(returns)
smart_sortino = qa.analytics.smart_sortino_ratio(returns)
smart_sortino_half = qa.analytics.smart_sortino_over_sqrt_two(returns)
```

- `prob_sharpe_ratio` returns the probability that your sample Sharpe exceeds a target.
- `smart_sharpe_ratio` and `smart_sortino_ratio` adjust the classical measures for skew/kurtosis.
- `/sqrt(2)` variants rescale Sortino-style ratios to account for half-normal downside distributions.
- `omega_ratio` compares upside vs downside partial moments around a target threshold.
- Use `max_drawdown_percent`, `longest_drawdown_days`, and `underwater_percent` to translate drawdown paths into presentation-ready KPIs.
- `average_drawdown`, `average_drawdown_days`, and `recovery_factor` capture typical drawdown depth/duration and how quickly returns reclaim new highs.
- `ulcer_index` and `serenity_index` summarize drawdown variability and annualized-growth efficiency in a single score.
