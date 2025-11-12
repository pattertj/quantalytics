---
sidebar_position: 2
---

# Stats

Use the stats helpers to calculate descriptive distribution properties before layering on performance metrics.

```python
import quantalytics as qa

skew = qa.stats.skew(returns)  # or qa.stats.skew(returns)
kurt = qa.stats.kurtosis(returns)
total = qa.stats.total_return(returns)
cagr = qa.stats.cagr(returns, periods_per_year=252)
cagr_pct = qa.stats.cagr_percent(returns, periods_per_year=252)
vol = qa.stats.volatility(returns)
best_day = qa.stats.best_period_return(returns, period="day")
worst_month = qa.stats.worst_period_return(returns, period="month")
win_rate = qa.stats.win_rate(returns, period="week")
```

- `skewness` and `kurtosis` operate directly on raw returns (or any numeric iterable) and drop missing values.
- `total_return` compounds the whole series to a single growth figure.
- `cagr`/`cagr_percent` annualize growth by the sampling frequency you specify (defaults to daily bars).
- `volatility` now returns annualized volatility by default (sqrt of the default 365 periods). Set `annualize=False` or adjust `periods` to work with arbitrary sampling frequencies.
- `best_period_return` / `worst_period_return` surface the highest/lowest compounded return for any period (`day|week|month|quarter|year`) and report the answer in percent terms for easy reporting.
`- win_rate` tells you how often the strategy finished positive for the selected period; it now returns a decimal (0–1) rather than a percentage.

Combine these stats with `qa.metrics.performance_summary` to get a complete picture of the distribution and risk-adjusted results.
