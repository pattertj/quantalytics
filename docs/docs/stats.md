---
sidebar_position: 2
---

# Stats

Use the stats helpers to calculate descriptive distribution properties before layering on performance metrics.

```python
import quantalytics as qa

skew = qa.stats.skewness(returns)  # or qa.stats.skew(returns)
kurt = qa.stats.kurtosis(returns)
total = qa.stats.total_return(returns)
cagr = qa.stats.cagr(returns, periods_per_year=252)
cagr_pct = qa.stats.cagr_percent(returns, periods_per_year=252)
vol = qa.stats.volatility(returns)
```

- `skewness` and `kurtosis` operate directly on raw returns (or any numeric iterable) and drop missing values.
- `total_return` compounds the whole series to a single growth figure.
- `cagr`/`cagr_percent` annualize growth by the sampling frequency you specify (defaults to daily bars).
- `volatility` returns the realized (non-annualized) standard deviation so you can annualize or compare across sampling windows manually.

Combine these stats with `qa.metrics.performance_summary` to get a complete picture of the distribution and risk-adjusted results.
