---
sidebar_position: 2
---

# Stats

Use the stats helpers to calculate descriptive distribution properties before layering on performance metrics.

```python
import quantalytics as qa

skew = qa.stats.skewness(returns)
kurt = qa.stats.kurtosis(returns)
total = qa.stats.total_return(returns)
cagr = qa.stats.cagr(returns, periods_per_year=252)
```

- `skewness` and `kurtosis` operate directly on raw returns (or any numeric iterable) and drop missing values.
- `total_return` compounds the whole series to a single growth figure.
- `cagr` annualizes that growth by the sampling frequency you specify (or defaults to daily bars).

Combine these stats with `qa.metrics.performance_summary` to get a complete picture of the distribution and risk-adjusted results.
