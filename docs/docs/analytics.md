# Analytics helpers

`quantalytics.analytics` bundles every descriptive and performance-centric helper into a consistent namespace so you can explore returns with minimal boilerplate.

The module covers basic stats (`skewness`, `kurtosis`, `total_return`, `cagr`, `volatility`) and also exposes specialties you just added, such as:

- **Risk balance** – `payoff_ratio`, `profit_ratio`, `win_loss_ratio`, `profit_factor`, `risk_of_ruin`, and `gain_to_pain_ratio` help you understand how wins compare to losses and what it would take to stay solvent.
- **Streaks & consistency** – `max_consecutive_wins`/`losses`, `avg_win`, `avg_loss`, and `win_rate` make it easy to quantify the cadence of profitable periods.
- **Advanced ratios** – `omega_ratio`, `tail_ratio`, `common_sense_ratio`, `information_ratio`, and `r_squared` highlight distributional skew, tail leverage, benchmark tracking, and explained variance.

Every helper works with pandas `Series` (or any iterable) and returns `float` results with consistent NaN/inf semantics. Use them alongside `qa.metrics` and `qa.charts` so you can summarize, visualize, and report on performance from a single code path.
