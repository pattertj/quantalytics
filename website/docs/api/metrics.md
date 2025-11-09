---
sidebar_position: 1
---

# Metrics API

Quantalytics exposes a comprehensive set of risk and performance metrics. All metric functions
operate on `pandas.Series` objects of periodic returns and default to daily frequency
(`periods_per_year=252`).

## `annualized_return(returns, periods_per_year=252)`
Compound growth rate assuming reinvestment.

## `annualized_volatility(returns, periods_per_year=252)`
Standard deviation of returns scaled to an annual horizon.

## `sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252)`
Annualized Sharpe ratio using the supplied risk-free rate.

## `sortino_ratio(returns, mar=0, periods_per_year=252)`
Focuses on downside risk by penalizing returns below the minimum acceptable return (MAR).

## `max_drawdown(returns)`
Worst peak-to-trough decline in the equity curve.

## `calmar_ratio(returns, periods_per_year=252)`
Compound annual growth rate divided by maximum drawdown magnitude.

## `tail_ratio(returns)`
95th percentile gain divided by the magnitude of the 5th percentile loss.

## `information_ratio(returns, benchmark, periods_per_year=252)`
Performance relative to a benchmark, scaled by tracking error.

## `rolling_beta(returns, benchmark, window=63)`
Rolling beta of the strategy relative to the benchmark using the specified window.
