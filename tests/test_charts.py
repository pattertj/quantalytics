import pandas as pd

from quantalytics.charts import (
    cumulative_returns_chart,
    drawdown_chart,
    rolling_volatility_chart,
)


def sample_returns():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    return pd.Series(0.001, index=dates)


def test_cumulative_returns_chart_includes_benchmark_trace():
    returns = sample_returns()
    benchmark = returns * 0.5
    fig = cumulative_returns_chart(returns, benchmark=benchmark)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Strategy"
    assert fig.data[1].name == "Benchmark"


def test_rolling_volatility_chart_annualizes_output():
    returns = sample_returns()
    fig = rolling_volatility_chart(returns, window=5, periods_per_year=252)
    assert len(fig.data) == 1
    values = pd.Series(fig.data[0].y).dropna()
    assert not values.empty
    assert values.min() >= 0


def test_drawdown_chart_fills_to_zero():
    returns = sample_returns()
    fig = drawdown_chart(returns)
    assert len(fig.data) == 1
    assert fig.data[0].fill == "tozeroy"
