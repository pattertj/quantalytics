import numpy as np
import pandas as pd
from quantalytics.metrics import max_drawdown, performance_summary, sharpe, sortino_ratio


def sample_returns():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.001, scale=0.01, size=252)
    return pd.Series(data, index=pd.date_range("2023-01-01", periods=252, freq="B"))


def test_performance_summary_runs():
    returns = sample_returns()
    summary = performance_summary(returns)
    assert summary.annualized_return is not None
    assert summary.max_drawdown <= 0


def test_sharpe_and_sortino_positive():
    returns = sample_returns()
    assert sharpe(returns) > 0
    assert sortino_ratio(returns) > 0


def test_max_drawdown_nonzero():
    returns = sample_returns()
    drawdown = max_drawdown(returns)
    assert drawdown <= 0
