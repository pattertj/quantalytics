import numpy as np
import pandas as pd

from quantalytics import metrics


def sample_returns():
    rng = np.random.default_rng(123)
    return pd.Series(rng.normal(0.001, 0.01, size=252))


def test_annualized_return_positive():
    returns = pd.Series([0.01] * 252)
    result = metrics.annualized_return(returns)
    assert result > 0


def test_sharpe_ratio_handles_zero_volatility():
    returns = pd.Series([0.0] * 252)
    assert np.isnan(metrics.sharpe_ratio(returns))


def test_max_drawdown_bounds():
    returns = pd.Series([0.01, -0.02, 0.03, -0.04])
    drawdown = metrics.max_drawdown(returns)
    assert drawdown <= 0


def test_information_ratio_matches_manual():
    returns = pd.Series([0.01, 0.02, 0.015])
    benchmark = pd.Series([0.005, 0.018, 0.01])
    ir = metrics.information_ratio(returns, benchmark, periods_per_year=3)
    active = returns - benchmark
    expected = metrics.annualized_return(active, periods_per_year=3) / metrics.annualized_volatility(
        active, periods_per_year=3
    )
    assert np.isclose(ir, expected)


def test_rolling_beta_length():
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    benchmark = pd.Series([0.005, 0.015, 0.025, 0.035, 0.045])
    beta = metrics.rolling_beta(returns, benchmark, window=3)
    assert len(beta) == 3
