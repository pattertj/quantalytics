import numpy as np
import pandas as pd
import pytest

from quantalytics.metrics import (
    conditional_value_at_risk,
    longest_drawdown_days,
    max_drawdown,
    max_drawdown_percent,
    omega_ratio,
    performance_summary,
    prob_sharpe_ratio,
    romad,
    sharpe,
    smart_sharpe_ratio,
    smart_sortino_over_sqrt_two,
    sortino_over_sqrt_two,
    sortino_ratio,
    underwater_percent,
    value_at_risk,
)


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


def test_max_drawdown_percent_is_absolute_value():
    returns = sample_returns()
    drawdown = max_drawdown(returns)
    percent = max_drawdown_percent(returns)
    assert percent == pytest.approx(abs(drawdown) * 100)


def test_romad_positive():
    returns = sample_returns()
    value = romad(returns)
    assert value > 0


def test_prob_sharpe_ratio_in_unit_interval():
    returns = sample_returns()
    value = prob_sharpe_ratio(returns)
    assert 0 <= value <= 1


def test_smart_vs_classic_sharpe():
    returns = sample_returns()
    classic = sharpe(returns)
    smart = smart_sharpe_ratio(returns)
    assert classic > 0
    assert smart > 0


def test_sortino_variants_scale_by_sqrt_two():
    returns = sample_returns()
    base = sortino_ratio(returns)
    scaled = sortino_over_sqrt_two(returns)
    assert scaled == pytest.approx(base / np.sqrt(2))


def test_smart_sortino_over_sqrt_two_positive():
    returns = sample_returns()
    value = smart_sortino_over_sqrt_two(returns)
    assert value > 0


def test_omega_ratio_positive():
    returns = sample_returns()
    assert omega_ratio(returns) > 0


def test_longest_drawdown_days_counts_calendar_days():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.02, -0.01, -0.01, -0.01, 0.05], index=dates)
    assert longest_drawdown_days(returns) == pytest.approx(3.0)


def test_underwater_percent_reflects_current_drawdown():
    returns = pd.Series(
        [0.02, -0.02, -0.01], index=pd.date_range("2024-01-01", periods=3, freq="D")
    )
    value = underwater_percent(returns)
    assert value > 0


def test_var_and_cvar_return_positive_losses():
    returns = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01])
    confidence = 0.8
    var = value_at_risk(returns, confidence=confidence)
    cvar = conditional_value_at_risk(returns, confidence=confidence)
    expected_var_threshold = returns.quantile(1 - confidence)
    expected_var = max(0.0, -expected_var_threshold)
    expected_cvar = max(0.0, -returns[returns <= expected_var_threshold].mean())
    assert var == pytest.approx(expected_var)
    assert cvar == pytest.approx(expected_cvar)
