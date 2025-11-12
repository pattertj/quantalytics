import pandas as pd
import pytest

from quantalytics.analytics import (
    avg_loss,
    avg_win,
    best_period_return,
    cagr,
    cagr_percent,
    gain_to_pain_ratio,
    kurtosis,
    max_consecutive_losses,
    max_consecutive_wins,
    skew,
    skewness,
    total_return,
    volatility,
    win_rate,
    worst_period_return,
)


def test_skewness_of_symmetric_series_is_zero():
    returns = pd.Series([-0.01, 0.0, 0.01])
    assert skewness(returns) == pytest.approx(0.0, abs=1e-12)
    assert skew(returns) == pytest.approx(0.0, abs=1e-12)


def test_kurtosis_matches_pandas():
    returns = pd.Series([0.02, -0.01, 0.015, 0.005, 0.01])
    expected = returns.kurtosis()
    assert kurtosis(returns) == pytest.approx(expected)


def test_total_return_compounds_returns():
    returns = pd.Series([0.01, -0.02, 0.015])
    expected = (1.01 * 0.98 * 1.015) - 1
    assert total_return(returns) == pytest.approx(expected)


def test_volatility_matches_std():
    returns = pd.Series([0.01, 0.02, -0.01])
    expected = returns.std(ddof=1)
    assert volatility(returns) == pytest.approx(expected)


def test_cagr_matches_manual_computation():
    # 252 trading days of 10 bps per day should annualize cleanly.
    returns = pd.Series([0.001] * 252)
    expected = (1.001**252) - 1
    assert cagr(returns, periods_per_year=252) == pytest.approx(expected)


def test_cagr_percent_scales_value():
    returns = pd.Series([0.001] * 252)
    value = cagr(returns, periods_per_year=252)
    assert cagr_percent(returns, periods_per_year=252) == pytest.approx(value * 100)


def test_best_and_worst_period_returns_per_day():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=dates)
    assert best_period_return(returns, period="day") == pytest.approx(3.0)
    assert worst_period_return(returns, period="day") == pytest.approx(-2.0)


def test_best_and_worst_period_returns_week():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    returns = pd.Series([0.01] * 5 + [-0.01] * 5, index=dates)
    best_week = (1.01**5 - 1) * 100
    worst_week = ((1 - 0.01) ** 5 - 1) * 100
    assert best_period_return(returns, period="week") == pytest.approx(best_week)
    assert worst_period_return(returns, period="week") == pytest.approx(worst_week)


def test_period_validation_raises_value_error():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.01, 0.02, 0.03], index=dates)
    with pytest.raises(ValueError):
        best_period_return(returns, period="decade")


def test_win_rate_daily_and_weekly():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.0], index=dates)
    assert win_rate(returns, period="day") == pytest.approx(50.0)
    weekly_dates = pd.date_range("2024-01-01", periods=10, freq="B")
    weekly_returns = pd.Series([0.01] * 5 + [-0.01] * 5, index=weekly_dates)
    assert win_rate(weekly_returns, period="week") == pytest.approx(50.0)


def test_avg_loss_and_win_magnitudes():
    series = pd.Series([-0.01, 0.02, -0.03, 0.04])
    assert avg_loss(series) == pytest.approx(0.02)
    assert avg_win(series) == pytest.approx(0.03)


def test_max_consecutive_win_loss_runs():
    series = pd.Series([-0.01, -0.02, 0.01, 0.02, 0.03, -0.05, -0.01, 0.02])
    assert max_consecutive_losses(series) == 2
    assert max_consecutive_wins(series) == 3


def test_gain_to_pain_ratio():
    series = pd.Series([0.01, -0.005, 0.02, -0.01])
    assert gain_to_pain_ratio(series) == pytest.approx(1.0)
