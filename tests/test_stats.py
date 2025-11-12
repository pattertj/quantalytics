import math

import pandas as pd
import pytest
from quantalytics.analytics import (
    avg_loss,
    avg_win,
    best_period_return,
    cagr,
    cagr_percent,
    common_sense_ratio,
    cpc_index,
    gain_to_pain_ratio,
    information_ratio,
    kelly_criterion,
    kurtosis,
    max_consecutive_losses,
    max_consecutive_wins,
    omega,
    payoff_ratio,
    profit_factor,
    profit_ratio,
    r_squared,
    risk_of_ruin,
    skew,
    tail_ratio,
    total_return,
    volatility,
    win_loss_ratio,
    win_rate,
    worst_period_return,
)


def test_skewness_of_symmetric_series_is_zero():
    returns = pd.Series([-0.01, 0.0, 0.01])
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
    expected = returns.std(ddof=1) * math.sqrt(365)
    assert volatility(returns) == pytest.approx(expected)


def test_volatility_no_annualize():
    returns = pd.Series([0.01, 0.02, -0.01])
    expected = returns.std(ddof=1)
    assert volatility(returns, annualize=False) == pytest.approx(expected)


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
    assert win_rate(returns, period="day") == pytest.approx(0.5)
    weekly_dates = pd.date_range("2024-01-01", periods=10, freq="B")
    weekly_returns = pd.Series([0.01] * 5 + [-0.01] * 5, index=weekly_dates)
    assert win_rate(weekly_returns, period="week") == pytest.approx(0.5)


def test_avg_loss_and_win_magnitudes():
    series = pd.Series([-0.01, 0.02, -0.03, 0.04])
    assert avg_loss(series) == pytest.approx(-0.02)
    assert avg_win(series) == pytest.approx(0.03)


def test_max_consecutive_win_loss_runs():
    series = pd.Series([-0.01, -0.02, 0.01, 0.02, 0.03, -0.05, -0.01, 0.02])
    assert max_consecutive_losses(series) == 2
    assert max_consecutive_wins(series) == 3


def test_gain_to_pain_ratio():
    series = pd.Series([0.01, -0.005, 0.02, -0.01])
    assert gain_to_pain_ratio(series) == pytest.approx(1.0)


def test_information_ratio():
    returns = pd.Series([0.01, 0.02, -0.005])
    benchmark = pd.Series([0.005, 0.015, -0.005])
    info = information_ratio(returns, benchmark, periods_per_year=252)
    assert info == pytest.approx(18.332, rel=1e-3)


def test_kelly_criterion():
    series = pd.Series([0.01, -0.005, 0.02, 0.015])
    expected = series.mean() / series.var(ddof=1)
    assert kelly_criterion(series) == pytest.approx(expected)


def test_omega():
    series = pd.Series([0.01, -0.005, 0.02, -0.03, 0.015])
    expected = (0.01 + 0.02 + 0.015) / (0.005 + 0.03)
    assert omega(series) == pytest.approx(expected)


def test_payoff_ratio_inf_and_zero():
    assert math.isinf(payoff_ratio(pd.Series([0.01, 0.02])))
    assert payoff_ratio(pd.Series([-0.01, -0.02])) == pytest.approx(0.0)


def test_payoff_and_profit_ratio_values():
    series = pd.Series([0.02, -0.01, 0.01])
    assert payoff_ratio(series) == pytest.approx(1.5)
    assert profit_ratio(series) == pytest.approx(2.0)


def test_win_loss_ratio_alias():
    series = pd.Series([0.02, -0.01])
    assert win_loss_ratio(series) == pytest.approx(2.0)


def test_profit_factor_and_derived_ratios():
    series = pd.Series([0.02, -0.01])
    pf = profit_factor(series)
    assert pf == pytest.approx(2.0)
    assert cpc_index(series) == pytest.approx(
        pf * win_rate(series) * win_loss_ratio(series)
    )
    assert common_sense_ratio(series) == pytest.approx(pf * tail_ratio(series))


def test_tail_ratio_value():
    series = pd.Series([0.01] * 19 + [-0.005])
    expected = abs(series.quantile(0.95) / series.quantile(0.05))
    assert tail_ratio(series) == pytest.approx(expected)


def test_r_squared_perfect_fit():
    series = pd.Series([0.01, 0.02, 0.03])
    assert r_squared(series, series) == pytest.approx(1.0)


def test_r_squared_partial_fit():
    returns = pd.Series([0.01, 0.02, 0.03])
    benchmark = pd.Series([0.01, 0.02, 0.01])
    value = r_squared(returns, benchmark)
    assert value < 1


def test_risk_of_ruin_simple_formula():
    series = pd.Series([0.01, -0.005, 0.005])
    wins = win_rate(series)
    expected = ((1 - wins) / (1 + wins)) ** len(series)
    assert risk_of_ruin(series) == pytest.approx(expected)


def test_risk_of_ruin_extreme():
    series = pd.Series([0.01] * 5)
    wins = win_rate(series)
    expected = ((1 - wins) / (1 + wins)) ** len(series)
    assert risk_of_ruin(series) == pytest.approx(expected)
    series = pd.Series([-0.01] * 5)
    assert risk_of_ruin(series) == pytest.approx(1.0)


def test_profit_ratio_handles_zero_losses():
    assert math.isinf(profit_ratio(pd.Series([0.01, 0.02])))
    assert profit_ratio(pd.Series([-0.01, -0.02])) == pytest.approx(0.0)


def test_skewness_empty_returns_nan():
    assert math.isnan(skew(pd.Series([], dtype=float)))


def test_kurtosis_pearson_adjustment():
    series = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert kurtosis(series, fisher=False) == pytest.approx(3.0)


def test_total_return_and_volatility_edgecases():
    empty = pd.Series([], dtype=float)
    assert math.isnan(total_return(empty))
    assert math.isnan(volatility(empty))


def test_cagr_string_periods_and_nan_returns():
    series = pd.Series([0.01] * 252)
    assert cagr(series, periods_per_year="D") > 0
    assert math.isnan(cagr(pd.Series([-1.0])))


def test_win_rate_weekly_and_monthly_combination():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    returns = pd.Series([0.01] * 10 + [-0.01] * 10, index=dates)
    assert win_rate(returns, period="weekly") == pytest.approx(0.5)
    monthly_rate = win_rate(returns, period="monthly")
    assert 0 <= monthly_rate <= 1


def test_gain_to_pain_handles_no_losses():
    series = pd.Series([0.01, 0.02])
    assert gain_to_pain_ratio(series) == pytest.approx(float("inf"))


def test_kelly_returns_infinite_when_variance_zero():
    series = pd.Series([0.01, 0.01, 0.01])
    assert math.isinf(kelly_criterion(series))


def test_information_ratio_with_zero_tracking_error():
    series = pd.Series([0.01, 0.02])
    info = information_ratio(series, series, periods_per_year=252)
    assert math.isnan(info)


def test_omega_handles_zero_losses():
    series = pd.Series([0.01, 0.02])
    assert math.isinf(omega(series))
    assert math.isinf(omega(series))
    assert math.isinf(omega(series))
    assert math.isinf(omega(series))
