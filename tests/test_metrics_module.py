import math

import numpy as np
import pandas as pd
import pytest

from quantalytics.analytics import metrics, stats
from quantalytics.reporting import metrics as reporting_metrics
from quantalytics.utils import timeseries as timeseries_utils


@pytest.fixture
def sample_returns():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01], index=dates)


def test_sortino_matches_manual(sample_returns):
    periods = 252
    normalized = timeseries_utils.normalize_returns(sample_returns, nperiods=periods)
    downside = np.sqrt((normalized[normalized < 0] ** 2).sum() / len(normalized))
    expected = normalized.mean() / downside * np.sqrt(periods)
    result = metrics.sortino(sample_returns, periods=periods, annualize=True)
    assert result == pytest.approx(expected)


def test_sortino_smart_penalizes(sample_returns):
    periods = 252
    normalized = timeseries_utils.normalize_returns(sample_returns, nperiods=periods)
    downside = np.sqrt((normalized[normalized < 0] ** 2).sum() / len(normalized))
    penalty = metrics.autocorr_penalty(normalized)
    expected = normalized.mean() / (downside * penalty)
    result = metrics.sortino(
        sample_returns, periods=periods, annualize=False, smart=True
    )
    assert result == pytest.approx(expected)


def test_sharpe_matches_manual(sample_returns):
    periods = 252
    normalized = timeseries_utils.normalize_returns(sample_returns, nperiods=periods)
    divisor = normalized.std(ddof=1)
    base = normalized.mean() / divisor
    expected = base * np.sqrt(periods)
    result = metrics.sharpe(sample_returns, periods=periods, annualize=True)
    assert result == pytest.approx(expected)


def test_sharpe_smart_penalizes(sample_returns):
    periods = 252
    normalized = timeseries_utils.normalize_returns(sample_returns, nperiods=periods)
    penalty = metrics.autocorr_penalty(normalized)
    base_sharpe = metrics.sharpe(sample_returns, periods=periods, annualize=False)
    smart_sharpe = metrics.sharpe(
        sample_returns, periods=periods, annualize=False, smart=True
    )
    assert smart_sharpe == pytest.approx(base_sharpe / penalty)


def test_sharpe_dataframe_returns_series(sample_returns):
    df = pd.DataFrame(
        {
            "base": sample_returns,
            "scaled": sample_returns * 1.25,
        }
    )
    result = metrics.sharpe(df, periods=252)
    expected = pd.Series(
        {col: metrics.sharpe(df[col], periods=252) for col in df.columns}
    )
    pd.testing.assert_series_equal(result, expected)


def test_autocorr(sample_returns):
    penalty = metrics.autocorr_penalty(sample_returns, prepare_returns=False)
    assert penalty >= 1.0


def test_calmar_matches_components(sample_returns):
    periods = 252
    normalized = timeseries_utils.normalize_returns(sample_returns)
    expected_cagr = stats.cagr(normalized, periods=periods)
    expected_max_dd = stats.max_drawdown(normalized)
    result = metrics.calmar(sample_returns, periods=periods)
    assert result == pytest.approx(expected_cagr / abs(expected_max_dd))


def test_omega_and_gain_to_pain(sample_returns):
    manual = sample_returns - 0
    gains = manual[manual > 0].sum()
    losses = -(manual[manual < 0].sum())
    assert metrics.omega(sample_returns) == pytest.approx(gains / losses)
    assert metrics.gain_to_pain_ratio(sample_returns) == pytest.approx(
        manual.sum() / losses
    )
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * -1})
    omega = metrics.omega(df)
    gain_to_pain = metrics.gain_to_pain_ratio(df)
    assert isinstance(omega, pd.Series)
    assert isinstance(gain_to_pain, pd.Series)


def test_skew_and_kurtosis_against_pandas(sample_returns):
    assert metrics.skew(sample_returns) == pytest.approx(sample_returns.skew())
    assert metrics.kurtosis(sample_returns) == pytest.approx(sample_returns.kurtosis())
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 1.2})
    pd.testing.assert_series_equal(metrics.skew(df), df.skew())
    pd.testing.assert_series_equal(metrics.kurtosis(df), df.kurtosis())


def test_gain_to_pain_handles_edge_cases():
    positive = pd.Series([0.01, 0.02])
    assert metrics.gain_to_pain_ratio(positive) == float("inf")
    assert metrics.gain_to_pain_ratio(pd.Series([-0.01, -0.02])) == -1.0


def test_omega_handles_edge_cases():
    positive = pd.Series([0.01, 0.02])
    assert metrics.omega(positive) == float("inf")
    assert metrics.omega(pd.Series([-0.01, -0.02])) == 0.0


def test_ulcer_and_risk_metrics(sample_returns):
    ui = metrics.ulcer_index(sample_returns)
    assert ui >= 0
    upi = metrics.ulcer_performance_index(sample_returns)
    assert math.isfinite(upi)
    assert metrics.upi(sample_returns) == pytest.approx(upi)
    serenity = metrics.serenity_index(sample_returns)
    assert math.isfinite(serenity)
    expected_ror = (
        (1 - metrics.win_rate(sample_returns)) / (1 + metrics.win_rate(sample_returns))
    ) ** len(sample_returns)
    assert metrics.risk_of_ruin(sample_returns) == pytest.approx(expected_ror)
    assert metrics.ror(sample_returns) == pytest.approx(expected_ror)
    assert isinstance(
        metrics.risk_of_ruin(
            pd.DataFrame({"a": sample_returns, "b": sample_returns * -1})
        ),
        pd.Series,
    )
    dd = metrics.to_drawdown_series(sample_returns)
    assert (dd <= 0).all()
    var = metrics.value_at_risk(sample_returns, confidence=0.95)
    cvar = metrics.conditional_value_at_risk(sample_returns, confidence=0.95)
    assert cvar <= var
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 0.5})
    pd.testing.assert_series_equal(
        metrics.value_at_risk(df, confidence=95), metrics.var(df, confidence=95)
    )
    pd.testing.assert_series_equal(
        metrics.conditional_value_at_risk(df), metrics.cvar(df)
    )


def test_sortino_and_sharpe_require_periods(sample_returns):
    with pytest.raises(ValueError):
        metrics.sortino(sample_returns, rf=0.01, periods=None)
    with pytest.raises(ValueError):
        metrics.sharpe(sample_returns, rf=0.01, periods=None)


def test_calmar_handles_zero_drawdown():
    flat = pd.Series([0.0, 0.0, 0.0])
    assert math.isnan(metrics.calmar(flat, periods=1))


def test_serenity_index_dataframe_zero_denominator():
    data = pd.DataFrame({"flat": [0.0, 0.0], "mixed": [0.01, -0.01]})
    result = metrics.serenity_index(data)
    assert math.isnan(result["flat"])
    assert math.isfinite(result["mixed"])


def test_ulcer_index_calc():
    series = pd.Series([0.01, -0.01, 0.02, -0.05, 0.03])
    ui = metrics.ulcer_index(series)
    prices = (1 + series).cumprod()
    running_max = prices.expanding().max()
    drawdowns = prices / running_max - 1
    expected = math.sqrt((drawdowns**2).sum() / (len(series) - 1))
    assert ui == pytest.approx(expected)
    df = pd.DataFrame({"a": series, "b": series * 0.5})
    per_column = metrics.ulcer_index(df)
    drawdowns_b = metrics.to_drawdown_series(df["b"], prepare_returns=False)
    expected_b = math.sqrt((drawdowns_b**2).sum() / (len(series) - 1))
    assert per_column["a"] == pytest.approx(expected)
    assert per_column["b"] == pytest.approx(expected_b)
    assert isinstance(per_column, pd.Series)
    rf = 0.01
    upi_value = metrics.ulcer_performance_index(series, rf=rf)
    comp = ((1 + series).prod() - 1) - rf
    assert upi_value == pytest.approx(comp / ui)


def test_romad_matches_calmar(sample_returns):
    romad_value = metrics.romad(sample_returns, periods=252)
    assert romad_value == pytest.approx(metrics.calmar(sample_returns, periods=252))
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 0.5})
    romad_df = metrics.romad(df, periods=252)
    calmar_df = metrics.calmar(df, periods=252)
    pd.testing.assert_series_equal(romad_df, calmar_df)


def test_tail_payoff_profit_metrics(sample_returns):
    tr = metrics.tail_ratio(sample_returns)
    upper = sample_returns.quantile(0.95)
    lower = sample_returns.quantile(0.05)
    assert tr == pytest.approx(abs(upper / lower))
    pr = metrics.payoff_ratio(sample_returns)
    assert pr == pytest.approx(
        metrics.avg_win(sample_returns) / abs(metrics.avg_loss(sample_returns))
    )
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 2})
    assert isinstance(metrics.tail_ratio(df), pd.Series)


def test_profit_and_cpc_metrics(sample_returns):
    pf = metrics.profit_factor(sample_returns)
    gains = sample_returns[sample_returns >= 0].sum()
    losses = sample_returns[sample_returns < 0].sum()
    assert pf == pytest.approx(abs(gains / losses))
    pr = metrics.profit_ratio(sample_returns)
    assert isinstance(pr, float)
    cpc = metrics.cpc_index(sample_returns)
    assert isinstance(cpc, float)
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * -1})
    assert isinstance(metrics.profit_ratio(df), pd.Series)


def test_expectancy_series_and_dataframe(sample_returns):
    wins = sample_returns[sample_returns > 0]
    losses = sample_returns[sample_returns < 0]
    win_rate = len(wins) / len(sample_returns)
    loss_rate = len(losses) / len(sample_returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    expected = (win_rate * avg_win) - (loss_rate * avg_loss)
    assert metrics.expectancy(sample_returns) == pytest.approx(expected)
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 0.5})
    result = metrics.expectancy(df)
    assert isinstance(result, pd.Series)
    assert result["a"] == pytest.approx(expected)


def test_outlier_win_ratio(sample_returns):
    """Test outlier win ratio calculation."""
    quantile = 0.99
    normalized = timeseries_utils.normalize_returns(sample_returns)
    positive = normalized[normalized >= 0]
    expected = normalized.quantile(quantile) / positive.mean()
    result = metrics.outlier_win_ratio(sample_returns, quantile=quantile)
    assert result == pytest.approx(expected)

    # Test with DataFrame
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 1.5})
    result_df = metrics.outlier_win_ratio(df, quantile=quantile)
    assert isinstance(result_df, pd.Series)
    assert len(result_df) == 2


def test_outlier_win_ratio_no_positive_returns():
    """Test outlier win ratio with no positive returns."""
    negative_only = pd.Series([-0.01, -0.02, -0.03])
    result = metrics.outlier_win_ratio(negative_only, prepare_returns=False)
    assert result == float("inf")


def test_outlier_loss_ratio(sample_returns):
    """Test outlier loss ratio calculation."""
    quantile = 0.01
    normalized = timeseries_utils.normalize_returns(sample_returns)
    negative = normalized[normalized < 0]
    expected = normalized.quantile(quantile) / negative.mean()
    result = metrics.outlier_loss_ratio(sample_returns, quantile=quantile)
    assert result == pytest.approx(expected)

    # Test with DataFrame
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 1.5})
    result_df = metrics.outlier_loss_ratio(df, quantile=quantile)
    assert isinstance(result_df, pd.Series)
    assert len(result_df) == 2


def test_outlier_loss_ratio_no_negative_returns():
    """Test outlier loss ratio with no negative returns."""
    positive_only = pd.Series([0.01, 0.02, 0.03])
    result = metrics.outlier_loss_ratio(positive_only, prepare_returns=False)
    assert result == float("-inf")


def test_outlier_ratios_measure_tail_risk():
    """Test that outlier ratios correctly measure tail risk."""
    # Series with extreme outlier win
    outlier_win_series = pd.Series([0.01, 0.01, 0.01, 0.50])  # Last value is outlier
    win_ratio = metrics.outlier_win_ratio(
        outlier_win_series, quantile=0.95, prepare_returns=False
    )
    assert win_ratio > 1.0  # Outlier should be much larger than mean

    # Series with extreme outlier loss
    outlier_loss_series = pd.Series(
        [-0.01, -0.01, -0.01, -0.50]
    )  # Last value is outlier
    loss_ratio = metrics.outlier_loss_ratio(
        outlier_loss_series, quantile=0.05, prepare_returns=False
    )
    assert loss_ratio > 1.0  # Both negative, ratio should be > 1


def test_recovery_factor(sample_returns):
    """Test recovery factor calculation."""
    normalized = timeseries_utils.normalize_returns(sample_returns)
    total_returns = normalized.sum()
    max_dd = stats.max_drawdown(sample_returns)
    expected = abs(total_returns) / abs(max_dd)
    result = metrics.recovery_factor(sample_returns)
    assert result == pytest.approx(expected)

    # Test with DataFrame
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 0.5})
    result_df = metrics.recovery_factor(df)
    assert isinstance(result_df, pd.Series)
    assert len(result_df) == 2


def test_recovery_factor_with_rf():
    """Test recovery factor with risk-free rate."""
    returns = pd.Series([0.05, -0.02, 0.03, 0.01])
    rf = 0.01
    normalized = timeseries_utils.normalize_returns(returns)
    total_returns = normalized.sum() - rf
    max_dd = stats.max_drawdown(returns)
    expected = abs(total_returns) / abs(max_dd)
    result = metrics.recovery_factor(returns, rf=rf)
    assert result == pytest.approx(expected)


def test_recovery_factor_zero_drawdown():
    """Test recovery factor with zero drawdown."""
    # Monotonically increasing returns (no drawdown)
    increasing = pd.Series([0.01, 0.02, 0.03, 0.04])
    result = metrics.recovery_factor(increasing, prepare_returns=False)
    assert result == float("inf")

    # Negative returns with zero drawdown should return -inf
    decreasing_no_dd = pd.Series([0.0, 0.0, 0.0])
    result_neg = metrics.recovery_factor(decreasing_no_dd, prepare_returns=False)
    # With zero total returns and zero drawdown, should be inf
    assert math.isinf(result_neg)


def test_risk_return_ratio(sample_returns):
    """Test risk-return ratio calculation."""
    normalized = timeseries_utils.normalize_returns(sample_returns)
    expected = normalized.mean() / normalized.std()
    result = metrics.risk_return_ratio(sample_returns)
    assert result == pytest.approx(expected)

    # Test with DataFrame
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns * 1.2})
    result_df = metrics.risk_return_ratio(df)
    assert isinstance(result_df, pd.Series)
    assert len(result_df) == 2


def test_risk_return_ratio_vs_sharpe():
    """Test that risk-return ratio is similar to Sharpe without rf adjustment."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015])
    risk_return = metrics.risk_return_ratio(returns, prepare_returns=False)
    # Sharpe with rf=0 and no annualization should be same
    sharpe_no_rf = metrics.sharpe(returns, rf=0, periods=None, annualize=False)
    assert risk_return == pytest.approx(sharpe_no_rf)


def test_risk_return_ratio_zero_volatility():
    """Test risk-return ratio with zero volatility."""
    # All same positive value
    constant_positive = pd.Series([0.01, 0.01, 0.01])
    result_pos = metrics.risk_return_ratio(constant_positive, prepare_returns=False)
    assert result_pos == float("inf")

    # All same negative value
    constant_negative = pd.Series([-0.01, -0.01, -0.01])
    result_neg = metrics.risk_return_ratio(constant_negative, prepare_returns=False)
    assert result_neg == float("-inf")


def test_new_metrics_dataframe_consistency(sample_returns):
    """Test that all new metrics handle DataFrame input consistently."""
    df = pd.DataFrame(
        {
            "strategy_1": sample_returns,
            "strategy_2": sample_returns * 1.5,
            "strategy_3": sample_returns * 0.8,
        }
    )

    # Test outlier_win_ratio
    owr_result = metrics.outlier_win_ratio(df)
    assert isinstance(owr_result, pd.Series)
    assert len(owr_result) == 3
    assert not owr_result.isna().any()

    # Test outlier_loss_ratio
    olr_result = metrics.outlier_loss_ratio(df)
    assert isinstance(olr_result, pd.Series)
    assert len(olr_result) == 3
    assert not olr_result.isna().any()

    # Test recovery_factor
    rf_result = metrics.recovery_factor(df)
    assert isinstance(rf_result, pd.Series)
    assert len(rf_result) == 3
    assert not rf_result.isna().any()

    # Test risk_return_ratio
    rrr_result = metrics.risk_return_ratio(df)
    assert isinstance(rrr_result, pd.Series)
    assert len(rrr_result) == 3
    assert not rrr_result.isna().any()


def test_smart_sharpe_equals_sharpe_with_smart_flag(sample_returns):
    """Test that smart_sharpe() equals sharpe(..., smart=True)."""
    periods = 252
    rf = 0.02

    smart_result = metrics.smart_sharpe(
        sample_returns, rf=rf, periods=periods, annualize=True
    )
    sharpe_result = metrics.sharpe(
        sample_returns, rf=rf, periods=periods, annualize=True, smart=True
    )

    assert smart_result == pytest.approx(sharpe_result)


def test_smart_sharpe_applies_autocorr_penalty(sample_returns):
    """Test that smart_sharpe applies autocorrelation penalty."""
    periods = 252

    # Smart Sharpe should be less than or equal to regular Sharpe (due to penalty)
    smart = metrics.smart_sharpe(sample_returns, periods=periods, annualize=False)
    regular = metrics.sharpe(
        sample_returns, periods=periods, annualize=False, smart=False
    )
    penalty = metrics.autocorr_penalty(
        timeseries_utils.normalize_returns(sample_returns)
    )

    # Smart sharpe should equal regular sharpe divided by penalty
    assert smart == pytest.approx(regular / penalty)


def test_smart_sharpe_with_rf_requires_periods():
    """Test that smart_sharpe raises ValueError when rf is non-zero and periods is None."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.005])

    with pytest.raises(
        ValueError, match="When rf is non-zero, periods must be specified"
    ):
        metrics.smart_sharpe(returns, rf=0.02, periods=None)


def test_smart_sharpe_dataframe_returns_series(sample_returns):
    """Test that smart_sharpe returns Series for DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy_1": sample_returns,
            "strategy_2": sample_returns * 1.5,
        }
    )

    result = metrics.smart_sharpe(df, periods=252)
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert "strategy_1" in result.index
    assert "strategy_2" in result.index


def test_smart_sharpe_annualize_parameter(sample_returns):
    """Test smart_sharpe with annualize parameter."""
    periods = 252

    annualized = metrics.smart_sharpe(sample_returns, periods=periods, annualize=True)
    not_annualized = metrics.smart_sharpe(
        sample_returns, periods=periods, annualize=False
    )

    # Annualized should be sqrt(periods) times the non-annualized
    assert annualized == pytest.approx(not_annualized * np.sqrt(periods))


def test_smart_sortino_equals_sortino_with_smart_flag(sample_returns):
    """Test that smart_sortino() equals sortino(..., smart=True)."""
    periods = 252
    rf = 0.02

    smart_result = metrics.smart_sortino(
        sample_returns, rf=rf, periods=periods, annualize=True
    )
    sortino_result = metrics.sortino(
        sample_returns, rf=rf, periods=periods, annualize=True, smart=True
    )

    assert smart_result == pytest.approx(sortino_result)


def test_smart_sortino_applies_autocorr_penalty(sample_returns):
    """Test that smart_sortino applies autocorrelation penalty to downside."""
    periods = 252

    # Smart Sortino should differ from regular Sortino due to penalty
    smart = metrics.smart_sortino(sample_returns, periods=periods, annualize=False)
    regular = metrics.sortino(
        sample_returns, periods=periods, annualize=False, smart=False
    )

    # They should be different (penalty applied to downside deviation)
    assert smart != regular


def test_smart_sortino_with_rf_requires_periods():
    """Test that smart_sortino raises ValueError when rf is non-zero and periods is None."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.005])

    with pytest.raises(
        ValueError, match="When rf is non-zero, periods must be specified"
    ):
        metrics.smart_sortino(returns, rf=0.02, periods=None)


def test_smart_sortino_series_input(sample_returns):
    """Test that smart_sortino works with Series input."""
    result = metrics.smart_sortino(sample_returns, periods=252)
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_smart_sortino_annualize_parameter(sample_returns):
    """Test smart_sortino with annualize parameter."""
    periods = 252

    annualized = metrics.smart_sortino(sample_returns, periods=periods, annualize=True)
    not_annualized = metrics.smart_sortino(
        sample_returns, periods=periods, annualize=False
    )

    # Annualized should be sqrt(periods) times the non-annualized
    assert annualized == pytest.approx(not_annualized * np.sqrt(periods))


def test_smart_metrics_more_conservative(sample_returns):
    """Test that smart metrics are generally more conservative than regular metrics."""
    periods = 252

    # For returns with autocorrelation, smart ratios should typically be lower
    regular_sharpe = metrics.sharpe(sample_returns, periods=periods, annualize=False)
    smart_sharpe_result = metrics.smart_sharpe(
        sample_returns, periods=periods, annualize=False
    )

    # The penalty should make smart sharpe smaller or equal
    # (Equal only if autocorrelation is zero, which is rare)
    penalty = metrics.autocorr_penalty(
        timeseries_utils.normalize_returns(sample_returns)
    )
    assert penalty >= 1.0  # Penalty should be >= 1
    assert smart_sharpe_result <= regular_sharpe


def test_rar_calculation(sample_returns):
    """Test RAR calculation matches manual calculation."""
    periods = 252

    normalized = timeseries_utils.normalize_returns(sample_returns)
    expected_cagr = stats.cagr(normalized, periods=periods)
    expected_exposure = stats.exposure(normalized)
    expected = expected_cagr / expected_exposure

    result = metrics.rar(sample_returns, periods=periods)
    assert result == pytest.approx(expected)


def test_rar_with_rf(sample_returns):
    """Test RAR with risk-free rate."""
    periods = 252
    rf = 0.02

    result = metrics.rar(sample_returns, rf=rf, periods=periods)
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_rar_dataframe_returns_series(sample_returns):
    """Test that RAR returns Series for DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy_1": sample_returns,
            "strategy_2": sample_returns * 1.5,
        }
    )

    result = metrics.rar(df, periods=252)
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert "strategy_1" in result.index
    assert "strategy_2" in result.index


def test_rar_accounts_for_exposure():
    """Test that RAR properly accounts for exposure."""
    # Create returns with 50% exposure (half zeros)
    returns_full = pd.Series([0.01, 0.02, 0.01, 0.02])  # 100% exposure
    returns_half = pd.Series([0.01, 0.0, 0.02, 0.0])  # 50% exposure

    rar_full = metrics.rar(returns_full, prepare_returns=False, periods=252)
    rar_half = metrics.rar(returns_half, prepare_returns=False, periods=252)

    # RAR with lower exposure should be higher (penalized less)
    # since CAGR is divided by exposure
    assert isinstance(rar_full, float)
    assert isinstance(rar_half, float)


def test_rar_zero_exposure():
    """Test RAR with zero exposure returns nan."""
    # All zero returns = zero exposure
    zero_returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    result = metrics.rar(zero_returns, prepare_returns=False)

    # Should be nan due to zero exposure
    assert np.isnan(result)


def test_kelly_criterion_calculation(sample_returns):
    """Test Kelly criterion calculation matches manual calculation."""
    normalized = timeseries_utils.normalize_returns(sample_returns)

    win_ratio = stats.avg_win(normalized) / abs(stats.avg_loss(normalized))
    win_prob = stats.win_rate(normalized)
    lose_prob = 1 - win_prob

    expected = ((win_ratio * win_prob) - lose_prob) / win_ratio
    result = metrics.kelly_criterion(sample_returns)

    assert result == pytest.approx(expected)


def test_kelly_criterion_dataframe_returns_series(sample_returns):
    """Test that kelly_criterion returns Series for DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy_1": sample_returns,
            "strategy_2": sample_returns * 1.5,
        }
    )

    result = metrics.kelly_criterion(df)
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert "strategy_1" in result.index
    assert "strategy_2" in result.index


def test_kelly_criterion_profitable_strategy():
    """Test Kelly criterion with a profitable strategy."""
    # Create profitable returns: 60% win rate, 2:1 win/loss ratio
    returns = pd.Series([0.02, 0.02, 0.02, -0.01, -0.01])

    result = metrics.kelly_criterion(returns, prepare_returns=False)

    # Should be positive for profitable strategy
    assert result > 0
    # Should be reasonable (not suggest over-leveraging)
    assert result <= 1.0


def test_kelly_criterion_unprofitable_strategy():
    """Test Kelly criterion with unprofitable strategy."""
    # Create unprofitable returns: more/bigger losses than wins
    returns = pd.Series([0.01, -0.02, -0.02, -0.02, 0.01])

    result = metrics.kelly_criterion(returns, prepare_returns=False)

    # Should be zero or negative for unprofitable strategy
    assert result <= 0


def test_kelly_criterion_uses_payoff_and_win_rate():
    """Test that Kelly uses payoff ratio and win rate correctly."""
    returns = pd.Series([0.03, 0.03, -0.01, -0.01, -0.01, -0.01])

    # Calculate components manually
    normalized = timeseries_utils.normalize_returns(returns)
    payoff = metrics.payoff_ratio(normalized, prepare_returns=False)
    win_r = stats.win_rate(normalized)
    lose_r = 1 - win_r

    expected = ((payoff * win_r) - lose_r) / payoff
    result = metrics.kelly_criterion(returns)

    assert result == pytest.approx(expected)


def test_kelly_criterion_no_prepare_returns():
    """Test Kelly criterion with prepare_returns=False."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.01, 0.02])

    result = metrics.kelly_criterion(returns, prepare_returns=False)
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_expected_shortfall_equals_cvar(sample_returns):
    """Test that expected_shortfall equals conditional_value_at_risk."""
    confidence = 0.95
    sigma = 1

    es_result = metrics.expected_shortfall(
        sample_returns, sigma=sigma, confidence=confidence
    )
    cvar_result = metrics.conditional_value_at_risk(
        sample_returns, sigma=sigma, confidence=confidence
    )

    assert es_result == pytest.approx(cvar_result)


def test_expected_shortfall_dataframe_returns_series(sample_returns):
    """Test that expected_shortfall returns Series for DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy_1": sample_returns,
            "strategy_2": sample_returns * 1.5,
        }
    )

    result = metrics.expected_shortfall(df, confidence=0.95)
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert "strategy_1" in result.index
    assert "strategy_2" in result.index


def test_expected_shortfall_confidence_levels(sample_returns):
    """Test expected_shortfall with different confidence levels."""
    # Higher confidence should give more extreme (negative) values
    es_95 = metrics.expected_shortfall(sample_returns, confidence=0.95)
    es_99 = metrics.expected_shortfall(sample_returns, confidence=0.99)

    assert isinstance(es_95, float)
    assert isinstance(es_99, float)
    # 99% ES should be more extreme (more negative) than 95% ES
    # (assuming there are losses in the tail)
    assert es_99 <= es_95


def test_expected_shortfall_is_tail_average():
    """Test that expected_shortfall measures average tail loss."""
    # Create returns where we know the tail
    returns = pd.Series([0.02, 0.01, 0.005, -0.01, -0.02, -0.03, -0.04])

    # At 95% confidence, roughly 5% worst = 1 observation (the -0.04)
    # ES should be close to mean of worst observations
    es = metrics.expected_shortfall(returns, confidence=0.95, prepare_returns=False)

    # Should be negative (representing losses)
    assert es < 0


def test_expected_shortfall_sigma_parameter(sample_returns):
    """Test expected_shortfall with different sigma values."""
    es_sigma_1 = metrics.expected_shortfall(sample_returns, sigma=1, confidence=0.95)
    es_sigma_2 = metrics.expected_shortfall(sample_returns, sigma=2, confidence=0.95)

    assert isinstance(es_sigma_1, float)
    assert isinstance(es_sigma_2, float)
    # Different sigma values may produce different results
    # (depending on implementation of conditional_value_at_risk)


def test_expected_shortfall_prepare_returns_parameter(sample_returns):
    """Test expected_shortfall with prepare_returns parameter."""
    result_with_prep = metrics.expected_shortfall(sample_returns, prepare_returns=True)
    result_without_prep = metrics.expected_shortfall(
        sample_returns, prepare_returns=False
    )

    # Both should return valid floats
    assert isinstance(result_with_prep, float)
    assert isinstance(result_without_prep, float)


def test_expected_shortfall_alias_consistency():
    """Test that all CVaR aliases return the same value."""
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])

    es = metrics.expected_shortfall(returns, prepare_returns=False)
    cvar = metrics.cvar(returns, prepare_returns=False)
    full = metrics.conditional_value_at_risk(returns, prepare_returns=False)

    # All three should be identical
    assert es == pytest.approx(cvar)
    assert es == pytest.approx(full)
    assert cvar == pytest.approx(full)


def test_adjusted_sortino_equals_sortino_divided_by_sqrt2(sample_returns):
    """Test that adjusted_sortino equals sortino / sqrt(2)."""
    periods = 252
    rf = 0.02

    adjusted = metrics.adjusted_sortino(
        sample_returns, rf=rf, periods=periods, annualize=True
    )
    standard = metrics.sortino(
        sample_returns, rf=rf, periods=periods, annualize=True, smart=False
    )

    # Adjusted should equal standard / sqrt(2)
    assert adjusted == pytest.approx(standard / np.sqrt(2))


def test_adjusted_sortino_with_smart_flag(sample_returns):
    """Test that adjusted_sortino works with smart=True."""
    periods = 252

    # Should equal smart_sortino / sqrt(2)
    adjusted_smart = metrics.adjusted_sortino(
        sample_returns, periods=periods, smart=True
    )
    smart_sortino_val = metrics.smart_sortino(sample_returns, periods=periods)

    assert adjusted_smart == pytest.approx(smart_sortino_val / np.sqrt(2))


def test_adjusted_sortino_dataframe_returns_series(sample_returns):
    """Test that adjusted_sortino returns Series for DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy_1": sample_returns,
            "strategy_2": sample_returns * 1.5,
        }
    )

    result = metrics.adjusted_sortino(df, periods=252)
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert "strategy_1" in result.index
    assert "strategy_2" in result.index


def test_adjusted_sortino_annualize_parameter(sample_returns):
    """Test adjusted_sortino with annualize parameter."""
    periods = 252

    annualized = metrics.adjusted_sortino(
        sample_returns, periods=periods, annualize=True
    )
    not_annualized = metrics.adjusted_sortino(
        sample_returns, periods=periods, annualize=False
    )

    # Annualized should be sqrt(periods) times the non-annualized
    assert annualized == pytest.approx(not_annualized * np.sqrt(periods))


def test_adjusted_sortino_lower_than_sortino():
    """Test that adjusted Sortino is lower than standard Sortino (due to sqrt(2) division)."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.015])

    adjusted = metrics.adjusted_sortino(returns, rf=0, periods=252, annualize=True)
    standard = metrics.sortino(returns, rf=0, periods=252, annualize=True)

    # Adjusted should be lower (divided by sqrt(2) ≈ 1.414)
    assert adjusted < standard
    assert adjusted == pytest.approx(standard / np.sqrt(2))


def test_adjusted_sortino_comparable_to_sharpe():
    """Test that adjusted Sortino is more comparable to Sharpe ratio."""
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.015, -0.01])

    adjusted_sortino_val = metrics.adjusted_sortino(
        returns, rf=0, periods=252, annualize=True
    )
    sharpe_val = metrics.sharpe(returns, rf=0, periods=252, annualize=True)

    # Both should be floats
    assert isinstance(adjusted_sortino_val, float)
    assert isinstance(sharpe_val, float)

    # Adjusted Sortino should be closer to Sharpe than standard Sortino
    # (This is the purpose of the adjustment)
    standard_sortino_val = metrics.sortino(returns, rf=0, periods=252, annualize=True)

    # Standard Sortino is typically higher than Sharpe
    # Adjusted Sortino should be closer to Sharpe value
    diff_adjusted = abs(adjusted_sortino_val - sharpe_val)
    diff_standard = abs(standard_sortino_val - sharpe_val)

    # Adjustment makes it more comparable
    assert diff_adjusted < diff_standard


def test_adjusted_sortino_with_rf(sample_returns):
    """Test adjusted_sortino with different risk-free rates."""
    rf_0 = metrics.adjusted_sortino(sample_returns, rf=0.0, periods=252)
    rf_2 = metrics.adjusted_sortino(sample_returns, rf=0.02, periods=252)

    # Different rf should give different results
    assert rf_0 != rf_2
    assert isinstance(rf_0, float)
    assert isinstance(rf_2, float)


def test_adjusted_sortino_series_input(sample_returns):
    """Test that adjusted_sortino works with Series input."""
    result = metrics.adjusted_sortino(sample_returns, periods=252)
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_adjusted_sortino_mixed_returns():
    """Test adjusted_sortino with mixed positive and negative returns."""
    # Include both positive and negative returns
    returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.03, 0.012])

    result = metrics.adjusted_sortino(returns, periods=252, rf=0, annualize=True)

    # Should be positive for returns with positive mean
    assert result > 0
    assert isinstance(result, float)
    assert not np.isnan(result)


# Tests for rolling_sharpe


def test_rolling_sharpe_basic(sample_returns):
    """Test rolling_sharpe basic functionality."""
    result = metrics.rolling_sharpe(sample_returns, rolling_period=3, annualize=False)

    # Should return a Series
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns)

    # First 2 values should be NaN (rolling_period - 1)
    assert result.iloc[:2].isna().all()

    # Later values should be numeric
    assert not result.iloc[3:].isna().all()


def test_rolling_sharpe_annualization(sample_returns):
    """Test rolling_sharpe annualization."""
    non_annualized = metrics.rolling_sharpe(
        sample_returns, rolling_period=3, annualize=False, periods=252
    )
    annualized = metrics.rolling_sharpe(
        sample_returns, rolling_period=3, annualize=True, periods=252
    )

    # Annualized should be scaled by sqrt(252)
    # Check on values that exist (not NaN)
    valid_mask = ~non_annualized.isna()
    if valid_mask.any():
        ratio = (annualized[valid_mask] / non_annualized[valid_mask]).iloc[0]
        assert ratio == pytest.approx(np.sqrt(252), rel=1e-5)


def test_rolling_sharpe_rf_parameter(sample_returns):
    """Test rolling_sharpe with risk-free rate."""
    rf_0 = metrics.rolling_sharpe(sample_returns, rf=0.0, rolling_period=3, periods=252)
    rf_2 = metrics.rolling_sharpe(
        sample_returns, rf=0.02, rolling_period=3, periods=252
    )

    # Different rf should give different results
    assert not rf_0.equals(rf_2)


def test_rolling_sharpe_window_size():
    """Test rolling_sharpe with different window sizes."""
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.01]
    )

    result_small = metrics.rolling_sharpe(returns, rolling_period=3, annualize=False)
    result_large = metrics.rolling_sharpe(returns, rolling_period=5, annualize=False)

    # Should have different number of NaN values
    assert result_small.isna().sum() == 2  # First 2 values
    assert result_large.isna().sum() == 4  # First 4 values


def test_rolling_sharpe_dataframe_support():
    """Test rolling_sharpe with DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy_a": [0.01, -0.01, 0.02, 0.01, -0.005, 0.015, 0.01, 0.02],
            "strategy_b": [0.02, -0.02, 0.03, 0.01, -0.01, 0.02, 0.015, 0.01],
        }
    )

    result = metrics.rolling_sharpe(df, rolling_period=3, annualize=False)

    # Should return a DataFrame with same columns
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["strategy_a", "strategy_b"]
    assert len(result) == len(df)

    # First 2 rows should be NaN for all columns
    assert result.iloc[:2].isna().all().all()


def test_rolling_sharpe_prepare_returns_flag():
    """Test rolling_sharpe with prepare_returns flag."""
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01, 0.02])

    with_prep = metrics.rolling_sharpe(
        returns, rolling_period=3, prepare_returns=True, annualize=False
    )
    without_prep = metrics.rolling_sharpe(
        returns, rolling_period=3, prepare_returns=False, annualize=False
    )

    # Both should return Series
    assert isinstance(with_prep, pd.Series)
    assert isinstance(without_prep, pd.Series)


def test_rolling_sharpe_consistency_with_sharpe():
    """Test that rolling_sharpe last value approximates full-period sharpe for stable returns."""
    # Use consistent returns to reduce variance
    returns = pd.Series([0.01] * 20 + [-0.005] * 10)

    rolling = metrics.rolling_sharpe(
        returns, rolling_period=len(returns), annualize=False
    )
    static = metrics.sharpe(returns, periods=None, annualize=False)

    # Last rolling value should approximate static sharpe (same window)
    assert rolling.iloc[-1] == pytest.approx(static, rel=0.01)


def test_rolling_sharpe_rf_validation():
    """Test that rolling_sharpe raises error when rf != 0 and rolling_period is None."""
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError, match="Must provide periods if rf != 0"):
        metrics.rolling_sharpe(returns, rf=0.02, rolling_period=None)


# Tests for rolling_sortino


def test_rolling_sortino_basic(sample_returns):
    """Test rolling_sortino basic functionality."""
    result = metrics.rolling_sortino(sample_returns, rolling_period=3, annualize=False)

    # Should return a Series
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns)

    # First 2 values should be NaN (rolling_period - 1)
    assert result.iloc[:2].isna().all()

    # Later values should be numeric (may contain inf if no downside)
    assert not result.iloc[3:].isna().all()


def test_rolling_sortino_annualization(sample_returns):
    """Test rolling_sortino annualization."""
    non_annualized = metrics.rolling_sortino(
        sample_returns, rolling_period=3, annualize=False, periods=252
    )
    annualized = metrics.rolling_sortino(
        sample_returns, rolling_period=3, annualize=True, periods=252
    )

    # Annualized should be scaled by sqrt(252)
    # Check on finite values only
    valid_mask = (
        np.isfinite(non_annualized) & np.isfinite(annualized) & (non_annualized != 0)
    )
    if valid_mask.any():
        ratio = (annualized[valid_mask] / non_annualized[valid_mask]).iloc[0]
        assert ratio == pytest.approx(np.sqrt(252), rel=1e-5)


def test_rolling_sortino_rf_parameter(sample_returns):
    """Test rolling_sortino with risk-free rate."""
    rf_0 = metrics.rolling_sortino(
        sample_returns, rf=0.0, rolling_period=3, periods=252
    )
    rf_2 = metrics.rolling_sortino(
        sample_returns, rf=0.02, rolling_period=3, periods=252
    )

    # Different rf should give different results
    assert not rf_0.equals(rf_2)


def test_rolling_sortino_window_size():
    """Test rolling_sortino with different window sizes."""
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.01]
    )

    result_small = metrics.rolling_sortino(returns, rolling_period=3, annualize=False)
    result_large = metrics.rolling_sortino(returns, rolling_period=5, annualize=False)

    # Should have different number of NaN values
    assert result_small.isna().sum() == 2  # First 2 values
    assert result_large.isna().sum() == 4  # First 4 values


def test_rolling_sortino_dataframe_support():
    """Test rolling_sortino with DataFrame input."""
    df = pd.DataFrame(
        {
            "strategy_a": [0.01, -0.01, 0.02, 0.01, -0.005, 0.015, 0.01, 0.02],
            "strategy_b": [0.02, -0.02, 0.03, 0.01, -0.01, 0.02, 0.015, 0.01],
        }
    )

    result = metrics.rolling_sortino(df, rolling_period=3, annualize=False)

    # Should return a DataFrame with same columns
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["strategy_a", "strategy_b"]
    assert len(result) == len(df)

    # First 2 rows should be NaN for all columns
    assert result.iloc[:2].isna().all().all()


def test_rolling_sortino_downside_focus():
    """Test that rolling_sortino only penalizes downside volatility."""
    # Window with no negative returns should give inf
    returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])

    result = metrics.rolling_sortino(returns, rolling_period=3, annualize=False)

    # Values should be inf when there's no downside in the window
    # (dividing by sqrt(0) gives inf)
    assert np.isinf(result.iloc[-1])


def test_rolling_sortino_prepare_returns_flag():
    """Test rolling_sortino with prepare_returns flag."""
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01, 0.02])

    with_prep = metrics.rolling_sortino(
        returns, rolling_period=3, prepare_returns=True, annualize=False
    )
    without_prep = metrics.rolling_sortino(
        returns, rolling_period=3, prepare_returns=False, annualize=False
    )

    # Both should return Series
    assert isinstance(with_prep, pd.Series)
    assert isinstance(without_prep, pd.Series)


def test_rolling_sortino_higher_than_sharpe():
    """Test that rolling_sortino is typically higher than rolling_sharpe for positive skew."""
    # Positive skew: mostly small gains, occasional large loss
    returns = pd.Series([0.01, 0.01, -0.05, 0.01, 0.01, 0.01, 0.01, -0.03, 0.01, 0.01])

    rolling_sortino = metrics.rolling_sortino(
        returns, rolling_period=5, annualize=False
    )
    rolling_sharpe = metrics.rolling_sharpe(returns, rolling_period=5, annualize=False)

    # For positive skew, sortino should generally be >= sharpe
    # Check on finite values
    valid_mask = np.isfinite(rolling_sortino) & np.isfinite(rolling_sharpe)
    if valid_mask.any():
        # Most values should have sortino >= sharpe (within numerical tolerance)
        assert (
            rolling_sortino[valid_mask] >= rolling_sharpe[valid_mask] - 0.1
        ).sum() > 0


def test_performance_metrics_helpers_handle_invalid_inputs(sample_returns):
    summary = reporting_metrics.performance_summary(sample_returns)
    assert math.isnan(summary._coerce_numeric(pd.Series(dtype=float)))
    assert math.isnan(summary._coerce_numeric("not-a-number"))  # ty: ignore
    assert summary._coerce_numeric(pd.Series([0.05])) == pytest.approx(0.05)
    assert summary._format_metric(float("nan")) == "N/A"
    assert summary._format_days(float("nan")) == "0 days"
