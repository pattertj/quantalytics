import numpy as np
import pandas as pd
import pytest

from quantalytics.analytics import metrics, stats
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
