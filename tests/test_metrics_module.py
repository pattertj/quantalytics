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
