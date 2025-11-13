import math
import numpy as np
import pandas as pd
import pytest

from quantalytics.analytics import stats


@pytest.fixture
def sample_returns():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.Series([0.01, -0.005, 0.02, -0.01, 0.005], index=dates)


def test_compsum_and_comp(sample_returns):
    cum = stats.compsum(sample_returns)
    assert cum.iloc[-1] == pytest.approx(stats.comp(sample_returns))
    assert cum.iloc[0] == pytest.approx(0.01)


def test_expected_return_and_geometric_alias(sample_returns):
    value = stats.expected_return(sample_returns)
    assert value == pytest.approx(stats.geometric_mean(sample_returns))


def test_distribution_returns_buckets(sample_returns):
    result = stats.distribution(sample_returns, compounded=False)
    assert all(period in result for period in ("Daily", "Weekly", "Monthly"))
    assert isinstance(result["Daily"]["values"], list)


def test_best_and_worst_returns_ignore_nan():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.01, np.nan, -0.02], index=dates)
    best = stats.best(returns, aggregate="day", compounded=False)
    worst = stats.worst(returns, aggregate="day", compounded=False)
    assert best == pytest.approx(0.01)
    assert worst == pytest.approx(-0.02)


def test_consecutive_runs(sample_returns):
    wins = stats.consecutive_wins(sample_returns)
    losses = stats.consecutive_losses(sample_returns)
    assert wins >= 1
    assert losses >= 1


def test_exposure_series_and_dataframe(sample_returns):
    exposure = stats.exposure(sample_returns)
    assert 0.0 <= exposure <= 1.0
    df = pd.DataFrame({"a": sample_returns, "b": sample_returns})
    exposures = stats.exposure(df)
    assert exposures["a"] == pytest.approx(exposures["b"])


def test_win_rate_and_avg_returns(sample_returns):
    series = sample_returns.copy()
    rate = stats.win_rate(series, prepare_returns=False)
    assert 0 <= rate <= 1
    avg_ret = stats.avg_return(series, prepare_returns=False)
    avg_win = stats.avg_win(series, prepare_returns=False)
    avg_loss = stats.avg_loss(series, prepare_returns=False)
    assert avg_win >= 0
    assert avg_loss <= 0
    assert avg_ret == pytest.approx(stats.comp(series) / len(series), rel=1e-2)


def test_volatility_and_rolling(sample_returns):
    vol = stats.volatility(sample_returns, periods=252, prepare_returns=False)
    assert vol == pytest.approx(sample_returns.std() * np.sqrt(252))
    rolling = stats.rolling_volatility(
        sample_returns, rolling_period=3, periods=252, prepare_returns=False
    )
    assert len(rolling) == len(sample_returns)
    assert rolling.isna().any()


def test_max_drawdown(sample_returns):
    value = stats.max_drawdown(sample_returns)
    cum = (1 + sample_returns).cumprod()
    running_max = cum.cummax()
    expected = float((cum / running_max - 1).min())
    assert value == pytest.approx(expected)


def test_implied_volatility(sample_returns):
    imp = stats.implied_volatility(sample_returns, annualize=False)
    assert imp >= 0


def test_max_drawdown_with_price_series():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, 105.0, 101.0, 110.0, 104.0], index=dates)
    value = stats.max_drawdown(prices)
    daily_returns = prices.pct_change().fillna(0)
    cum = (1 + daily_returns).cumprod()
    expected = float((cum / cum.cummax() - 1).min())
    assert value == pytest.approx(expected)


def test_pct_rank_edges():
    outidx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.Series([3.0, 1.0, 2.0], index=outidx)
    ranks = stats.pct_rank(prices, window=3)
    assert pytest.approx(100.0) == ranks.iloc[0]
    assert 0.0 <= ranks.min() <= ranks.max() <= 100.0


def test_ghpr_alias(sample_returns):
    assert stats.ghpr(sample_returns) == pytest.approx(
        stats.expected_return(sample_returns)
    )


def test_outliers_and_remove_outliers():
    returns = pd.Series(
        [0.01, 0.02, 0.03, 0.5], index=pd.date_range("2024-01-01", periods=4, freq="D")
    )
    high = stats.outliers(returns, quantile=0.9)
    cleaned = stats.remove_outliers(returns, quantile=0.9)
    assert high.iloc[0] == pytest.approx(0.5)
    assert 0.5 not in cleaned.tolist()


def test_avg_returns(sample_returns):
    expected_avg = sample_returns.mean()
    expected_win = sample_returns[sample_returns > 0].mean()
    expected_loss = sample_returns[sample_returns < 0].mean()

    assert stats.avg_return(sample_returns) == pytest.approx(expected_avg)
    assert stats.avg_win(sample_returns) == pytest.approx(expected_win)
    assert stats.avg_loss(sample_returns) == pytest.approx(expected_loss)


def test_rolling_volatility_prefix_nans(sample_returns):
    rolling = stats.rolling_volatility(
        sample_returns, rolling_period=3, prepare_returns=False
    )
    assert len(rolling) == len(sample_returns)
    assert rolling.iloc[:2].isna().all()


def test_cagr_simple_case():
    returns = pd.Series(
        [0.05, 0.05], index=pd.date_range("2024-01-01", periods=2, freq="D")
    )
    expected = (1 + 0.05) ** 2 - 1
    assert stats.cagr(returns, periods=2) == pytest.approx(expected)


def test_distribution_with_dataframe(sample_returns):
    df = pd.DataFrame(
        {
            "open": [0.01, -0.005, 0.02],
            "close": [0.01, -0.002, 0.01],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    with pytest.warns(UserWarning):
        result = stats.distribution(df, compounded=True)
    assert "Daily" in result
    assert isinstance(result["Daily"]["values"], list)


def test_expected_return_dataframe(sample_returns):
    df = pd.DataFrame(
        {
            "a": sample_returns,
            "b": sample_returns * 1.1,
        }
    )
    result = stats.expected_return(df)
    assert isinstance(result, pd.Series)


def test_win_rate_dataframe_and_zero_series(sample_returns):
    df = pd.DataFrame(
        {
            "a": sample_returns,
            "b": sample_returns * -1,
        }
    )
    result = stats.win_rate(df, aggregate="month")
    assert isinstance(result, pd.Series)
    assert stats.win_rate(pd.Series([0.0, 0.0])) == 0.0


def test_avg_metrics_dataframe(sample_returns):
    df = pd.DataFrame(
        {
            "a": sample_returns,
            "b": sample_returns * 2,
        }
    )
    assert isinstance(stats.avg_return(df), pd.Series)
    assert isinstance(stats.avg_win(df), pd.Series)
    assert isinstance(stats.avg_loss(df), pd.Series)


def test_volatility_and_rolling_dataframe(sample_returns):
    df = pd.DataFrame(
        {
            "a": sample_returns,
            "b": sample_returns * 1.5,
        }
    )
    vol = stats.volatility(df, periods=252)
    rolling = stats.rolling_volatility(df, rolling_period=2, prepare_returns=False)
    assert isinstance(vol, pd.Series)
    assert isinstance(rolling, pd.DataFrame)


def test_implied_volatility_non_annualized(sample_returns):
    result = stats.implied_volatility(sample_returns, annualize=False)
    assert isinstance(result, float)


def test_max_drawdown_dataframe(sample_returns):
    df = pd.DataFrame(
        {
            "a": sample_returns,
            "b": sample_returns * 0.5,
        }
    )
    result = stats.max_drawdown(df)
    assert isinstance(result, pd.Series)


def test_cagr_dataframe_and_edge_cases(sample_returns):
    df = pd.DataFrame(
        {
            "a": sample_returns,
            "b": sample_returns * 1.2,
        }
    )
    result = stats.cagr(df, periods=252)
    assert isinstance(result, pd.Series)
    empty = pd.Series([], dtype=float)
    assert math.isnan(stats.cagr(empty, periods=1))
    assert math.isnan(stats.cagr(pd.Series([-1.1, 0.05]), periods=2))
    with pytest.raises(ValueError):
        stats.cagr(pd.Series([0.01, 0.02]))
