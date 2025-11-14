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
    """Test CAGR with simple compounding using periods parameter."""
    returns = pd.Series([0.05, 0.05])  # No DatetimeIndex
    # With 2 data points and periods=2, years = 2/2 = 1 year
    # Total return = 1.05 * 1.05 = 1.1025
    # CAGR = 1.1025^(1/1) - 1 = 0.1025
    expected = (1 + 0.05) * (1 + 0.05) - 1  # = 0.1025
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


def test_drawdown_details_basic():
    """Test basic drawdown_details functionality with simple drawdown pattern."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Create a simple drawdown: 0, -0.05, -0.10, -0.08, 0, 0, -0.03, -0.05, -0.02, 0
    drawdown = pd.Series(
        [0.0, -0.05, -0.10, -0.08, 0.0, 0.0, -0.03, -0.05, -0.02, 0.0], index=dates
    )

    result = stats.drawdown_details(drawdown)

    # Should identify 2 drawdown periods
    assert len(result) == 2
    assert list(result.columns) == [
        "start",
        "valley",
        "end",
        "days",
        "max drawdown",
        "99% max drawdown",
    ]

    # First drawdown: index 1-3 (Jan 2-4), recovers on index 4 (Jan 5)
    assert result.iloc[0]["days"] == 3
    assert result.iloc[0]["max drawdown"] == pytest.approx(10.0)  # 10% drawdown

    # Second drawdown: index 6-8 (Jan 7-9), recovers on index 9 (Jan 10)
    assert result.iloc[1]["days"] == 3
    assert result.iloc[1]["max drawdown"] == pytest.approx(5.0)  # 5% drawdown


def test_drawdown_details_no_drawdowns():
    """Test drawdown_details with no drawdowns (all zeros)."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates)

    result = stats.drawdown_details(drawdown)

    assert len(result) == 0
    assert list(result.columns) == [
        "start",
        "valley",
        "end",
        "days",
        "max drawdown",
        "99% max drawdown",
    ]


def test_drawdown_details_starts_in_drawdown():
    """Test drawdown_details when series starts in drawdown."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    # Starts at -0.05 (in drawdown), gets worse, then recovers
    drawdown = pd.Series([-0.05, -0.08, -0.10, -0.05, 0.0], index=dates)

    result = stats.drawdown_details(drawdown)

    assert len(result) == 1
    # Should use first date as start
    assert result.iloc[0]["start"] == dates[0].strftime("%Y-%m-%d")
    # Days in drawdown: Jan 1-4 (4 days), recovers on Jan 5
    assert result.iloc[0]["days"] == 4
    assert result.iloc[0]["max drawdown"] == pytest.approx(10.0)


def test_drawdown_details_ends_in_drawdown():
    """Test drawdown_details when series ends in drawdown."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    # Starts at 0, enters drawdown, never recovers
    drawdown = pd.Series([0.0, -0.02, -0.05, -0.08, -0.10], index=dates)

    result = stats.drawdown_details(drawdown)

    assert len(result) == 1
    # Should use last date as end
    assert result.iloc[0]["end"] == dates[-1].strftime("%Y-%m-%d")
    assert result.iloc[0]["days"] == 4
    assert result.iloc[0]["max drawdown"] == pytest.approx(10.0)


def test_drawdown_details_dataframe_input():
    """Test drawdown_details with DataFrame input (multiple columns)."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "strategy_a": [0.0, -0.05, -0.10, 0.0, -0.03, 0.0],
            "strategy_b": [0.0, -0.02, -0.04, -0.03, 0.0, 0.0],
        },
        index=dates,
    )

    result = stats.drawdown_details(df)

    # Should have multi-level columns
    assert isinstance(result.columns, pd.MultiIndex)
    assert "strategy_a" in result.columns.get_level_values(0)
    assert "strategy_b" in result.columns.get_level_values(0)

    # Each strategy should have 2 drawdown periods
    assert len(result) == 2


def test_drawdown_details_valley_identification():
    """Test that valley (max drawdown point) is correctly identified."""
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    # Drawdown that gets progressively worse then recovers
    drawdown = pd.Series([0.0, -0.02, -0.05, -0.10, -0.08, -0.03, 0.0], index=dates)

    result = stats.drawdown_details(drawdown)

    assert len(result) == 1
    # Valley should be at index 3 (where -0.10 occurs)
    assert result.iloc[0]["valley"] == dates[3].strftime("%Y-%m-%d")


def test_drawdown_details_percentages_positive():
    """Test that drawdown percentages are returned as positive values."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    drawdown = pd.Series([0.0, -0.15, -0.20, -0.10, 0.0], index=dates)

    result = stats.drawdown_details(drawdown)

    # All drawdown values should be positive (easier interpretation)
    assert result.iloc[0]["max drawdown"] > 0
    assert result.iloc[0]["99% max drawdown"] > 0
    assert result.iloc[0]["max drawdown"] == pytest.approx(20.0)  # 20%


def test_drawdown_details_99th_percentile():
    """Test that 99% max drawdown excludes outliers."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    # Create drawdown with mostly small values and one outlier
    drawdown_values = [-0.01] * 98 + [-0.50, 0.0]  # One extreme outlier
    drawdown_values[0] = 0.0  # Start at no drawdown
    drawdown = pd.Series(drawdown_values, index=dates)

    result = stats.drawdown_details(drawdown)

    assert len(result) == 1
    # Max drawdown should include outlier
    assert result.iloc[0]["max drawdown"] == pytest.approx(50.0)
    # 99% max drawdown should exclude it and be much smaller
    assert result.iloc[0]["99% max drawdown"] < result.iloc[0]["max drawdown"]
    assert result.iloc[0]["99% max drawdown"] == pytest.approx(1.0, rel=0.5)
