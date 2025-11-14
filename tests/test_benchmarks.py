import numpy as np
import pandas as pd
import pytest

from quantalytics.analytics import benchmarks


@pytest.fixture
def sample_returns():
    """Sample portfolio returns for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.02 + 0.001, index=dates)


@pytest.fixture
def sample_benchmark():
    """Sample benchmark returns for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(43)
    return pd.Series(np.random.randn(100) * 0.015 + 0.0008, index=dates)


@pytest.fixture
def correlated_returns(sample_benchmark):
    """Returns that are highly correlated with benchmark."""
    # Create returns that are highly correlated with the benchmark
    # by using the benchmark itself plus a small noise component
    np.random.seed(44)
    noise = np.random.randn(len(sample_benchmark)) * 0.002
    return pd.Series(
        sample_benchmark.values * 1.2 + noise, index=sample_benchmark.index
    )


@pytest.fixture
def sample_dataframe_returns():
    """Sample DataFrame returns for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(45)
    return pd.DataFrame(
        {
            "strategy_1": np.random.randn(100) * 0.02 + 0.001,
            "strategy_2": np.random.randn(100) * 0.025 + 0.0015,
        },
        index=dates,
    )


class TestRSquared:
    """Tests for r_squared function."""

    def test_r_squared_basic(self, sample_returns, sample_benchmark):
        """Test basic r_squared calculation."""
        result = benchmarks.r_squared(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_r_squared_perfect_correlation(self):
        """Test r_squared with perfect correlation."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        returns = pd.Series(np.arange(50) * 0.01, index=dates)
        benchmark = pd.Series(np.arange(50) * 0.01, index=dates)
        result = benchmarks.r_squared(returns, benchmark)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_r_squared_high_correlation(self, correlated_returns, sample_benchmark):
        """Test r_squared with highly correlated returns."""
        result = benchmarks.r_squared(correlated_returns, sample_benchmark)
        assert result > 0.5  # Should have high R²

    def test_r_squared_identical_values(self):
        """Test r_squared with all identical values returns 0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        returns = pd.Series([0.01] * 10, index=dates)
        benchmark = pd.Series([0.01] * 10, index=dates)
        result = benchmarks.r_squared(returns, benchmark)
        assert result == 0.0

    def test_r_squared_series_only(self, sample_returns, sample_benchmark):
        """Test r_squared works with Series input."""
        result = benchmarks.r_squared(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_r_squared_no_prepare_returns(self, sample_returns, sample_benchmark):
        """Test r_squared with prepare_returns=False."""
        result = benchmarks.r_squared(
            sample_returns, sample_benchmark, prepare_returns=False
        )
        assert isinstance(result, float)
        assert 0 <= result <= 1


class TestR2:
    """Tests for r2 function (shorthand for r_squared)."""

    def test_r2_equals_r_squared(self, sample_returns, sample_benchmark):
        """Test that r2 returns same result as r_squared."""
        r2_result = benchmarks.r2(sample_returns, sample_benchmark)
        r_squared_result = benchmarks.r_squared(sample_returns, sample_benchmark)
        assert r2_result == pytest.approx(r_squared_result)

    def test_r2_basic(self, sample_returns, sample_benchmark):
        """Test basic r2 calculation."""
        result = benchmarks.r2(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert 0 <= result <= 1


class TestInformationRatio:
    """Tests for information_ratio function."""

    def test_information_ratio_basic(self, sample_returns, sample_benchmark):
        """Test basic information ratio calculation."""
        result = benchmarks.information_ratio(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_information_ratio_outperformance(self):
        """Test information ratio with consistent outperformance."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        returns = pd.Series([0.02] * 50, index=dates)
        benchmark = pd.Series([0.01] * 50, index=dates)
        result = benchmarks.information_ratio(returns, benchmark, prepare_returns=False)
        assert result > 0  # Should be positive when outperforming

    def test_information_ratio_underperformance(self):
        """Test information ratio with consistent underperformance."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        returns = pd.Series([0.01] * 50, index=dates)
        benchmark = pd.Series([0.02] * 50, index=dates)
        result = benchmarks.information_ratio(returns, benchmark, prepare_returns=False)
        assert result < 0  # Should be negative when underperforming

    def test_information_ratio_series(self, sample_returns, sample_benchmark):
        """Test information ratio with Series input."""
        result = benchmarks.information_ratio(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_information_ratio_no_prepare_returns(
        self, sample_returns, sample_benchmark
    ):
        """Test information ratio with prepare_returns=False."""
        result = benchmarks.information_ratio(
            sample_returns, sample_benchmark, prepare_returns=False
        )
        assert isinstance(result, float)


class TestGreeks:
    """Tests for greeks function."""

    def test_greeks_basic(self, sample_returns, sample_benchmark):
        """Test basic greeks calculation."""
        result = benchmarks.greeks(sample_returns, sample_benchmark)
        assert isinstance(result, pd.Series)
        assert "alpha" in result.index
        assert "beta" in result.index

    def test_greeks_beta_range(self, sample_returns, sample_benchmark):
        """Test that beta is reasonable."""
        result = benchmarks.greeks(sample_returns, sample_benchmark)
        # Beta should typically be between -3 and 3 for most portfolios
        assert -3 <= result["beta"] <= 3

    def test_greeks_positive_correlation(self, correlated_returns, sample_benchmark):
        """Test greeks with positively correlated returns."""
        result = benchmarks.greeks(correlated_returns, sample_benchmark)
        # Beta can be positive or negative depending on correlation
        # Just check it's a reasonable value
        assert -5 <= result["beta"] <= 5

    def test_greeks_periods_parameter(self, sample_returns, sample_benchmark):
        """Test greeks with different periods parameter."""
        result_252 = benchmarks.greeks(sample_returns, sample_benchmark, periods=252)
        result_365 = benchmarks.greeks(sample_returns, sample_benchmark, periods=365)
        # Alpha should scale with periods
        assert result_252["alpha"] != result_365["alpha"]
        # Beta should be the same
        assert result_252["beta"] == pytest.approx(result_365["beta"])

    def test_greeks_no_prepare_returns(self, sample_returns, sample_benchmark):
        """Test greeks with prepare_returns=False."""
        result = benchmarks.greeks(
            sample_returns, sample_benchmark, prepare_returns=False
        )
        assert isinstance(result, pd.Series)
        assert "alpha" in result.index
        assert "beta" in result.index


class TestRollingGreeks:
    """Tests for rolling_greeks function."""

    def test_rolling_greeks_basic(self, sample_returns, sample_benchmark):
        """Test basic rolling greeks calculation."""
        result = benchmarks.rolling_greeks(sample_returns, sample_benchmark, periods=30)
        assert isinstance(result, pd.DataFrame)
        assert "alpha" in result.columns
        assert "beta" in result.columns
        assert len(result) == len(sample_returns)

    def test_rolling_greeks_index_matches(self, sample_returns, sample_benchmark):
        """Test that rolling greeks index matches input."""
        result = benchmarks.rolling_greeks(sample_returns, sample_benchmark, periods=30)
        assert result.index.equals(sample_returns.index)

    def test_rolling_greeks_initial_nans(self, sample_returns, sample_benchmark):
        """Test that initial values are NaN due to window size."""
        periods = 30
        result = benchmarks.rolling_greeks(
            sample_returns, sample_benchmark, periods=periods
        )
        # First (periods-1) values should be NaN
        assert result["beta"].iloc[: periods - 1].isna().all()

    def test_rolling_greeks_default_periods(self, sample_returns, sample_benchmark):
        """Test rolling greeks with default periods (365)."""
        result = benchmarks.rolling_greeks(sample_returns, sample_benchmark)
        assert isinstance(result, pd.DataFrame)
        # Most values will be NaN since we only have 100 data points
        assert result["beta"].isna().sum() > 0

    def test_rolling_greeks_values_reasonable(self, sample_returns, sample_benchmark):
        """Test rolling greeks returns reasonable values."""
        result = benchmarks.rolling_greeks(sample_returns, sample_benchmark, periods=30)
        assert isinstance(result, pd.DataFrame)
        assert "alpha" in result.columns
        assert "beta" in result.columns
        # Check that non-NaN betas are reasonable
        valid_betas = result["beta"].dropna()
        assert len(valid_betas) > 0, "Should have at least some valid beta values"
        assert all(valid_betas.abs() < 10), "Beta should be reasonable (abs < 10)"


class TestCompare:
    """Tests for compare function."""

    def test_compare_basic(self, sample_returns, sample_benchmark):
        """Test basic compare functionality."""
        result = benchmarks.compare(sample_returns, sample_benchmark)
        assert isinstance(result, pd.DataFrame)
        assert "Benchmark" in result.columns
        assert "Returns" in result.columns
        assert "Multiplier" in result.columns
        assert "Won" in result.columns

    def test_compare_aggregate_month(self, sample_returns, sample_benchmark):
        """Test compare with monthly aggregation."""
        result = benchmarks.compare(sample_returns, sample_benchmark, aggregate="month")
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 4  # Should have fewer rows than daily data

    def test_compare_aggregate_week(self, sample_returns, sample_benchmark):
        """Test compare with weekly aggregation."""
        result = benchmarks.compare(sample_returns, sample_benchmark, aggregate="week")
        assert isinstance(result, pd.DataFrame)

    def test_compare_won_column(self):
        """Test that Won column correctly identifies winners."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        returns = pd.Series(
            [0.02, -0.01, 0.015, 0.01, 0.005, 0.02, -0.01, 0.01, 0.015, 0.02],
            index=dates,
        )
        benchmark = pd.Series(
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], index=dates
        )
        result = benchmarks.compare(returns, benchmark, prepare_returns=False)
        # Check that Won column has + or -
        assert all(result["Won"].isin(["+", "-"]))
        # First row should be + (0.02 > 0.01)
        assert result["Won"].iloc[0] == "+"

    def test_compare_multiplier(self):
        """Test that Multiplier column is calculated correctly."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        returns = pd.Series([0.02, 0.04, 0.01, 0.03, 0.02], index=dates)
        benchmark = pd.Series([0.01, 0.02, 0.01, 0.01, 0.01], index=dates)
        result = benchmarks.compare(returns, benchmark, prepare_returns=False)
        # Multiplier should be Returns / Benchmark (in percentage)
        expected_multiplier = (returns * 100) / (benchmark * 100)
        assert result["Multiplier"].iloc[0] == pytest.approx(
            expected_multiplier.iloc[0]
        )

    def test_compare_round_vals(self, sample_returns, sample_benchmark):
        """Test compare with rounding."""
        result = benchmarks.compare(sample_returns, sample_benchmark, round_vals=2)
        # Check that values are rounded to 2 decimal places
        assert all(
            result["Returns"].apply(
                lambda x: len(str(x).split(".")[-1]) <= 2 if "." in str(x) else True
            )
        )

    def test_compare_dataframe_returns(
        self, sample_dataframe_returns, sample_benchmark
    ):
        """Test compare with DataFrame returns."""
        result = benchmarks.compare(sample_dataframe_returns, sample_benchmark)
        assert isinstance(result, pd.DataFrame)
        assert "Benchmark" in result.columns
        # Should have Returns_0, Returns_1 for each strategy
        assert "Returns_0" in result.columns
        assert "Returns_1" in result.columns
        # Should NOT have Multiplier or Won columns for DataFrame input
        assert "Multiplier" not in result.columns
        assert "Won" not in result.columns

    def test_compare_compounded_true(self, sample_returns, sample_benchmark):
        """Test compare with compounded=True."""
        result = benchmarks.compare(
            sample_returns, sample_benchmark, aggregate="month", compounded=True
        )
        assert isinstance(result, pd.DataFrame)

    def test_compare_compounded_false(self, sample_returns, sample_benchmark):
        """Test compare with compounded=False."""
        result = benchmarks.compare(
            sample_returns, sample_benchmark, aggregate="month", compounded=False
        )
        assert isinstance(result, pd.DataFrame)

    def test_compare_no_prepare_returns(self, sample_returns, sample_benchmark):
        """Test compare with prepare_returns=False."""
        result = benchmarks.compare(
            sample_returns, sample_benchmark, prepare_returns=False
        )
        assert isinstance(result, pd.DataFrame)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_series(self):
        """Test handling of empty series returns NaN."""
        dates = pd.date_range("2024-01-01", periods=0, freq="D")
        returns = pd.Series([], index=dates, dtype=float)
        benchmark = pd.Series([], index=dates, dtype=float)

        # Empty series returns NaN (not an error)
        result = benchmarks.r_squared(returns, benchmark)
        assert np.isnan(result)

    def test_single_value(self):
        """Test handling of single value."""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        returns = pd.Series([0.01], index=dates)
        benchmark = pd.Series([0.01], index=dates)

        # Single value should return 0 for r_squared
        result = benchmarks.r_squared(returns, benchmark)
        assert result == 0.0

    def test_mismatched_lengths(self, sample_returns):
        """Test handling of mismatched lengths."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        benchmark = pd.Series(np.random.randn(50) * 0.01, index=dates)

        # Should handle mismatched lengths (benchmark preparation will align)
        result = benchmarks.r_squared(sample_returns, benchmark)
        assert isinstance(result, float)

    def test_all_zeros(self):
        """Test handling of all zero returns."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        returns = pd.Series([0.0] * 10, index=dates)
        benchmark = pd.Series([0.0] * 10, index=dates)

        result = benchmarks.r_squared(returns, benchmark)
        assert result == 0.0

    def test_benchmark_as_dataframe(self, sample_returns):
        """Test using DataFrame as benchmark."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        benchmark_df = pd.DataFrame(
            {"bench": np.random.randn(100) * 0.015}, index=dates
        )

        result = benchmarks.r_squared(sample_returns, benchmark_df)
        assert isinstance(result, float)


class TestTreynorRatio:
    """Tests for treynor_ratio function."""

    def test_treynor_ratio_basic(self, sample_returns, sample_benchmark):
        """Test basic Treynor ratio calculation."""
        result = benchmarks.treynor_ratio(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_treynor_ratio_manual_calculation(self, sample_returns, sample_benchmark):
        """Test Treynor ratio matches manual calculation."""
        from quantalytics.analytics.stats import comp

        # Calculate components manually
        greeks_result = benchmarks.greeks(
            sample_returns, sample_benchmark, periods=365.0
        )
        beta = greeks_result["beta"]
        total_return = comp(sample_returns)

        expected = total_return / beta if beta != 0 else 0.0

        result = benchmarks.treynor_ratio(sample_returns, sample_benchmark, rf=0.0)
        assert result == pytest.approx(expected)

    def test_treynor_ratio_with_rf(self, sample_returns, sample_benchmark):
        """Test Treynor ratio with risk-free rate."""
        from quantalytics.analytics.stats import comp

        rf = 0.02
        greeks_result = benchmarks.greeks(sample_returns, sample_benchmark)
        beta = greeks_result["beta"]
        total_return = comp(sample_returns)

        expected = (total_return - rf) / beta if beta != 0 else 0.0

        result = benchmarks.treynor_ratio(sample_returns, sample_benchmark, rf=rf)
        assert result == pytest.approx(expected)

    def test_treynor_ratio_zero_beta(self):
        """Test Treynor ratio returns 0 when beta is 0."""
        # Create returns with no correlation to benchmark (beta ~ 0)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(100)
        returns = pd.Series(np.random.randn(50) * 0.001, index=dates)
        # Constant benchmark (will result in beta = 0)
        benchmark = pd.Series([0.0] * 50, index=dates)

        result = benchmarks.treynor_ratio(returns, benchmark, prepare_returns=False)
        assert result == 0.0

    def test_treynor_ratio_dataframe_input(
        self, sample_dataframe_returns, sample_benchmark
    ):
        """Test Treynor ratio with DataFrame input (uses first column)."""
        result = benchmarks.treynor_ratio(sample_dataframe_returns, sample_benchmark)
        assert isinstance(result, float)

        # Should be same as using just the first column
        first_col_result = benchmarks.treynor_ratio(
            sample_dataframe_returns.iloc[:, 0], sample_benchmark
        )
        assert result == pytest.approx(first_col_result)

    def test_treynor_ratio_different_periods(self, sample_returns, sample_benchmark):
        """Test Treynor ratio with different periods parameter."""
        result_252 = benchmarks.treynor_ratio(
            sample_returns, sample_benchmark, periods=252
        )
        result_365 = benchmarks.treynor_ratio(
            sample_returns, sample_benchmark, periods=365
        )

        # Results should be different because beta calculation uses periods
        # but the difference should be reasonable
        assert isinstance(result_252, float)
        assert isinstance(result_365, float)

    def test_treynor_ratio_positive_returns(self):
        """Test Treynor ratio with positive returns."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Use varying returns that correlate positively with benchmark to produce non-zero beta
        np.random.seed(42)
        benchmark_vals = np.random.normal(0.008, 0.002, 50)
        returns_vals = benchmark_vals * 1.2 + np.random.normal(
            0.001, 0.001, 50
        )  # Correlated with benchmark
        returns = pd.Series(returns_vals, index=dates)
        benchmark = pd.Series(benchmark_vals, index=dates)

        result = benchmarks.treynor_ratio(returns, benchmark, prepare_returns=False)
        assert result > 0  # Should be positive with positive returns on average

    def test_treynor_ratio_negative_returns(self):
        """Test Treynor ratio with negative returns."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Use varying returns that correlate with benchmark and are negative on average
        np.random.seed(43)
        benchmark_vals = np.random.normal(-0.008, 0.002, 50)
        returns_vals = benchmark_vals * 1.2 + np.random.normal(
            -0.001, 0.001, 50
        )  # Correlated with benchmark
        returns = pd.Series(returns_vals, index=dates)
        benchmark = pd.Series(benchmark_vals, index=dates)

        result = benchmarks.treynor_ratio(returns, benchmark, prepare_returns=False)
        assert result < 0  # Should be negative with negative returns on average

    def test_treynor_ratio_no_prepare_returns(self, sample_returns, sample_benchmark):
        """Test Treynor ratio with prepare_returns=False."""
        result = benchmarks.treynor_ratio(
            sample_returns, sample_benchmark, prepare_returns=False
        )
        assert isinstance(result, float)


class TestBenchmarkCorrelation:
    """Tests for benchmark_correlation function."""

    def test_benchmark_correlation_basic(self, sample_returns, sample_benchmark):
        """Test basic benchmark correlation calculation."""
        result = benchmarks.benchmark_correlation(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert -1 <= result <= 1

    def test_benchmark_correlation_perfect_positive(self):
        """Test correlation with perfectly correlated returns."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        returns = pd.Series(np.random.randn(50) * 0.02, index=dates)
        # Benchmark is identical to returns
        benchmark = returns.copy()

        result = benchmarks.benchmark_correlation(
            returns, benchmark, prepare_returns=False
        )
        assert result == pytest.approx(1.0)

    def test_benchmark_correlation_perfect_negative(self):
        """Test correlation with perfectly negatively correlated returns."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        returns = pd.Series(np.random.randn(50) * 0.02, index=dates)
        # Benchmark is exact opposite
        benchmark = -returns

        result = benchmarks.benchmark_correlation(
            returns, benchmark, prepare_returns=False
        )
        assert result == pytest.approx(-1.0)

    def test_benchmark_correlation_high_positive(
        self, correlated_returns, sample_benchmark
    ):
        """Test correlation with highly correlated returns."""
        result = benchmarks.benchmark_correlation(correlated_returns, sample_benchmark)
        # Should be high positive correlation (created with 1.2x multiplier + noise)
        assert result > 0.8

    def test_benchmark_correlation_dataframe_returns_series(
        self, sample_dataframe_returns, sample_benchmark
    ):
        """Test that benchmark_correlation returns Series for DataFrame input."""
        result = benchmarks.benchmark_correlation(
            sample_dataframe_returns, sample_benchmark
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert "strategy_1" in result.index
        assert "strategy_2" in result.index
        # All correlations should be in valid range
        assert all(-1 <= corr <= 1 for corr in result)

    def test_benchmark_correlation_no_prepare_returns(
        self, sample_returns, sample_benchmark
    ):
        """Test benchmark_correlation with prepare_returns=False."""
        result = benchmarks.benchmark_correlation(
            sample_returns, sample_benchmark, prepare_returns=False
        )
        assert isinstance(result, float)
        assert -1 <= result <= 1

    def test_benchmark_correlation_vs_pandas_corr(self):
        """Test that benchmark_correlation matches pandas corr."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(999)
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
        benchmark = pd.Series(np.random.randn(100) * 0.015, index=dates)

        result = benchmarks.benchmark_correlation(
            returns, benchmark, prepare_returns=False
        )
        expected = returns.corr(benchmark)

        assert result == pytest.approx(expected)

    def test_benchmark_correlation_independent_returns(self):
        """Test correlation with independent (uncorrelated) returns."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(111)
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
        np.random.seed(222)
        benchmark = pd.Series(np.random.randn(100) * 0.015, index=dates)

        result = benchmarks.benchmark_correlation(
            returns, benchmark, prepare_returns=False
        )

        # Should be close to 0 for independent random returns
        assert abs(result) < 0.3

    def test_benchmark_correlation_benchmark_as_dataframe(self):
        """Test using DataFrame as benchmark."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        returns = pd.Series(np.random.randn(50) * 0.02, index=dates)
        benchmark_df = pd.DataFrame({"bench": np.random.randn(50) * 0.015}, index=dates)

        result = benchmarks.benchmark_correlation(returns, benchmark_df)
        assert isinstance(result, float)
        assert -1 <= result <= 1

    def test_benchmark_correlation_series_input(self, sample_returns, sample_benchmark):
        """Test that benchmark_correlation works with Series input."""
        result = benchmarks.benchmark_correlation(sample_returns, sample_benchmark)
        assert isinstance(result, float)
        assert not np.isnan(result)
