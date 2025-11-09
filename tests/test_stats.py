import pandas as pd
import pytest

from quantalytics.stats import cagr, kurtosis, skewness, total_return


def test_skewness_of_symmetric_series_is_zero():
    returns = pd.Series([-0.01, 0.0, 0.01])
    assert skewness(returns) == pytest.approx(0.0, abs=1e-12)


def test_kurtosis_matches_pandas():
    returns = pd.Series([0.02, -0.01, 0.015, 0.005, 0.01])
    expected = returns.kurtosis()
    assert kurtosis(returns) == pytest.approx(expected)


def test_total_return_compounds_returns():
    returns = pd.Series([0.01, -0.02, 0.015])
    expected = (1.01 * 0.98 * 1.015) - 1
    assert total_return(returns) == pytest.approx(expected)


def test_cagr_matches_manual_computation():
    # 252 trading days of 10 bps per day should annualize cleanly.
    returns = pd.Series([0.001] * 252)
    expected = (1.001**252) - 1
    assert cagr(returns, periods_per_year=252) == pytest.approx(expected)
