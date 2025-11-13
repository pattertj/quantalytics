import pandas as pd
import pytest
from quantalytics.utils import timeseries as ts_utils


def test_ensure_datetime_index_preserves_dates():
    series = pd.Series([1, 2, 3], index=["2024-01-01", "2024-01-02", "2024-01-03"])
    indexed = ts_utils.ensure_datetime_index(series)
    assert isinstance(indexed.index, pd.DatetimeIndex)
    assert indexed.index.is_monotonic_increasing


def test_rolling_statistic_std():
    series = pd.Series([1, 2, 3, 4, 5])
    roll_std = ts_utils.rolling_statistic(series, window=3, function="std")
    assert roll_std.iloc[2] == pytest.approx(series[:3].std(ddof=1))


def test_infer_periods_datetime_index():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    years, periods = ts_utils.infer_periods(pd.Series([0.01, 0.02, 0.03], index=dates))
    assert years == pytest.approx(2 / 365, rel=1e-3)
    assert periods == pytest.approx(len(dates) / years, rel=1e-3)


def test_infer_periods_non_datetime():
    values = pd.Series([0.01, 0.02, 0.03])
    years, periods = ts_utils.infer_periods(values, fallback_periods=365)
    assert years == pytest.approx(len(values) / 365)
    assert periods == 365


def test_prepare_returns_detects_prices_and_applies_rf():
    prices = pd.Series([100, 102, 101, 103])
    returns = ts_utils.normalize_returns(prices, data_type="prices")
    assert returns.iloc[0] == pytest.approx(0.0)
    assert returns.iloc[1] == pytest.approx(0.02)

    excess = ts_utils.normalize_returns(
        prices, rf=0.01, data_type="prices", nperiods=252
    )
    assert not excess.equals(returns)


def test_prepare_returns_fill_none():
    data = pd.Series([1.0, None, 1.02])
    cleaned = ts_utils.normalize_returns(data, fill_method="none", data_type="returns")
    assert cleaned.isna().sum() == 1


def test_prepare_returns_invalid_fill_method():
    data = pd.Series([1.0])
    with pytest.raises(ValueError):
        ts_utils.normalize_returns(data, fill_method="invalid", data_type="returns")


def test_prepare_returns_with_list_like_input():
    values = [100, 102, 101, 103]
    result = ts_utils.normalize_returns(values)
    assert isinstance(result, pd.Series)
    assert result.iloc[1] == pytest.approx(0.02)


def test_prepare_returns_auto_detect_dataframe():
    prices = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = ts_utils.normalize_returns(prices, data_type="auto")
    assert "A" in result.columns


def test_prepare_returns_drop_method():
    data = pd.Series([1.0, None, 1.02])
    cleaned = ts_utils.normalize_returns(data, data_type="returns", fill_method="drop")
    assert cleaned.isna().sum() == 0


def test_prepare_returns_no_excess():
    prices = pd.Series([100, 102, 104])
    returns = ts_utils.normalize_returns(
        prices, data_type="prices", apply_excess_returns=False
    )
    assert returns.iloc[1] == pytest.approx(0.02)


def test_convert_to_returns_auto_switch():
    series = pd.Series([1, 2, 1, 2])
    converted = ts_utils._convert_to_returns(series, data_type="auto")
    assert converted.equals(series.pct_change())


def test_likely_prices_logic():
    series = pd.Series([100, 105, 110])
    assert ts_utils._is_likely_prices(series)
    returns = pd.Series([0.01, -0.01, 0.02])
    assert not ts_utils._is_likely_prices(returns)


def test_multi_shift_column_suffixes():
    series = pd.Series([1, 2, 3, 4])
    shifted = ts_utils.multi_shift(series, shift=3)
    assert [str(col) for col in shifted.columns.tolist()] == ["0", "01", "02"]
    assert shifted.shape[1] == 3


def test_count_consecutive():
    data = pd.Series([1, 1, 0, 1, 1, 1])
    counted = ts_utils._count_consecutive(data)
    assert counted.tolist() == [1, 2, 0, 1, 2, 3]


def test_group_returns_and_aggregate_periods():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    returns = pd.Series([0.01] * 10, index=idx)
    grouped = ts_utils.group_returns(returns, idx.to_period("M"))
    assert grouped.sum() == pytest.approx(0.1)
    monthly = ts_utils.aggregate_returns(returns, period="month")
    assert isinstance(monthly, pd.Series)


def test_aggregate_returns_invalid_index():
    data = pd.Series([0.01, 0.02], index=[1, 2])
    with pytest.raises(ValueError):
        ts_utils.aggregate_returns(data, period="month")


def test_ensure_datetime_index_non_datetime():
    series = pd.Series([1, 2, 3])
    result = ts_utils.ensure_datetime_index(series)
    assert isinstance(result.index, pd.DatetimeIndex)
