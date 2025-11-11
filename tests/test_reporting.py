from pathlib import Path

import pandas as pd

from quantalytics.reporting import tearsheet


def _sample_series():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    values = pd.Series([0.01 * (1 + i / 100) for i in range(len(dates))], index=dates)
    return values


def test_heatmap_matrix_creates_year_rows():
    series = _sample_series()
    months, years, matrix = tearsheet._heatmap_matrix(series, years=2)
    assert len(months) == 12
    assert 1 <= len(years) <= 2
    assert all(len(row) == 12 for row in matrix)


def test_win_rate_and_rolling_helpers():
    series = _sample_series()
    assert tearsheet._win_rate(series, "D") >= 0
    window = min(len(series), 63)
    ann = 252
    assert len(tearsheet._rolling_sharpe(series, window, ann)) == len(series)
    assert len(tearsheet._rolling_sortino(series, window, ann)) == len(series)
    assert len(tearsheet._rolling_volatility(series, window, ann)) == len(series)


def test_html_output_writes_file(tmp_path: Path):
    series = pd.Series(
        [0.01, -0.005, 0.02], index=pd.date_range("2020-01-01", periods=3, freq="B")
    )
    target = tmp_path / "report.html"
    report = tearsheet.html(
        series, title="Test Report", output=target, subtitle="Custom subtitle"
    )
    assert "Test Report" in report.html
    assert target.exists()
    assert "Data coverage" in report.html


def test_html_handles_log_returns(tmp_path: Path):
    series = pd.Series(
        [0.01, 0.02, 0.03], index=pd.date_range("2020-01-01", periods=3, freq="B")
    )
    report = tearsheet.html(series, log_returns=True)
    assert "Quantalytics" in report.html
