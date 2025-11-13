import pandas as pd
import pytest

from quantalytics.reporting import tearsheet


def sample_returns(periods=60):
    """Deterministic series used in multiple tearsheet helpers."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    return pd.Series([0.01 * (1 + i / 100) for i in range(periods)], index=dates)


def test_heatmap_matrix_creates_year_rows():
    series = pd.Series(
        [0.01] * 12, index=pd.date_range("2020-01-31", periods=12, freq="M")
    )
    months, years, matrix = tearsheet._heatmap_matrix(series, years=1)
    assert months == [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    assert len(years) == 1
    assert len(matrix) == 1
    assert all(len(row) == 12 for row in matrix)


def test_win_rate_and_rolling_helpers():
    series = sample_returns(periods=20)
    assert 0.0 <= tearsheet._win_rate(series, "D") <= 100.0
    monthly_win = tearsheet._win_rate(series, "M")
    assert monthly_win == pytest.approx(100.0)
    window = min(len(series), 63)
    ann = 252
    assert len(tearsheet._rolling_sharpe(series, window, ann)) == len(series)
    assert len(tearsheet._rolling_sortino(series, window, ann)) == len(series)
    assert len(tearsheet._rolling_volatility(series, window, ann)) == len(series)


def test_format_and_nan_helpers():
    assert tearsheet._format_percent(0.1234, decimals=1) == "12.3%"
    sequence: list[float | None] = [0.1, None, float("nan"), 0.2]
    assert tearsheet._nan_safe(sequence) == [0.1, None, None, 0.2]


def test_rolling_helpers_emit_nones():
    series = pd.Series([0.01, 0.02, -0.005, 0.01, 0.0])
    window = 3
    ann = 252
    sharpe = tearsheet._rolling_sharpe(series, window, ann)
    sortino = tearsheet._rolling_sortino(series, window, ann)
    vol = tearsheet._rolling_volatility(series, window, ann)
    assert len(sharpe) == len(series)
    assert sharpe[:2] == [None, None]
    assert all(value is None or isinstance(value, float) for value in sortino)
    assert len(vol) == len(series)


def test_html_render_defaults():
    returns = sample_returns(periods=20)
    report = tearsheet.html(
        returns,
        title="Custom Tearsheet",
        subtitle="Metrics for testing",
        parameters={"Name": "testbot"},
    )
    assert isinstance(report.html, str)
    assert "Custom Tearsheet" in report.html
    assert "Data coverage" in report.html
    assert "testbot" in report.html


def test_html_validations():
    with pytest.raises(ValueError):
        tearsheet.html([])
    with pytest.raises(TypeError):
        tearsheet.html(returns=["a", "b", "c"])  # ty: ignore[invalid-argument-type, unresolved-attribute]


def test_render_basic_tearsheet_with_config():
    series = sample_returns(periods=15)
    config = tearsheet.TearsheetConfig(
        title="Config Title",
        subtitle="Custom Subtitle",
        sections=[
            tearsheet.TearsheetSection(title="Custom", description="desc"),
        ],
    )
    sheet = tearsheet.render_basic_tearsheet(series, config=config)
    assert sheet.html.startswith("<!DOCTYPE html>")


def test_period_return_and_yearly_table():
    series = sample_returns(periods=12)
    last = series.index[-1]
    assert tearsheet._period_return(series, last + pd.DateOffset(days=1)) == 0.0
    assert tearsheet._period_return(series, series.index[0]) == pytest.approx(
        (1 + series).prod() - 1
    )
    table = tearsheet._yearly_table(series)
    assert all(len(row) == 3 for row in table)


def test_drawdown_segments_and_heatmap():
    series = pd.Series(
        [0.01, -0.02, 0.015, -0.01, 0.03, -0.005],
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )
    drawdown, segments = tearsheet._drawdown_segments(series)
    assert isinstance(drawdown, pd.Series)
    assert isinstance(segments, list)
    months, years, matrix = tearsheet._heatmap_matrix(series, years=1)
    assert len(months) == 12
    assert len(matrix) == len(years)


def test_ensure_datetime_index_and_to_html(tmp_path):
    series = pd.Series([0.01, 0.02])
    result = tearsheet._ensure_datetime_index(series)
    assert isinstance(result.index, pd.DatetimeIndex)
    sheet = tearsheet.Tearsheet(html="<p>ok</p>")
    target = tmp_path / "temp.html"
    sheet.to_html(target)
    assert target.exists()


# TODO: Put back
# def test_html_output_writes_file(tmp_path: Path):
#     series = pd.Series(
#         [0.01, -0.005, 0.02], index=pd.date_range("2020-01-01", periods=3, freq="B")
#     )
#     target = tmp_path / "report.html"
#     report = tearsheet.html(
#         series, title="Test Report", output=target, subtitle="Custom subtitle"
#     )
#     assert "Test Report" in report.html
#     assert target.exists()
#     assert "Data coverage" in report.html

# TODO: Put back
# def test_html_handles_log_returns(tmp_path: Path):
#     series = pd.Series(
#         [0.01, 0.02, 0.03], index=pd.date_range("2020-01-01", periods=3, freq="B")
#     )
#     report = tearsheet.html(series, log_returns=True)
#     assert "Quantalytics" in report.html
