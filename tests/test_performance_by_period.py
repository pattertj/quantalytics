"""Test for Performance by Period table in tearsheet."""

import pandas as pd

from quantalytics.reports import html
from quantalytics.reports.metrics import format_consistency_rows, metrics


def sample_returns() -> pd.Series:
    """Create sample returns for testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    # Create some realistic returns
    import numpy as np

    np.random.seed(42)
    returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    return returns


def test_performance_by_period_table_populated():
    """Test that the Performance by Period table is populated with actual values, not 0.00%."""
    returns = sample_returns()

    # Generate tearsheet
    report = html(returns, title="Performance Test", risk_free_rate=0.02, periods=252)

    # Find the Performance by Period section
    perf_section_start = report.html.find("Performance by Period")
    assert perf_section_start != -1, "Performance by Period section not found"

    # Extract JavaScript section that contains the table data
    js_start = report.html.find("const performanceRows", perf_section_start - 5000)
    assert js_start != -1, "performanceRows JavaScript not found"

    js_section = report.html[js_start : js_start + 3000]

    # Check that the rows have actual values, not just 0.00%
    # The table should have: Best, Avg Up, Expected, Avg Down, Worst rows
    assert "Best Day" in js_section or "consistencyLookup[" in js_section

    # Count how many times we see "0.00%" which would indicate missing data
    zero_percent_count = js_section.count('"0.00%"')

    # We should not have ALL values as 0.00% - allow a few but not all 25 cells (5 rows x 5 cols)
    assert zero_percent_count < 20, (
        f"Too many 0.00% values in performance table: {zero_percent_count}"
    )

    # Verify specific rows are referenced
    assert 'consistencyLookup["Best Day"]' in js_section or "Best Day" in js_section
    assert 'consistencyLookup["Best Week"]' in js_section or "Best Week" in js_section
    assert 'consistencyLookup["Avg Up Day"]' in js_section or "Avg Up Day" in js_section
    assert (
        'consistencyLookup["Expected Daily%"]' in js_section
        or "Expected Daily" in js_section
    )


def test_format_consistency_rows_uses_new_metric_names():
    """Test that format_consistency_rows uses the new metric names from the metrics function."""
    returns = sample_returns()

    # Get metrics using the metrics function (which uses new names)
    metrics_series = metrics(returns, risk_free_rate=0.02, periods=252)

    # Format consistency rows
    consistency_rows = format_consistency_rows(metrics_series)

    # Convert to lookup dict for easier checking
    consistency_lookup = {row["label"]: row["value"] for row in consistency_rows}

    # These are the keys that the JavaScript template expects to find
    expected_keys = [
        "Best Day",
        "Best Week",
        "Best Month",
        "Best Quarter",
        "Best Year",
        "Avg Up Day",
        "Avg Up Week",
        "Avg Up Month",
        "Avg Up Quarter",
        "Avg Up Year",
        "Expected Daily%",
        "Expected Weekly%",
        "Expected Monthly%",
        "Expected Quarterly%",
        "Expected Yearly%",
        "Avg Down Day",
        "Avg Down Week",
        "Avg Down Month",
        "Avg Down Quarter",
        "Avg Down Year",
        "Worst Day",
        "Worst Week",
        "Worst Month",
        "Worst Quarter",
        "Worst Year",
    ]

    # Check that all expected keys are present
    for key in expected_keys:
        assert key in consistency_lookup, f"Missing key: {key}"
        # Values should not all be "0.00%" - at least some should have actual values

    # Check that values are not all "0.00%"
    non_zero_count = sum(1 for v in consistency_lookup.values() if v != "0.00%")
    assert non_zero_count >= 15, (
        f"Too many 0.00% values, only {non_zero_count} non-zero"
    )


def test_metrics_function_returns_new_names():
    """Test that the metrics function returns the new metric names with % suffix."""
    returns = sample_returns()

    # Get metrics
    metrics_series = metrics(returns, risk_free_rate=0.02, periods=252)
    metrics_dict = metrics_series.to_dict()

    # Check that new names exist (note: "Avg." has a period)
    assert "Best Day %" in metrics_dict, "Best Day % not found in metrics"
    assert "Best Week %" in metrics_dict, "Best Week % not found in metrics"
    assert "Avg. Up Day %" in metrics_dict, "Avg. Up Day % not found in metrics"
    assert "Expected Daily %" in metrics_dict, "Expected Daily % not found in metrics"
    assert "Worst Day %" in metrics_dict, "Worst Day % not found in metrics"

    # Values should not all be 0.0
    assert metrics_dict["Best Day %"] != 0.0 or metrics_dict["Worst Day %"] != 0.0
