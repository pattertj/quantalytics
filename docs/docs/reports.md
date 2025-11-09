---
sidebar_position: 5
---

# Reports

Generate interactive, responsive tear sheets that combine metrics and visuals.

## Quickstart

```python
import quantalytics as qa

tearsheet = qa.reports.render_basic_tearsheet(returns, benchmark=benchmark_returns)
tearsheet.to_html("tearsheet.html")
```

## Custom Sections

Create your own layout by passing `TearsheetConfig` with custom sections:

```python
import quantalytics as qa

sections = [
    qa.reports.TearsheetSection(
        title="Performance",
        description="Strategy vs benchmark cumulative returns.",
        figure_html=qa.charts.cumulative_returns_chart(returns, benchmark=benchmark_returns).to_html(
            full_html=False,
            include_plotlyjs="cdn",
        ),
    ),
]

config = qa.reports.TearsheetConfig(
    title="Q1 2024 Tear Sheet",
    subtitle="Mid-frequency strategy",
    sections=sections,
)
tearsheet = qa.reports.render_basic_tearsheet(returns, benchmark=benchmark_returns, config=config)
```

Use the `tearsheet.html` output as-is or export to PDF with your preferred toolchain.
