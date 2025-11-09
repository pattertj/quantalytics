---
sidebar_position: 4
---

# Reports

Generate interactive, responsive tear sheets that combine metrics and visuals.

## Quickstart

```python
from quantalytics import render_basic_tearsheet

tearsheet = render_basic_tearsheet(returns, benchmark=benchmark_returns)
tearsheet.to_html("tearsheet.html")
```

## Custom Sections

Create your own layout by passing `TearsheetConfig` with custom sections:

```python
from quantalytics.reporting import TearsheetConfig, TearsheetSection
from quantalytics.charts import cumulative_returns_chart

sections = [
    TearsheetSection(
        title="Performance",
        description="Strategy vs benchmark cumulative returns.",
        figure_html=cumulative_returns_chart(returns, benchmark=benchmark_returns).to_html(
            full_html=False,
            include_plotlyjs="cdn",
        ),
    ),
]

config = TearsheetConfig(title="Q1 2024 Tear Sheet", subtitle="Mid-frequency strategy", sections=sections)
tearsheet = render_basic_tearsheet(returns, benchmark=benchmark_returns, config=config)
```

Use the `tearsheet.html` output as-is or export to PDF with your preferred toolchain.
