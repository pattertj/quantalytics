"""Quantalytics: fast modern quantitative analytics library."""

# Organize the public surface into focused namespaces so consumers
# access helpers via `quantalytics.metrics`, `quantalytics.stats`, `quantalytics.charts`, and `quantalytics.reports`.
from . import charts as charts
from . import metrics as metrics
from . import stats as stats
from . import reporting as reports

__all__ = ["metrics", "stats", "charts", "reports"]
