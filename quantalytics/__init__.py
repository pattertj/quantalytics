"""Quantalytics: fast modern quantitative analytics library."""

# Organize the public surface into focused namespaces so consumers
# access helpers via `quantalytics.analytics`, `quantalytics.charts`, and `quantalytics.reports`.
from . import analytics as analytics
from . import charts as charts
from . import reporting as reports

__all__ = ["analytics", "charts", "reports"]
