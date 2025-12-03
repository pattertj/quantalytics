"""Quantalytics: fast modern quantitative analytics library."""

from importlib import import_module

from quantalytics import analytics, charts, reports

__all__: list[str] = ["analytics", "charts", "reports"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"quantalytics.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__ + list(globals().keys())
