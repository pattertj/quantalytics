from __future__ import annotations

from dataclasses import dataclass

from pandas.core.frame import DataFrame
from pandas.core.series import Series

from quantalytics.analytics import calmar, max_drawdown, sharpe, sortino
from quantalytics.analytics.stats import cagr, comp, volatility
from quantalytics.utils.timeseries import normalize_returns


@dataclass
class PerformanceMetrics:
    """Container for common performance statistics."""

    annualized_return: float | Series
    annualized_volatility: float | Series
    sharpe: float | Series
    sortino: float | Series
    calmar: float | Series
    max_drawdown: float | Series
    cumulative_return: float | Series

    def as_dict(self) -> dict[str, float | Series]:
        """Return a dictionary representation suitable for DataFrames."""

        return {
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "cumulative_return": self.cumulative_return,
        }


def performance_summary(
    returns: Series | DataFrame,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods: int = 365,
) -> PerformanceMetrics:
    """Generate a collection of core performance metrics."""

    series = normalize_returns(data=returns)
    cum_return = comp(returns=series)
    ann_ret = cagr(returns=series, periods=periods)
    ann_vol = volatility(returns=series, periods=periods)
    sharpe_ratio = sharpe(returns=series, rf=risk_free_rate, periods=periods)
    sortino_ratio = sortino(returns=series, rf=risk_free_rate, periods=periods)
    calmar_ratio = calmar(returns=series, periods=periods)
    mdd = max_drawdown(returns=series)

    return PerformanceMetrics(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe=sharpe_ratio,
        sortino=sortino_ratio,
        calmar=calmar_ratio,
        max_drawdown=mdd,
        cumulative_return=cum_return,
    )
