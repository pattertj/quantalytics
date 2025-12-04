from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


@dataclass(frozen=True)
class SummaryMetricSpec:
    """Metadata describing how to render a summary metric."""

    value_key: str
    label: str
    scale: float = 1.0
    suffix: str = ""
    decimals: int = 2
    tooltip: str | None = None


# Explicit registry of analytics metrics that can be rendered in summary cards.
SUMMARY_METRIC_REGISTRY: dict[str, SummaryMetricSpec] = {
    # Returns / performance
    "annualized_return": SummaryMetricSpec(
        "CAGR﹪",
        "CAGR",
        scale=1,
        suffix="",  # New metrics already scaled to %
    ),
    "cagr": SummaryMetricSpec("CAGR﹪", "CAGR", scale=1, suffix=""),
    "cumulative_return": SummaryMetricSpec(
        "Cumulative Return %", "Cumulative Return", scale=1, suffix="%"
    ),
    "expected_daily": SummaryMetricSpec(
        "Expected Daily %", "Expected Daily", scale=1, suffix="%"
    ),
    "expected_weekly": SummaryMetricSpec(
        "Expected Weekly %", "Expected Weekly", scale=1, suffix="%"
    ),
    "expected_monthly": SummaryMetricSpec(
        "Expected Monthly %", "Expected Monthly", scale=1, suffix="%"
    ),
    "expected_quarterly": SummaryMetricSpec(
        "Expected Quarterly %", "Expected Quarterly", scale=1, suffix="%"
    ),
    "expected_yearly": SummaryMetricSpec(
        "Expected Yearly %", "Expected Yearly", scale=1, suffix="%"
    ),
    "best_day": SummaryMetricSpec("Best Day %", "Best Day", scale=1, suffix="%"),
    "worst_day": SummaryMetricSpec("Worst Day %", "Worst Day", scale=1, suffix="%"),
    "best_month": SummaryMetricSpec("Best Month %", "Best Month", scale=1, suffix="%"),
    "worst_month": SummaryMetricSpec(
        "Worst Month %", "Worst Month", scale=1, suffix="%"
    ),
    "best_year": SummaryMetricSpec("Best Year %", "Best Year", scale=1, suffix="%"),
    "worst_year": SummaryMetricSpec("Worst Year %", "Worst Year", scale=1, suffix="%"),
    "avg_up_month": SummaryMetricSpec(
        "Avg. Up Month %", "Avg Up Month", scale=1, suffix="%"
    ),
    "avg_down_month": SummaryMetricSpec(
        "Avg. Down Month %", "Avg Down Month", scale=1, suffix="%"
    ),
    # Risk / ratios
    "sharpe": SummaryMetricSpec("Sharpe", "Sharpe Ratio"),
    "sortino": SummaryMetricSpec("Sortino", "Sortino Ratio"),
    "smart_sharpe": SummaryMetricSpec("Smart Sharpe", "Smart Sharpe"),
    "smart_sortino": SummaryMetricSpec("Smart Sortino", "Smart Sortino"),
    "calmar": SummaryMetricSpec("Calmar", "Calmar Ratio"),
    "romad": SummaryMetricSpec("RoMaD", "RoMaD"),
    "omega": SummaryMetricSpec("Omega", "Omega Ratio"),
    "serenity_index": SummaryMetricSpec("Serenity Index", "Serenity Index"),
    "time_in_market": SummaryMetricSpec(
        "Time in Market %", "Time in Market", scale=1, suffix="%"
    ),
    "recovery_factor": SummaryMetricSpec("Recovery Factor", "Recovery Factor"),
    "ulcer_index": SummaryMetricSpec("Ulcer Index", "Ulcer Index"),
    "ulcer_performance_index": SummaryMetricSpec(
        "ulcer_index", "Ulcer Performance Index"
    ),
    "gain_to_pain_ratio": SummaryMetricSpec("gain_to_pain_ratio", "Gain to Pain"),
    "payoff_ratio": SummaryMetricSpec("payoff_ratio", "Payoff Ratio"),
    "profit_factor": SummaryMetricSpec("profit_factor", "Profit Factor"),
    "profit_ratio": SummaryMetricSpec("profit_ratio", "Profit Ratio"),
    "win_loss_ratio": SummaryMetricSpec("win_loss_ratio", "Win/Loss Ratio"),
    # Volatility / drawdown
    "annualized_volatility": SummaryMetricSpec(
        "Volatility (ann.) %", "Annualized Vol", scale=1, suffix="%"
    ),
    "volatility": SummaryMetricSpec(
        "Volatility (ann.) %", "Annualized Vol", scale=1, suffix="%"
    ),
    "max_drawdown": SummaryMetricSpec(
        "Max Drawdown %", "Max Drawdown", scale=1, suffix="%"
    ),
    "longest_drawdown_days": SummaryMetricSpec(
        "Longest DD Days", "Longest DD Days", decimals=0, suffix=" days"
    ),
    "average_drawdown": SummaryMetricSpec(
        "Avg. Drawdown %", "Average Drawdown", scale=1, suffix="%"
    ),
    "average_drawdown_days": SummaryMetricSpec(
        "Avg. Drawdown Days", "Average DD Days", decimals=0, suffix=" days"
    ),
    "underwater_pct": SummaryMetricSpec(
        "Underwater %", "Underwater %", scale=1, suffix="%"
    ),
    # Tails / distribution
    "value_at_risk": SummaryMetricSpec(
        "Daily Value-at-Risk %", "Daily VaR", scale=1, suffix="%"
    ),
    "var": SummaryMetricSpec("Daily Value-at-Risk %", "Daily VaR", scale=1, suffix="%"),
    "expected_shortfall": SummaryMetricSpec(
        "Expected Shortfall (cVaR) %", "Expected Shortfall", scale=1, suffix="%"
    ),
    "conditional_value_at_risk": SummaryMetricSpec(
        "Expected Shortfall (cVaR) %", "Expected Shortfall", scale=1, suffix="%"
    ),
    "skewness": SummaryMetricSpec("Skew", "Skewness"),
    "kurtosis": SummaryMetricSpec("Kurtosis", "Kurtosis"),
    # Win/loss counts and rates
    "win_rate": SummaryMetricSpec("win_rate", "Win Rate", scale=100, suffix="%"),
    "winning_days": SummaryMetricSpec("Winning Days", "Winning Days", decimals=0),
    "losing_days": SummaryMetricSpec("Losing Days", "Losing Days", decimals=0),
    "consecutive_wins": SummaryMetricSpec(
        "winning_days", "Max Consecutive Wins", decimals=0
    ),
    "consecutive_losses": SummaryMetricSpec(
        "losing_days", "Max Consecutive Losses", decimals=0
    ),
}

DEFAULT_SUMMARY_METRICS: list[str] = [
    "annualized_return",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "romad",
    "sortino",
]


def resolve_summary_specs(keys: Iterable[str] | None) -> list[SummaryMetricSpec]:
    requested = list(keys) if keys is not None else DEFAULT_SUMMARY_METRICS
    specs: list[SummaryMetricSpec] = []
    for key in requested:
        spec = SUMMARY_METRIC_REGISTRY.get(key)
        if spec is None:
            raise ValueError(f"Unsupported summary metric '{key}'.")
        specs.append(spec)
    return specs


class SummaryMetric(Enum):
    ANNUALIZED_RETURN = "annualized_return"
    CAGR = "cagr"
    CUMULATIVE_RETURN = "cumulative_return"
    EXPECTED_DAILY = "expected_daily"
    EXPECTED_WEEKLY = "expected_weekly"
    EXPECTED_MONTHLY = "expected_monthly"
    EXPECTED_QUARTERLY = "expected_quarterly"
    EXPECTED_YEARLY = "expected_yearly"
    BEST_DAY = "best_day"
    WORST_DAY = "worst_day"
    BEST_MONTH = "best_month"
    WORST_MONTH = "worst_month"
    BEST_YEAR = "best_year"
    WORST_YEAR = "worst_year"
    AVG_UP_MONTH = "avg_up_month"
    AVG_DOWN_MONTH = "avg_down_month"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    SMART_SHARPE = "smart_sharpe"
    SMART_SORTINO = "smart_sortino"
    CALMAR = "calmar"
    ROMAD = "romad"
    OMEGA = "omega"
    SERENITY_INDEX = "serenity_index"
    TIME_IN_MARKET = "time_in_market"
    RECOVERY_FACTOR = "recovery_factor"
    ULCER_INDEX = "ulcer_index"
    ULCER_PERFORMANCE_INDEX = "ulcer_performance_index"
    GAIN_TO_PAIN_RATIO = "gain_to_pain_ratio"
    PAYOFF_RATIO = "payoff_ratio"
    PROFIT_FACTOR = "profit_factor"
    PROFIT_RATIO = "profit_ratio"
    WIN_LOSS_RATIO = "win_loss_ratio"
    ANNUALIZED_VOLATILITY = "annualized_volatility"
    VOLATILITY = "volatility"
    MAX_DRAWDOWN = "max_drawdown"
    LONGEST_DRAWDOWN_DAYS = "longest_drawdown_days"
    AVERAGE_DRAWDOWN = "average_drawdown"
    AVERAGE_DRAWDOWN_DAYS = "average_drawdown_days"
    UNDERWATER_PCT = "underwater_pct"
    VALUE_AT_RISK = "value_at_risk"
    VAR = "var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    CONDITIONAL_VALUE_AT_RISK = "conditional_value_at_risk"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    WIN_RATE = "win_rate"
    WINNING_DAYS = "winning_days"
    LOSING_DAYS = "losing_days"
    CONSECUTIVE_WINS = "consecutive_wins"
    CONSECUTIVE_LOSSES = "consecutive_losses"
