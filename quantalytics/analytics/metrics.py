from __future__ import annotations

from math import sqrt
from typing import Optional, overload

import numpy as _np
from numpy.ma.extras import corrcoef
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.stats import norm as _norm

from quantalytics.analytics.stats import (
    avg_loss,
    avg_win,
    cagr,
    comp,
    exposure,
    max_drawdown,
    volatility,
    win_rate,
)
from quantalytics.utils import timeseries as _utils


@overload
def sharpe(
    returns: Series,
    rf: float = 0,
    periods: Optional[int] = 365,
    annualize: bool = True,
    smart: bool = ...,
) -> float: ...
@overload
def sharpe(
    returns: DataFrame,
    rf: float = 0,
    periods: Optional[int] = 365,
    annualize: bool = True,
    smart: bool = ...,
) -> Series: ...
def sharpe(
    returns: Series | DataFrame,
    rf: float = 0,
    periods: Optional[int] = 365,
    annualize: bool = True,
    smart: bool = False,
) -> float | Series:
    """Calculates the sharpe ratio of access returns

    Args:
        returns (Series | DataFrame): returns series in $ or $
        rf (float, optional): Risk-free rate expressed as a yearly (annualized) return. Defaults to 0.
        periods (int, optional): Freq. of returns. Defaults to 365.
        annualize (bool, optional): return annualize sharpe?. Defaults to True.
        smart (bool, optional): return smart sharpe ratio. Defaults to False.

    Raises:
        ValueError: When rf is non-zero, periods must be specified

    Returns:
        float | Series: Series input → returns float (single Sharpe ratio). DataFrame input → returns pd.Series (one Sharpe per column)
    """
    if rf != 0 and periods is None:
        raise ValueError("When rf is non-zero, periods must be specified")

    returns: Series | DataFrame = _utils.normalize_returns(
        data=returns, rf=rf, nperiods=periods
    )
    divisor = returns.std(ddof=1)

    if smart:
        # penalize sharpe with auto correlation
        divisor = divisor * autocorr_penalty(returns=returns)

    res = returns.mean() / divisor

    return res * sqrt(1 if periods is None else periods) if annualize else res


@overload
def rolling_sharpe(
    returns: Series,
    rf: float = 0.0,
    rolling_period: int | None = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> Series: ...
@overload
def rolling_sharpe(
    returns: DataFrame,
    rf: float = 0.0,
    rolling_period: int | None = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> DataFrame: ...
def rolling_sharpe(
    returns: Series | DataFrame,
    rf: float = 0.0,
    rolling_period: int | None = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> Series | DataFrame:
    """Calculate rolling Sharpe ratio over a specified window.

    The rolling Sharpe ratio computes the Sharpe ratio using a rolling window,
    providing a time-varying measure of risk-adjusted performance. This allows
    you to track how the strategy's risk-adjusted returns evolve over time and
    adapt to changing market conditions.

    Args:
        returns (Series | DataFrame): Portfolio returns data. If DataFrame, returns a DataFrame
            with rolling Sharpe values for each column.
        rf (float, optional): Risk-free rate (annualized). Defaults to 0.0.
        rolling_period (int, optional): Rolling window size in periods.
            Defaults to 126 (~6 months of daily data).
        annualize (bool, optional): Whether to annualize the ratio. Defaults to True.
        periods_per_year (int, optional): Number of periods per year for annualization.
            Defaults to 252 (daily data).
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        Series | DataFrame: Rolling Sharpe ratio time series. Returns Series for Series input,
            DataFrame for DataFrame input (one column per input column).
            Earlier values will be NaN until the rolling window is filled.

    Raises:
        ValueError: If rf != 0 and rolling_period is None.

    Examples:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])
        >>> rolling_sharpe(returns, rolling_period=3, annualize=False)
        0         NaN
        1         NaN
        2    0.577350
        3   -0.707107
        4    0.816497
        5    0.707107
        dtype: float64

        >>> df = pd.DataFrame({"strategy_a": [0.01, -0.01, 0.02, 0.01],
        ...                    "strategy_b": [0.02, -0.02, 0.03, 0.01]})
        >>> rolling_sharpe(df, rolling_period=3, periods_per_year=252)
        # Returns DataFrame with rolling Sharpe for both strategies

    Notes:
        - First (rolling_period - 1) values will be NaN
        - Useful for monitoring strategy performance over time
        - Captures regime changes and varying market conditions
        - Annualization uses sqrt(periods_per_year) scaling
        - When prepare_returns=True, adjusts for risk-free rate
        - Standard rolling window is 126 days (~6 months) or 63 days (~3 months)

    See Also:
        sharpe: Static Sharpe ratio for entire period
        rolling_sortino: Rolling Sortino ratio (downside risk focus)
        smart_sharpe: Sharpe with autocorrelation penalty
    """
    # Validate parameters for risk-free rate handling
    if rf != 0 and rolling_period is None:
        raise ValueError("Must provide periods if rf != 0")

    normalized = (
        _utils.normalize_returns(data=returns, rf=rf, nperiods=rolling_period)
        if prepare_returns
        else returns
    )

    # Calculate rolling mean and standard deviation
    res = (
        normalized.rolling(rolling_period).mean()
        / normalized.rolling(rolling_period).std()
    )

    # Annualize if requested
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)

    return res


@overload
def rolling_sortino(
    returns: Series,
    rf: float = 0.0,
    rolling_period: int | None = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> Series: ...
@overload
def rolling_sortino(
    returns: DataFrame,
    rf: float = 0.0,
    rolling_period: int | None = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> DataFrame: ...
def rolling_sortino(
    returns: Series | DataFrame,
    rf: float = 0.0,
    rolling_period: int | None = 126,
    annualize: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
) -> Series | DataFrame:
    """Calculate rolling Sortino ratio over a specified window.

    The rolling Sortino ratio computes the Sortino ratio using a rolling window,
    providing a time-varying measure of downside risk-adjusted performance. Unlike
    the rolling Sharpe ratio, it only penalizes downside volatility, making it
    more suitable for strategies with asymmetric return distributions.

    Args:
        returns (Series | DataFrame): Portfolio returns data. If DataFrame, returns a DataFrame
            with rolling Sortino values for each column.
        rf (float, optional): Risk-free rate (annualized). Defaults to 0.0.
        rolling_period (int, optional): Rolling window size in periods.
            Defaults to 126 (~6 months of daily data).
        annualize (bool, optional): Whether to annualize the ratio. Defaults to True.
        periods_per_year (int, optional): Number of periods per year for annualization.
            Defaults to 252 (daily data).
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        Series | DataFrame: Rolling Sortino ratio time series. Returns Series for Series input,
            DataFrame for DataFrame input (one column per input column).
            Earlier values will be NaN until the rolling window is filled.
            May contain inf values when downside deviation is zero.

    Raises:
        ValueError: If rf != 0 and rolling_period is None.

    Examples:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])
        >>> rolling_sortino(returns, rolling_period=3, annualize=False)
        0         NaN
        1         NaN
        2    0.816497
        3   -1.000000
        4    1.154701
        5    1.000000
        dtype: float64

        >>> df = pd.DataFrame({"strategy_a": [0.01, -0.01, 0.02, 0.01],
        ...                    "strategy_b": [0.02, -0.02, 0.03, 0.01]})
        >>> rolling_sortino(df, rolling_period=3, periods_per_year=252)
        # Returns DataFrame with rolling Sortino for both strategies

    Notes:
        - First (rolling_period - 1) values will be NaN
        - Only penalizes downside volatility (returns below zero)
        - More forgiving than rolling Sharpe for strategies with positive skew
        - May return inf when window has no negative returns (zero downside)
        - Annualization uses sqrt(periods_per_year) scaling
        - Downside deviation calculated from squared negative returns only
        - Useful for monitoring asymmetric risk profiles over time

    See Also:
        sortino: Static Sortino ratio for entire period
        rolling_sharpe: Rolling Sharpe ratio (total risk focus)
        adjusted_sortino: Sortino adjusted for Sharpe comparability
        smart_sortino: Sortino with autocorrelation penalty
    """
    # Validate parameters for risk-free rate handling
    if rf != 0 and rolling_period is None:
        raise ValueError("Must provide periods if rf != 0")

    normalized = (
        _utils.normalize_returns(data=returns, rf=rf, nperiods=rolling_period)
        if prepare_returns
        else returns
    )

    # Optimized downside calculation using vectorized operations
    def calc_downside(x):
        """
        Calculate downside variance more efficiently.

        This function computes the sum of squared negative returns,
        which is used to calculate downside deviation.
        """
        negative_returns = x[x < 0]
        return (negative_returns**2).sum() if len(negative_returns) > 0 else 0

    # Calculate rolling downside deviation
    downside = (
        normalized.rolling(rolling_period).apply(calc_downside, raw=True)
        / rolling_period
    )

    # Calculate rolling Sortino ratio
    res = normalized.rolling(rolling_period).mean() / _np.sqrt(downside)

    # Annualize if requested
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)

    return res


@overload
def sortino(
    returns: Series,
    rf: float = 0,
    periods: Optional[int] = 365,
    annualize: bool = True,
    smart: bool = ...,
) -> float: ...
@overload
def sortino(
    returns: DataFrame,
    rf: float = 0,
    periods: Optional[int] = 365,
    annualize: bool = True,
    smart: bool = ...,
) -> Series: ...
def sortino(
    returns: Series | DataFrame,
    rf: float = 0,
    periods: Optional[int] = 365,
    annualize: bool = True,
    smart: bool = False,
) -> float | Series:
    """
    Calculates the sortino ratio of excess returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    https://en.wikipedia.org/wiki/Sortino_ratio
    """
    if rf != 0 and periods is None:
        raise ValueError("When rf is non-zero, periods must be specified")

    returns: Series | DataFrame = _utils.normalize_returns(
        data=returns, rf=rf, nperiods=periods
    )

    # Calculate downside deviation
    squared_neg = (returns[returns < 0] ** 2).sum() / len(returns)
    # Use numpy sqrt to handle both Series and float
    downside = _np.sqrt(squared_neg)

    if smart:
        downside = downside * autocorr_penalty(returns)

    mean_returns = returns.mean()
    if isinstance(downside, Series):
        safe = downside.replace(0, _np.nan)
        res = mean_returns / safe
        res = res.where(~downside.eq(0), float("nan"))
    elif downside == 0:
        return float("nan")
    else:
        res = mean_returns / downside

    return res * sqrt(1 if periods is None else periods) if annualize else res


@overload
def adjusted_sortino(
    returns: Series,
    rf: float = 0.0,
    periods: int = 365,
    annualize: bool = True,
    smart: bool = False,
) -> float: ...
@overload
def adjusted_sortino(
    returns: DataFrame,
    rf: float = 0.0,
    periods: int = 365,
    annualize: bool = True,
    smart: bool = False,
) -> Series: ...
def adjusted_sortino(
    returns: Series | DataFrame,
    rf: float = 0.0,
    periods: int = 365,
    annualize: bool = True,
    smart: bool = False,
) -> float | Series:
    """Calculate Jack Schwager's adjusted Sortino ratio for direct comparison with Sharpe.

    The adjusted Sortino ratio is the standard Sortino ratio divided by sqrt(2), allowing
    for direct numerical comparisons with the Sharpe ratio. This adjustment accounts for
    the difference in how downside deviation and total volatility are calculated.

    Args:
        returns (Series | DataFrame): Portfolio returns data. If DataFrame, returns a Series
            with one adjusted Sortino value per column.
        rf (float, optional): Risk-free rate (annualized). Defaults to 0.0.
        periods (int, optional): Number of periods per year for annualization.
            Defaults to 252 (daily data).
        annualize (bool, optional): Whether to annualize the ratio. Defaults to True.
        smart (bool, optional): Whether to apply autocorrelation penalty to downside deviation.
            Defaults to False.

    Returns:
        float | Series: Adjusted Sortino ratio value(s). Returns float for Series input,
            Series for DataFrame input (one value per column).

    Examples:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> adjusted_sortino(returns, rf=0.02, periods=252)
        1.23  # Can be directly compared to Sharpe ratio

        >>> df = pd.DataFrame({"strategy_a": [0.01, -0.01, 0.02], "strategy_b": [0.02, -0.02, 0.03]})
        >>> adjusted_sortino(df, periods=252)
        strategy_a    1.15
        strategy_b    1.08
        dtype: float64

    Notes:
        - Divides the standard Sortino ratio by sqrt(2) for comparability with Sharpe
        - This adjustment was proposed by Jack Schwager for practical comparison purposes
        - Use this when you need to compare Sortino and Sharpe ratios numerically
        - Without adjustment, Sortino ratios tend to be higher than Sharpe ratios
        - The smart parameter applies autocorrelation penalty via the sortino function
        - Reference: https://archive.is/wip/2rwFW

    See Also:
        sortino: Standard Sortino ratio without adjustment
        sharpe: Sharpe ratio for comparison
        smart_sortino: Sortino with autocorrelation penalty
    """
    # Calculate standard Sortino ratio
    data = sortino(returns, rf, periods=periods, annualize=annualize, smart=smart)

    # Apply Schwager's adjustment factor
    return data / sqrt(2)


@overload
def calmar(
    returns: Series, prepare_returns: bool = True, periods: int | None = None
) -> float: ...
@overload
def calmar(
    returns: DataFrame, prepare_returns: bool = True, periods: int | None = None
) -> Series: ...
def calmar(
    returns: Series | DataFrame,
    prepare_returns: bool = True,
    periods: int | None = None,
) -> float | Series:
    """Calculates the calmar ratio (CAGR% / MaxDD%)"""
    if prepare_returns:
        returns = _utils.normalize_returns(data=returns)
    cagr_pct = cagr(returns=returns, periods=periods)
    max_dd = max_drawdown(returns=returns)
    if isinstance(max_dd, Series):
        safe = max_dd.replace(0, _np.nan)
        return cagr_pct / safe.abs()
    return float("nan") if max_dd == 0 else cagr_pct / abs(max_dd)


@overload
def romad(
    returns: Series, prepare_returns: bool = True, periods: int | None = None
) -> float: ...
@overload
def romad(
    returns: DataFrame, prepare_returns: bool = True, periods: int | None = None
) -> Series: ...
def romad(
    returns: Series | DataFrame,
    prepare_returns: bool = True,
    periods: int | None = None,
) -> float | Series:
    """Alias for `calmar`; RoMaD is return over max drawdown."""
    return calmar(returns=returns, prepare_returns=prepare_returns, periods=periods)


def autocorr_penalty(returns: Series | DataFrame, prepare_returns=False) -> float:
    """Metric to account for auto correlation"""
    if prepare_returns:
        returns = _utils.normalize_returns(returns)

    if isinstance(returns, DataFrame):
        returns = returns[returns.columns[0]]

    num = len(returns)
    coef = abs(corrcoef(returns[:-1], returns[1:])[0, 1])
    corr = [((num - x) / num) * coef**x for x in range(1, num)]
    return sqrt(1 + 2 * sum(corr))


@overload
def omega(returns: Series, threshold: float = 0.0) -> float: ...
@overload
def omega(returns: DataFrame, threshold: float = 0.0) -> Series: ...
def omega(returns: Series | DataFrame, threshold: float = 0.0) -> float | Series:
    """Omega ratio: upside deviation divided by downside deviation relative to `threshold`."""

    def _omega(series: Series) -> float:
        clean = series
        diff = clean - threshold
        gains = diff[diff > 0].sum()
        losses = -diff[diff < 0].sum()
        if losses == 0:
            return float("inf") if gains > 0 else float("nan")
        return float(gains / losses)

    normalized = _utils.normalize_returns(data=returns)
    if isinstance(normalized, DataFrame):
        return Series({col: _omega(normalized[col]) for col in normalized.columns})
    return _omega(normalized)


@overload
def gain_to_pain_ratio(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def gain_to_pain_ratio(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def gain_to_pain_ratio(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """Gain-to-pain ratio computed as sum all returns over absolute sum of losses."""

    def _ratio(series: Series) -> float:
        clean = series
        total = clean.sum()
        negative = -clean[clean < 0].sum()
        if negative == 0:
            return float("inf") if total > 0 else float("nan")
        return float(total / negative)

    if prepare_returns:
        returns = _utils.normalize_returns(data=returns)
    if isinstance(returns, DataFrame):
        return Series({col: _ratio(returns[col]) for col in returns.columns})
    return _ratio(returns)


@overload
def skew(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def skew(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def skew(returns: Series | DataFrame, prepare_returns: bool = True) -> float | Series:
    """Skewness of the return distribution."""

    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns
    return normalized.skew()


@overload
def kurtosis(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def kurtosis(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def kurtosis(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """Kurtosis of the return distribution."""

    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns
    return normalized.kurtosis()


@overload
def ulcer_index(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def ulcer_index(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def ulcer_index(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """Calculate Ulcer Index."""

    normalized: Series | DataFrame = (
        _utils.normalize_returns(data=returns) if prepare_returns else returns
    )

    drawdowns: Series | DataFrame = to_drawdown_series(
        returns=normalized, prepare_returns=False
    )

    if isinstance(drawdowns, DataFrame):
        return _np.sqrt(_np.divide((drawdowns**2).sum(), normalized.shape[0] - 1))
    return float(_np.sqrt(_np.divide((drawdowns**2).sum(), normalized.shape[0] - 1)))


@overload
def ulcer_performance_index(
    returns: Series, rf: float = 0, prepare_returns: bool = True
) -> float: ...
@overload
def ulcer_performance_index(
    returns: DataFrame, rf: float = 0, prepare_returns: bool = True
) -> Series: ...
def ulcer_performance_index(
    returns: Series | DataFrame, rf: float = 0, prepare_returns: bool = True
) -> float | Series:
    """Return comp / ulcer index ratio."""

    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns
    base = comp(normalized)
    ui = ulcer_index(normalized, prepare_returns=False)
    if isinstance(base, Series):
        return base.subtract(rf).divide(ui)
    return (base - rf) / ui


@overload
def upi(returns: Series, rf: float = 0, prepare_returns: bool = True) -> float: ...
@overload
def upi(returns: DataFrame, rf: float = 0, prepare_returns: bool = True) -> Series: ...
def upi(
    returns: Series | DataFrame, rf: float = 0, prepare_returns: bool = True
) -> float | Series:
    """Alias for ulcer_performance_index."""

    return ulcer_performance_index(returns, rf=rf, prepare_returns=prepare_returns)


@overload
def serenity_index(
    returns: Series, rf: float = 0, prepare_returns: bool = True
) -> float: ...
@overload
def serenity_index(
    returns: DataFrame, rf: float = 0, prepare_returns: bool = True
) -> Series: ...
def serenity_index(
    returns: Series | DataFrame, rf: float = 0, prepare_returns: bool = True
) -> float | Series:
    """
    Serenity index (annualized return divided by ulcer index * CVaR pitfall).
    https://www.keyquant.com/Download/GetFile8e2a.pdf?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf
    """

    normalized: Series | DataFrame = (
        _utils.normalize_returns(data=returns) if prepare_returns else returns
    )

    pitfall = -cdar(returns=normalized, prepare_returns=False) / volatility(
        returns=normalized, prepare_returns=False
    )
    ulcer = ulcer_index(returns=normalized, prepare_returns=False)
    cagr_pct = cagr(returns=normalized, rf=rf)

    return cagr_pct / (ulcer * pitfall)


@overload
def risk_of_ruin(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def risk_of_ruin(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def risk_of_ruin(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """Return the risk of ruin after a sequence of returns."""

    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns
    if isinstance(normalized, DataFrame):
        return Series(
            {
                col: risk_of_ruin(normalized[col], prepare_returns=False)
                for col in normalized.columns
            }
        )
    wins = win_rate(normalized, prepare_returns=False)
    return ((1 - wins) / (1 + wins)) ** len(normalized)


@overload
def ror(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def ror(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def ror(returns: Series | DataFrame, prepare_returns: bool = True) -> float | Series:
    """Alias for risk_of_ruin."""

    return risk_of_ruin(returns, prepare_returns=prepare_returns)


@overload
def to_drawdown_series(returns: Series, prepare_returns: bool = True) -> Series: ...
@overload
def to_drawdown_series(
    returns: DataFrame, prepare_returns: bool = True
) -> DataFrame: ...
def to_drawdown_series(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> Series | DataFrame:
    """Convert return series into cumulative drawdown series."""

    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns
    prices = (1 + normalized).cumprod()
    running_max = prices.expanding().max()

    dd = prices / running_max - 1
    return dd.fillna(0)


def _value_at_risk(
    series: Series, sigma: float, confidence: float, prepare_returns: bool
) -> float:
    clean = _utils.normalize_returns(data=series) if prepare_returns else series
    mu = clean.mean()
    vol = sigma * clean.std(ddof=1)
    conf = confidence / 100 if confidence > 1 else confidence
    return _norm.ppf(1 - conf, mu, vol)


@overload
def value_at_risk(
    returns: Series,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float: ...
@overload
def value_at_risk(
    returns: DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> Series: ...
def value_at_risk(
    returns: Series | DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | Series:
    """Variance-covariance value at risk using Gaussian approximation."""

    if isinstance(returns, DataFrame):
        return Series(
            {
                col: _value_at_risk(
                    returns[col],
                    sigma=sigma,
                    confidence=confidence,
                    prepare_returns=prepare_returns,
                )
                for col in returns.columns
            }
        )
    return _value_at_risk(
        returns, sigma=sigma, confidence=confidence, prepare_returns=prepare_returns
    )


def var(
    returns: Series | DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | Series:
    """Alias for value_at_risk."""

    return value_at_risk(
        returns, sigma=sigma, confidence=confidence, prepare_returns=prepare_returns
    )


@overload
def conditional_value_at_risk(
    returns: Series,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float: ...
@overload
def conditional_value_at_risk(
    returns: DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> Series: ...
def conditional_value_at_risk(
    returns: Series | DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | Series:
    """Expected shortfall below the VaR threshold."""

    if isinstance(returns, DataFrame):
        return Series(
            {
                col: conditional_value_at_risk(
                    returns[col],
                    sigma=sigma,
                    confidence=confidence,
                    prepare_returns=prepare_returns,
                )
                for col in returns.columns
            }
        )

    clean = _utils.normalize_returns(data=returns) if prepare_returns else returns
    threshold = value_at_risk(
        clean, sigma=sigma, confidence=confidence, prepare_returns=False
    )
    tail = clean[clean < threshold]
    mean_tail = float("nan") if tail.empty else tail.mean()
    return threshold if _np.isnan(mean_tail) else float(mean_tail)


@overload
def cvar(
    returns: Series,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float: ...
@overload
def cvar(
    returns: DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> Series: ...
def cvar(
    returns: Series | DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | Series:
    """Alias for conditional_value_at_risk."""

    return conditional_value_at_risk(
        returns, sigma=sigma, confidence=confidence, prepare_returns=prepare_returns
    )


@overload
def expected_shortfall(
    returns: Series,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float: ...
@overload
def expected_shortfall(
    returns: DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> Series: ...
def expected_shortfall(
    returns: Series | DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | Series:
    """Calculate the Expected Shortfall (ES), also known as CVaR.

    Expected Shortfall measures the average loss in the worst-case scenarios
    beyond the Value at Risk (VaR) threshold. It provides a more comprehensive
    risk measure than VaR by considering the severity of tail losses, not just
    their probability.

    This is an alias for conditional_value_at_risk() with identical functionality.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        sigma (float, optional): Volatility multiplier for tail risk estimation.
            Higher values assume more extreme tail events. Defaults to 1.
        confidence (float, optional): Confidence level for the calculation.
            Must be between 0 and 1. Common values:
            - 0.95 (95%): Standard risk threshold
            - 0.99 (99%): Conservative risk threshold
            Defaults to 0.95.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Expected shortfall value (negative indicates losses).
            Returns float for Series input, Series for DataFrame input (one value per column).
            Calculated as the mean of returns below the VaR threshold.

    Examples:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.03, 0.02, -0.01])
        >>> expected_shortfall(returns, confidence=0.95)
        -0.025  # Average of worst 5% losses

    Notes:
        - Also known as Conditional Value at Risk (CVaR) or Average Value at Risk (AVaR)
        - More conservative than VaR as it considers tail severity
        - Preferred by many regulators for risk management
        - Satisfies sub-additivity property (unlike VaR)
        - Returns NaN if no returns fall below VaR threshold
        - Higher confidence levels result in more extreme (negative) ES values

    See Also:
        conditional_value_at_risk: The full implementation (ES is an alias)
        cvar: Another alias for the same function
        value_at_risk: VaR calculation (threshold only, not average)
        var: Alias for value_at_risk
    """
    return conditional_value_at_risk(returns, sigma, confidence, prepare_returns)


@overload
def cdar(
    returns: Series,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float: ...
@overload
def cdar(
    returns: DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> Series: ...
def cdar(
    returns: Series | DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | Series:
    """Alias for conditional_drawdown_at_risk."""

    return conditional_drawdown_at_risk(
        returns=returns,
        sigma=sigma,
        confidence=confidence,
        prepare_returns=prepare_returns,
    )


@overload
def conditional_drawdown_at_risk(
    returns: Series,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float: ...
@overload
def conditional_drawdown_at_risk(
    returns: DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> Series: ...
def conditional_drawdown_at_risk(
    returns: Series | DataFrame,
    sigma: float = 1,
    confidence: float = 0.95,
    prepare_returns: bool = True,
) -> float | Series:
    """
    CDaR: CVaR of drawdowns
    In the worst drawdowns, I'm down 21.7% from my peak
    """

    normalized = _utils.normalize_returns(returns) if prepare_returns else returns
    dd = to_drawdown_series(returns=normalized, prepare_returns=False)

    return conditional_value_at_risk(
        returns=dd, sigma=sigma, confidence=confidence, prepare_returns=prepare_returns
    )


@overload
def tail_ratio(
    returns: Series, cutoff: float = 0.95, prepare_returns: bool = True
) -> float: ...
@overload
def tail_ratio(
    returns: DataFrame, cutoff: float = 0.95, prepare_returns: bool = True
) -> Series: ...
def tail_ratio(
    returns: Series | DataFrame, cutoff: float = 0.95, prepare_returns: bool = True
) -> float | Series:
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    def _ratio(series: Series) -> float:
        upper = series.quantile(cutoff)
        lower = series.quantile(1 - cutoff)
        return float(abs(upper / lower if lower != 0 else float("inf")))

    if isinstance(normalized, DataFrame):
        return Series({col: _ratio(normalized[col]) for col in normalized.columns})
    return _ratio(normalized)


@overload
def payoff_ratio(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def payoff_ratio(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def payoff_ratio(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """
    Calculate the Payoff Ratio (Average Win / Average Loss).

    Also known as Win/Loss Ratio. Measures the average size of winning trades
    relative to losing trades. A value > 1 indicates average wins exceed average losses.

    Parameters
    ----------
    returns : Series or DataFrame
        Returns data
    prepare_returns : bool, default True
        Whether to normalize returns before calculation

    Returns
    -------
    float or Series
        Payoff ratio (avg win / avg loss)

    Examples
    --------
    >>> returns = pd.Series([0.10, -0.05, 0.03, -0.02, 0.04])
    >>> payoff_ratio(returns)  # (0.0567 / 0.035) = 1.62
    1.62
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    def _ratio(series: Series) -> float:
        win = avg_win(series)
        loss = abs(avg_loss(series))
        return float(win / loss) if loss != 0 else float("inf")

    if isinstance(normalized, DataFrame):
        return Series({col: _ratio(normalized[col]) for col in normalized.columns})
    return _ratio(normalized)


def win_loss_ratio(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """Alias for payoff_ratio. See payoff_ratio for documentation."""
    return payoff_ratio(returns, prepare_returns)


@overload
def profit_factor(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def profit_factor(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def profit_factor(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """
    Calculate the Profit Factor (Total Wins / Total Losses).

    Measures the ratio of gross profits to gross losses. A value > 1 indicates
    profitability, > 2 is considered good, > 3 is excellent.

    Parameters
    ----------
    returns : Series or DataFrame
        Returns data
    prepare_returns : bool, default True
        Whether to normalize returns before calculation

    Returns
    -------
    float or Series
        Profit factor (sum of wins / abs(sum of losses))

    Examples
    --------
    >>> returns = pd.Series([0.10, -0.05, 0.03, -0.02, 0.04])
    >>> profit_factor(returns)  # (0.17 / 0.07) = 2.43
    2.43
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    def _ratio(series: Series) -> float:
        gains = series[series >= 0].sum()
        losses = abs(series[series < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    if isinstance(normalized, DataFrame):
        return Series({col: _ratio(normalized[col]) for col in normalized.columns})
    return _ratio(normalized)


def profit_ratio(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """
    Alias for profit_factor. See profit_factor for documentation.

    Note: The term 'profit ratio' is often used interchangeably with
    'profit factor' in trading literature.
    """
    return profit_factor(returns, prepare_returns)


@overload
def expectancy(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def expectancy(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def expectancy(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """
    Calculate the Expectancy (expected value per trade).

    Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

    Parameters
    ----------
    returns : Series or DataFrame
        Returns data
    prepare_returns : bool, default True
        Whether to normalize returns before calculation

    Returns
    -------
    float or Series
        Expected return per trade
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    def _expectancy(series: Series) -> float:
        if len(series) == 0:
            return 0.0

        wins = series[series > 0]
        losses = series[series < 0]

        win_rate = len(wins) / len(series) if len(series) > 0 else 0
        loss_rate = len(losses) / len(series) if len(series) > 0 else 0

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        return float((win_rate * avg_win) - (loss_rate * avg_loss))

    if isinstance(normalized, DataFrame):
        return Series({col: _expectancy(normalized[col]) for col in normalized.columns})
    return _expectancy(normalized)


@overload
def cpc_index(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def cpc_index(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def cpc_index(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    def _index(series: Series) -> float:
        pf = profit_factor(series)
        wr = win_rate(series, prepare_returns=False)
        wl = win_loss_ratio(series, prepare_returns=False)
        return float(pf * wr * wl)

    if isinstance(normalized, DataFrame):
        return Series({col: _index(normalized[col]) for col in normalized.columns})
    return _index(normalized)


def common_sense_ratio(returns, prepare_returns=True):
    """Measures the common sense ratio (profit factor * tail ratio)"""
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns
    return profit_factor(normalized) * tail_ratio(normalized)


@overload
def outlier_win_ratio(
    returns: Series, quantile: float = 0.99, prepare_returns: bool = True
) -> float: ...
@overload
def outlier_win_ratio(
    returns: DataFrame, quantile: float = 0.99, prepare_returns: bool = True
) -> Series: ...
def outlier_win_ratio(
    returns: Series | DataFrame, quantile: float = 0.99, prepare_returns: bool = True
) -> float | Series:
    """Calculate the outlier winners ratio of returns.

    Measures the magnitude of extreme positive returns relative to average gains.
    A higher ratio indicates that outlier wins are significantly larger than typical wins.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        quantile (float, optional): Percentile to use for outlier threshold.
            Defaults to 0.99 (99th percentile).
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Outlier win ratio. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Calculated as: quantile value / mean of positive returns.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.15, 0.01])
        >>> outlier_win_ratio(returns, quantile=0.95)
        7.5  # The 95th percentile is much larger than average wins

    Notes:
        - Returns inf if there are no positive returns
        - Higher values suggest that outlier wins drive performance
        - Useful for understanding return distribution skewness
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    if not isinstance(normalized, Series):
        return normalized.apply(
            lambda col: outlier_win_ratio(col, quantile, prepare_returns=False)
        )
    positive_returns = normalized[normalized >= 0]
    if len(positive_returns) == 0:
        return float("inf")
    return float(normalized.quantile(quantile) / positive_returns.mean())


@overload
def outlier_loss_ratio(
    returns: Series, quantile: float = 0.01, prepare_returns: bool = True
) -> float: ...
@overload
def outlier_loss_ratio(
    returns: DataFrame, quantile: float = 0.01, prepare_returns: bool = True
) -> Series: ...
def outlier_loss_ratio(
    returns: Series | DataFrame, quantile: float = 0.01, prepare_returns: bool = True
) -> float | Series:
    """Calculate the outlier losers ratio of returns.

    Measures the magnitude of extreme negative returns relative to average losses.
    A higher ratio (closer to 0) indicates that outlier losses are not much worse
    than typical losses. A lower ratio indicates severe tail risk.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        quantile (float, optional): Percentile to use for outlier threshold.
            Defaults to 0.01 (1st percentile, representing worst losses).
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Outlier loss ratio. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Calculated as: quantile value / mean of negative returns.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, -0.15, -0.01])
        >>> outlier_loss_ratio(returns, quantile=0.05)
        7.5  # The 5th percentile loss is much worse than average losses

    Notes:
        - Returns -inf if there are no negative returns
        - Lower values indicate severe tail risk (large outlier losses)
        - Both the quantile and mean will be negative
        - Ratio will be positive (negative / negative)
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    if not isinstance(normalized, Series):
        return normalized.apply(
            lambda col: outlier_loss_ratio(col, quantile, prepare_returns=False)
        )
    negative_returns = normalized[normalized < 0]
    if len(negative_returns) == 0:
        return float("-inf")
    return float(normalized.quantile(quantile) / negative_returns.mean())


@overload
def recovery_factor(
    returns: Series, rf: float = 0.0, prepare_returns: bool = True
) -> float: ...
@overload
def recovery_factor(
    returns: DataFrame, rf: float = 0.0, prepare_returns: bool = True
) -> Series: ...
def recovery_factor(
    returns: Series | DataFrame, rf: float = 0.0, prepare_returns: bool = True
) -> float | Series:
    """Calculate the recovery factor of returns.

    Measures how effectively a strategy recovers from drawdowns by comparing
    total returns to the maximum drawdown. Higher values indicate better
    recovery characteristics.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        rf (float, optional): Risk-free rate to subtract from returns.
            Defaults to 0.0.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Recovery factor. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Calculated as: |total_returns - rf| / |max_drawdown|.

    Examples:
        >>> returns = pd.Series([0.02, -0.05, 0.03, 0.02])
        >>> recovery_factor(returns)
        2.0  # Recovered twice the maximum drawdown

    Notes:
        - Higher values indicate better recovery from losses
        - A value of 1.0 means total return equals maximum drawdown
        - Values > 1.0 indicate recovery exceeds drawdown depth
        - Useful for evaluating resilience after losses
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    if not isinstance(normalized, Series):
        return normalized.apply(
            lambda col: recovery_factor(col, rf, prepare_returns=False)
        )
    total_returns = normalized.sum() - rf
    # max_drawdown already normalizes returns internally
    max_dd = max_drawdown(returns) if prepare_returns else max_drawdown(normalized)
    if max_dd == 0:
        return float("inf") if total_returns >= 0 else float("-inf")
    return float(abs(total_returns) / abs(max_dd))


@overload
def risk_return_ratio(returns: Series, prepare_returns: bool = True) -> float: ...
@overload
def risk_return_ratio(returns: DataFrame, prepare_returns: bool = True) -> Series: ...
def risk_return_ratio(
    returns: Series | DataFrame, prepare_returns: bool = True
) -> float | Series:
    """Calculate the risk-return ratio of returns.

    Measures the return per unit of risk, similar to Sharpe ratio but without
    accounting for the risk-free rate. This is the raw mean/volatility relationship.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Risk-return ratio. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Calculated as: mean(returns) / std(returns).

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        >>> risk_return_ratio(returns)
        1.5  # Mean return is 1.5x the standard deviation

    Notes:
        - Similar to Sharpe ratio but without risk-free rate adjustment
        - Positive values indicate positive average returns
        - Higher values indicate better risk-adjusted performance
        - Not annualized by default
    """
    normalized = _utils.normalize_returns(data=returns) if prepare_returns else returns

    if not isinstance(normalized, Series):
        return normalized.apply(
            lambda col: risk_return_ratio(col, prepare_returns=False)
        )
    std = normalized.std()
    if std == 0:
        return float("inf") if normalized.mean() > 0 else float("-inf")
    return float(normalized.mean() / std)


@overload
def smart_sharpe(
    returns: Series,
    rf: float = 0.0,
    periods: Optional[int] = 365,
    annualize: bool = True,
) -> float: ...
@overload
def smart_sharpe(
    returns: DataFrame,
    rf: float = 0.0,
    periods: Optional[int] = 365,
    annualize: bool = True,
) -> Series: ...
def smart_sharpe(
    returns: Series | DataFrame,
    rf: float = 0.0,
    periods: Optional[int] = 365,
    annualize: bool = True,
) -> float | Series:
    """Calculate the Smart Sharpe ratio with autocorrelation penalty.

    The Smart Sharpe ratio adjusts the traditional Sharpe ratio by penalizing
    for autocorrelation in returns. This provides a more conservative estimate
    of risk-adjusted performance when returns exhibit serial correlation, which
    can artificially inflate the standard Sharpe ratio.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        rf (float, optional): Risk-free rate expressed as a yearly (annualized) return.
            Defaults to 0.0.
        periods (int, optional): Number of periods per year for annualization.
            Defaults to 365 (daily data). Required when rf is non-zero.
        annualize (bool, optional): Whether to annualize the ratio.
            Defaults to True.

    Returns:
        float | Series: Smart Sharpe ratio value. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Calculated as: sharpe_ratio * autocorr_penalty.

    Raises:
        ValueError: When rf is non-zero and periods is None.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        >>> smart_sharpe(returns, rf=0.02, periods=252)
        1.8  # Sharpe ratio adjusted for autocorrelation

    Notes:
        - Applies autocorrelation penalty to volatility in denominator
        - More conservative than standard Sharpe when returns are autocorrelated
        - Useful for strategies with momentum or mean-reversion characteristics
        - Penalty based on correlation between consecutive returns
        - Higher autocorrelation leads to larger penalty (lower Smart Sharpe)

    See Also:
        sharpe: Standard Sharpe ratio without autocorrelation adjustment
        smart_sortino: Sortino ratio with autocorrelation penalty
        autocorr_penalty: The autocorrelation penalty calculation
    """
    return sharpe(returns, rf, periods, annualize, smart=True)


@overload
def smart_sortino(
    returns: Series,
    rf: float = 0.0,
    periods: Optional[int] = 365,
    annualize: bool = True,
) -> float: ...
@overload
def smart_sortino(
    returns: DataFrame,
    rf: float = 0.0,
    periods: Optional[int] = 365,
    annualize: bool = True,
) -> Series: ...
def smart_sortino(
    returns: Series | DataFrame,
    rf: float = 0.0,
    periods: Optional[int] = 365,
    annualize: bool = True,
) -> float | Series:
    """Calculate the Smart Sortino ratio with autocorrelation penalty.

    The Smart Sortino ratio adjusts the traditional Sortino ratio by penalizing
    for autocorrelation in returns. Like Smart Sharpe, this provides a more
    conservative estimate when returns exhibit serial correlation, but focuses
    only on downside volatility rather than total volatility.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        rf (float, optional): Risk-free rate expressed as a yearly (annualized) return.
            Defaults to 0.0.
        periods (int, optional): Number of periods per year for annualization.
            Defaults to 365 (daily data). Required when rf is non-zero.
        annualize (bool, optional): Whether to annualize the ratio.
            Defaults to True.

    Returns:
        float | Series: Smart Sortino ratio value. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Returns NaN if there is no downside volatility.

    Raises:
        ValueError: When rf is non-zero and periods is None.

    Examples:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.005])
        >>> smart_sortino(returns, rf=0.02, periods=252)
        2.3  # Sortino ratio adjusted for autocorrelation

    Notes:
        - Applies autocorrelation penalty to downside deviation in denominator
        - More conservative than standard Sortino when returns are autocorrelated
        - Only penalizes downside volatility, not upside volatility
        - Useful for asymmetric return distributions
        - Preferred over Smart Sharpe when focusing on downside risk
        - Returns NaN when there are no negative returns (zero downside deviation)

    See Also:
        sortino: Standard Sortino ratio without autocorrelation adjustment
        smart_sharpe: Sharpe ratio with autocorrelation penalty
        autocorr_penalty: The autocorrelation penalty calculation
    """
    return sortino(returns, rf, periods, annualize, smart=True)


@overload
def rar(
    returns: Series,
    rf: float = 0.0,
    periods: int = 365,
    prepare_returns: bool = True,
) -> float: ...
@overload
def rar(
    returns: DataFrame,
    rf: float = 0.0,
    periods: int = 365,
    prepare_returns: bool = True,
) -> Series: ...
def rar(
    returns: Series | DataFrame,
    rf: float = 0.0,
    periods: int = 365,
    prepare_returns: bool = True,
) -> float | Series:
    """Calculate the Risk-Adjusted Return (RAR) accounting for market exposure.

    RAR divides the Compound Annual Growth Rate (CAGR) by market exposure time,
    providing a more accurate risk-adjusted return metric that penalizes strategies
    with lower market participation. This is particularly useful for strategies that
    are not fully invested at all times.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        rf (float, optional): Risk-free rate expressed as a yearly (annualized) return.
            Defaults to 0.0.
        periods (int, optional): Number of periods per year for annualization.
            Defaults to 365 (daily data).
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Risk-adjusted return value. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Calculated as: CAGR / exposure.

    Examples:
        >>> returns = pd.Series([0.01, 0.0, 0.02, 0.0, 0.03])  # 60% exposure
        >>> rar(returns, periods=252)
        0.12  # CAGR adjusted for 60% exposure time

    Notes:
        - Divides CAGR by exposure to account for time in market
        - Higher exposure leads to lower RAR for same CAGR
        - Useful for comparing strategies with different holding periods
        - Exposure calculated as proportion of non-zero return periods
        - Returns NaN if exposure is 0 (no market participation)

    See Also:
        cagr: Compound Annual Growth Rate
        exposure: Market exposure time calculation
        sharpe: Alternative risk-adjusted return metric
    """
    normalized = _utils.normalize_returns(returns, rf) if prepare_returns else returns
    exp = exposure(normalized)

    # Handle zero exposure edge case
    if isinstance(exp, Series):
        # For DataFrame input, handle zero exposure per column
        result = cagr(returns=normalized, periods=periods) / exp
        return result.replace([_np.inf, -_np.inf], _np.nan)
    else:
        # For Series input
        if exp == 0:
            return float("nan")
        return cagr(returns=normalized, periods=periods) / exp


@overload
def kelly_criterion(
    returns: Series,
    prepare_returns: bool = True,
) -> float: ...
@overload
def kelly_criterion(
    returns: DataFrame,
    prepare_returns: bool = True,
) -> Series: ...
def kelly_criterion(
    returns: Series | DataFrame,
    prepare_returns: bool = True,
) -> float | Series:
    """Calculate the Kelly Criterion for optimal position sizing.

    The Kelly Criterion determines the optimal fraction of capital to allocate
    to a strategy to maximize long-term growth rate. It balances the trade-off
    between maximizing returns and minimizing risk of ruin.

    Args:
        returns (Series | DataFrame): Portfolio returns data.
        prepare_returns (bool, optional): Whether to normalize returns before calculation.
            Defaults to True.

    Returns:
        float | Series: Optimal capital allocation fraction. Returns float for Series input,
            Series for DataFrame input (one value per column).
            Calculated as: ((win_ratio * win_prob) - lose_prob) / win_ratio.
            Values typically range from 0 to 1, where:
            - 0 means do not invest
            - 1 means invest 100% of capital
            - Values > 1 suggest leverage (use with caution)
            - Negative values suggest the strategy is not profitable

    Examples:
        >>> returns = pd.Series([0.02, -0.01, 0.03, -0.01, 0.02])
        >>> kelly_criterion(returns)
        0.25  # Suggests allocating 25% of capital

    Notes:
        - Based on win rate, loss rate, and win/loss ratio
        - Assumes returns are independent and identically distributed
        - Often considered aggressive; many practitioners use fractional Kelly (e.g., Kelly/2)
        - Returns 0 or negative if strategy is not profitable
        - Does not account for correlation between trades
        - Sensitive to estimation errors in win rate and payoff ratio
        - Should be used as a guideline, not an absolute rule

    References:
        Kelly, J. L. (1956). "A New Interpretation of Information Rate".
        Bell System Technical Journal. 35 (4): 917–926.
        https://en.wikipedia.org/wiki/Kelly_criterion

    See Also:
        win_rate: Probability of winning trades
        payoff_ratio: Average win to average loss ratio
        risk_of_ruin: Probability of losing all capital
    """
    normalized: Series | DataFrame = (
        _utils.normalize_returns(data=returns) if prepare_returns else returns
    )
    win_loss_ratio = payoff_ratio(normalized, prepare_returns=False)
    win_prob = win_rate(normalized, prepare_returns=False)
    lose_prob = 1 - win_prob

    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio
