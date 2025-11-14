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
    smart: bool = False,
) -> float: ...
@overload
def sharpe(
    returns: DataFrame,
    rf: float = 0,
    periods: int = 365,
    annualize: bool = True,
    smart: bool = False,
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
def sortino(
    returns: Series,
    rf: float = 0,
    periods: Optional[int] = 365,
    annualize: bool = True,
    smart: bool = False,
) -> float: ...
@overload
def sortino(
    returns: DataFrame,
    rf: float = 0,
    periods: int = 365,
    annualize: bool = True,
    smart: bool = False,
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

    downside = sqrt((returns[returns < 0] ** 2).sum() / len(returns))

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
