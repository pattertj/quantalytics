from __future__ import annotations

from math import sqrt
from typing import overload

from numpy.ma.extras import corrcoef
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from quantalytics.analytics.stats import cagr, max_drawdown
from quantalytics.utils import timeseries as _utils


@overload
def sharpe(
    returns: Series,
    rf: float = 0,
    periods: int = 365,
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
    periods: int = 365,
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
    periods: int = 365,
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
    periods: int = 365,
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
        # Apply autocorrelation penalty
        downside = downside * autocorr_penalty(returns)

    res = returns.mean() / downside

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
    return cagr_pct / abs(max_dd)


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
