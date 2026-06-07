"""Cross-sectional quantile long/short portfolio formation and a cost-aware backtest.

Given a panel of signals ``s_{i,t}`` (dates x tickers), we go long the top quantile and short the bottom
quantile each rebalance, hold dollar-neutral, and run the turnover-cost-aware backtest from
``shared.validation``.

No look-ahead: the signal observed at ``t`` cannot be traded until the next session's open, so target weights
are **lagged one period** before being multiplied by per-period returns. ``cost_aware_backtest`` expects
weights "known at the OPEN of each date (already lagged)" and per-period asset returns realized over that
date — the lag here satisfies that contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from shared.validation import cost_aware_backtest

__all__ = ["quantile_long_short_weights", "rebalance_dates", "backtest_signal"]

_FREQ_ALIASES = {"daily": "D", "weekly": "W", "monthly": "M", "D": "D", "W": "W", "M": "M"}


def quantile_long_short_weights(signals: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Dollar-neutral long/short weights from cross-sectional signal ranks.

    For each date, names in the **top** quantile are equally weighted to sum to ``+1`` and names in the
    **bottom** quantile to sum to ``-1`` (row sum ``0``, gross exposure ``~2``). Dates with fewer than
    ``quantiles`` valid signals get all-zero weights (cannot form quantiles).

    Parameters
    ----------
    signals : pd.DataFrame
        Dates x tickers signal panel; NaN = no signal for that name/date.
    quantiles : int
        Number of quantile buckets (5 = quintiles).

    Returns
    -------
    pd.DataFrame
        Dates x tickers weights, same shape as ``signals``.
    """
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for t, row in signals.iterrows():
        valid = row.dropna()
        if valid.size < quantiles:
            continue
        # method="first" breaks ties deterministically so leg membership is reproducible.
        pct = valid.rank(pct=True, method="first")
        long_mask = pct > (quantiles - 1) / quantiles
        short_mask = pct <= 1.0 / quantiles
        n_long, n_short = int(long_mask.sum()), int(short_mask.sum())
        if n_long == 0 or n_short == 0:
            continue
        w = pd.Series(0.0, index=valid.index)
        w[long_mask] = 1.0 / n_long
        w[short_mask] = -1.0 / n_short
        weights.loc[t, w.index] = w.values
    return weights


def rebalance_dates(index: pd.DatetimeIndex, rule: str = "weekly") -> pd.DatetimeIndex:
    """First trading session within each calendar period (daily/weekly/monthly)."""
    freq = _FREQ_ALIASES.get(rule)
    if freq is None:
        raise ValueError(f"unknown rebalance rule: {rule!r}")
    if freq == "D":
        return index
    ser = pd.Series(1, index=index)
    return ser.groupby(index.to_period(freq)).head(1).index


def backtest_signal(
    signals: pd.DataFrame,
    asset_returns: pd.DataFrame,
    quantiles: int = 5,
    cost_bps: float = 10.0,
    rebalance: str = "weekly",
    lag: int = 1,
) -> dict:
    """Form quantile L/S weights, hold between rebalances, lag to next open, and backtest with costs.

    Parameters
    ----------
    signals : pd.DataFrame
        Dates x tickers signal panel (observed at each date).
    asset_returns : pd.DataFrame
        Dates x tickers **per-period** returns realized over each date.
    quantiles, cost_bps, rebalance : see module docstring.
    lag : int
        Sessions to lag weights before applying returns (default 1 = enter at next open; prevents look-ahead).

    Returns
    -------
    dict
        Output of :func:`shared.validation.cost_aware_backtest` (gross/net returns, turnover, net Sharpe
        stats), plus ``weights`` (the lagged, held weights actually applied).
    """
    targets = quantile_long_short_weights(signals, quantiles)
    reb = rebalance_dates(signals.index, rebalance)
    # Update weights only on rebalance dates; carry them forward between rebalances.
    keep = signals.index.isin(reb)
    held = targets.copy()
    held.loc[~keep] = np.nan
    held = held.ffill().fillna(0.0)
    held = held.shift(lag).fillna(0.0)  # lag to next open: signal@t -> position effective t+lag
    result = cost_aware_backtest(held, asset_returns, cost_bps=cost_bps)
    result["weights"] = held
    return result
