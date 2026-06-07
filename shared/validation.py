"""Reusable validation utilities shared across all four portfolio projects.

Implements the López de Prado / Bailey toolkit for honest performance evaluation:
- Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
- Minimum Track Record Length
- Purged K-Fold cross-validation with embargo
- Combinatorial Purged Cross-Validation (CPCV) path generation
- Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric Cross-Validation (CSCV)
- A simple transaction-cost-aware long/short backtest

References
----------
Bailey, D. and López de Prado, M. (2014). The Deflated Sharpe Ratio. J. Portfolio Management 40(5).
Bailey, D., Borwein, J., López de Prado, M., Zhu, Q. (2016). The Probability of Backtest Overfitting.
López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.

These are clean-room implementations for educational/portfolio use. Validate against `skfolio` / `pypbo`
before relying on them in production. Do NOT use `mlfinlab` (not free/open-source).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------------------
# Sharpe-ratio statistics
# --------------------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    sr: float, n_obs: int, skew: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio.

    Probability that the true (per-period) Sharpe ratio exceeds ``sr_benchmark``, correcting for
    sample length and the non-normality (skew, kurtosis) of the return distribution.

    Parameters
    ----------
    sr : float
        Observed per-period Sharpe ratio (NOT annualized).
    n_obs : int
        Number of return observations.
    skew : float
        Sample skewness of returns.
    kurt : float
        Sample kurtosis of returns (non-excess; normal == 3).
    sr_benchmark : float
        Benchmark Sharpe to beat (default 0).

    Returns
    -------
    float
        PSR in [0, 1].
    """
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2)
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe ratio across ``n_trials`` independent strategies under the null.

    Uses the extreme-value approximation from Bailey & López de Prado (2014):

        E[max SR] ~ sqrt(V) * [ (1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)) ]

    where V is the cross-trial variance of Sharpe ratios, g is the Euler-Mascheroni constant,
    and Z^-1 is the standard-normal quantile function.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    sqrt_v = np.sqrt(sr_variance)
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sqrt_v * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2))


def deflated_sharpe_ratio(
    sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_variance: float
) -> float:
    """Deflated Sharpe Ratio.

    PSR evaluated against the *expected maximum* Sharpe under ``n_trials`` (selection-bias correction).
    A DSR > 0.95 is the usual bar for "this is unlikely to be a fluke of multiple testing".
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_benchmark=sr0)


def min_track_record_length(
    sr: float, skew: float, kurt: float, sr_benchmark: float = 0.0, prob: float = 0.95
) -> float:
    """Minimum number of observations needed for PSR(sr_benchmark) >= ``prob``."""
    if sr <= sr_benchmark:
        return float("inf")
    denom = (sr - sr_benchmark) ** 2
    factor = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    return float(1.0 + factor * (norm.ppf(prob) / np.sqrt(denom)) ** 2 if denom > 0 else float("inf"))


def sharpe_stats(returns: np.ndarray) -> dict:
    """Convenience: per-period Sharpe, n, skew, kurtosis from a return series."""
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew

    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    return {
        "sr": float(sr),
        "n_obs": int(r.size),
        "skew": float(_skew(r)) if r.size > 2 else 0.0,
        "kurt": float(_kurt(r, fisher=False)) if r.size > 3 else 3.0,
    }


# --------------------------------------------------------------------------------------
# Purged / embargoed cross-validation (López de Prado, AFML ch. 7)
# --------------------------------------------------------------------------------------
@dataclass
class PurgedKFold:
    """K-Fold cross-validation that purges and embargoes to prevent label leakage in time series.

    Each observation i has a label that is realized over the interval [index[i], t1[i]]. When a test
    fold covers some interval, any training observation whose label interval overlaps the test interval
    is *purged*. An additional *embargo* removes a fraction of observations immediately after the test set.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    t1 : pd.Series
        Series indexed by event start time; values are label end times (must be sorted by index).
    embargo_pct : float
        Fraction of total observations to embargo after each test fold (e.g., 0.01).
    """

    n_splits: int
    t1: pd.Series
    embargo_pct: float = 0.0

    def split(self, X: pd.DataFrame):
        if not X.index.equals(self.t1.index):
            raise ValueError("X and t1 must share the same index")
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.embargo_pct)
        test_starts = [
            (i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)
        ]
        for start, end in test_starts:
            t0 = self.t1.index[start]                      # test interval start time
            test_idx = indices[start:end]
            max_t1_idx = X.index.searchsorted(self.t1.iloc[test_idx].max())
            train_idx = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            train_idx = train_idx[train_idx < start]       # purge: drop overlapping labels
            if max_t1_idx < X.shape[0]:                     # embargo after the test window
                train_idx = np.concatenate(
                    (train_idx, indices[max_t1_idx + embargo:])
                )
            yield train_idx, test_idx


def combinatorial_purged_splits(
    n_groups: int, k_test: int, t1: pd.Series, embargo_pct: float = 0.0
):
    """Generate CPCV train/test index pairs.

    Splits observations into ``n_groups`` contiguous groups and forms every combination of ``k_test``
    test groups, purging+embargoing the training groups each time. With N groups and k test groups you
    obtain C(N, k) splits, which can be recombined into multiple backtest paths.

    Yields
    ------
    (train_idx, test_idx) : tuple[np.ndarray, np.ndarray]
    """
    n = len(t1)
    indices = np.arange(n)
    groups = np.array_split(indices, n_groups)
    embargo = int(n * embargo_pct)
    for combo in itertools.combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in combo])
        test_idx.sort()
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        # purge: drop train obs whose label window overlaps any test obs window
        test_start, test_end = test_idx.min(), test_idx.max()
        for i in np.where(train_mask)[0]:
            # crude index-overlap purge; refine with real timestamps in your project
            if test_start <= i <= test_end:
                train_mask[i] = False
        # embargo after each contiguous test block
        if embargo > 0:
            train_mask[test_end + 1 : test_end + 1 + embargo] = False
        yield indices[train_mask], test_idx


# --------------------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------------------
def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, n_partitions: int = 16
) -> dict:
    """Estimate PBO via Combinatorially-Symmetric Cross-Validation.

    Parameters
    ----------
    returns_matrix : np.ndarray, shape (T, N)
        T time observations of per-period returns for N candidate strategies/configurations.
    n_partitions : int
        Even number S of disjoint time partitions. Forms C(S, S/2) train/test combinations.

    Returns
    -------
    dict with keys:
        'pbo'    : probability the in-sample-best strategy underperforms the median out-of-sample.
        'logits' : array of logit values (one per combination).
    """
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    if n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even")
    rows = np.array_split(np.arange(T), n_partitions)
    logits = []
    for combo in itertools.combinations(range(n_partitions), n_partitions // 2):
        is_rows = np.concatenate([rows[i] for i in combo])
        oos_rows = np.concatenate([rows[i] for i in range(n_partitions) if i not in combo])
        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)
        best_is = int(np.argmax(is_perf))
        # out-of-sample rank (1..N) of the in-sample-best strategy
        rank_oos = (np.argsort(np.argsort(oos_perf))[best_is] + 1) / (N + 1)
        rank_oos = min(max(rank_oos, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank_oos / (1.0 - rank_oos)))
    logits = np.asarray(logits)
    pbo = float((logits < 0).mean())
    return {"pbo": pbo, "logits": logits}


# --------------------------------------------------------------------------------------
# Cost-aware backtest
# --------------------------------------------------------------------------------------
def cost_aware_backtest(
    weights: pd.DataFrame, asset_returns: pd.DataFrame, cost_bps: float = 5.0
) -> dict:
    """Vectorized long/short backtest with turnover-based transaction costs.

    Parameters
    ----------
    weights : pd.DataFrame (dates x assets)
        Target portfolio weights known at the OPEN of each date (already lagged to avoid look-ahead).
    asset_returns : pd.DataFrame (dates x assets)
        Per-period asset returns realized over each date.
    cost_bps : float
        Round-trip-equivalent cost per unit turnover, in basis points (e.g., 5 bps).

    Returns
    -------
    dict with gross/net return series, turnover series, and summary Sharpe stats (net).
    """
    w = weights.reindex_like(asset_returns).fillna(0.0)
    gross = (w * asset_returns).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    cost = turnover * (cost_bps * 1e-4)
    net = gross - cost
    stats = sharpe_stats(net.values)
    return {
        "gross_returns": gross,
        "net_returns": net,
        "turnover": turnover,
        "net_sharpe_stats": stats,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, size=1000)
    s = sharpe_stats(r)
    print("PSR:", probabilistic_sharpe_ratio(s["sr"], s["n_obs"], s["skew"], s["kurt"]))
    print("DSR (50 trials):", deflated_sharpe_ratio(s["sr"], s["n_obs"], s["skew"], s["kurt"], 50, 0.01))
