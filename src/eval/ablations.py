"""Ablation harness: sweep methods × leakage levels, evaluate with multiple-testing-honest statistics.

This is where the headline result is produced. For each (method, leakage-level) cell we form a quantile L/S
portfolio, run the cost-aware backtest, and report:

- **Net Sharpe** (per-period and annualized).
- **Deflated Sharpe Ratio** against the *expected maximum* Sharpe over every configuration tried
  (selection-bias correction; ``n_trials_for_dsr`` from config).
- **CPCV robustness**: mean/std of out-of-sample Sharpe across combinatorially-purged test folds.
- **PBO**: a single Probability of Backtest Overfitting across all cells (CSCV).
- **Decay by horizon**: how net Sharpe falls as the forecast horizon lengthens.

The harness takes *precomputed* signal panels (one per cell) so it stays decoupled from the heavy scorers and
runs end-to-end on synthetic data with no network or model downloads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shared.validation import (
    combinatorial_purged_splits,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_stats,
)

from src.backtest.portfolio import backtest_signal

__all__ = ["AblationConfig", "run_ablation", "decay_by_horizon", "plot_ablation"]


@dataclass
class AblationConfig:
    """Configuration for the ablation sweep (mirrors configs/config.yaml)."""

    quantiles: int = 5
    cost_bps: float = 10.0
    rebalance: str = "weekly"
    n_trials_for_dsr: int = 60
    cpcv_groups: int = 8
    cpcv_k_test: int = 2
    embargo_pct: float = 0.01
    pbo_partitions: int = 8
    periods_per_year: int = 252


def _cell_net_returns(signals: pd.DataFrame, asset_returns: pd.DataFrame, cfg: AblationConfig) -> pd.Series:
    res = backtest_signal(
        signals, asset_returns,
        quantiles=cfg.quantiles, cost_bps=cfg.cost_bps, rebalance=cfg.rebalance,
    )
    return res["net_returns"]


def _cpcv_sharpe_distribution(net_returns: pd.Series, cfg: AblationConfig) -> tuple[float, float]:
    """Mean and std of per-test-fold Sharpe across combinatorially-purged CV splits."""
    r = net_returns.to_numpy(dtype=float)
    n = r.size
    if n < cfg.cpcv_groups:
        return float("nan"), float("nan")
    t1 = pd.Series(np.arange(n), index=np.arange(n))
    sharpes = []
    for _train, test in combinatorial_purged_splits(cfg.cpcv_groups, cfg.cpcv_k_test, t1, cfg.embargo_pct):
        seg = r[test]
        seg = seg[~np.isnan(seg)]
        sd = seg.std(ddof=1) if seg.size > 2 else 0.0
        if sd > 0:
            sharpes.append(seg.mean() / sd)
    if not sharpes:
        return float("nan"), float("nan")
    arr = np.asarray(sharpes)
    return float(arr.mean()), float(arr.std())


def _pbo_across_cells(returns_matrix: pd.DataFrame, cfg: AblationConfig) -> float:
    """PBO over the per-cell net-return columns; clamps partitions to fit short samples."""
    n_rows, n_cols = returns_matrix.shape
    if n_cols < 2 or n_rows < 4:
        return float("nan")
    parts = min(cfg.pbo_partitions, n_rows)
    if parts % 2 != 0:
        parts -= 1
    if parts < 2:
        return float("nan")
    return float(probability_of_backtest_overfitting(returns_matrix.to_numpy(), n_partitions=parts)["pbo"])


def run_ablation(
    signal_panels: dict[tuple[str, str], pd.DataFrame],
    asset_returns: pd.DataFrame,
    cfg: AblationConfig | None = None,
) -> dict:
    """Run the full sweep and return a results table plus a single PBO for the experiment.

    Parameters
    ----------
    signal_panels : dict[(method, leakage_level) -> pd.DataFrame]
        One dates x tickers signal panel per cell.
    asset_returns : pd.DataFrame
        Dates x tickers per-period returns.
    cfg : AblationConfig, optional

    Returns
    -------
    dict
        ``table`` (DataFrame: method, leakage, n_obs, net_sharpe, net_sharpe_ann, dsr, cpcv_sharpe_mean/std),
        ``pbo`` (float), ``sr_variance`` (cross-trial Sharpe variance), ``n_trials`` (used for DSR).
    """
    cfg = cfg or AblationConfig()
    if not signal_panels:
        raise ValueError("signal_panels is empty")

    net = {cell: _cell_net_returns(sig, asset_returns, cfg) for cell, sig in signal_panels.items()}
    stats = {cell: sharpe_stats(s.dropna().to_numpy()) for cell, s in net.items()}

    sr_list = [st["sr"] for st in stats.values()]
    sr_variance = float(np.var(sr_list, ddof=1)) if len(sr_list) > 1 else 0.0
    n_trials = max(cfg.n_trials_for_dsr, len(signal_panels))
    ann = math.sqrt(cfg.periods_per_year)

    rows = []
    for (method, leakage), st in stats.items():
        dsr = (
            deflated_sharpe_ratio(st["sr"], st["n_obs"], st["skew"], st["kurt"], n_trials, sr_variance)
            if sr_variance > 0 else float("nan")
        )
        cpcv_mean, cpcv_std = _cpcv_sharpe_distribution(net[(method, leakage)], cfg)
        rows.append({
            "method": method,
            "leakage": leakage,
            "n_obs": st["n_obs"],
            "net_sharpe": st["sr"],
            "net_sharpe_ann": st["sr"] * ann,
            "dsr": dsr,
            "cpcv_sharpe_mean": cpcv_mean,
            "cpcv_sharpe_std": cpcv_std,
        })
    table = pd.DataFrame(rows)

    matrix = pd.DataFrame({f"{m}|{lv}": s for (m, lv), s in net.items()}).dropna(how="any")
    pbo = _pbo_across_cells(matrix, cfg)
    return {"table": table, "pbo": pbo, "sr_variance": sr_variance, "n_trials": n_trials}


def decay_by_horizon(
    signal_panel: pd.DataFrame,
    returns_by_horizon: dict[int, pd.DataFrame],
    cfg: AblationConfig | None = None,
) -> pd.Series:
    """Annualized net Sharpe of one signal across forecast horizons (the decay curve)."""
    cfg = cfg or AblationConfig()
    ann = math.sqrt(cfg.periods_per_year)
    out = {}
    for horizon, ret in sorted(returns_by_horizon.items()):
        st = sharpe_stats(_cell_net_returns(signal_panel, ret, cfg).dropna().to_numpy())
        out[horizon] = st["sr"] * ann
    return pd.Series(out, name="net_sharpe_ann").rename_axis("horizon_days")


def plot_ablation(table: pd.DataFrame, decay: pd.Series, out_path: str) -> str:
    """Save a two-panel figure (method×leakage Sharpe heatmap + decay curve) to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    pivot = table.pivot(index="method", columns="leakage", values="net_sharpe_ann")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    axes[0].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(pivot.index)), pivot.index)
    axes[0].set_title("Net Sharpe (ann) by method × leakage control")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].plot(decay.index.to_numpy(), decay.to_numpy(), marker="o")
    axes[1].axhline(0.0, color="grey", lw=0.8, ls="--")
    axes[1].set_xlabel("horizon (trading days)")
    axes[1].set_ylabel("net Sharpe (annualized)")
    axes[1].set_title("Signal decay by horizon")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
