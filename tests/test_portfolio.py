"""Tests for quantile long/short portfolio formation and the cost-aware backtest (no network)."""
import numpy as np
import pandas as pd
import pytest
from src.backtest.portfolio import (
    backtest_signal,
    quantile_long_short_weights,
    rebalance_dates,
)


def _panel(seed=0, n_days=30, n_assets=10):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_assets)]
    signals = pd.DataFrame(rng.normal(size=(n_days, n_assets)), index=idx, columns=cols)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(n_days, n_assets)), index=idx, columns=cols)
    return signals, returns


def test_weights_are_dollar_neutral_and_quantile_split():
    # 10 distinct signals, quintiles -> long top 2, short bottom 2.
    idx = pd.bdate_range("2020-01-01", periods=1)
    row = pd.DataFrame([list(range(10))], index=idx, columns=[f"S{i}" for i in range(10)])
    w = quantile_long_short_weights(row, quantiles=5).iloc[0]
    assert w.sum() == pytest.approx(0.0)            # dollar neutral
    assert (w > 0).sum() == 2 and (w < 0).sum() == 2  # top/bottom quintile each = 2 names
    assert w["S9"] == pytest.approx(0.5) and w["S8"] == pytest.approx(0.5)   # highest signals long
    assert w["S0"] == pytest.approx(-0.5) and w["S1"] == pytest.approx(-0.5)  # lowest signals short
    assert w.abs().sum() == pytest.approx(2.0)      # gross exposure ~2


def test_too_few_names_gives_zero_weights():
    idx = pd.bdate_range("2020-01-01", periods=1)
    row = pd.DataFrame([[1.0, 2.0, np.nan]], index=idx, columns=["A", "B", "C"])
    w = quantile_long_short_weights(row, quantiles=5).iloc[0]
    assert (w == 0).all()


def test_every_row_sums_to_zero_on_full_panel():
    signals, _ = _panel()
    w = quantile_long_short_weights(signals, quantiles=5)
    assert np.allclose(w.sum(axis=1).values, 0.0, atol=1e-12)


def test_weekly_rebalance_dates():
    # 2020-01-01 is a Wednesday; 10 business days span three ISO weeks (W01: 1-3, W02: 6-10, W03: 13-14).
    idx = pd.bdate_range("2020-01-01", periods=10)
    reb = rebalance_dates(idx, "weekly")
    assert list(reb) == [idx[0], idx[3], idx[8]]
    assert list(rebalance_dates(idx, "daily")) == list(idx)


def test_backtest_lags_weights_and_reports_turnover():
    signals, returns = _panel()
    res = backtest_signal(signals, returns, quantiles=5, cost_bps=10.0, rebalance="weekly")
    # Lag means no position is held on the very first session.
    assert (res["weights"].iloc[0] == 0).all()
    # Turnover and net returns are produced and finite.
    assert "turnover" in res and "net_returns" in res
    assert np.isfinite(res["turnover"]).all()
    assert res["net_returns"].shape[0] == returns.shape[0]
    # Net <= gross because costs are non-negative.
    assert (res["net_returns"] <= res["gross_returns"] + 1e-12).all()
