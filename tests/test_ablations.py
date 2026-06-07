"""End-to-end ablation harness test on synthetic data (no network, no models)."""
import numpy as np
import pandas as pd
from src.eval.ablations import AblationConfig, decay_by_horizon, plot_ablation, run_ablation


def _synthetic(seed=1, n_days=160, n_assets=12):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"A{i}" for i in range(n_assets)]
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(n_days, n_assets)), index=idx, columns=cols)
    # One cell's signal is mildly predictive (correlated with next-period returns); others are noise.
    informed = returns.shift(-1).fillna(0.0) + rng.normal(0, 0.02, size=(n_days, n_assets))
    noise = pd.DataFrame(rng.normal(size=(n_days, n_assets)), index=idx, columns=cols)
    panels = {
        ("finbert", "L1_timestamp"): informed,
        ("finbert", "L0_naive"): informed + rng.normal(0, 0.01, size=(n_days, n_assets)),
        ("lm_dictionary", "L1_timestamp"): noise,
        ("lm_dictionary", "L0_naive"): noise + rng.normal(0, 0.01, size=(n_days, n_assets)),
    }
    return panels, returns


def test_run_ablation_produces_full_table_and_pbo():
    panels, returns = _synthetic()
    cfg = AblationConfig(n_trials_for_dsr=60)
    res = run_ablation(panels, returns, cfg)
    table = res["table"]

    assert len(table) == 4
    for col in ["method", "leakage", "n_obs", "net_sharpe", "net_sharpe_ann", "dsr",
                "cpcv_sharpe_mean", "cpcv_sharpe_std"]:
        assert col in table.columns
    # DSR is a probability in [0, 1] (or NaN if degenerate); sr_variance > 0 with differing cells.
    assert res["sr_variance"] > 0
    dsr = table["dsr"].dropna()
    assert ((dsr >= 0) & (dsr <= 1)).all()
    # n_trials honestly counts at least every configuration tried.
    assert res["n_trials"] >= len(panels)
    # PBO is a probability in [0, 1].
    assert np.isnan(res["pbo"]) or (0.0 <= res["pbo"] <= 1.0)


def test_informed_signal_beats_noise_on_average():
    panels, returns = _synthetic()
    res = run_ablation(panels, returns)
    t = res["table"].set_index(["method", "leakage"])
    finbert = t.loc[("finbert", "L1_timestamp"), "net_sharpe"]
    lm = t.loc[("lm_dictionary", "L1_timestamp"), "net_sharpe"]
    assert finbert > lm  # the predictive signal should out-Sharpe pure noise


def test_decay_curve_and_plot(tmp_path):
    panels, _ = _synthetic()
    rng = np.random.default_rng(2)
    idx = panels[("finbert", "L1_timestamp")].index
    cols = panels[("finbert", "L1_timestamp")].columns
    returns_by_h = {
        h: pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), len(cols))), index=idx, columns=cols)
        for h in (1, 3, 5, 10)
    }
    decay = decay_by_horizon(panels[("finbert", "L1_timestamp")], returns_by_h)
    assert list(decay.index) == [1, 3, 5, 10]
    assert decay.notna().all()

    res = run_ablation(panels, panels[("finbert", "L1_timestamp")] * 0 + 0.001)  # trivial returns ok for plot
    out = plot_ablation(res["table"], decay, str(tmp_path / "reports" / "ablation.png"))
    assert (tmp_path / "reports" / "ablation.png").exists()
    assert (tmp_path / "reports" / "ablation.png").stat().st_size > 0
    assert out.endswith("ablation.png")
