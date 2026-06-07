# The Rigor Doctrine

Every project in this kit lives or dies on rigor. This file defines the shared standard and explains the
reusable utilities in `shared/validation.py`.

## Non-negotiables

1. **Transaction costs.** No backtest is reported gross-only. Use a turnover-based cost model and report both
   gross and net. A signal that dies after costs is a *finding*, not a failure to hide.
2. **Multiple-testing correction.** If you tried N configurations, the best raw Sharpe is upward-biased. Report
   the **Deflated Sharpe Ratio (DSR)** (Bailey & López de Prado, 2014) and the **Probability of Backtest
   Overfitting (PBO)** (Bailey et al., 2016, via CSCV).
3. **Leakage-safe validation.** Use **purged K-fold with embargo** and **combinatorial purged cross-validation
   (CPCV)** when labels span time (López de Prado, *Advances in Financial Machine Learning*, 2018).
4. **Look-ahead control.** Features known at time t must use only data with timestamp <= t. For text, gate on
   the *dissemination* timestamp, not the period covered. For LLMs, beware training-cutoff leakage/memorization.
5. **Survivorship awareness.** Free price data usually reflects today's surviving universe. Disclose it.
6. **Ablations + honest negatives.** Show what each component contributes; report where the edge disappears.
7. **Reproducibility.** Seed everything; config-drive everything; pin dependencies; provide a one-command repro.

## What `shared/validation.py` gives you

- `probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_benchmark=0.0)` — PSR: probability the true Sharpe
  exceeds a benchmark, correcting for sample length, skew and kurtosis.
- `deflated_sharpe_ratio(sr, n_obs, skew, kurt, n_trials, sr_variance)` — DSR: PSR against the *expected
  maximum* Sharpe under `n_trials`, i.e. corrects for selection bias.
- `min_track_record_length(sr, skew, kurt, sr_benchmark=0.0, prob=0.95)` — minimum sample size to be
  confident a Sharpe is above a benchmark.
- `PurgedKFold(n_splits, t1, embargo_pct)` — sklearn-style splitter that purges training labels overlapping
  the test window and embargoes a fraction afterwards.
- `combinatorial_purged_splits(n_groups, k_test, t1, embargo_pct)` — CPCV: yields many backtest paths.
- `probability_of_backtest_overfitting(returns_matrix, n_partitions)` — PBO via CSCV.
- `cost_aware_backtest(weights, asset_returns, cost_bps)` — net/gross returns, turnover, Sharpe.

## How to report results in your README

State the headline as: "Net Sharpe X.XX (Deflated Sharpe Y.YY over N trials); gross Z.ZZ. Signal concentrated
in <regime>; decays to ~0 after <condition>. Costs assumed <bps>. Universe <described>; no survivorship
correction." That sentence alone signals senior judgment.
