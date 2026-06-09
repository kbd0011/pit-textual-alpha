"""Real end-to-end textual-alpha pipeline on free data (SEC EDGAR + yfinance + FinBERT).

Demonstrates the **leakage decomposition** on real filings: it scores each company's annual-report tone with
FinBERT, then forms the daily cross-sectional signal twice — once **naively** (treating a filing as knowable on
its period-end date, L0) and once **dissemination-gated** (knowable only at `acceptanceDateTime`, L1) — and
runs the cost-aware CPCV ablation. The L0−L1 gap is the look-ahead the naive setup injects.

This is a deliberately small free-data demonstration (a handful of large caps, annual 10-Ks): the signal is
weak and the cross-section is thin, so treat the numbers as an honest illustration of the *method*, not a
tradable result. Scale the universe (S&P 1500 via a PIT membership CSV) for a real study.

Usage:  SEC_USER_AGENT="Name email@x.com" python scripts/run_pipeline.py
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from src.eval.ablations import AblationConfig, plot_ablation, run_ablation
from src.features.sentiment import FinBERTScorer, split_sentences
from src.ingest.edgar import EdgarClient, extract_mdna
from src.ingest.prices import forward_returns_from_prices

# Small, sector-spread large-cap universe (ticker -> CIK).
UNIVERSE = {
    "AAPL": 320193, "MSFT": 789019, "JNJ": 200406, "JPM": 19617, "XOM": 34088,
    "PG": 80424, "KO": 21344, "INTC": 50863, "WMT": 104169, "CAT": 18230,
}
START, END = "2017-01-01", "2023-12-31"
MAX_SENTENCES = 30   # cap FinBERT work per filing for CPU speed
HORIZON = 5          # trading-day forward-return horizon


def collect_filing_scores(scorer: FinBERTScorer) -> pd.DataFrame:
    """Pull 10-Ks, extract MD&A/risk text, FinBERT-score a truncated slice. Returns one row per filing."""
    client = EdgarClient(user_agent=os.environ["SEC_USER_AGENT"])
    rows = []
    for ticker, cik in UNIVERSE.items():
        for f in client.iter_filings(cik, forms={"10-K"}):
            yr = int(f.filing_date[:4])
            if yr < int(START[:4]) or yr > int(END[:4]):
                continue
            try:
                text = extract_mdna(client.fetch_document(f))
            except Exception:  # noqa: BLE001 — skip filings whose section can't be located
                continue
            snippet = " ".join(split_sentences(text)[:MAX_SENTENCES])
            score = scorer.score(snippet, f.dissemination_time)
            rows.append({
                "ticker": ticker,
                "period_date": pd.Timestamp(f.report_date) if f.report_date else pd.Timestamp(f.filing_date),
                "disseminated": pd.Timestamp(f.dissemination_time).tz_localize(None),
                "score": score,
            })
            print(f"  {ticker} {f.filing_date} score={score:+.3f}")
    return pd.DataFrame(rows)


def build_panel(filings: pd.DataFrame, dates: pd.DatetimeIndex, time_col: str) -> pd.DataFrame:
    """Daily signal panel: each (date, ticker) gets the most recent filing knowable as of `time_col`."""
    panel = pd.DataFrame(index=dates, columns=sorted(UNIVERSE), dtype=float)
    for ticker, g in filings.groupby("ticker"):
        g = g.sort_values(time_col)
        knowable_time = g[time_col].to_numpy()
        scores = g["score"].to_numpy()
        idx = np.searchsorted(knowable_time, dates.to_numpy(), side="right") - 1
        panel[ticker] = np.where(idx >= 0, scores[idx.clip(min=0)], np.nan)
    return panel


def main() -> None:
    if "@" not in os.environ.get("SEC_USER_AGENT", ""):
        raise SystemExit("Set SEC_USER_AGENT='Your Name your@email.com' (SEC requires it).")

    import yfinance as yf

    print("1/4 scoring filings with FinBERT (real EDGAR text)...")
    filings = collect_filing_scores(FinBERTScorer())
    print(f"  -> {len(filings)} filings scored across {filings['ticker'].nunique()} names")

    print("2/4 downloading prices + forward returns...")
    raw = yf.download(list(UNIVERSE), start=START, end=END, auto_adjust=True, progress=False)["Open"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    fwd = forward_returns_from_prices(raw.reindex(columns=sorted(UNIVERSE)), HORIZON)
    dates = fwd.index

    print("3/4 building L0 (naive period-dated) and L1 (dissemination-gated) signal panels...")
    panels = {
        ("finbert", "L0_naive"): build_panel(filings, dates, "period_date"),
        ("finbert", "L1_timestamp"): build_panel(filings, dates, "disseminated"),
    }

    print("4/4 running cost-aware CPCV ablation...")
    cfg = AblationConfig(quantiles=3, cost_bps=10.0, rebalance="weekly", n_trials_for_dsr=8,
                         cpcv_groups=6, cpcv_k_test=2, pbo_partitions=6)
    res = run_ablation(panels, fwd, cfg)
    table = res["table"]
    print("\n" + table.round(4).to_string(index=False))
    print(f"PBO across cells: {res['pbo']:.3f}")

    naive = table.loc[table["leakage"] == "L0_naive", "net_sharpe_ann"].iloc[0]
    gated = table.loc[table["leakage"] == "L1_timestamp", "net_sharpe_ann"].iloc[0]
    print(f"\nNaive (L0) net Sharpe {naive:.2f}  vs  dissemination-gated (L1) {gated:.2f}  "
          f"-> look-ahead gap {naive - gated:+.2f}")

    decay = pd.Series({HORIZON: gated}, name="net_sharpe_ann").rename_axis("horizon_days")
    from pathlib import Path
    Path("assets").mkdir(exist_ok=True)
    plot_ablation(table, decay, "assets/leakage_decomposition.png")
    print("figure -> assets/leakage_decomposition.png")
    print(f"\nrun at {datetime.now().isoformat(timespec='seconds')}; FREE-DATA DEMONSTRATION — small universe.")


if __name__ == "__main__":
    main()
