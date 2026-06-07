# Project 4 — Point-in-Time NLP/LLM Textual Alpha (RANK #1)

**Thesis.** *How much of the apparent return predictability from LLM analysis of corporate disclosures
survives a rigorous removal of look-ahead bias and memorization?*

The contribution is **not** "an LLM predicts returns" (already shown by Lopez-Lira & Tang, 2023). It is a
clean, reproducible, point-in-time-correct benchmark that **decomposes measured alpha** into (a) genuine
real-time textual signal vs. (b) leakage/memorization — across a ladder of methods (dictionary → fine-tuned
encoder → instruction-tuned LLM) and a ladder of leakage controls.

## Why this is the #1 project for you
- Sits on the hottest 2024–2026 hiring theme (LLM/NLP "textual alpha").
- Uses your strengths: ML/AI research, synthetic data, Python.
- Runs entirely on **free** data (EDGAR, FinBERT, ChronoBERT) + your OpenAI API access.
- Engages a *live, unsettled* research question, so an honest result is genuinely differentiating.

## What "done" looks like
A repo whose README states an honest headline like:
> "Real-time long/short net Sharpe 0.6 (Deflated Sharpe 0.4 over 60 trials); naive (leaky) Sharpe 1.9.
> ~70% of the naive signal is attributable to look-ahead/memorization. Signal concentrates in small-cap,
> high-uncertainty filings and decays after T+2. Costs 10 bps; universe = S&P 1500 as-of each date; no CRSP."

## Layout
```
project4_textual_alpha/
├── IMPLEMENTATION_LOGIC.md   # read this first
├── DATA.md                   # free data sources + the acceptanceDateTime rule
├── LLM_PROMPTS.md            # ordered prompts to build each component
├── requirements.txt
├── configs/config.yaml
├── src/
│   ├── ingest/edgar.py       # IMPLEMENTED: rate-limited, acceptanceDateTime-gated EDGAR client
│   ├── pit/leakage.py        # IMPLEMENTED: point-in-time guards + leakage assertions
│   ├── features/sentiment.py # IMPLEMENTED interfaces: LM dict, FinBERT, ChronoBERT scorers
│   ├── llm/openai_signal.py  # IMPLEMENTED: cached OpenAI scorer w/ anonymization + cutoff guard
│   ├── backtest/             # TODO via prompts: cost-aware portfolio formation (use shared/validation.py)
│   └── eval/                 # TODO via prompts: memorization probes, ablations, plots
├── tests/test_lookahead.py   # IMPLEMENTED: example leakage tests
└── .github/workflows/ci.yml
```

## Run (once built)
```bash
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your@email.com"   # required by SEC
export OPENAI_API_KEY=...                           # for the generative-LLM signal
python -m src.ingest.edgar --cik 0000320193 --forms 10-K   # example: Apple 10-Ks
```

## Data caveats (disclosed, not hidden)
- **Survivorship bias.** `src/ingest/universe.py` separates two regimes. A *PIT-accurate* `UniverseTable`
  (built from a historical membership CSV with `start_date`/`end_date` per name) gives true point-in-time
  constituents. The bundled `fetch_current_sp1500()` returns **today's** constituents only — its
  `members_as_of(date)` is identical for every date and raises a `SurvivorshipWarning`. Backtests run on a
  current snapshot are upward-biased; supply a historical CSV for an honest universe, or report results as
  survivorship-uncorrected.
- **No CRSP.** Prices come from yfinance (Stooq fallback); the Stooq fallback is **not** split/dividend
  adjusted. Forward returns enter at the next session's open (`src/ingest/prices.py`) to avoid look-ahead.
