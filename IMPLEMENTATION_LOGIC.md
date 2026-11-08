# Implementation Logic — Point-in-Time Textual Alpha

## 1. The research question, precisely
Define a daily cross-sectional signal s_{i,t} for stock i from the text of its most recent disclosure
*knowable* at t. Predict forward returns r_{i,t -> t+h}. Estimate the long/short performance of the signal
**twice**: once naively, once under strict leakage controls. The headline result is the *difference*.

## 2. The two ladders

**Method ladder (increasing capability):**
1. Loughran-McDonald (2011) finance sentiment dictionary — count positive/negative tokens (the baseline).
2. FinBERT (Huang/Yang et al.) — fine-tuned encoder tone at sentence level, aggregated.
3. Instruction-tuned LLM (OpenAI API) — prompt the model for a forward-looking sentiment/stance score.

**Leakage-control ladder (increasing strictness):**
- L0 Naive: score any text with a modern model, ignore timestamps. (Demonstrates the trap.)
- L1 Timestamp gating: only use text with dissemination time <= t (the `acceptanceDateTime` rule).
- L2 Entity anonymization: mask company names/tickers/dates so the model cannot recall "what happened to X".
- L3 Chronological consistency: use a model whose training data ends before t (ChronoBERT/ChronoGPT vintages).
- L4 Memorization probe: no-context recall test — ask the model to "predict" with the entity but no real text;
  any predictive power here is pure memorization and must be subtracted.

## 3. Data flow
```
EDGAR submissions API ──> filing index (with acceptanceDateTime)
                               │  gate on dissemination time
                               ▼
        primary document  ──> MD&A / risk-factor extraction ──> cleaned text
                               │
   prices (Stooq/yfinance) ────┤
                               ▼
   features/sentiment.py  ──> s_{i,t} per method × per leakage level
                               │
   backtest/ (shared/validation.py: cost_aware_backtest, CPCV, DSR/PBO)
                               ▼
   eval/ ──> ablation table, memorization-adjusted Sharpe, decay curves, plots
```

## 4. Module responsibilities
- `ingest/edgar.py` — fetch submissions, enumerate filings, attach `acceptanceDateTime`, compute dissemination
  time (next business day if accepted after 17:30 ET; 22:00 ET for ownership forms), download primary docs.
- `ingest/transcripts.py` (planned) — load earnings-call transcripts from a free HF dataset; attach call datetime.
- `ingest/prices.py` — daily prices from yfinance (Stooq fallback); align to trading calendar; compute forward returns.
- `pit/leakage.py` — `latest_knowable(items, as_of)`, `assert_no_lookahead(info_time, as_of)`, a `PITStore` helper.
- `features/sentiment.py` — `LoughranMcDonaldScorer`, `FinBERTScorer`, `ChronoBERTScorer`; common `.score(text, as_of)`.
- `llm/openai_signal.py` — cached OpenAI scorer with anonymization and a knowledge-cutoff guard.
- `backtest/portfolio.py` — form quantile long/short weights from s_{i,t}, lag to next open, call `cost_aware_backtest`.
- `eval/memorization.py` — L4 probe; compute memorization-adjusted performance.
- `eval/ablations.py` — sweep method × leakage level; produce the headline table with DSR + PBO.
- `scripts/run_pipeline.py` — the end-to-end EDGAR run: FinBERT at L0 and L1 on a 10-name universe (see README).

## 5. Methodology
- Gate on `acceptanceDateTime`; `tests/test_lookahead.py` fails if a future timestamp is swapped in.
- Anonymize entities to separate sentiment from memorized associations.
- Use ChronoBERT/ChronoGPT as a leakage-free control and reproduce the "is look-ahead bias modest?" question.
- Cost-aware backtest; CPCV with purge+embargo; Deflated Sharpe Ratio + Probability of Backtest Overfitting.
- Report honest negatives and decay curves.

## 6. Pitfalls
- Using `filingDate` (date only) or `periodOfReport` instead of `acceptanceDateTime` (intraday leakage).
- Scoring text with an LLM whose training window covers the period (memorization).
- Survivorship bias from a current-constituents universe.
- Ignoring the after-hours dissemination cutoff.
- Costs that make the signal vanish; multiple-testing inflation; treating the OpenAI cutoff as a clean boundary.
