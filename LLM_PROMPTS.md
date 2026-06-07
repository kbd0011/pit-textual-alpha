# LLM Prompts — Project 4 (Textual Alpha)

Use the SYSTEM prompt from `../LLM_PROMPTING_GUIDE.md` once per session. Then run these TASK prompts in order.
Paste the cited section of `IMPLEMENTATION_LOGIC.md` and any required interfaces as context.

---
### Prompt 1 — prices + forward returns (`src/ingest/prices.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §3–4. INTERFACES: produce a function
> `forward_returns(tickers, start, end, horizon_days) -> pd.DataFrame (dates x tickers)`.
> TASK: implement `src/ingest/prices.py` that downloads daily adjusted closes via yfinance (fallback Stooq),
> aligns to the NYSE calendar (pandas-market-calendars), and computes h-day forward returns **lagged so the
> signal at t maps to the return from next open to t+h**. CONSTRAINTS: libraries = yfinance, pandas,
> pandas-market-calendars, numpy. No look-ahead: returns for date t must start at t+1 open.
> ACCEPTANCE: implementation + pytest using a tiny synthetic price frame (no network) asserting the lag is correct.

### Prompt 2 — universe membership, point-in-time (`src/ingest/universe.py`)
> TASK: build a point-in-time S&P 1500 membership table loader (free source or a static historical CSV you
> provide), exposing `members_as_of(date) -> list[str]`. CONSTRAINTS: explicitly DOCUMENT the survivorship
> limitation if only current membership is available. ACCEPTANCE: code + test + a README note on the caveat.

### Prompt 3 — MD&A extraction (`extract_mdna` in `src/ingest/edgar.py`)
> CONTEXT: the `extract_mdna` stub + its docstring strategy. TASK: implement robust Item 7 / "Management's
> Discussion and Analysis" extraction from 10-K/10-Q HTML using BeautifulSoup: strip tags/tables/exhibits,
> locate the section between Item 7 and Item 7A/Item 8, normalize whitespace, fallback to Item 1A risk factors.
> CONSTRAINTS: beautifulsoup4 + lxml + re only. ACCEPTANCE: code + tests on 2–3 saved HTML fixtures (committed
> to tests/fixtures/) covering different layouts.

### Prompt 4 — Loughran-McDonald loader (`LoughranMcDonaldScorer.from_master_csv`)
> TASK: parse the official LM master dictionary CSV into positive/negative/uncertainty word sets and wire up
> `from_master_csv`. CONSTRAINTS: pandas + stdlib. ACCEPTANCE: code + test against a 10-row fixture CSV; verify
> a known positive and negative word land in the right sets.

### Prompt 5 — FinBERT scoring (`FinBERTScorer.score`)
> CONTEXT: the FinBERT stub. TASK: implement sentence splitting, batched inference with the FinBERT pipeline,
> map {positive, negative, neutral} -> net tone in [-1, 1], aggregate by mean. CONSTRAINTS: transformers, torch,
> nltk or a regex splitter. The unit test must MOCK the pipeline (no model download). ACCEPTANCE: code + mocked test.

### Prompt 6 — ChronoBERT vintage control (`ChronoBERTScorer`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §2 (L3) + the stub. TASK: implement `pick_vintage(as_of)` (newest cutoff
> <= as_of, else raise) and `score` using the chosen vintage's embeddings fed to a lightweight probe head that
> is FIT ONLY on pre-as_of labels (pass the fitted head in). CONSTRAINTS: transformers, torch, scikit-learn.
> ACCEPTANCE: code + test for vintage selection logic (mock the model); assert it refuses when no vintage qualifies.

### Prompt 7 — OpenAI signal (`OpenAISignal.score`)
> CONTEXT: the pseudocode in `OpenAISignal.score`. TASK: implement the cached API call + robust numeric parsing
> + the cutoff guard + anonymization hook. CONSTRAINTS: openai>=1.40, stdlib. temperature=0. The unit test must
> MOCK the OpenAI client and verify: (a) cutoff guard raises for post-cutoff as_of when not allowed, (b) caching
> avoids a second API call, (c) parsing handles "-0.3 because ...". ACCEPTANCE: code + mocked tests.

### Prompt 8 — portfolio formation (`src/backtest/portfolio.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §3 + `../../shared/validation.py` (`cost_aware_backtest`). TASK: from a panel
> of signals s_{i,t}, form weekly quantile long/short weights (long top quintile, short bottom), lag to next
> open, and call `cost_aware_backtest`. CONSTRAINTS: pandas, numpy, import from shared.validation. No look-ahead.
> ACCEPTANCE: code + test on synthetic signals asserting weights sum to ~0 and turnover is computed.

### Prompt 9 — memorization probe (`src/eval/memorization.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §2 (L4). TASK: implement the no-context recall probe — score with entity
> identifiers but with the real disclosure text removed/blanked — and a function that subtracts probe
> performance from full performance to yield a memorization-adjusted signal. ACCEPTANCE: code + test.

### Prompt 10 — ablation harness (`src/eval/ablations.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §5 + `shared/validation.py` (CPCV, DSR, PBO). TASK: sweep methods × leakage
> levels, run the cost-aware backtest under CPCV, and emit a results table with net Sharpe, Deflated Sharpe
> (use config `n_trials_for_dsr`), PBO, and a decay-by-horizon curve. ACCEPTANCE: code + a small end-to-end test
> on synthetic data + a Matplotlib figure saved to reports/.

### Prompt 11 — README results paragraph
> TASK: given my final results table, draft the honest README headline paragraph per `shared/RIGOR.md` ("How to
> report results"), stating Sharpe with and without leakage controls, DSR, costs, universe, and decay. Do NOT
> inflate; if the controlled signal is weak, say so and frame the contribution as the leakage decomposition.
