# Data Sources — Project 4 (all FREE)

## SEC EDGAR (primary text source)
- **No API key, no auth.** A `User-Agent` header is REQUIRED (format: "Company Name contact@email.com").
- Rate limit: no more than **10 requests/second** (exceeding returns HTTP 403 + temporary IP block).
- Submissions metadata: `https://data.sec.gov/submissions/CIK##########.json` (10-digit zero-padded CIK).
- Full-text search (2001–present): `https://efts.sec.gov/LATEST/search-index?q=...`
- Structured form indexes back to 1994Q3; bulk files (`submissions.zip`, `companyfacts.zip`, DERA datasets) are free.

### THE point-in-time rule
Gate every text on **`acceptanceDateTime`** (the timestamp EDGAR accepted/disseminated the filing),
NOT `filingDate` (date only) and NOT `reportDate`/`periodOfReport` (the period covered).
Filings accepted after **17:30 ET** (ownership Forms 3/4/5: **22:00 ET**) are disseminated the **next business day**.
Your `dissemination_time` = acceptance time, rolled to next business day open if after the cutoff.

## Earnings-call transcripts (optional second text source)
- Hugging Face datasets (e.g., search "earnings call transcript"); some Motley Fool scrapes on Kaggle.
- Attach the call datetime; treat text as knowable only after the call ends. Coverage is partial — disclose it.

## Prices / returns
- **Stooq** (free CSV) and **Yahoo Finance** via `yfinance` for daily OHLCV. **FRED** for the risk-free rate.
- No CRSP — your universe likely reflects survivors. **Disclose survivorship bias** in the README.

## Models (all free on Hugging Face)
- FinBERT: `yiyanghkust/finbert-tone`, `ProsusAI/finbert`.
- ChronoBERT / ChronoGPT (chronologically-consistent, vintage by cutoff year): `manelalab/chrono-bert-*`,
  `manelalab/chrono-gpt-*`. Pick the vintage whose cutoff <= your as-of date.
- OpenAI API for the instruction-tuned generative signal (cache responses; respect the cutoff guard).

## Sanity-check benchmarks
- Financial PhraseBank, FinBen, PIXIU/FinMA — to validate your sentiment classifiers before backtesting.

## Loughran-McDonald word lists
- Download the master dictionary CSV from the official Notre Dame SRAF page; load positive/negative/uncertainty
  word sets. (Cite the source; do not redistribute the list in your repo if the license restricts it.)
