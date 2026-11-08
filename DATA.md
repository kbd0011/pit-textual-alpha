# Data Sources (all free)

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
`dissemination_time` = acceptance time, rolled to the next business day's open if after the cutoff.

## Earnings-call transcripts (optional second text source)
- Hugging Face datasets (e.g., search "earnings call transcript"); some Motley Fool scrapes on Kaggle.
- Attach the call datetime; treat text as knowable only after the call ends. Coverage is partial — disclose it.

## Prices / returns
- **Stooq** (free CSV) and **Yahoo Finance** via `yfinance` for daily OHLCV. **FRED** for the risk-free rate.
- No CRSP, so a free-data universe likely reflects survivors; the README discloses survivorship bias.

## Models (all free on Hugging Face)
- FinBERT: `yiyanghkust/finbert-tone`, `ProsusAI/finbert`.
- ChronoBERT / ChronoGPT (chronologically-consistent, vintage by cutoff year): `manelalab/chrono-bert-*`,
  `manelalab/chrono-gpt-*`. The scorer picks the newest vintage whose cutoff <= the as-of date.
- OpenAI API for the instruction-tuned generative signal (cache responses; respect the cutoff guard).

## Sanity-check benchmarks
- Financial PhraseBank, FinBen, PIXIU/FinMA — candidate checks for the sentiment classifiers before backtesting (not yet run).

## Loughran-McDonald word lists
- Download the master dictionary CSV from the official Notre Dame SRAF page; load positive/negative/uncertainty
  word sets. (Cited, not redistributed: the tests use a small sample CSV fixture.)
