.PHONY: setup test lint typecheck check run
# One-command workflows. Seeds are fixed in code; the run is point-in-time gated (reproducible).

setup:          ## install dependencies
	pip install -r requirements.txt

test:           ## run the test suite (no network/GPU; models mocked)
	pytest -q

lint:           ## ruff lint
	ruff check src tests

typecheck:      ## mypy type check
	mypy src --ignore-missing-imports

check: lint typecheck test   ## lint + typecheck + test (what CI runs)

run:            ## real EDGAR -> FinBERT -> leakage-decomposition ablation (needs SEC_USER_AGENT)
	PYTHONPATH=. SEC_USER_AGENT="$${SEC_USER_AGENT:?set SEC_USER_AGENT='Your Name your@email.com'}" \
		python scripts/run_pipeline.py
