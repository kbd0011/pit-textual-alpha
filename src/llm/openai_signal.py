"""Instruction-tuned LLM signal via the OpenAI API, with the leakage controls that make this rigorous.

Three safeguards baked in:
1. On-disk response cache (so re-runs are deterministic and cheap).
2. Entity anonymization (mask company names/tickers/dates) to separate sentiment from memorized facts.
3. A knowledge-cutoff guard: refuse to score text dated after the model's training cutoff unless the caller
   explicitly opts in (and logs that the result is for the L0-naive arm only).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass
class CutoffPolicy:
    """Knowledge-cutoff guard configuration."""

    model_cutoff: date           # the model's stated training cutoff (treat as FUZZY, not exact)
    allow_post_cutoff: bool = False  # if True, permit scoring text after cutoff (naive/L0 arm only)


PROMPT_TEMPLATE = (
    "You are a financial analyst. Based ONLY on the disclosure text below, assess the likely direction of "
    "the company's stock over the next {horizon} trading days. Do not use any outside knowledge about the "
    "company or what happened after this filing. Respond with a single number from -1 (very negative) to "
    "+1 (very positive), then a one-sentence justification.\n\n"
    "DISCLOSURE:\n{text}\n\nSCORE:"
)

_TICKER = re.compile(r"\b[A-Z]{1,5}\b")
_DATE = re.compile(r"\b(19|20)\d{2}\b")


def anonymize(text: str, company_names: list[str]) -> str:
    """Mask company names, plausible tickers, and 4-digit years to blunt memorization (L2 control)."""
    out = text
    for name in sorted(company_names, key=len, reverse=True):
        if name:
            out = re.sub(re.escape(name), "[COMPANY]", out, flags=re.IGNORECASE)
    out = _DATE.sub("[YEAR]", out)
    # NOTE: tickers are risky to mask blindly; do it only if you maintain a real ticker list.
    return out


class OpenAISignal:
    """Cached OpenAI scorer producing a signal in [-1, 1]."""

    def __init__(
        self,
        model: str,
        cutoff: CutoffPolicy,
        cache_dir: str = ".cache/openai",
        horizon: int = 5,
    ):
        self.model = model
        self.cutoff = cutoff
        self.horizon = horizon
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # lazy

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256((self.model + "::" + prompt).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _guard_cutoff(self, as_of: datetime) -> None:
        if as_of.date() > self.cutoff.model_cutoff and not self.cutoff.allow_post_cutoff:
            raise ValueError(
                f"as_of {as_of.date()} is after model cutoff {self.cutoff.model_cutoff}; "
                "scoring would risk memorization. Set allow_post_cutoff=True only for the naive L0 arm."
            )

    def score(self, text: str, as_of: datetime, company_names: list[str] | None = None) -> float:
        r"""Score a disclosure. Applies the cutoff guard; uses cache; parses the leading number.

        TODO (prompt #7): implement the API call + robust number parsing. Pseudocode:
            self._guard_cutoff(as_of)
            text2 = anonymize(text, company_names or [])          # for the L2 arm
            prompt = PROMPT_TEMPLATE.format(horizon=self.horizon, text=text2[:MAX_CHARS])
            cached = self._cache_path(prompt)
            if cached.exists(): return json.loads(cached.read_text())["score"]
            resp = self._client_lazy().chat.completions.create(model=self.model,
                       messages=[{"role": "user", "content": prompt}], temperature=0)
            raw = resp.choices[0].message.content
            val = float(re.search(r"-?\d+(\.\d+)?", raw).group())
            val = max(-1.0, min(1.0, val))
            cached.write_text(json.dumps({"score": val, "raw": raw}))
            return val
        """
        raise NotImplementedError("Implement via LLM_PROMPTS.md prompt #7.")
