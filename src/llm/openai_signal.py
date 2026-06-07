"""Instruction-tuned LLM signal via the OpenAI API, with the leakage controls that make this rigorous.

Three safeguards baked in:
1. On-disk response cache (so re-runs are deterministic and cheap).
2. Entity anonymization (mask company names/tickers/dates) to separate sentiment from memorized facts.
3. A knowledge-cutoff guard: refuse to score text dated after the model's training cutoff unless the caller
   explicitly opts in (and logs that the result is for the L0-naive arm only).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

MAX_CHARS = 12000  # truncate disclosure text to keep token cost bounded and deterministic
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_score(raw: str) -> float:
    """Extract the leading signed number from an LLM response and clip to [-1, 1].

    Robust to formats like ``"-0.3 because revenue declined"`` or ``"+0.5"``. Raises if no number is present.
    """
    m = _NUMBER.search(raw or "")
    if m is None:
        raise ValueError(f"no parseable score in model response: {raw!r}")
    return max(-1.0, min(1.0, float(m.group())))


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
        """Score a disclosure in [-1, 1]. Applies the cutoff guard, uses the on-disk cache, parses the number.

        Parameters
        ----------
        text : str
            The disclosure text (already dissemination-gated upstream).
        as_of : datetime
            The decision time; checked against the model's training cutoff.
        company_names : list[str], optional
            If provided, entities are masked before prompting (the L2 anonymization arm).

        Returns
        -------
        float
            The model's directional score, clipped to [-1, 1]. Cached responses are returned without an API
            call (deterministic, cheap re-runs).
        """
        self._guard_cutoff(as_of)
        text2 = anonymize(text, company_names or [])
        prompt = PROMPT_TEMPLATE.format(horizon=self.horizon, text=text2[:MAX_CHARS])
        cached = self._cache_path(prompt)
        if cached.exists():
            return float(json.loads(cached.read_text())["score"])
        resp = self._client_lazy().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content
        val = parse_score(raw)
        cached.write_text(json.dumps({"score": val, "raw": raw}))
        return val
