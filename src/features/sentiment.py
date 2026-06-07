"""Sentiment / textual-signal scorers: the method ladder.

Common interface: every scorer implements `.score(text, as_of) -> float` returning a signal in [-1, 1].
Heavy models (transformers) are loaded lazily so importing this module is cheap and test-friendly.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime


class Scorer(ABC):
    """Abstract textual-signal scorer."""

    name: str = "base"

    @abstractmethod
    def score(self, text: str, as_of: datetime) -> float:
        """Return a signal in [-1, 1] for `text`, asserting no look-ahead relative to `as_of`."""
        raise NotImplementedError


class LoughranMcDonaldScorer(Scorer):
    """Baseline: net tone from the Loughran-McDonald finance dictionary.

    score = (n_positive - n_negative) / (n_positive + n_negative + eps), clipped to [-1, 1].
    Load the official word lists (positive/negative sets) from the Notre Dame SRAF master CSV.
    """

    name = "lm_dictionary"
    _WORD = re.compile(r"[A-Za-z]+")

    def __init__(self, positive: set[str], negative: set[str]):
        self.positive = {w.lower() for w in positive}
        self.negative = {w.lower() for w in negative}

    @classmethod
    def from_master_csv(cls, path: str) -> LoughranMcDonaldScorer:
        """TODO (prompt #4): parse the LM master dictionary CSV into positive/negative word sets."""
        raise NotImplementedError("Implement via LLM_PROMPTS.md prompt #4.")

    def score(self, text: str, as_of: datetime) -> float:  # noqa: D401
        tokens = [t.lower() for t in self._WORD.findall(text)]
        pos = sum(t in self.positive for t in tokens)
        neg = sum(t in self.negative for t in tokens)
        if pos + neg == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))


class FinBERTScorer(Scorer):
    """Fine-tuned encoder tone (FinBERT). Aggregates sentence-level prob(positive) - prob(negative)."""

    name = "finbert"

    def __init__(self, model_id: str = "yiyanghkust/finbert-tone", device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._pipe_obj = None  # lazy, per-instance (avoids lru_cache-on-method leak)

    def _pipe(self):
        # Lazy import + per-instance cache so the module imports without transformers/torch installed.
        if self._pipe_obj is None:
            from transformers import pipeline

            self._pipe_obj = pipeline(
                "text-classification", model=self.model_id, top_k=None, device=self.device
            )
        return self._pipe_obj

    def score(self, text: str, as_of: datetime) -> float:
        """TODO (prompt #5): split text into sentences, run the pipeline in batches, map
        {positive, negative, neutral} probabilities to a net tone in [-1, 1], aggregate (mean)."""
        raise NotImplementedError("Implement via LLM_PROMPTS.md prompt #5.")


class ChronoBERTScorer(Scorer):
    """Chronologically-consistent encoder (ChronoBERT) — the leakage-free control.

    Select the model vintage whose training cutoff <= as_of.year so the model could not have seen the future.
    """

    name = "chronobert"

    # Map an as-of year to the latest permissible vintage repo (fill with real vintages you download).
    VINTAGES: dict[int, str] = {
        # 2008: "manelalab/chrono-bert-v1-19991231",  # example placeholders — verify exact repo names
        # 2015: "manelalab/chrono-bert-v1-20141231",
        # 2020: "manelalab/chrono-bert-v1-20191231",
    }

    def pick_vintage(self, as_of: datetime) -> str:
        """TODO (prompt #6): choose the newest vintage with cutoff <= as_of; raise if none qualifies."""
        raise NotImplementedError("Implement via LLM_PROMPTS.md prompt #6.")

    def score(self, text: str, as_of: datetime) -> float:
        """TODO (prompt #6): embed text with the chosen vintage; map to a signal (e.g., via a probe head
        trained only on pre-as_of data) in [-1, 1]."""
        raise NotImplementedError("Implement via LLM_PROMPTS.md prompt #6.")
