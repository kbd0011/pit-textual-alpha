"""Sentiment / textual-signal scorers: the method ladder.

Common interface: every scorer implements `.score(text, as_of) -> float` returning a signal in [-1, 1].
Heavy models (transformers) are loaded lazily so importing this module is cheap and test-friendly.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

# Split on sentence-final punctuation followed by whitespace. Deliberately dependency-free (no nltk) so the
# module imports cheaply and unit tests need no downloads.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentences using a lightweight regex."""
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def net_tone(label_scores: Sequence[dict]) -> float:
    """Map a list of ``{'label', 'score'}`` dicts to ``p(positive) - p(negative)`` in [-1, 1].

    Labels are matched case-insensitively so FinBERT-tone ('Positive') and ProsusAI/finbert ('positive')
    both work. Missing labels contribute 0.
    """
    probs = {str(d["label"]).lower(): float(d["score"]) for d in label_scores}
    return probs.get("positive", 0.0) - probs.get("negative", 0.0)


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

    def __init__(
        self, positive: set[str], negative: set[str], uncertainty: set[str] | None = None
    ):
        self.positive = {w.lower() for w in positive}
        self.negative = {w.lower() for w in negative}
        self.uncertainty = {w.lower() for w in (uncertainty or set())}

    @classmethod
    def from_master_csv(cls, path: str) -> LoughranMcDonaldScorer:
        """Build the scorer from the official Notre Dame SRAF master dictionary CSV.

        In the master file each sentiment column (``Negative``, ``Positive``, ``Uncertainty``, ...) holds the
        *year* a word was first assigned to that category, or ``0`` if it never was. A word belongs to a
        category iff its value in that column is non-zero. Words are stored uppercase; we lowercase them to
        match :meth:`score`'s tokenization. Column matching is case-insensitive.

        Parameters
        ----------
        path : str
            Path to ``Loughran-McDonald_MasterDictionary_*.csv``.
        """
        import pandas as pd

        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        word_col = cols.get("word", df.columns[0])
        words = df[word_col].astype(str)

        def category(name: str) -> set[str]:
            col = cols.get(name)
            if col is None:
                return set()
            flags = pd.to_numeric(df[col], errors="coerce").fillna(0) != 0
            return {w.lower() for w in words[flags]}

        return cls(
            positive=category("positive"),
            negative=category("negative"),
            uncertainty=category("uncertainty"),
        )

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

    def __init__(
        self,
        model_id: str = "yiyanghkust/finbert-tone",
        device: str | None = None,
        batch_size: int = 16,
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
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
        """Mean sentence-level net tone from FinBERT, in [-1, 1].

        Splits ``text`` into sentences, runs them through the FinBERT classification pipeline in batches,
        maps each sentence's {positive, negative, neutral} probabilities to ``p(pos) - p(neg)``, and averages.

        ``as_of`` is part of the common :class:`Scorer` interface but is not used for gating here: FinBERT is a
        fixed encoder with no training-cutoff leakage concern (see :class:`ChronoBERTScorer` for that control);
        the text passed in is already dissemination-gated upstream.
        """
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        outputs = self._pipe()(sentences, batch_size=self.batch_size, truncation=True)
        tones = [net_tone(item) for item in outputs]
        if not tones:
            return 0.0
        mean_tone = sum(tones) / len(tones)
        return float(max(-1.0, min(1.0, mean_tone)))


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
