"""Sentiment / textual-signal scorers: the method ladder.

Common interface: every scorer implements `.score(text, as_of) -> float` returning a signal in [-1, 1].
Heavy models (transformers) are loaded lazily so importing this module is cheap and test-friendly.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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

    The whole point of this scorer is that the model cannot have memorized the future: we select the newest
    *vintage* whose training **cutoff date** is on or before ``as_of``. If no vintage qualifies (``as_of`` is
    earlier than every cutoff) we refuse to score rather than risk look-ahead.

    Signal generation is a two-stage design: (1) embed the text with the chosen vintage (a frozen encoder),
    (2) map the embedding to a [-1, 1] signal with a lightweight *probe head* that the caller fits **only on
    labels dated before ``as_of``**. The probe is injected (``probe=``) so the fit-window discipline lives with
    the experiment harness, not here.
    """

    name = "chronobert"

    # Default registry: training cutoff -> HF repo. Verify exact repo names against the manelalab releases
    # before downloading; these follow the published vintage naming (chrono-bert-v1-YYYYMMDD).
    DEFAULT_VINTAGES: dict[date, str] = {
        date(1999, 12, 31): "manelalab/chrono-bert-v1-19991231",
        date(2009, 12, 31): "manelalab/chrono-bert-v1-20091231",
        date(2014, 12, 31): "manelalab/chrono-bert-v1-20141231",
        date(2019, 12, 31): "manelalab/chrono-bert-v1-20191231",
    }

    def __init__(
        self,
        probe=None,
        vintages: dict | None = None,
        device: str | None = None,
    ):
        # Normalize vintage keys to datetime.date so str/date inputs both work.
        raw = self.DEFAULT_VINTAGES if vintages is None else vintages
        self.vintages: dict[date, str] = {
            (k if isinstance(k, date) else datetime.fromisoformat(str(k)).date()): v
            for k, v in raw.items()
        }
        self.probe = probe
        self.device = device
        self._models: dict[str, object] = {}  # repo -> (tokenizer, model), lazily loaded

    def pick_vintage(self, as_of: datetime) -> str:
        """Return the repo of the newest vintage with cutoff <= ``as_of``; raise if none qualifies."""
        as_of_date = as_of.date() if isinstance(as_of, datetime) else date.fromisoformat(str(as_of))
        eligible = [(cut, repo) for cut, repo in self.vintages.items() if cut <= as_of_date]
        if not eligible:
            earliest = min(self.vintages) if self.vintages else None
            raise ValueError(
                f"No ChronoBERT vintage with cutoff <= {as_of_date} (earliest available: {earliest}); "
                "refusing to score to avoid training-cutoff look-ahead."
            )
        return max(eligible, key=lambda x: x[0])[1]

    def _embed(self, text: str, repo: str) -> np.ndarray:
        """Mean-pooled last-hidden-state embedding from a vintage encoder (lazy transformers/torch)."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        if repo not in self._models:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModel.from_pretrained(repo)
            if self.device:
                mdl = mdl.to(self.device)
            mdl.eval()
            self._models[repo] = (tok, mdl)
        tok, mdl = self._models[repo]  # type: ignore[misc]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        if self.device:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
        last_hidden = out.last_hidden_state[0]                 # (seq_len, hidden)
        mask = enc["attention_mask"][0].unsqueeze(-1)          # (seq_len, 1)
        pooled = (last_hidden * mask).sum(0) / mask.sum().clamp(min=1)
        return pooled.cpu().numpy()

    def _apply_probe(self, embedding: np.ndarray) -> float:
        """Map an embedding to a scalar via the injected probe (callable or sklearn-style ``predict``)."""
        if self.probe is None:
            raise ValueError("ChronoBERTScorer requires a fitted probe head (probe=...) to produce a signal")
        if callable(self.probe):
            return float(self.probe(embedding))
        return float(self.probe.predict(embedding.reshape(1, -1))[0])

    def score(self, text: str, as_of: datetime) -> float:
        """Embed with the cutoff-safe vintage, apply the probe head, clip to [-1, 1]."""
        repo = self.pick_vintage(as_of)
        embedding = self._embed(text, repo)
        raw = self._apply_probe(embedding)
        return float(max(-1.0, min(1.0, raw)))
