"""Tests for FinBERTScorer — the transformers pipeline is MOCKED (no model download, no network)."""
from datetime import datetime

import pytest
from src.features.sentiment import FinBERTScorer, net_tone, split_sentences

AS_OF = datetime(2020, 1, 1)


def _fake_pipe(sentences, **kwargs):
    """Stand-in for a FinBERT pipeline. Content-driven so output aligns 1:1 with the input list."""
    out = []
    for s in sentences:
        low = s.lower()
        if "good" in low:
            out.append([{"label": "Positive", "score": 0.9}, {"label": "Negative", "score": 0.05},
                        {"label": "Neutral", "score": 0.05}])
        elif "bad" in low:
            out.append([{"label": "positive", "score": 0.1}, {"label": "negative", "score": 0.8},
                        {"label": "neutral", "score": 0.1}])
        else:
            out.append([{"label": "Positive", "score": 0.2}, {"label": "Negative", "score": 0.2},
                        {"label": "Neutral", "score": 0.6}])
    return out


def _scorer():
    s = FinBERTScorer()
    s._pipe_obj = _fake_pipe  # inject mock; bypasses the lazy transformers import entirely
    return s


def test_split_sentences():
    assert split_sentences("One. Two! Three?  ") == ["One.", "Two!", "Three?"]
    assert split_sentences("   ") == []


def test_net_tone_case_insensitive():
    assert net_tone([{"label": "Positive", "score": 0.7}, {"label": "negative", "score": 0.2}]) == pytest.approx(0.5)
    assert net_tone([{"label": "Neutral", "score": 1.0}]) == pytest.approx(0.0)


def test_mean_over_sentences():
    s = _scorer()
    # sent1 net = 0.9-0.05 = 0.85 ; sent2 net = 0.1-0.8 = -0.70 ; mean = 0.075
    assert s.score("This is good. This is bad.", AS_OF) == pytest.approx((0.85 - 0.70) / 2)


def test_empty_text_returns_zero_without_calling_pipe():
    s = FinBERTScorer()
    # No _pipe_obj injected and not loaded; if score touched the pipeline it would try to import transformers.
    s._pipe_obj = None
    assert s.score("   ", AS_OF) == 0.0


def test_output_clipped_to_unit_interval():
    s = _scorer()
    assert -1.0 <= s.score("good good good", AS_OF) <= 1.0
