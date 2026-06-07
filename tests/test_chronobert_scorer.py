"""Tests for ChronoBERTScorer vintage selection and probe wiring (model embedding is MOCKED)."""
from datetime import date, datetime

import numpy as np
import pytest
from src.features.sentiment import ChronoBERTScorer

VINTAGES = {
    date(1999, 12, 31): "repo-1999",
    date(2009, 12, 31): "repo-2009",
    date(2014, 12, 31): "repo-2014",
    date(2019, 12, 31): "repo-2019",
}


def test_pick_newest_vintage_at_or_before_as_of():
    s = ChronoBERTScorer(vintages=VINTAGES)
    assert s.pick_vintage(datetime(2016, 6, 1)) == "repo-2014"
    assert s.pick_vintage(datetime(2014, 12, 31)) == "repo-2014"  # boundary inclusive
    assert s.pick_vintage(datetime(2025, 1, 1)) == "repo-2019"   # newest available


def test_refuses_when_no_vintage_qualifies():
    s = ChronoBERTScorer(vintages=VINTAGES)
    with pytest.raises(ValueError, match="refusing to score"):
        s.pick_vintage(datetime(1990, 1, 1))  # earlier than every cutoff


def test_string_keyed_vintages_normalized():
    s = ChronoBERTScorer(vintages={"2014-12-31": "v14", "2019-12-31": "v19"})
    assert s.pick_vintage(datetime(2020, 1, 1)) == "v19"


def test_score_uses_cutoff_safe_vintage_and_probe():
    s = ChronoBERTScorer(vintages=VINTAGES, probe=lambda emb: float(emb.mean()))
    used = {}

    def fake_embed(text, repo):
        used["repo"] = repo
        return np.array([0.3, 0.3, 0.3])  # mean 0.3

    s._embed = fake_embed  # inject mock; no transformers/torch needed
    val = s.score("some 2013 disclosure text", datetime(2016, 1, 1))
    assert used["repo"] == "repo-2014"          # cutoff-safe vintage chosen
    assert val == pytest.approx(0.3)


def test_score_clips_and_requires_probe():
    s = ChronoBERTScorer(vintages=VINTAGES, probe=lambda emb: 5.0)  # out-of-range
    s._embed = lambda text, repo: np.zeros(3)
    assert s.score("x", datetime(2016, 1, 1)) == 1.0  # clipped to +1

    s_noprobe = ChronoBERTScorer(vintages=VINTAGES)
    s_noprobe._embed = lambda text, repo: np.zeros(3)
    with pytest.raises(ValueError, match="requires a fitted probe"):
        s_noprobe.score("x", datetime(2016, 1, 1))
