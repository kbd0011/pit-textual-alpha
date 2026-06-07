"""Tests for the Loughran-McDonald dictionary loader and net-tone scoring (no network)."""
from datetime import datetime
from pathlib import Path

import pytest
from src.features.sentiment import LoughranMcDonaldScorer

FIX = Path(__file__).parent / "fixtures"
AS_OF = datetime(2020, 1, 1)


def _scorer() -> LoughranMcDonaldScorer:
    return LoughranMcDonaldScorer.from_master_csv(str(FIX / "lm_master_sample.csv"))


def test_words_land_in_correct_category():
    s = _scorer()
    assert {"gain", "improve"} <= s.positive
    assert {"loss", "adverse"} <= s.negative
    assert {"uncertain", "approximately", "may"} <= s.uncertainty
    # Neutral words appear in no sentiment set.
    for neutral in ("the", "company"):
        assert neutral not in s.positive
        assert neutral not in s.negative
        assert neutral not in s.uncertainty


def test_zero_year_words_excluded():
    s = _scorer()
    # LITIGATION is Litigious-only (no Positive/Negative/Uncertainty) -> in none of our three sets.
    assert "litigation" not in s.positive | s.negative | s.uncertainty


def test_net_tone_scoring():
    s = _scorer()
    assert s.score("We expect a gain and an improvement", AS_OF) == pytest.approx(1.0)
    assert s.score("A loss and an adverse decline", AS_OF) == pytest.approx(-1.0)
    # pos=1 (gain), neg=2 (loss, adverse) -> (1-2)/3.
    assert s.score("gain loss adverse", AS_OF) == pytest.approx(-1 / 3)
    # No sentiment words -> 0.
    assert s.score("the company building", AS_OF) == pytest.approx(0.0)
