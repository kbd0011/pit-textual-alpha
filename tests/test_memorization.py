"""Tests for the L4 memorization probe (no network)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from src.eval.memorization import (
    DISCLOSURE_REMOVED,
    build_probe_text,
    memorization_adjusted_performance,
    memorization_adjusted_signal,
    probe_signal_series,
)

AS_OF = datetime(2021, 3, 31)


def test_probe_text_has_entity_but_no_disclosure():
    text = build_probe_text("Apple Inc.", AS_OF, period_label="Q1 2021")
    assert "Apple Inc." in text
    assert "Q1 2021" in text
    assert "2021-03-31" in text
    assert DISCLOSURE_REMOVED in text
    # The probe must not contain any real disclosure body.
    assert "revenue" not in text.lower()


class _EntityRecallScorer:
    """Mock model that 'remembers' a known winner regardless of (absent) disclosure text."""

    def score(self, text, as_of):
        return 0.6 if "WinnerCo" in text else 0.0


def test_probe_signal_series_captures_entity_only_recall():
    entities = {"WIN": {"name": "WinnerCo"}, "MEH": {"name": "NeutralCo"}}
    probe = probe_signal_series(_EntityRecallScorer(), entities, AS_OF)
    assert probe["WIN"] == pytest.approx(0.6)  # memorized
    assert probe["MEH"] == pytest.approx(0.0)


def test_adjusted_signal_subtracts_probe_and_clips():
    idx = pd.bdate_range("2021-01-01", periods=2)
    full = pd.DataFrame({"WIN": [0.9, 0.9], "MEH": [0.2, 0.2]}, index=idx)
    probe = pd.Series({"WIN": 0.6, "MEH": 0.1})
    adj = memorization_adjusted_signal(full, probe)
    assert adj["WIN"].iloc[0] == pytest.approx(0.3)  # 0.9 - 0.6
    assert adj["MEH"].iloc[0] == pytest.approx(0.1)  # 0.2 - 0.1


def test_pure_memorization_yields_zero_residual():
    idx = pd.bdate_range("2021-01-01", periods=3)
    full = pd.DataFrame({"WIN": [0.6, 0.6, 0.6]}, index=idx)
    probe = pd.Series({"WIN": 0.6})  # full signal IS the memorized signal
    adj = memorization_adjusted_signal(full, probe)
    assert np.allclose(adj["WIN"].values, 0.0)  # no genuine real-time alpha left


def test_clipping_on_adjustment():
    idx = pd.bdate_range("2021-01-01", periods=1)
    full = pd.DataFrame({"X": [0.9]}, index=idx)
    probe = pd.Series({"X": -0.5})
    adj = memorization_adjusted_signal(full, probe)
    assert adj["X"].iloc[0] == pytest.approx(1.0)  # 1.4 clipped to 1.0


def test_adjusted_performance_scalar():
    assert memorization_adjusted_performance(1.2, 0.4) == pytest.approx(0.8)
