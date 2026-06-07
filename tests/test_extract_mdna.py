"""Tests for MD&A / risk-factor section extraction across filing layouts (no network)."""
from pathlib import Path

import pytest
from src.ingest.edgar import extract_mdna

FIX = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIX / name).read_text()


def test_10k_extracts_item7_body_not_toc():
    text = extract_mdna(_read("filing_10k.html"))
    assert "Revenue increased during the fiscal year" in text
    assert "Cash flow from operations was sufficient" in text
    # Must stop at Item 7A — the market-risk derivative sentence belongs to the next section.
    assert "derivative instruments" not in text
    # Must not bleed forward into Item 8.
    assert "consolidated financial statements and accompanying notes" not in text
    # The TOC dotted-leader entry must not be what we returned (body is long).
    assert len(text) > 400


def test_10k_drops_table_digits():
    text = extract_mdna(_read("filing_10k.html"))
    # Numbers live only inside a <table>, which must be stripped.
    assert "123456" not in text
    assert "987654" not in text


def test_10q_extracts_item2_with_entity_encoded_apostrophe():
    text = extract_mdna(_read("filing_10q.html"))
    assert "Net sales for the quarter rose" in text
    assert "seasonal demand to support results" in text
    # Stops before Item 3 market risk and Item 4 controls.
    assert "no material changes to our market-risk" not in text
    assert "disclosure controls and procedures were effective" not in text


def test_fallback_to_risk_factors_when_no_mdna():
    text = extract_mdna(_read("filing_riskonly.html"))
    assert "highly competitive markets" in text
    assert "Cybersecurity incidents" in text
    # Stops at Item 1B / Item 2.
    assert "headquarters and principal facilities" not in text


def test_raises_when_no_section_found():
    with pytest.raises(ValueError):
        extract_mdna("<html><body><p>Nothing relevant here at all.</p></body></html>")
