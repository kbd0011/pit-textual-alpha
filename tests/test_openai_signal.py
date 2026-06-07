"""Tests for OpenAISignal — the OpenAI client is MOCKED (no API calls, no network)."""
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from src.llm.openai_signal import CutoffPolicy, OpenAISignal, anonymize, parse_score


class _MockClient:
    """Records calls and returns a canned chat-completion response."""

    def __init__(self, content: str):
        self.content = content
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _signal(tmp_path, content="-0.3 because revenue declined", allow_post=False):
    policy = CutoffPolicy(model_cutoff=date(2021, 12, 31), allow_post_cutoff=allow_post)
    sig = OpenAISignal(model="gpt-test", cutoff=policy, cache_dir=str(tmp_path), horizon=5)
    sig._client = _MockClient(content)  # inject mock; bypasses _client_lazy / OpenAI()
    return sig


def test_parse_score_handles_justification_text():
    assert parse_score("-0.3 because revenue declined") == pytest.approx(-0.3)
    assert parse_score("+0.8 strong guidance") == pytest.approx(0.8)
    assert parse_score("2.0 very bullish") == pytest.approx(1.0)  # clipped
    with pytest.raises(ValueError):
        parse_score("no number here")


def test_cutoff_guard_raises_post_cutoff_when_not_allowed(tmp_path):
    sig = _signal(tmp_path)
    with pytest.raises(ValueError, match="after model cutoff"):
        sig.score("some text", datetime(2022, 6, 1))  # after 2021-12-31 cutoff
    assert sig._client.calls == 0  # guarded before any API call


def test_post_cutoff_allowed_for_naive_arm(tmp_path):
    sig = _signal(tmp_path, allow_post=True)
    val = sig.score("some text", datetime(2022, 6, 1))
    assert val == pytest.approx(-0.3)


def test_caching_avoids_second_api_call(tmp_path):
    sig = _signal(tmp_path)
    as_of = datetime(2020, 1, 1)
    v1 = sig.score("identical disclosure text", as_of)
    v2 = sig.score("identical disclosure text", as_of)
    assert v1 == v2 == pytest.approx(-0.3)
    assert sig._client.calls == 1  # second call served from cache


def test_anonymize_masks_company_and_years():
    out = anonymize("Apple grew in 2021 and Apple expects more.", ["Apple"])
    assert "Apple" not in out
    assert "[COMPANY]" in out
    assert "2021" not in out and "[YEAR]" in out
