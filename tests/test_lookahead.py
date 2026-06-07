"""Leakage tests — these must FAIL if point-in-time discipline is broken."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from src.pit.leakage import LookaheadError, assert_no_lookahead, latest_knowable

ET = ZoneInfo("America/New_York")


def test_assert_no_lookahead_passes_for_past_info():
    assert_no_lookahead(datetime(2020, 1, 1, tzinfo=ET), datetime(2020, 1, 2, tzinfo=ET))


def test_assert_no_lookahead_raises_for_future_info():
    with pytest.raises(LookaheadError):
        assert_no_lookahead(datetime(2020, 1, 3, tzinfo=ET), datetime(2020, 1, 2, tzinfo=ET))


def test_latest_knowable_excludes_future():
    items = [
        (datetime(2020, 1, 1, tzinfo=ET), "a"),
        (datetime(2020, 6, 1, tzinfo=ET), "b"),   # future relative to as_of below
    ]
    rec = latest_knowable(items, datetime(2020, 3, 1, tzinfo=ET))
    assert rec is not None and rec[1] == "a"


def test_dissemination_after_hours_rolls_to_next_day():
    from src.ingest.edgar import _dissemination_time

    accepted = datetime(2023, 11, 3, 18, 1, tzinfo=ET)   # Friday after 17:30
    disseminated = _dissemination_time(accepted, "10-K")
    assert disseminated.date() > accepted.date()          # rolled forward
    assert disseminated.weekday() < 5                      # lands on a business day
