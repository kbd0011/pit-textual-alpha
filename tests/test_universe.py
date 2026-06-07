"""Tests for point-in-time index membership (no network)."""
import pandas as pd
import pytest
from src.ingest.universe import SurvivorshipWarning, UniverseTable


def _pit_table():
    # AAA: member 2010-2015 (dropped). BBB: member from 2012 onward (still in). CCC: sp400 from 2018.
    rows = [
        {"ticker": "AAA", "index_name": "sp500", "start_date": "2010-01-01", "end_date": "2015-06-30"},
        {"ticker": "BBB", "index_name": "sp500", "start_date": "2012-03-15", "end_date": None},
        {"ticker": "CCC", "index_name": "sp400", "start_date": "2018-01-01", "end_date": None},
    ]
    return UniverseTable(pd.DataFrame(rows), pit_accurate=True)


def test_membership_respects_start_and_end_dates():
    u = _pit_table()
    assert u.members_as_of("2011-01-01") == ["AAA"]            # only AAA yet
    assert u.members_as_of("2013-01-01") == ["AAA", "BBB"]      # both active
    assert u.members_as_of("2016-01-01") == ["BBB"]            # AAA dropped
    assert u.members_as_of("2019-01-01") == ["BBB", "CCC"]      # CCC added


def test_boundary_dates_are_inclusive():
    u = _pit_table()
    assert "AAA" in u.members_as_of("2010-01-01")  # exactly start
    assert "AAA" in u.members_as_of("2015-06-30")  # exactly end
    assert "AAA" not in u.members_as_of("2015-07-01")


def test_index_filter():
    u = _pit_table()
    assert u.members_as_of("2019-01-01", index_name="sp500") == ["BBB"]
    assert u.members_as_of("2019-01-01", index_name="sp400") == ["CCC"]


def test_current_snapshot_warns_and_is_date_invariant():
    u = UniverseTable.from_current_snapshot({"sp500": ["XYZ", "QRS"]})
    assert u.pit_accurate is False
    with pytest.warns(SurvivorshipWarning):
        m1 = u.members_as_of("1990-01-01")
    with pytest.warns(SurvivorshipWarning):
        m2 = u.members_as_of("2025-01-01")
    assert m1 == m2 == ["QRS", "XYZ"]  # same for every date -> the survivorship trap, made explicit


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        UniverseTable(pd.DataFrame({"ticker": ["AAA"]}))
