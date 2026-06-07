"""Point-in-time guards and leakage assertions.

The single most important discipline in this project. Anything that could let information from a time
later than `as_of` enter a feature/label must pass through these guards.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class LookaheadError(AssertionError):
    """Raised when information postdating the as-of time would leak into a feature/label."""


def assert_no_lookahead(info_time: datetime, as_of: datetime, what: str = "information") -> None:
    """Fail loudly if `info_time` is after `as_of`.

    Parameters
    ----------
    info_time : datetime
        Dissemination/observation time of the information being used.
    as_of : datetime
        The decision time. Only info with info_time <= as_of is permissible.
    what : str
        Label for the error message.
    """
    if info_time.tzinfo is None:
        info_time = info_time.replace(tzinfo=ET)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=ET)
    if info_time > as_of:
        raise LookaheadError(f"Look-ahead: {what} at {info_time.isoformat()} > as_of {as_of.isoformat()}")


def latest_knowable(items: Sequence[tuple[datetime, object]], as_of: datetime):
    """Return the most recent (time, payload) whose time <= as_of, else None.

    `items` is a list of (dissemination_time, payload). Use for "most recent filing knowable at t".
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=ET)
    eligible = [(t if t.tzinfo else t.replace(tzinfo=ET), p) for t, p in items]
    eligible = [(t, p) for t, p in eligible if t <= as_of]
    if not eligible:
        return None
    return max(eligible, key=lambda x: x[0])


# A tiny point-in-time store concept (back it with DuckDB/SQLite in the project).
class PITStore:
    """In-memory point-in-time store: append-only records keyed by (entity, dissemination_time)."""

    def __init__(self) -> None:
        self._rows: dict[str, list[tuple[datetime, dict]]] = {}

    def put(self, entity: str, dissemination_time: datetime, payload: dict) -> None:
        self._rows.setdefault(entity, []).append((dissemination_time, payload))

    def as_of(self, entity: str, when: datetime) -> dict | None:
        rec = latest_knowable(self._rows.get(entity, []), when)
        return rec[1] if rec else None
