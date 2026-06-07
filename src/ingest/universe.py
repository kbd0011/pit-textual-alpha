"""Point-in-time index membership (S&P 1500 = S&P 500 + 400 + 600).

The honest hard part of a cross-sectional study is *survivorship bias*: free sources usually expose only
**today's** constituents, so a backtest silently assumes you held names that were added later and never held
the ones that were dropped. This module separates the two regimes explicitly:

- **PIT-accurate** (``pit_accurate=True``): built from a membership table with ``start_date``/``end_date`` per
  ticker. ``members_as_of(date)`` is then truly point-in-time.
- **Current snapshot** (``pit_accurate=False``): built from today's constituents only. ``members_as_of`` returns
  the same set for every date and a :class:`SurvivorshipWarning` is raised. Results from this regime are
  upward-biased and must be disclosed as such (see README).

Provide your own historical CSV to get the accurate regime; the bundled Wikipedia fetch is current-only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

__all__ = ["SurvivorshipWarning", "UniverseTable", "fetch_current_sp1500"]

_REQUIRED_COLS = {"ticker", "index_name", "start_date", "end_date"}


class SurvivorshipWarning(UserWarning):
    """Raised when membership queries run against a current-only (non-PIT) snapshot."""


def _norm_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


@dataclass
class UniverseTable:
    """Index-membership intervals: one row per (ticker, index, membership spell).

    Columns: ``ticker``, ``index_name`` (e.g. ``sp500``/``sp400``/``sp600``), ``start_date`` (inclusive),
    ``end_date`` (inclusive; ``NaT`` means still a member). ``pit_accurate`` flags whether the spells reflect
    real add/remove history or a current snapshot.
    """

    table: pd.DataFrame
    pit_accurate: bool = True

    def __post_init__(self) -> None:
        missing = _REQUIRED_COLS - set(self.table.columns)
        if missing:
            raise ValueError(f"membership table missing columns: {sorted(missing)}")
        df = self.table.copy()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["index_name"] = df["index_name"].astype(str).str.lower().str.strip()
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.tz_localize(None)
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.tz_localize(None)  # NaT stays NaT
        self.table = df.reset_index(drop=True)

    @classmethod
    def from_csv(cls, path: str, pit_accurate: bool = True) -> UniverseTable:
        """Load a membership table from CSV with the required columns."""
        return cls(pd.read_csv(path), pit_accurate=pit_accurate)

    @classmethod
    def from_current_snapshot(cls, tickers: dict[str, list[str]]) -> UniverseTable:
        """Build a *current-only* table from ``{index_name: [tickers]}``; flags survivorship bias.

        Every name is treated as a member for all time (``start_date`` = epoch, ``end_date`` = NaT), which is
        exactly the survivorship-biased assumption — hence ``pit_accurate=False``.
        """
        rows = [
            {"ticker": t, "index_name": idx, "start_date": pd.Timestamp("1900-01-01"), "end_date": pd.NaT}
            for idx, names in tickers.items()
            for t in names
        ]
        return cls(pd.DataFrame(rows, columns=sorted(_REQUIRED_COLS)), pit_accurate=False)

    def members_as_of(self, date, index_name: str | None = None) -> list[str]:
        """Tickers that are members on ``date`` (inclusive of start, inclusive of end).

        Parameters
        ----------
        date : str | datetime-like
            The as-of date. Membership holds when ``start_date <= date`` and
            (``end_date`` is NaT or ``date <= end_date``).
        index_name : str, optional
            Restrict to one sub-index (e.g. ``"sp500"``). Default: union across all indices.

        Returns
        -------
        list[str]
            Sorted, de-duplicated tickers. For a current snapshot this is identical for every ``date`` and
            emits a :class:`SurvivorshipWarning`.
        """
        if not self.pit_accurate:
            warnings.warn(
                "members_as_of called on a current-only snapshot: results assume today's constituents held "
                "for all history (survivorship bias). Supply a historical membership CSV for PIT accuracy.",
                SurvivorshipWarning,
                stacklevel=2,
            )
        d = _norm_ts(date)
        df = self.table
        mask = (df["start_date"] <= d) & (df["end_date"].isna() | (d <= df["end_date"]))
        if index_name is not None:
            mask &= df["index_name"] == index_name.lower().strip()
        return sorted(df.loc[mask, "ticker"].unique().tolist())


def fetch_current_sp1500() -> UniverseTable:
    """Fetch *current* S&P 500/400/600 constituents from Wikipedia (network; current-only -> not PIT).

    Lazy-imports so the module stays import-light for tests. The result is ``pit_accurate=False``.
    """
    sources = {
        "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        "sp400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol"),
        "sp600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol"),
    }
    members: dict[str, list[str]] = {}
    for idx, (url, col) in sources.items():
        tables = pd.read_html(url)  # uses lxml; raises if no table found
        for tbl in tables:
            if col in tbl.columns:
                members[idx] = [str(s).replace(".", "-").upper().strip() for s in tbl[col].tolist()]
                break
    return UniverseTable.from_current_snapshot(members)
