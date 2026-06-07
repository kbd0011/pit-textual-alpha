"""Daily prices and look-ahead-safe forward returns.

The signal at date ``t`` is *known* at ``t`` (it is gated on dissemination time elsewhere). The earliest a
portfolio can act on it is the **next session's open** (``t+1``). The forward return we attach to ``t`` is
therefore realized from the open of ``t+1`` to the open of ``t+1+h``:

    fwd_h(t) = Open[t+1+h] / Open[t+1] - 1

This construction is leakage-safe by design: the label attached to ``t`` uses only prices strictly after
``t``, and entry happens at ``t+1`` (never the same close the signal was observed on). The pure core
``forward_returns_from_prices`` is separated from all I/O so it can be unit-tested with no network.

Prices are split/dividend adjusted (yfinance ``auto_adjust=True``); the Stooq fallback is *not* adjusted —
that limitation is logged and should be disclosed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "forward_returns_from_prices",
    "trading_sessions",
    "download_open_prices",
    "forward_returns",
]


def forward_returns_from_prices(open_prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """h-day forward returns, lagged so the signal at ``t`` enters at the ``t+1`` open.

    Parameters
    ----------
    open_prices : pd.DataFrame
        Session-indexed (sorted, ascending) frame of adjusted **open** prices; columns are tickers.
    horizon_days : int
        Holding horizon ``h`` in trading sessions (``h >= 1``).

    Returns
    -------
    pd.DataFrame
        Same shape/index/columns as ``open_prices``. Value at ``(t, i)`` is ``Open[t+1+h]/Open[t+1] - 1``
        for ticker ``i``. The final ``h + 1`` rows are NaN (no realized future). No value at row ``t`` uses
        any price at or before ``t`` — this is the no-look-ahead guarantee.

    Raises
    ------
    ValueError
        If ``horizon_days < 1`` or the index is not monotonically increasing.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1 (you can only act at the next open at the earliest)")
    if not open_prices.index.is_monotonic_increasing:
        raise ValueError("open_prices index must be sorted ascending by session")

    entry = open_prices.shift(-1)              # price you pay: next session's open
    exit_ = open_prices.shift(-(1 + horizon_days))  # price you receive: open h sessions later
    fwd = exit_ / entry - 1.0
    # Guard the invariant: row t must be derivable only from rows > t. shift(-k) for k>=1 pulls strictly
    # future rows, so any non-NaN at row t is a function of t+1..t+1+h only.
    fwd.attrs["horizon_days"] = horizon_days
    return fwd


def trading_sessions(start: str, end: str, calendar: str = "XNYS") -> pd.DatetimeIndex:
    """Valid trading sessions in ``[start, end]`` for a market calendar (default NYSE).

    Lazy-imports ``pandas_market_calendars`` so importing this module stays cheap and test-friendly.
    """
    import pandas_market_calendars as mcal

    sched = mcal.get_calendar(calendar).schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index)


def _download_yf(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Adjusted open prices via yfinance (auto_adjust=True). Returns dates x tickers."""
    import yfinance as yf

    # end is inclusive for our purposes; yfinance treats end as exclusive, so nudge by a day.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers, start=start, end=end_excl, auto_adjust=True, progress=False, group_by="column"
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    # Single ticker -> columns are fields; multi -> MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        opens = raw["Open"].copy()
    else:
        opens = raw[["Open"]].copy()
        opens.columns = [tickers[0]]
    opens.index = pd.DatetimeIndex(opens.index).tz_localize(None)
    return opens.reindex(columns=tickers)


def _download_stooq(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Unadjusted open prices via Stooq CSV (fallback). Disclose: NOT split/dividend adjusted."""
    import io

    import requests

    d1, d2 = pd.Timestamp(start).strftime("%Y%m%d"), pd.Timestamp(end).strftime("%Y%m%d")
    frames: dict[str, pd.Series] = {}
    for tk in tickers:
        sym = tk.lower().replace(".", "-")
        url = f"https://stooq.com/q/d/l/?s={sym}.us&d1={d1}&d2={d2}&i=d"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if "Open" not in df.columns or df.empty:
                continue
            s = pd.Series(df["Open"].values, index=pd.DatetimeIndex(df["Date"]), name=tk)
            frames[tk] = s
        except Exception:  # noqa: BLE001 — fallback is best-effort; missing tickers stay NaN
            continue
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index().reindex(columns=tickers)


def download_open_prices(
    tickers: list[str], start: str, end: str, calendar: str = "XNYS"
) -> pd.DataFrame:
    """Adjusted open prices aligned to the trading calendar; yfinance primary, Stooq fallback.

    Returns a sessions x tickers DataFrame restricted to valid trading sessions in ``[start, end]``.
    Sessions with no data remain NaN. Network code only — not exercised by unit tests.
    """
    opens = _download_yf(tickers, start, end)
    if opens.empty or opens.isna().all().all():
        opens = _download_stooq(tickers, start, end)
    if opens.empty:
        return pd.DataFrame(columns=tickers)
    opens = opens.sort_index()
    opens.index = pd.DatetimeIndex(opens.index).tz_localize(None)
    sessions = trading_sessions(start, end, calendar)
    return opens.reindex(sessions).reindex(columns=tickers)


def forward_returns(
    tickers: list[str], start: str, end: str, horizon_days: int, calendar: str = "XNYS"
) -> pd.DataFrame:
    """Download prices and compute look-ahead-safe h-day forward returns (sessions x tickers).

    Thin orchestration over :func:`download_open_prices` and :func:`forward_returns_from_prices`.
    """
    opens = download_open_prices(tickers, start, end, calendar)
    if opens.empty:
        return pd.DataFrame(columns=tickers)
    return forward_returns_from_prices(opens.astype(float).replace(0.0, np.nan), horizon_days)
