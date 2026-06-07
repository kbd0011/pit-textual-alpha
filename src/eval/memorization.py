"""L4 memorization probe: isolate and subtract the memorized component of a textual signal.

A modern LLM may "predict" a stock's move not because it *read* the disclosure but because it *remembers* what
happened to that company around that date (training-data leakage). The probe measures exactly that: score the
same model with the **entity identifiers present but the real disclosure text removed**. Any predictive power
from this no-context input is memorization, not real-time reading — so we subtract it.

Two granularities are provided:
- :func:`memorization_adjusted_signal` strips the memorized component at the *signal* level (preferred: backtest
  the residual signal).
- :func:`memorization_adjusted_performance` subtracts at the *metric* level (e.g. adjusted Sharpe = full - probe)
  for the headline ablation table.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

__all__ = [
    "DISCLOSURE_REMOVED",
    "build_probe_text",
    "probe_signal_series",
    "memorization_adjusted_signal",
    "memorization_adjusted_performance",
]

DISCLOSURE_REMOVED = "[DISCLOSURE TEXT REMOVED]"


def build_probe_text(
    company_name: str,
    as_of: datetime,
    period_label: str | None = None,
    extra_identifiers: dict[str, str] | None = None,
) -> str:
    """Construct the no-context probe body: entity identifiers only, real disclosure blanked.

    The model is handed who/when but not what-was-said. Whatever signal it returns is memorized association.
    """
    parts = [f"Company: {company_name}"]
    if period_label:
        parts.append(f"Period: {period_label}")
    parts.append(f"As of: {as_of.date().isoformat()}")
    for key, val in (extra_identifiers or {}).items():
        parts.append(f"{key}: {val}")
    parts.append(DISCLOSURE_REMOVED)
    return "\n".join(parts)


def probe_signal_series(scorer, entities: dict[str, dict], as_of: datetime) -> pd.Series:
    """Score the no-context probe for each entity; returns a Series indexed by ticker.

    Parameters
    ----------
    scorer : object
        Anything with ``.score(text, as_of) -> float``. NOTE: do not anonymize here — the entity must be
        present for the probe to measure memorization.
    entities : dict[str, dict]
        ``{ticker: {"name": ..., "period_label": ...}}``. Only ``name`` is required.
    as_of : datetime
        Decision time passed through to the scorer (and its cutoff guard, if any).
    """
    scores = {
        ticker: scorer.score(
            build_probe_text(meta["name"], as_of, meta.get("period_label")), as_of
        )
        for ticker, meta in entities.items()
    }
    return pd.Series(scores, dtype=float)


def memorization_adjusted_signal(full: pd.DataFrame, probe) -> pd.DataFrame:
    """Residual signal after removing memorization: ``full - probe``, clipped to [-1, 1].

    Parameters
    ----------
    full : pd.DataFrame
        Dates x tickers full-text signals.
    probe : pd.Series | pd.DataFrame
        Per-ticker probe signals (Series, broadcast across dates) or a full dates x tickers probe panel.

    Returns
    -------
    pd.DataFrame
        The memorization-adjusted signal panel. If ``full == probe`` everywhere (signal is *entirely*
        memorization), the result is all zeros — i.e. no genuine real-time alpha.
    """
    adj = full.sub(probe, axis=1) if isinstance(probe, pd.Series) else full.sub(probe)
    return adj.clip(lower=-1.0, upper=1.0)


def memorization_adjusted_performance(full_value, probe_value):
    """Subtract probe performance from full performance (scalars or aligned pandas objects).

    A positive residual is the share of performance *not* explained by memorization; a residual near zero means
    the apparent edge was mostly the model recalling the future.
    """
    return full_value - probe_value
