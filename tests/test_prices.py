"""Tests for look-ahead-safe forward returns (no network)."""
import numpy as np
import pandas as pd
import pytest
from src.ingest.prices import forward_returns_from_prices


def _frame(values, ticker="AAA"):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({ticker: values}, index=idx)


def test_forward_return_uses_next_open_to_t_plus_1_plus_h():
    # Opens: 10, 11, 12, 13, 14, 15. h=2.
    # Signal at t=0 -> enter at open[1]=11, exit at open[1+2]=open[3]=13 -> 13/11 - 1.
    prices = _frame([10, 11, 12, 13, 14, 15])
    fwd = forward_returns_from_prices(prices, horizon_days=2)
    assert fwd["AAA"].iloc[0] == pytest.approx(13 / 11 - 1)
    assert fwd["AAA"].iloc[1] == pytest.approx(14 / 12 - 1)
    assert fwd["AAA"].iloc[2] == pytest.approx(15 / 13 - 1)


def test_last_h_plus_1_rows_are_nan():
    prices = _frame([10, 11, 12, 13, 14, 15])
    fwd = forward_returns_from_prices(prices, horizon_days=2)
    # With h=2, the last h+1 = 3 rows have no realized future.
    assert fwd["AAA"].iloc[-3:].isna().all()
    assert fwd["AAA"].iloc[:-3].notna().all()


def test_no_lookahead_swap_future_price_changes_only_future_rows():
    """Adversarial leakage test: mutating a future price must NOT change earlier forward returns.

    If row t's value depended on any price at or before t (look-ahead), perturbing the *last* price would
    leak backwards. It must only affect rows whose window includes that price.
    """
    base = _frame([10, 11, 12, 13, 14, 15])
    fwd_base = forward_returns_from_prices(base, horizon_days=2)
    perturbed = base.copy()
    perturbed.iloc[-1] = 99.0  # change open of the very last session
    fwd_pert = forward_returns_from_prices(perturbed, horizon_days=2)
    # Row 0 window is open[1..3] -> unaffected; row 2 window is open[3..5] -> affected.
    assert fwd_base["AAA"].iloc[0] == pytest.approx(fwd_pert["AAA"].iloc[0])
    assert not np.isclose(fwd_base["AAA"].iloc[2], fwd_pert["AAA"].iloc[2])


def test_multi_ticker_alignment():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"AAA": [10, 20, 40, 80], "BBB": [5, 5, 5, 5]}, index=idx)
    fwd = forward_returns_from_prices(prices, horizon_days=1)
    # h=1: signal at t -> open[t+1] to open[t+2].
    assert fwd["AAA"].iloc[0] == pytest.approx(40 / 20 - 1)
    assert fwd["BBB"].iloc[0] == pytest.approx(0.0)
    assert fwd.iloc[-2:].isna().all().all()  # last h+1 = 2 rows NaN


def test_rejects_bad_horizon_and_unsorted_index():
    prices = _frame([10, 11, 12])
    with pytest.raises(ValueError):
        forward_returns_from_prices(prices, horizon_days=0)
    with pytest.raises(ValueError):
        forward_returns_from_prices(prices.iloc[::-1], horizon_days=1)
