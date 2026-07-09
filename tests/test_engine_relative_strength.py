"""
Relative Strength Engine: known-answer excess returns, an exact-beta linear
construction, beta's index-intersection alignment, rs_rank percentile, and the
totality / JSON-safety contract of evaluate().

Fully offline and deterministic — no RNG, no fetching; every frame is built
from hand-chosen returns so the expected numbers are computed by hand.

    venv/Scripts/python.exe -m pytest tests/test_engine_relative_strength.py -q
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines import relative_strength as rs  # noqa: E402


# ---------- deterministic frame builders (no RNG) ----------

# A fixed, varied return path (nonzero variance) used to build price series.
# 24 returns → 25 closes; long enough to exercise the default beta_lookback=20.
IDX_RETS = [0.010, -0.020, 0.015, 0.030, -0.010, 0.020, 0.005, -0.015,
            0.025, 0.010, -0.005, 0.018, -0.012, 0.022, 0.008, -0.018,
            0.014, 0.006, -0.009, 0.017, 0.011, -0.013, 0.019, 0.007]


def _closes_from_returns(rets, start=100.0):
    """Compound a return path into a Close list: closes[k] = closes[k-1](1+r).
    pct_change() of the result recovers the input returns exactly."""
    closes = [float(start)]
    for r in rets:
        closes.append(closes[-1] * (1.0 + r))
    return closes


def _frame(closes, start_date="2025-01-01"):
    """Minimal daily OHLC frame with a DatetimeIndex — matched timestamps let
    beta()'s inner join line the two series up correctly."""
    idx = pd.date_range(start_date, periods=len(closes), freq="D")
    c = pd.Series([float(x) for x in closes], index=idx)
    return pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 100000.0},
        index=idx)


# ---------- pct_return ----------

def test_pct_return_known_answer():
    # last=110, ref (5 bars back) = closes[0] = 100 → +10%.
    df = _frame([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    assert abs(rs.pct_return(df, 5) - 10.0) < 1e-9
    # Accepts a bare Close Series too.
    assert abs(rs.pct_return(df["Close"], 5) - 10.0) < 1e-9


def test_pct_return_too_short_is_none():
    df = _frame([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])  # 6 bars
    assert rs.pct_return(df, 20) is None          # need 21 bars
    assert rs.pct_return(df, 5) is not None        # 6 bars is exactly enough


def test_pct_return_zero_reference_is_none():
    # Reference price 0 → division undefined → None, not an exception.
    df = _frame([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert rs.pct_return(df, 5) is None


# ---------- relative_strength ----------

def test_relative_strength_excess_return():
    # stock +10% over 5 bars, index +4% over 5 bars → rs_5 ≈ +6 pp.
    stock = _frame([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    index = _frame([100.0, 101.0, 102.0, 103.0, 103.5, 104.0])
    out = rs.relative_strength(stock, index, windows=(5, 20))
    assert abs(out["stock_ret_5"] - 10.0) < 1e-9
    assert abs(out["index_ret_5"] - 4.0) < 1e-9
    assert abs(out["rs_5"] - 6.0) < 1e-9
    # Only 6 bars → the 20-bar window is unavailable everywhere.
    assert out["rs_20"] is None
    assert out["stock_ret_20"] is None and out["index_ret_20"] is None


def test_relative_strength_missing_index_rs_none():
    stock = _frame([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    out = rs.relative_strength(stock, None, windows=(5,))
    assert out["stock_ret_5"] is not None   # stock leg still computable
    assert out["index_ret_5"] is None
    assert out["rs_5"] is None


# ---------- beta ----------

def test_beta_exact_linear_construction():
    # stock returns are EXACTLY 1.5× index returns → beta = 1.5.
    index = _frame(_closes_from_returns(IDX_RETS))
    stock = _frame(_closes_from_returns([1.5 * r for r in IDX_RETS]))
    b = rs.beta(stock, index, lookback=20)
    assert b is not None and abs(b - 1.5) < 1e-9


def test_beta_intersection_ignores_unshared_bars():
    # Stock carries extra TRAILING dates the index lacks; the inner join drops
    # them, so beta on the shared range is still exactly 1.5. This exercises the
    # index-intersection alignment (mismatched calendars must not corrupt it).
    index = _frame(_closes_from_returns(IDX_RETS))
    stock_rets = [1.5 * r for r in IDX_RETS] + [0.02, -0.01, 0.03]
    stock = _frame(_closes_from_returns(stock_rets))
    assert len(stock) > len(index)
    b = rs.beta(stock, index, lookback=20)
    assert b is not None and abs(b - 1.5) < 1e-9


def test_beta_zero_index_variance_is_none():
    # Constant index → all index returns 0 → var 0 → beta undefined → None.
    index = _frame([100.0] * 25)
    stock = _frame(_closes_from_returns(IDX_RETS))
    assert rs.beta(stock, index, lookback=20) is None


def test_beta_insufficient_overlap_is_none():
    one = _frame([100.0])
    assert rs.beta(one, one, lookback=20) is None


# ---------- rs_rank ----------

def test_rs_rank_known_percentile():
    peers = [1.0, 2.0, 3.0, 4.0]
    assert rs.rs_rank(3.5, peers) == 75.0     # 3 of 4 peers strictly below
    assert rs.rs_rank(5.0, peers) == 100.0    # beats all
    assert rs.rs_rank(0.0, peers) == 0.0      # below all


def test_rs_rank_empty_or_missing_is_none():
    assert rs.rs_rank(2.0, []) is None
    assert rs.rs_rank(None, [1.0, 2.0, 3.0]) is None


# ---------- evaluate: contract + totality ----------

def test_evaluate_no_index_rs_none_with_diagnostic():
    stock = _frame([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    res = rs.evaluate(stock, index_df=None)
    assert res.engine == "relative_strength"
    assert res.ok is True
    # RS / index legs None, but the stock leg is still populated.
    assert res.values["rs_5"] is None
    assert res.values["index_ret_5"] is None
    assert res.values["stock_ret_5"] is not None
    assert any("no index" in d for d in res.diagnostics)


def test_evaluate_one_row_all_none_no_exception():
    one = _frame([100.0])
    res = rs.evaluate(one, one)            # must not raise
    assert res.ok is True
    assert all(v is None for v in res.values.values())
    assert len(res.diagnostics) >= 1       # the all-None case leaves a note


def test_evaluate_thin_stock_frame_all_none():
    # A None stock frame is honestly all-None with a diagnostic, never a raise.
    res = rs.evaluate(None, None)
    assert res.ok is True
    assert all(v is None for v in res.values.values())
    assert res.diagnostics


def test_evaluate_json_safe_scalar_values():
    index = _frame(_closes_from_returns(IDX_RETS))
    stock = _frame(_closes_from_returns([1.5 * r for r in IDX_RETS]))
    res = rs.evaluate(stock, index, peers=[-5.0, 0.0, 5.0, 10.0])
    # json.dumps must not raise → no numpy scalars, no NaN leaked through.
    json.dumps(res.values)
    for k, v in res.values.items():
        assert v is None or isinstance(v, float), f"{k}: {type(v)}"
    # 25 bars is enough for both windows and the beta estimate.
    assert res.values["rs_5"] is not None
    assert res.values["rs_20"] is not None
    assert abs(res.values["beta"] - 1.5) < 1e-6
    # rs_rank was requested (peers given) and must be a valid percentile.
    assert res.values["rs_rank"] is not None
    assert 0.0 <= res.values["rs_rank"] <= 100.0
