"""
engines/price_action.py — known-answer tests for swing structure, BOS/CHoCH,
and support/resistance. All offline and deterministic (no RNG): each series is
a hand-built OHLC staircase whose pivots, trend, and break direction are known
by construction, so the assertions pin exact structural reads.

    venv/Scripts/python.exe -m pytest tests/test_engine_price_action.py -q
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.base import EngineResult                      # noqa: E402
from engines.price_action import (                         # noqa: E402
    break_of_structure,
    evaluate,
    structure,
    support_resistance,
    swing_points,
)


def _ohlc(bases, spread=0.5):
    """Build an OHLC frame from a list of base closes; High/Low straddle the
    base by `spread`, so swing highs track the base peaks and swing lows the
    base troughs. spread=0 makes price == base for exact pivot assertions."""
    idx = pd.date_range("2026-01-01", periods=len(bases), freq="D")
    return pd.DataFrame(
        {
            "Open": bases,
            "High": [b + spread for b in bases],
            "Low": [b - spread for b in bases],
            "Close": bases,
            "Volume": [100000] * len(bases),
        },
        index=idx,
    )


# Clean staircase UPtrend: peaks 106@3, 109@9 (HH); troughs 103@6, 106@12 (HL).
UPTREND = [100, 102, 104, 106, 105, 104, 103, 105, 107, 109,
           108, 107, 106, 108, 110, 112]

# Mirror DOWNtrend: peaks 109@6, 103@12 (LH); troughs 106@3, 100@9 (LL).
DOWNTREND = [112, 110, 108, 106, 107, 108, 109, 106, 103, 100,
             101, 102, 103, 100, 97, 94]


# ---------------------------------------------------------------------------
# swing_points — exact pivots on a hand-built zigzag
# ---------------------------------------------------------------------------

def test_swing_points_finds_exact_pivots():
    # zigzag: up to 14@2, down to 10@4, up to 14@6. With n=2 the confirmable
    # centres are 2..6, so exactly high@2, low@4, high@6 are pivots.
    df = _ohlc([10, 12, 14, 12, 10, 12, 14, 12, 10], spread=0.0)
    pts = swing_points(df, n=2, atr_mult=0.0)   # filter off → pure fractals
    assert pts == [
        {"idx": 2, "type": "high", "price": 14.0},
        {"idx": 4, "type": "low", "price": 10.0},
        {"idx": 6, "type": "high", "price": 14.0},
    ]


def test_swing_points_empty_when_too_short():
    # Fewer than 2n+1 bars ⇒ nothing can be confirmed.
    df = _ohlc([10, 11, 12, 11], spread=0.0)
    assert swing_points(df, n=2, atr_mult=0.0) == []


def test_swing_points_chronological_and_typed():
    pts = swing_points(_ohlc(UPTREND))
    assert [p["type"] for p in pts]                 # non-empty
    assert pts == sorted(pts, key=lambda p: p["idx"])   # chronological
    assert all(p["type"] in ("high", "low") for p in pts)


# ---------------------------------------------------------------------------
# structure — trend classification
# ---------------------------------------------------------------------------

def test_structure_uptrend():
    st = structure(_ohlc(UPTREND))
    assert st["trend"] == "up"
    assert st["hh"] is True and st["hl"] is True
    assert st["lh"] is False and st["ll"] is False


def test_structure_downtrend():
    st = structure(_ohlc(DOWNTREND))
    assert st["trend"] == "down"
    assert st["lh"] is True and st["ll"] is True
    assert st["hh"] is False and st["hl"] is False


def test_structure_range_on_broadening():
    # HH (112>108) with LL (100<104): neither clean up nor down ⇒ range.
    bases = [102, 104, 106, 108, 106, 105, 104, 107, 110, 112,
             108, 104, 100, 103, 106, 108]
    st = structure(_ohlc(bases))
    assert st["trend"] == "range"
    assert st["hh"] is True and st["ll"] is True
    assert st["hl"] is False and st["lh"] is False


def test_structure_none_when_insufficient():
    st = structure(_ohlc([10, 11, 12, 11, 10], spread=0.0))  # <2 highs & lows
    assert st["trend"] is None


# ---------------------------------------------------------------------------
# break_of_structure — BOS vs CHoCH
# ---------------------------------------------------------------------------

def test_choch_bearish_break_against_uptrend():
    # Clean uptrend, then a decisive drop: last close 100 slices below the most
    # recent confirmed swing low (106@12) while structure still reads "up" —
    # the textbook change-of-character.
    df = _ohlc(UPTREND + [108, 104, 100])
    assert structure(df)["trend"] == "up"          # prevailing trend intact
    res = break_of_structure(df)
    assert res["bos"] == "bearish"
    assert res["choch"] is True


def test_bos_bullish_continuation_with_uptrend():
    # Last close pushes ABOVE the last swing high, with the trend → BOS, not
    # CHoCH (continuation).
    df = _ohlc(UPTREND)                             # ends at 112, above high 109
    res = break_of_structure(df)
    assert res["bos"] == "bullish"
    assert res["choch"] is False


def test_choch_bullish_break_against_downtrend():
    # Symmetric case: a downtrend whose last close reclaims the last swing high.
    df = _ohlc(DOWNTREND + [96, 100, 105])
    assert structure(df)["trend"] == "down"
    res = break_of_structure(df)
    assert res["bos"] == "bullish"
    assert res["choch"] is True


# ---------------------------------------------------------------------------
# support_resistance
# ---------------------------------------------------------------------------

def test_support_resistance_brackets_price():
    # Uptrend then a pull-back so the last close (107.5) sits between a support
    # (swing low ~106.5) and a resistance (swing high ~109.5).
    df = _ohlc(UPTREND + [111, 107.5])
    sr = support_resistance(df)
    last_close = 107.5
    assert sr["nearest_support"] is not None and sr["nearest_resistance"] is not None
    assert sr["nearest_support"] < last_close < sr["nearest_resistance"]
    # Distances are positive: support below, resistance above.
    assert sr["dist_support_pct"] > 0
    assert sr["dist_resistance_pct"] > 0


# ---------------------------------------------------------------------------
# evaluate — EngineResult contract + totality
# ---------------------------------------------------------------------------

def test_evaluate_uptrend_populated_and_json_safe():
    res = evaluate(_ohlc(UPTREND))
    assert isinstance(res, EngineResult)
    assert res.engine == "price_action"
    assert res.ok is True                           # feature engine never gates
    assert res.values["trend"] == "up"
    assert res.values["hh"] is True and res.values["hl"] is True
    assert res.values["n_swings"] >= 4
    assert set(res.values) == {
        "trend", "last_high", "last_low", "hh", "hl", "lh", "ll", "bos",
        "choch", "n_swings", "nearest_support", "nearest_resistance",
        "dist_support_pct", "dist_resistance_pct",
    }
    json.dumps(res.values)                          # numpy scalars would raise
    for v in res.values.values():                   # every value is a scalar
        assert v is None or isinstance(v, (bool, int, float, str))
    assert "price_action_trend" in res.feature_dict()


def test_evaluate_choch_flags_in_values():
    res = evaluate(_ohlc(UPTREND + [108, 104, 100]))
    assert res.values["choch"] is True
    assert res.values["bos"] == "bearish"
    assert res.values["trend"] == "up"


def test_evaluate_thin_df_all_none_no_exception():
    res = evaluate(_ohlc([100, 101, 102], spread=0.0))   # 3 bars < 2n+1
    assert res.ok is True
    assert res.diagnostics                          # says WHY it is empty
    assert res.values["n_swings"] == 0
    for key, val in res.values.items():
        if key == "n_swings":
            continue
        assert val is None
    json.dumps(res.values)


@pytest.mark.parametrize("bad", [None, pd.DataFrame(), "not a frame", 42])
def test_evaluate_total_on_garbage(bad):
    res = evaluate(bad)
    assert isinstance(res, EngineResult) and res.ok is True
    assert res.values["n_swings"] == 0
    assert res.values["trend"] is None
    assert res.diagnostics


def test_swing_points_total_on_garbage():
    for bad in (None, pd.DataFrame(), "nope", 7):
        assert swing_points(bad) == []
