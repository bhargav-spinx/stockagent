"""
Phase-3 extensions to the Phase-2 gap/volume/vwap engines: gap classification,
volume curve/acceleration/institutional-proxy, VWAP state machine. Known-answer
on synthetic frames. The scorer composition stays byte-identical — pinned
separately by test_golden_parity.

    venv/Scripts/python.exe -m pytest tests/test_engine_extensions.py
"""
import json
import math
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines import gap as gap_engine       # noqa: E402
from engines import volume as vol_engine     # noqa: E402
from engines import vwap as vwap_engine      # noqa: E402
from scanner_indicators import split_sessions  # noqa: E402


def _sessions(days, per_day=25, start_price=500.0):
    """Build a multi-session 5-min IST frame from a list of per-day dicts."""
    rows, idx = [], []
    for d, spec in enumerate(days):
        day = pd.Timestamp("2025-06-02", tz="Asia/Kolkata") + pd.Timedelta(days=d)
        base = spec.get("base", start_price)
        vol = spec.get("vol", 100_000)
        for i in range(spec.get("n", per_day)):
            ts = day + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(minutes=5 * i)
            o = base + spec.get("drift", 0.0) * i
            c = o + spec.get("cc", 0.1)
            hi = max(o, c) + 1.0
            lo = min(o, c) - 1.0
            rows.append((o, hi, lo, c, vol))
            idx.append(ts)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(idx))


# ---------- gap classification ----------

def test_classify_gap_size_and_character():
    cfg = gap_engine.CONFIG.gap_class
    # gap 2%, atr 1% → gap_atr 2.0 (>= normal 1.5). High RVOL → breakaway.
    c = gap_engine.classify_gap(2.0, atr_pct=1.0, rvol=2.0)
    assert c["gap_atr"] == pytest.approx(2.0)
    assert c["gap_class"] == "breakaway"
    # same size, thin volume → exhaustion
    assert gap_engine.classify_gap(2.0, 1.0, rvol=1.0)["gap_class"] == "exhaustion"
    # very large gap + volume → runaway
    assert gap_engine.classify_gap(4.0, 1.0, rvol=2.0)["gap_class"] == "runaway"
    # tiny
    assert gap_engine.classify_gap(0.3, 1.0, rvol=2.0)["gap_class"] == "tiny"
    # normal
    assert gap_engine.classify_gap(1.0, 1.0, rvol=2.0)["gap_class"] == "normal"


def test_classify_gap_totality_on_unknowns():
    assert gap_engine.classify_gap(None, 1.0, 1.0)["gap_class"] is None
    # no ATR → cannot size, but fill_candidate from retrace still resolves
    c = gap_engine.classify_gap(2.0, None, 1.0, retrace_frac=0.7)
    assert c["gap_atr"] is None and c["gap_class"] is None
    assert c["fill_candidate"] is True


def test_gap_retrace_and_evaluate_json_safe():
    # day1 close 500; day2 opens 510 (gap +2%) then trades back to 505 (half fill)
    df = _sessions([
        {"base": 500.0, "cc": 0.0, "drift": 0.0, "n": 25, "vol": 100_000},
        {"base": 510.0, "cc": 0.0, "drift": -0.2, "n": 25, "vol": 200_000},
    ])
    today, priors = split_sessions(df)
    res = gap_engine.evaluate(today, priors, "long", atr_pct=1.0, rvol=2.0)
    assert res.values["gap_pct"] == pytest.approx(2.0, abs=0.2)
    assert res.values["retrace_frac"] is not None
    json.dumps(res.values)


# ---------- volume extensions ----------

def test_volume_acceleration_known():
    # last 3 vols mean 300, prior 3 mean 100 → accel 3.0
    v = [100] * 3 + [300] * 3
    df = pd.DataFrame({"Open": [1]*6, "High": [1]*6, "Low": [1]*6,
                       "Close": [1]*6, "Volume": v})
    assert vol_engine.volume_acceleration(df, window=3) == pytest.approx(3.0)
    assert vol_engine.volume_acceleration(df.head(3), window=3) is None


def test_volume_curve_ratio_same_position():
    # today's current (last) candle vs prior days' volume at the same position
    df = _sessions([
        {"base": 500, "n": 5, "vol": 100_000},
        {"base": 500, "n": 5, "vol": 100_000},
        {"base": 500, "n": 3, "vol": 300_000},   # today, 3 candles, 3x volume
    ])
    today, priors = split_sessions(df)
    r = vol_engine.volume_curve_ratio(today, priors)
    assert r == pytest.approx(3.0)


def test_institutional_proxy_bounds_and_none():
    assert vol_engine.institutional_proxy(None, None, None) is None
    p = vol_engine.institutional_proxy(3.0, 80.0, 0.9)
    assert 0.0 <= p <= 1.0 and p > 0.6           # strong on all three
    low = vol_engine.institutional_proxy(0.5, 10.0, 0.1)
    assert low < p


# ---------- vwap state machine ----------

def test_vwap_state_reclaim_and_time_above():
    # a session that starts below VWAP then closes above on the last bar
    rows, idx = [], []
    day = pd.Timestamp("2025-06-03 09:15", tz="Asia/Kolkata")
    prices = [100, 99, 98, 99, 101, 103]        # dips then reclaims
    for i, p in enumerate(prices):
        ts = day + pd.Timedelta(minutes=5 * i)
        rows.append((p, p + 0.5, p - 0.5, p, 1000))
        idx.append(ts)
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                      index=pd.DatetimeIndex(idx))
    st = vwap_engine.vwap_state(df, atr_val=1.0)
    assert st["above"] is True                   # closed above VWAP
    assert 0.0 <= st["time_above_frac"] <= 1.0
    assert st["dist_atr"] is not None
    json.dumps(st)


def test_phase3_features_aggregator():
    import features
    df = _sessions([
        {"base": 500, "cc": 0.1, "drift": 0.1, "n": 25, "vol": 100_000},
        {"base": 500, "cc": 0.1, "drift": 0.1, "n": 25, "vol": 120_000},
        {"base": 505, "cc": 0.2, "drift": 0.2, "n": 20, "vol": 300_000},
    ])
    f = features.phase3_features(df, index_df=None, delivery_pct=55.0,
                                 direction="long")
    assert isinstance(f, dict) and f, "expected non-empty p3 features"
    assert all(k.startswith("p3_") for k in f), list(f)[:5]
    # engine groups present
    assert any(k.startswith("p3_vol_") for k in f)      # volatility
    assert any(k.startswith("p3_pa_") for k in f)       # price action
    assert any(k.startswith("p3_vwap_") for k in f)     # vwap state
    assert any(k.startswith("p3_gap_") for k in f)      # gap class
    assert any(k.startswith("p3_vx_") for k in f)       # volume extras
    json.dumps(f)                                        # JSON-safe throughout


def test_phase3_features_total_on_garbage():
    import features
    import pandas as pd
    assert features.phase3_features(pd.DataFrame()) == {}   # empty → {} no raise
    assert isinstance(features.phase3_features(None), dict)


def test_vwap_state_totality_on_thin_df():
    df = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1],
                       "Volume": [1]},
                      index=pd.DatetimeIndex([pd.Timestamp("2025-06-03 09:15",
                                                           tz="Asia/Kolkata")]))
    st = vwap_engine.vwap_state(df)
    assert st["above"] in (True, False, None)    # no exception
    json.dumps(st)
