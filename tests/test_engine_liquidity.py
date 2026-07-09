"""
Liquidity engine: known-answer arithmetic for every proxy, each rejection
reason, unknown-not-gated (fail-open) semantics, and total behavior on
None / empty / garbage input.

    venv/Scripts/python.exe -m pytest tests/test_engine_liquidity.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiquidityConfig  # noqa: E402
from constants import IST  # noqa: E402
from engines.liquidity import assess  # noqa: E402


# ---------------------------------------------------------------- fixtures

def daily(close, volume):
    return pd.DataFrame({"Close": close, "Volume": volume})


def intraday_index(dates, candles=12):
    """5-min IST index, `candles` bars per session starting 09:15."""
    parts = [pd.date_range(f"{d} 09:15", periods=candles, freq="5min", tz=IST)
             for d in dates]
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def flat_intraday(dates, price, volume, candles=12):
    """Zero-range 5-min frame (High=Low=Close) — spread proxy = 0%."""
    idx = intraday_index(dates, candles)
    return pd.DataFrame({"High": price, "Low": price, "Close": price,
                         "Volume": volume}, index=idx, dtype=float)


GOOD_DAILY = daily([100.0] * 6, [1e6, 2e6, 3e6, 4e6, 5e6, 6e6])
# turnover ₹: 1e8..6e8; median = (3e8+4e8)/2 = 3.5e8 = 35.0 crore


# ------------------------------------------------------------ known answers

def test_median_turnover_known_answer():
    r = assess("TEST", daily_df=GOOD_DAILY)
    # median(1e8..6e8) = 3.5e8 ₹ ÷ 1e7 = 35.0 cr
    assert r.values["median_turnover_cr"] == 35.0
    assert r.ok and r.values["liquid"] is True
    assert r.engine == "liquidity"


def test_turnover_uses_lookback_tail_only():
    # 25 rows: first 5 at ₹100cr/day, then 10×₹10cr + 10×₹20cr.
    # tail(20) median = (10+20)/2 = 15cr; a full-25 median would be 20cr.
    close = [1000.0] * 5 + [100.0] * 20
    vol = [1e6] * 5 + [1e6] * 10 + [2e6] * 10
    r = assess("TEST", daily_df=daily(close, vol))
    assert r.values["median_turnover_cr"] == 15.0


def test_spread_proxy_last_two_sessions_only():
    # 3 sessions × 12 candles. Session 1 spread = 10/100 = 10% (must be
    # EXCLUDED); session 2 = 0.2/100 = 0.2%; session 3 = 0.4/100 = 0.4%.
    # Median over last 2 sessions (12×0.2, 12×0.4) = 0.3%. If session 1
    # leaked in, the 36-row median would be 0.4%.
    idx = intraday_index(["2026-07-06", "2026-07-07", "2026-07-08"])
    df = pd.DataFrame({
        "High": [110.0] * 12 + [100.2] * 12 + [100.4] * 12,
        "Low": [100.0] * 36,
        "Close": [100.0] * 36,
        "Volume": [1000.0] * 36,
    }, index=idx)
    r = assess("TEST", intraday_df=df)
    assert r.values["spread_proxy_pct"] == pytest.approx(0.3)
    # per-min value uses the FULL frame: median(100×1000)=1e5 ÷ 5 = 20000
    assert r.values["per_min_traded_value_inr"] == 20000.0
    assert r.ok and r.values["liquid"] is True  # 0.3 ≤ 0.35 default ceiling


def test_spread_proxy_naive_index_localized():
    # Same frame, tz-naive index — must be treated as IST, not crash.
    df = flat_intraday(["2026-07-07", "2026-07-08"], 100.0, 1000.0)
    df.index = df.index.tz_localize(None)
    r = assess("TEST", intraday_df=df)
    assert r.values["spread_proxy_pct"] == 0.0


def test_participation_and_slippage_known_answer():
    # per-min: median(200×1000)=200000 ₹/5-min ÷ 5 = 40000 ₹/min.
    # participation = 1000/40000×100 = 2.5%; slippage = 0.05×2.5 = 0.125%.
    df = flat_intraday(["2026-07-07", "2026-07-08"], 200.0, 1000.0)
    r = assess("TEST", intraday_df=df, order_value_inr=1000.0)
    assert r.values["per_min_traded_value_inr"] == 40000.0
    assert r.values["participation_pct"] == 2.5
    assert r.values["est_slippage_pct"] == pytest.approx(0.125)
    assert r.ok and r.values["liquid"] is True


# --------------------------------------------------------------- rejections

def test_reject_low_turnover():
    # 10×1000 = ₹10,000/day = 0.001cr < 5cr default floor.
    r = assess("ILLIQ", daily_df=daily([10.0] * 6, [1000.0] * 6))
    assert not r.ok
    assert "turnover" in r.reject_reason.lower()
    assert r.values["liquid"] is False


def test_reject_wide_spread():
    # (101−100)/100×100 = 1.0% > 0.35% default ceiling.
    idx = intraday_index(["2026-07-07", "2026-07-08"])
    df = pd.DataFrame({"High": 101.0, "Low": 100.0, "Close": 100.0,
                       "Volume": 1000.0}, index=idx, dtype=float)
    r = assess("WIDE", intraday_df=df)
    assert not r.ok
    assert "spread" in r.reject_reason.lower()
    assert r.values["liquid"] is False


def test_reject_participation():
    # per-min = 100×10/5 = 200 ₹/min; order ₹1e5 → 50000% > 5% cap.
    # Turnover (35cr) and spread (0%) both pass, so participation is the
    # first — and only — breach.
    df = flat_intraday(["2026-07-07", "2026-07-08"], 100.0, 10.0)
    r = assess("BIG", daily_df=GOOD_DAILY, intraday_df=df,
               order_value_inr=1e5)
    assert not r.ok
    assert "participation" in r.reject_reason.lower()
    assert r.values["participation_pct"] == 50000.0


def test_first_breach_wins_turnover_before_spread():
    bad_daily = daily([10.0] * 6, [1000.0] * 6)          # 0.001cr — breach
    idx = intraday_index(["2026-07-07", "2026-07-08"])
    wide = pd.DataFrame({"High": 101.0, "Low": 100.0, "Close": 100.0,
                         "Volume": 1000.0}, index=idx, dtype=float)  # 1% — breach
    r = assess("BOTH", daily_df=bad_daily, intraday_df=wide)
    assert not r.ok
    assert "turnover" in r.reject_reason.lower()
    assert "spread" not in r.reject_reason.lower()


def test_cfg_override_is_respected():
    # Same 35cr stock rejects once the floor is raised above it.
    cfg = LiquidityConfig(min_median_turnover_cr=50.0)
    r = assess("TEST", daily_df=GOOD_DAILY, cfg=cfg)
    assert not r.ok and r.values["liquid"] is False


# ----------------------------------------------- unknown-not-gated / total

def test_all_unknown_fails_open():
    r = assess("NODATA")
    assert r.ok and r.reject_reason == ""
    assert all(v is None for v in r.values.values())    # incl. liquid=None
    assert any("turnover unknown" in d for d in r.diagnostics)
    assert any("spread proxy unknown" in d for d in r.diagnostics)
    assert any("participation unknown" in d for d in r.diagnostics)


def test_empty_frames_are_total():
    r = assess("EMPTY", daily_df=pd.DataFrame(),
               intraday_df=pd.DataFrame(), order_value_inr=5000.0)
    assert r.ok and r.values["liquid"] is None


def test_thin_frames_yield_none_not_reject():
    thin_daily = daily([100.0] * 4, [1e6] * 4)              # 4 < 5 rows
    thin_intra = flat_intraday(["2026-07-08"], 100.0, 1000.0, candles=19)
    r = assess("THIN", daily_df=thin_daily, intraday_df=thin_intra)
    assert r.ok
    assert r.values["median_turnover_cr"] is None
    assert r.values["spread_proxy_pct"] is None
    assert r.values["per_min_traded_value_inr"] is None
    assert r.values["liquid"] is None


def test_missing_columns_are_total():
    junk = pd.DataFrame({"Foo": range(30)})
    r = assess("JUNK", daily_df=junk, intraday_df=junk)
    assert r.ok
    assert r.values["median_turnover_cr"] is None
    assert r.values["spread_proxy_pct"] is None


def test_unknown_turnover_does_not_block_known_pass():
    # Only intraday known and passing ⇒ liquid True despite unknown turnover.
    df = flat_intraday(["2026-07-07", "2026-07-08"], 100.0, 1000.0)
    r = assess("PART", intraday_df=df)
    assert r.ok and r.values["liquid"] is True
    assert any("turnover unknown" in d for d in r.diagnostics)


def test_values_are_plain_python_scalars():
    # np.float64 subclasses float, so check EXACT type, not isinstance.
    df = flat_intraday(["2026-07-07", "2026-07-08"], 200.0, 1000.0)
    r = assess("TYPES", daily_df=GOOD_DAILY, intraday_df=df,
               order_value_inr=1000.0)
    for key, val in r.values.items():
        assert type(val) in (float, bool, int, str, type(None)), \
            f"{key} is {type(val)}"
        assert not isinstance(val, np.generic), f"{key} is a numpy scalar"
    # feature snapshot keying comes from the base contract
    assert "liquidity_median_turnover_cr" in r.feature_dict()
