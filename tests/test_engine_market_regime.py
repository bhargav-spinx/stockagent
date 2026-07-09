"""
engines/market_regime.py (Phase 2): known-answer tests for the deterministic
regime labels — day_type branches on synthetic 5-min frames, VIX/breadth
threshold edges, index bias, holiday-shifted expiry, and the live_snapshot
wrapper's fetch-failure resilience + TTL cache. All offline: data_provider
fetchers are monkeypatched, never called for real.

    venv/Scripts/python.exe -m pytest tests/test_engine_market_regime.py
"""
import json
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_provider          # noqa: E402
import market_calendar        # noqa: E402
from config import CONFIG, RegimeConfig   # noqa: E402
from constants import IST     # noqa: E402
import engines.market_regime as mr        # noqa: E402

CFG = RegimeConfig()   # pinned defaults, immune to $STOCKAGENT_CONFIG overrides


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _session(rows, day):
    """5-min OHLCV session on `day` from (O, H, L, C) tuples."""
    idx = pd.date_range(f"{day} 09:15", periods=len(rows), freq="5min",
                        tz="Asia/Kolkata")
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}
         for (o, h, l, c) in rows],
        index=idx,
    )

# Prior session: close 100.0, range = 100.5 − 99.5 = 1.0 (the avg_range base).
PRIOR = [
    (100.0, 100.5, 99.5, 100.0),
    (100.0, 100.3, 99.8, 100.1),
    (100.1, 100.4, 99.6, 100.0),
]


def _two_day(today_rows):
    return pd.concat([_session(PRIOR, "2026-07-07"),
                      _session(today_rows, "2026-07-08")])


def _classify(index_df, **kw):
    kw.setdefault("now", IST.localize(datetime(2026, 7, 8, 10, 0)))
    return mr.classify_regime(index_df=index_df, cfg=CFG, **kw)


# ---------------------------------------------------------------------------
# day_type branches
# ---------------------------------------------------------------------------

def test_trend_up_day():
    # gap = (100.05−100)/100·100 = 0.05% < 0.3 → not a gap day.
    # today_range = 101.5 − 100.0 = 1.5 ≥ 1.3 × avg_range(1.0)   ✓
    # trend_strength = |101.45 − 100.05| / 1.5 = 0.9333 ≥ 0.6    ✓  up
    res = _classify(_two_day([
        (100.05, 100.4, 100.0, 100.3),
        (100.3, 101.0, 100.2, 100.9),
        (100.9, 101.5, 100.8, 101.45),
    ]))
    assert res.values["day_type"] == "trend_up"
    assert res.values["gap_open_pct"] == pytest.approx(0.05)
    assert res.values["trend_strength"] == pytest.approx(0.9333, abs=1e-4)
    assert res.ok is True and res.reject_reason == ""


def test_trend_down_day():
    # gap = −0.05%; range = 100.0 − 98.5 = 1.5 ≥ 1.3; net move down.
    # trend_strength = |98.55 − 99.95| / 1.5 = 0.9333
    res = _classify(_two_day([
        (99.95, 100.0, 99.6, 99.7),
        (99.7, 99.8, 99.0, 99.1),
        (99.1, 99.2, 98.5, 98.55),
    ]))
    assert res.values["day_type"] == "trend_down"
    assert res.values["trend_strength"] == pytest.approx(0.9333, abs=1e-4)


def test_gap_go_day():
    # gap = (101−100)/100·100 = 1.0% ≥ 0.3 → gap day.
    # fill = (open 101.0 − low 100.9) / gap 1.0 = 0.1 < 0.5 → gap_go
    res = _classify(_two_day([
        (101.0, 101.5, 100.9, 101.3),
        (101.3, 101.8, 101.2, 101.7),
    ]))
    assert res.values["day_type"] == "gap_go"
    assert res.values["gap_open_pct"] == pytest.approx(1.0)


def test_gap_fill_day_up_gap():
    # gap up 1.0%; fill = (101.0 − low 100.4) / 1.0 = 0.6 ≥ 0.5 → gap_fill
    res = _classify(_two_day([
        (101.0, 101.2, 100.4, 100.6),
        (100.6, 100.9, 100.5, 100.8),
    ]))
    assert res.values["day_type"] == "gap_fill"


def test_gap_fill_day_down_gap():
    # gap down −1.0%; fill = (high 99.6 − open 99.0) / 1.0 = 0.6 → gap_fill
    res = _classify(_two_day([
        (99.0, 99.6, 98.9, 99.5),
        (99.5, 99.55, 99.2, 99.3),
    ]))
    assert res.values["day_type"] == "gap_fill"
    assert res.values["gap_open_pct"] == pytest.approx(-1.0)


def test_range_day():
    # gap 0.05% < 0.3; range = 100.6 − 100.0 = 0.6 < 1.3 × 1.0 → range
    res = _classify(_two_day([
        (100.05, 100.4, 100.0, 100.2),
        (100.2, 100.6, 100.1, 100.3),
    ]))
    assert res.values["day_type"] == "range"


@pytest.mark.parametrize("bad", [
    None,
    pd.DataFrame(),
    "not a frame",
])
def test_unknown_on_garbage(bad):
    res = _classify(bad)
    assert res.values["day_type"] == "unknown"
    assert res.values["gap_open_pct"] is None
    assert res.values["trend_strength"] is None
    assert res.ok is True                    # feature engine never gates
    assert res.diagnostics                   # says WHY it is unknown


def test_unknown_on_single_session():
    # No prior session ⇒ no gap/avg_range baseline ⇒ honest "unknown".
    res = _classify(_session(PRIOR, "2026-07-08"))
    assert res.values["day_type"] == "unknown"


def test_values_flat_and_json_safe():
    res = _classify(_two_day([(100.05, 100.4, 100.0, 100.2)]),
                    vix=14.5, breadth_above_vwap=0.7,
                    idx_trend={"NIFTY": "bullish"})
    assert set(res.values) == {
        "day_type", "vol_state", "vix", "breadth_state", "breadth_frac",
        "expiry_day", "trend_strength", "gap_open_pct", "index_bias",
    }
    json.dumps(res.values)   # numpy scalars would blow up here
    # feature_dict prefixes for the alert snapshot
    assert "market_regime_day_type" in res.feature_dict()


# ---------------------------------------------------------------------------
# vol / breadth / bias labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vix,state", [
    (11.99, "low"),        # < vix_low 12.0
    (12.0, "normal"),      # boundary is NOT low (strict <)
    (17.0, "normal"),      # boundary is NOT high (strict >)
    (17.01, "high"),
    (None, None),
])
def test_vol_state_edges(vix, state):
    res = _classify(None, vix=vix)
    assert res.values["vol_state"] == state
    assert res.values["vix"] == (pytest.approx(vix) if vix is not None else None)


@pytest.mark.parametrize("frac,state", [
    (0.60, "bullish"),     # >= breadth_bullish 0.60
    (0.59, "mixed"),
    (0.50, "mixed"),
    (0.40, "bearish"),     # <= breadth_bearish 0.40
    (0.41, "mixed"),
    (None, None),
])
def test_breadth_state_edges(frac, state):
    res = _classify(None, breadth_above_vwap=frac)
    assert res.values["breadth_state"] == state
    assert res.values["breadth_frac"] == (pytest.approx(frac)
                                          if frac is not None else None)


@pytest.mark.parametrize("trend,bias", [
    ({"NIFTY": "bullish", "BANKNIFTY": "bullish"}, "bullish"),
    ({"NIFTY": "bearish", "BANKNIFTY": "bearish"}, "bearish"),
    ({"NIFTY": "bullish", "BANKNIFTY": "bearish"}, "mixed"),
    ({"NIFTY": "bullish", "BANKNIFTY": "sideways"}, "mixed"),
    ({}, None),
    (None, None),
])
def test_index_bias(trend, bias):
    res = _classify(None, idx_trend=trend)
    assert res.values["index_bias"] == bias


@pytest.mark.parametrize("garbage", [
    "bullish",                 # stray string instead of {"NIFTY": "bullish"}
    ["bullish", "bearish"],    # list of states
    42,
])
def test_index_bias_garbage_degrades_to_none(garbage):
    # Totality: non-dict idx_trend ⇒ None + diagnostic, never AttributeError.
    res = _classify(None, idx_trend=garbage)
    assert res.values["index_bias"] is None
    assert res.ok is True
    assert any("index_bias" in d for d in res.diagnostics)


def test_index_bias_garbage_survives_live_snapshot(monkeypatch):
    # live_snapshot forwards idx_trend unguarded — the guard must hold there too.
    mr._reset_cache()

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(data_provider, "fetch_data", boom)
    monkeypatch.setattr(data_provider, "fetch_yfinance", boom)
    res = mr.live_snapshot(idx_trend="bullish")
    assert res.ok is True
    assert res.values["index_bias"] is None
    mr._reset_cache()


def test_now_garbage_degrades_to_clock(monkeypatch):
    # Totality: non-datetime `now` ⇒ fall back to current IST date + diagnostic,
    # never AttributeError from now.date().
    monkeypatch.setattr(market_calendar, "is_trading_holiday", lambda d: False)
    res = mr.classify_regime(now="2026-07-09", cfg=CFG)
    assert res.ok is True
    assert isinstance(res.values["expiry_day"], bool)
    assert any("now" in d for d in res.diagnostics)


def test_now_accepts_plain_date(monkeypatch):
    # A datetime.date is enough (we only need .date()-level precision).
    monkeypatch.setattr(market_calendar, "is_trading_holiday", lambda d: False)
    res = mr.classify_regime(now=date(2026, 7, 9), cfg=CFG)   # Thursday expiry
    assert res.values["expiry_day"] is True
    assert not any("now" in d for d in res.diagnostics)


# ---------------------------------------------------------------------------
# expiry day (weekday from config, holiday shifts it BACK one trading day)
# ---------------------------------------------------------------------------
# 2026-07-06 is a Monday; nominal expiry weekday 3 ⇒ Thursday 2026-07-09.

def test_expiry_normal_thursday(monkeypatch):
    monkeypatch.setattr(market_calendar, "is_trading_holiday", lambda d: False)
    assert mr.is_expiry_day(date(2026, 7, 9), CFG) is True    # Thursday
    assert mr.is_expiry_day(date(2026, 7, 8), CFG) is False   # Wednesday
    assert mr.is_expiry_day(date(2026, 7, 10), CFG) is False  # Friday: past it


def test_expiry_shifts_back_on_holiday(monkeypatch):
    holidays = {date(2026, 7, 9)}                             # Thursday closed
    monkeypatch.setattr(market_calendar, "is_trading_holiday",
                        lambda d: d in holidays)
    assert mr.is_expiry_day(date(2026, 7, 9), CFG) is False   # holiday itself
    assert mr.is_expiry_day(date(2026, 7, 8), CFG) is True    # Wed takes it


def test_expiry_skips_holiday_run(monkeypatch):
    holidays = {date(2026, 7, 9), date(2026, 7, 8)}           # Thu + Wed closed
    monkeypatch.setattr(market_calendar, "is_trading_holiday",
                        lambda d: d in holidays)
    assert mr.is_expiry_day(date(2026, 7, 7), CFG) is True    # Tuesday takes it
    assert mr.is_expiry_day(date(2026, 7, 8), CFG) is False


def test_classify_reports_expiry_flag(monkeypatch):
    monkeypatch.setattr(market_calendar, "is_trading_holiday", lambda d: False)
    on = mr.classify_regime(now=IST.localize(datetime(2026, 7, 9, 10, 0)), cfg=CFG)
    off = mr.classify_regime(now=IST.localize(datetime(2026, 7, 8, 10, 0)), cfg=CFG)
    assert on.values["expiry_day"] is True
    assert off.values["expiry_day"] is False


# ---------------------------------------------------------------------------
# live_snapshot: fetch resilience + TTL cache
# ---------------------------------------------------------------------------

def _vix_frame(last=14.5):
    idx = pd.date_range("2026-07-06", periods=2, freq="1D", tz="Asia/Kolkata")
    return pd.DataFrame({"Close": [13.0, last]}, index=idx)


def test_live_snapshot_survives_fetch_failures(monkeypatch):
    mr._reset_cache()

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(data_provider, "fetch_data", boom)
    monkeypatch.setattr(data_provider, "fetch_yfinance", boom)
    res = mr.live_snapshot()
    assert res.ok is True                       # never a gate, never a raise
    assert res.values["day_type"] == "unknown"
    assert res.values["vix"] is None and res.values["vol_state"] is None
    joined = " ".join(res.diagnostics)
    assert "index_df" in joined and "vix" in joined


def test_live_snapshot_ttl_cache(monkeypatch):
    mr._reset_cache()
    calls = {"nifty": 0, "vix": 0}
    frame = _two_day([(101.0, 101.5, 100.9, 101.3)])   # gap_go by arithmetic

    def fake_fetch_data(symbol, period="6mo", interval="1d"):
        assert symbol == "NIFTY"
        calls["nifty"] += 1
        return frame

    def fake_fetch_yf(symbol, period, interval):
        assert symbol == "^INDIAVIX"
        calls["vix"] += 1
        return _vix_frame(14.5)

    monkeypatch.setattr(data_provider, "fetch_data", fake_fetch_data)
    monkeypatch.setattr(data_provider, "fetch_yfinance", fake_fetch_yf)

    r1 = mr.live_snapshot()
    r2 = mr.live_snapshot()
    assert calls == {"nifty": 1, "vix": 1}      # second call served from cache
    assert r1.values["day_type"] == r2.values["day_type"] == "gap_go"
    assert r1.values["vix"] == r2.values["vix"] == 14.5
    assert r1.values["vol_state"] == "normal"   # 12 ≤ 14.5 ≤ 17

    # Age both entries past the TTL → next call refetches.
    ttl = timedelta(minutes=CONFIG.regime.cache_ttl_min + 1)
    for key, (val, ts) in list(mr._cache.items()):
        mr._cache[key] = (val, ts - ttl)
    mr.live_snapshot()
    assert calls == {"nifty": 2, "vix": 2}
    mr._reset_cache()


def test_failed_fetch_is_not_cached(monkeypatch):
    """A transient failure must not occupy the cache for a whole TTL."""
    mr._reset_cache()
    calls = {"n": 0}

    def flaky(symbol, period="6mo", interval="1d"):
        calls["n"] += 1
        raise RuntimeError("transient")

    monkeypatch.setattr(data_provider, "fetch_data", flaky)
    monkeypatch.setattr(data_provider, "fetch_yfinance",
                        lambda *a, **k: _vix_frame())
    mr.live_snapshot()
    mr.live_snapshot()
    assert calls["n"] == 2                      # retried, not poisoned
    mr._reset_cache()
