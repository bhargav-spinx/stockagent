"""
Golden characterization tests for the signal path.

These pin the EXACT numeric behavior of the scoring engine, universal filters,
setup detection, trade-level model and swing analyzer on fixed synthetic
fixtures, captured from the pre-config-extraction code (2026-07-04).

If one of these fails after a refactor, live signal behavior CHANGED. That is
either a bug in the refactor or a deliberate strategy change — a deliberate
change must update the pins in the same commit and say so in the message.

Offline: no network, no DB. CI-safe (pandas + pytz only).

Regenerate pins (only for a deliberate behavior change):
    venv/Scripts/python.exe tests/test_golden_parity.py --emit
"""
import math
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer                                    # noqa: E402
import narrative                                   # noqa: E402
from intraday_score import score_stock             # noqa: E402
from scanner_filters import (                      # noqa: E402
    apply_universal_filters, atr_bounds_filter, trigger_volume_filter,
    vwap_slope_filter, round_number_filter,
)
from scanner_indicators import trade_levels, vwap, ema  # noqa: E402
from scanner_setups import detect_setup_a          # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic fixtures — pure arithmetic, no RNG, stable on every platform.
# ---------------------------------------------------------------------------

def _session_ts(day: datetime, n: int) -> list[datetime]:
    """First n 5-min candle timestamps of an NSE session (09:15 open)."""
    start = day.replace(hour=9, minute=15)
    return [start + timedelta(minutes=5 * i) for i in range(n)]


def five_min_df() -> pd.DataFrame:
    """Two IST sessions of 5-min candles engineered to exercise the interesting
    branches: day 2 gaps up ~1%, holds a tight opening range, breaks out long
    with rising volume and a spiking trigger candle."""
    rows, idx = [], []

    # Day 1 (Tue 2025-06-03): gentle climb around 500, full 75-candle session.
    for i, ts in enumerate(_session_ts(datetime(2025, 6, 3), 75)):
        o = 500.0 + 0.05 * i + 0.6 * math.sin(i / 4)
        c = o + 0.05 + 0.25 * math.sin(i / 3 + 1)
        hi = max(o, c) + 1.2 + 0.4 * math.sin(i / 5)
        lo = min(o, c) - 1.2 - 0.4 * math.cos(i / 6)
        v = 100_000 + 15_000 * math.sin(i / 7) ** 2 + 300 * i
        rows.append((o, hi, lo, c, v))
        idx.append(ts)

    # Day 2 (Wed 2025-06-04): gap up, 3 tight ORB candles, breakout climb.
    # 25 candles → "now" is 11:15 IST, well inside the session.
    for i, ts in enumerate(_session_ts(datetime(2025, 6, 4), 25)):
        if i < 3:
            o = 510.0 + 0.10 * i
            c = o + 0.15
            hi = max(o, c) + 0.9
            lo = min(o, c) - 0.9
            v = 220_000.0 - 8_000 * i
        else:
            # High-frequency close oscillation → mixed up/down deltas so
            # RSI(14) stays moderate instead of pegging above 70.
            o = 510.5 + 0.22 * (i - 3) + 0.9 * math.sin(i / 3)
            c = o + 0.10 + 1.15 * math.sin(i * 1.3)
            hi = max(o, c) + 1.1 + 0.3 * math.sin(i / 4)
            lo = min(o, c) - 1.1 - 0.3 * math.cos(i / 5)
            v = 130_000 + 3_000 * i + 15_000 * math.sin(i / 6) ** 2
        if i == 24:                       # trigger candle: volume spike
            v *= 3.0
        rows.append((o, hi, lo, c, v))
        idx.append(ts)

    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                      index=pd.DatetimeIndex(idx))
    return df


def daily_df() -> pd.DataFrame:
    """120 daily bars around 1500 with a late uptrend — enough history for
    SMA(50), RSI(14), MACD, Bollinger and the swing trade-setup model."""
    rows, idx = [], []
    day = datetime(2025, 1, 1)
    made = 0
    i = 0
    while made < 120:
        day_i = day + timedelta(days=i)
        i += 1
        if day_i.weekday() >= 5:
            continue
        t = made
        base = 1500.0 + 0.8 * t + 18.0 * math.sin(t / 9) + 7.0 * math.sin(t / 3.7)
        o = base - 2.0 * math.sin(t / 5)
        c = base + 3.5 * math.sin(t / 6 + 0.5) + (2.0 if t > 100 else 0.0)
        hi = max(o, c) + 9.0 + 3.0 * math.sin(t / 4)
        lo = min(o, c) - 9.0 - 3.0 * math.cos(t / 5)
        v = 1_000_000 + 200_000 * math.sin(t / 8) ** 2
        rows.append((o, hi, lo, c, v))
        idx.append(day_i.replace(hour=15, minute=30))
        made += 1
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(idx))


IDX_TREND = {"NIFTY": "bullish", "BANKNIFTY": "flat"}


# ---------------------------------------------------------------------------
# Capture — everything we pin, computed from the fixtures.
# ---------------------------------------------------------------------------

def _py(obj):
    """Recursively coerce numpy scalars to native Python types so pins are
    clean literals and comparisons are type-stable."""
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_py(v) for v in obj)
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def _capture() -> dict:
    df5 = five_min_df()
    dfd = daily_df()

    card = score_stock(df5, "GOLDEN", idx_trend=IDX_TREND, daily_df=dfd,
                       skip_external=True)

    filters_all = apply_universal_filters(df5, "long", check_time=False)
    f_atr = atr_bounds_filter(df5)
    f_vol = trigger_volume_filter(df5)
    f_slope = vwap_slope_filter(df5)
    f_round_l = round_number_filter(df5, "long")
    f_round_s = round_number_filter(df5, "short")

    sig = detect_setup_a(df5, "GOLDEN")
    setup_a = None
    if sig is not None:
        setup_a = {
            "direction": sig.direction,
            "entry": sig.entry,
            "stop_loss": sig.stop_loss,
            "target1": sig.target1,
            "target2": sig.target2,
            "n_confluences": len(sig.confluences),
        }

    lv_long = trade_levels(518.0, "long", df5)
    lv_short = trade_levels(518.0, "short", df5)

    ind = {
        "rsi5_last": float(analyzer.rsi(df5["Close"], 14).iloc[-1]),
        "atr5_last": float(analyzer.atr(df5, 14).iloc[-1]),
        "vwap_last": float(vwap(df5).iloc[-1]),
        "ema9_last": float(ema(df5["Close"], 9).iloc[-1]),
        "ema20_last": float(ema(df5["Close"], 20).iloc[-1]),
        "rsi_d_last": float(analyzer.rsi(dfd["Close"], 14).iloc[-1]),
        "atr_d_last": float(analyzer.atr(dfd, 14).iloc[-1]),
    }

    price_d = float(dfd["Close"].iloc[-1])
    bts_buy = analyzer.build_trade_setup(dfd, "BUY", price_d, ind["atr_d_last"], "swing")
    bts_sell = analyzer.build_trade_setup(dfd, "SELL", price_d, ind["atr_d_last"], "swing")
    bts_hold = analyzer.build_trade_setup(dfd, "HOLD", price_d, ind["atr_d_last"], "swing")

    # analyze() end-to-end with the data fetch stubbed out.
    orig_fetch = analyzer.fetch_data
    analyzer.fetch_data = lambda sym, period=None, interval=None: dfd.copy()
    try:
        res = analyzer.analyze("GOLDEN", "swing")
    finally:
        analyzer.fetch_data = orig_fetch

    gates = narrative.evaluate_gates_intraday(card)

    return _py({
        "score": {
            "direction": card.direction,
            "score": card.score,
            "rating": card.rating,
            "breakdown": dict(card.breakdown),
            "entry": card.entry,
            "stop_loss": card.stop_loss,
            "target1": card.target1,
            "target2": card.target2,
            "bull_conviction": card.bull_conviction,
            "bear_conviction": card.bear_conviction,
            "gap_pct": card.gap_pct,
            "rvol": card.rvol,
            "orb_window": card.orb_window,
            "regime_ok": card.regime_ok,
            "event_ok": card.event_ok,
        },
        "filters": {
            "all": (filters_all.passed, filters_all.reason),
            "atr": (f_atr.passed, f_atr.reason),
            "vol": (f_vol.passed, f_vol.reason),
            "slope": (f_slope.passed, f_slope.reason),
            "round_long": (f_round_l.passed, f_round_l.reason),
            "round_short": (f_round_s.passed, f_round_s.reason),
        },
        "setup_a": setup_a,
        "levels": {"long": lv_long, "short": lv_short},
        "indicators": ind,
        "bts_buy": bts_buy,
        "bts_sell": bts_sell,
        "bts_hold": bts_hold,
        "analyze": {
            "signal": res["signal"],
            "confidence": res["confidence"],
            "rsi": res["rsi"],
            "price": res["price"],
            "change_pct": res["change_pct"],
            "votes": [(n, s) for (n, s, _r) in res["indicators"]],
            "trade_setup": res["trade_setup"],
        },
        "gates": gates,
    })


# ---------------------------------------------------------------------------
# Pinned expectations (paste from `--emit`).
# ---------------------------------------------------------------------------

_EXPECTED: dict = {
    'analyze': {'change_pct': 0.25,
                'confidence': 50.0,
                'price': 1616.42,
                'rsi': 94.97,
                'signal': 'HOLD',
                'trade_setup': {'action': 'WAIT',
                                'atr': 19.11,
                                'buy_breakout': 1625.67,
                                'buy_breakout_sl': 1606.21,
                                'sell_breakdown': 1551.94,
                                'sell_breakdown_sl': 1570.61,
                                'swing_high': 1622.43,
                                'swing_low': 1555.05},
                'votes': [('SMA Trend', 'BUY'),
                          ('RSI', 'SELL'),
                          ('MACD', 'BUY'),
                          ('Bollinger', 'SELL')]},
    'bts_buy': {'action': 'BUY',
                'atr': 19.11,
                'entry': 1616.42,
                'risk_level': 'Medium',
                'risk_pct': 1.77,
                'risk_per_share': 28.66,
                'rr_ratio': 1.5,
                'stop_loss': 1587.75,
                'swing_high': 1622.43,
                'swing_low': 1555.05,
                'target1': 1659.41,
                'target2': 1702.4},
    'bts_hold': {'action': 'WAIT',
                 'atr': 19.11,
                 'buy_breakout': 1625.67,
                 'buy_breakout_sl': 1606.21,
                 'sell_breakdown': 1551.94,
                 'sell_breakdown_sl': 1570.61,
                 'swing_high': 1622.43,
                 'swing_low': 1555.05},
    'bts_sell': {'action': 'SELL',
                 'atr': 19.11,
                 'entry': 1616.42,
                 'risk_level': 'Low',
                 'risk_pct': 0.87,
                 'risk_per_share': 14.13,
                 'rr_ratio': 1.5,
                 'stop_loss': 1630.54,
                 'swing_high': 1622.43,
                 'swing_low': 1555.05,
                 'target1': 1595.23,
                 'target2': 1574.04},
    'filters': {'all': (True, ''),
                'atr': (True, ''),
                'round_long': (True, ''),
                'round_short': (True, ''),
                'slope': (True, ''),
                'vol': (True, '')},
    'gates': (True,
              [],
              ['Multi-timeframe / market-structure / relative-strength: not provided']),
    'indicators': {'atr5_last': 2.7320881835383957,
                   'atr_d_last': 19.107417919415326,
                   'ema20_last': 512.8204020801068,
                   'ema9_last': 514.535898345408,
                   'rsi5_last': 68.8795140002885,
                   'rsi_d_last': 94.97093925160715,
                   'vwap_last': 512.9585460422222},
    'levels': {'long': (513.9018677246924, 526.1962645506152, 530.2943968259228),
               'short': (522.0981322753076, 509.8037354493848, 505.7056031740772)},
    'score': {'bear_conviction': 20,
              'breakdown': {'EMA Trend': 15,
                            'Gap Up/Down': 10,
                            'ORB Breakout': 15,
                            'Relative Volume': 18,
                            'VWAP Confirmation': 15,
                            'Volume Breakout': 10},
              'bull_conviction': 80,
              'direction': 'long',
              'entry': 515.8640320006208,
              'event_ok': True,
              'gap_pct': 1.2564253215448278,
              'orb_window': 15,
              'rating': 'Strong Buy',
              'regime_ok': True,
              'rvol': 1.8024683426530068,
              'score': 83,
              'stop_loss': 511.7658997253132,
              'target1': 524.060296551236,
              'target2': 528.1584288265436},
    'setup_a': {'direction': 'long',
                'entry': 515.8640320006208,
                'n_confluences': 3,
                'stop_loss': 511.7658997253132,
                'target1': 524.060296551236,
                'target2': 528.1584288265436},
}


# ---------------------------------------------------------------------------
# Comparison helper — exact for str/bool/int, tight approx for floats.
# ---------------------------------------------------------------------------

def _assert_same(got, exp, path=""):
    if isinstance(exp, dict):
        assert isinstance(got, dict), f"{path}: {type(got)} != dict"
        assert set(got.keys()) == set(exp.keys()), (
            f"{path}: keys {sorted(got.keys())} != {sorted(exp.keys())}")
        for k in exp:
            _assert_same(got[k], exp[k], f"{path}.{k}")
    elif isinstance(exp, (list, tuple)):
        assert len(got) == len(exp), f"{path}: len {len(got)} != {len(exp)}"
        for i, (g, e) in enumerate(zip(got, exp)):
            _assert_same(g, e, f"{path}[{i}]")
    elif isinstance(exp, bool) or exp is None or isinstance(exp, str):
        assert got == exp, f"{path}: {got!r} != {exp!r}"
    elif isinstance(exp, (int, float)):
        assert got == pytest.approx(exp, rel=1e-9, abs=1e-9), (
            f"{path}: {got!r} != {exp!r}")
    else:
        assert got == exp, f"{path}: {got!r} != {exp!r}"


def test_signal_path_golden_parity():
    assert _EXPECTED, ("Pins missing — run "
                       "`python tests/test_golden_parity.py --emit` and paste.")
    got = _capture()
    _assert_same(got, _EXPECTED, "golden")


if __name__ == "__main__":
    if "--emit" in sys.argv:
        import pprint
        cap = _capture()
        print("_EXPECTED = \\")
        pprint.pprint(cap, width=88, sort_dicts=True)
    else:
        test_signal_path_golden_parity()
        print("golden parity OK")
