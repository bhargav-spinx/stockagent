"""
Pins eod_report.resolve_intraday / resolve_swing P&L to the ACTUAL trade levels.

This is the regression test for BLOCKER-1: the resolver used to return a
hardcoded +1.5% (t2_hit) / +0.5% (breakeven) regardless of where T1/T2 actually
sat. Targets are ATR-sized (distance varies per stock), so a fixed percentage is
pure fiction. These assertions FAIL against the old hardcoded code and PASS once
P&L is blended from real entry/T1/T2 prices.

Runs with plain python (no pytest needed):
    venv/Scripts/python.exe tests/test_resolver.py
or under pytest if installed.
"""
import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eod_report  # noqa: E402


def _df(rows, start="2026-01-05 09:30"):
    """Build an IST-aware 5-min OHLCV frame from (o,h,l,c) tuples."""
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}
         for (o, h, l, c) in rows],
        index=idx,
    )


def _alert(direction="long", entry=100.0, sl=98.0, t1=106.0, t2=109.0, gen=None):
    return {
        "symbol": "TEST.NS", "entry": entry, "stop_loss": sl,
        "target1": t1, "target2": t2, "direction": direction,
        "generated_at": gen,
    }


def _approx(a, b, tol=0.011):
    return abs(a - b) < tol


def test_t2_hit_long_blends_actual_levels():
    # entry 100, T1 106 (+6%), T2 109 (+9%) -> blended 7.5%, NOT the old 1.5
    df = _df([(100, 100, 100, 100),    # trigger candle (gen_time)
              (100, 110, 100, 108)])   # next candle tags both T1 and T2
    gen = df.index[0].to_pydatetime()
    out = eod_report.resolve_intraday(_alert(gen=gen), df=df)
    assert out["status"] == "t2_hit", out
    assert _approx(out["pnl_pct"], 7.5), out


def test_sl_hit_long_is_real_loss():
    df = _df([(100, 100, 100, 100),
              (100, 101, 97, 98)])     # low 97 <= SL 98
    gen = df.index[0].to_pydatetime()
    out = eod_report.resolve_intraday(_alert(gen=gen), df=df)
    assert out["status"] == "sl_hit", out
    assert _approx(out["pnl_pct"], -2.0), out      # (98-100)/100 = -2%


def test_t1_then_breakeven_long():
    # T1 tagged, then price trails back to entry -> blended 0.5*6 + 0.5*0 = 3.0
    df = _df([(100, 100, 100, 100),
              (100, 106, 100, 105),    # T1 (106) hit, not T2
              (105, 105, 99, 100)])    # back to entry (low 99 <= 100)
    gen = df.index[0].to_pydatetime()
    out = eod_report.resolve_intraday(_alert(gen=gen), df=df)
    assert out["status"] == "t1_then_breakeven", out
    assert _approx(out["pnl_pct"], 3.0), out       # NOT the old 0.5


def test_t1_then_squareoff_long():
    # T1 hit, no T2, no breakeven, session ends at close 103
    # blended 0.5*6 + 0.5*((103-100)/100*100=3) = 4.5
    df = _df([(100, 100, 100, 100),
              (100, 106, 105, 106),    # T1 hit
              (106, 107, 102, 103)])   # last close 103, never T2, never back to entry
    gen = df.index[0].to_pydatetime()
    out = eod_report.resolve_intraday(_alert(gen=gen), df=df)
    assert out["status"] == "t1_then_squareoff", out
    assert _approx(out["pnl_pct"], 4.5), out


def test_t2_hit_short_blends_actual_levels():
    # short: entry 100, SL 102, T1 94 (+6% gain), T2 91 (+9% gain) -> blended 7.5
    df = _df([(100, 100, 100, 100),
              (100, 100, 90, 92)])     # low 90 <= T2 91 and <= T1 94
    gen = df.index[0].to_pydatetime()
    a = _alert(direction="short", sl=102.0, t1=94.0, t2=91.0, gen=gen)
    out = eod_report.resolve_intraday(a, df=df)
    assert out["status"] == "t2_hit", out
    assert _approx(out["pnl_pct"], 7.5), out


def test_sl_before_t1_same_candle_is_conservative():
    # both SL and T1 inside one candle -> assume SL first (loss), never the win
    df = _df([(100, 100, 100, 100),
              (100, 106, 97, 100)])    # high 106 (T1) AND low 97 (SL) same candle
    gen = df.index[0].to_pydatetime()
    out = eod_report.resolve_intraday(_alert(gen=gen), df=df)
    assert out["status"] == "sl_hit", out
    assert _approx(out["pnl_pct"], -2.0), out


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
