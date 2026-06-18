"""
Repaint guard: drop_forming_candle removes the still-forming candle at live
signal sites so calls don't repaint. CI-safe (analyzer only).

    venv/Scripts/python.exe tests/test_repaint.py
"""
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer  # noqa: E402
from constants import IST  # noqa: E402


def _intraday(n, start="2026-01-05 09:30"):
    idx = pd.date_range(start, periods=n, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                         "Close": 1.0, "Volume": 1}, index=idx)


def _daily(n, start="2026-01-05"):
    idx = pd.date_range(start, periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                         "Close": 1.0, "Volume": 1}, index=idx)


def test_intraday_drops_still_forming_candle():
    df = _intraday(5)                                   # last bar opens 09:50
    now = IST.localize(datetime(2026, 1, 5, 9, 52))     # inside [09:50, 09:55)
    assert len(analyzer.drop_forming_candle(df, "5m", now=now)) == 4


def test_intraday_keeps_closed_candle():
    df = _intraday(5)
    now = IST.localize(datetime(2026, 1, 5, 9, 56))     # 09:50 bar has closed
    assert len(analyzer.drop_forming_candle(df, "5m", now=now)) == 5


def test_daily_drops_todays_open_session_bar():
    df = _daily(3)                                      # last bar = 2026-01-07
    now = IST.localize(datetime(2026, 1, 7, 11, 0))     # market still open
    assert len(analyzer.drop_forming_candle(df, "1d", now=now)) == 2


def test_daily_keeps_bar_after_close():
    df = _daily(3)
    now = IST.localize(datetime(2026, 1, 7, 16, 0))     # after 15:30 close
    assert len(analyzer.drop_forming_candle(df, "1d", now=now)) == 3


def test_too_few_rows_unchanged():
    df = _intraday(1)
    assert len(analyzer.drop_forming_candle(df, "5m")) == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1; print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
