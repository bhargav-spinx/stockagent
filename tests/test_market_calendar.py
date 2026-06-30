"""
NSE holiday gate (#2): holidays loaded from nse_holidays.txt and honoured by the
intraday entry-window check. CI-safe (needs pandas via scanner_filters→analyzer).

    venv/Scripts/python.exe tests/test_market_calendar.py
"""
import os
import sys
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market_calendar as mc          # noqa: E402
from scanner_filters import is_intraday_entry_window  # noqa: E402
from constants import IST             # noqa: E402


def test_fixed_holidays_are_loaded():
    assert mc.is_trading_holiday(date(2026, 1, 26))    # Republic Day
    assert mc.is_trading_holiday(date(2026, 8, 15))    # Independence Day
    assert mc.is_trading_holiday(date(2026, 12, 25))   # Christmas


def test_normal_day_is_not_a_holiday():
    assert not mc.is_trading_holiday(date(2026, 1, 27))  # Tue, ordinary day


def test_entry_window_blocked_on_holiday_weekday():
    # 2026-01-26 is a Monday (would be open) but it's Republic Day → blocked.
    dt = IST.localize(datetime(2026, 1, 26, 10, 0))
    assert dt.weekday() < 5                  # ensure the block is the holiday, not weekend
    assert is_intraday_entry_window(dt) is False


def test_entry_window_open_on_ordinary_weekday():
    # 2026-01-27 (Tue) 10:00 — inside the window, not a holiday.
    dt = IST.localize(datetime(2026, 1, 27, 10, 0))
    assert is_intraday_entry_window(dt) is True


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
