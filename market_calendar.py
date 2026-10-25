"""
NSE equity trading-holiday calendar.

Full-day exchange closures on which the bot should NOT scan (the data would be
stale / nonexistent). Weekends are handled separately by the callers.

Source of truth is `nse_holidays.txt` (one ISO date per line; inline `#`
comments allowed). It ships pre-filled with the FIXED-date holidays only —
those are always correct — and you add the variable/lunar ones (Holi, Eid,
Ram Navami, Good Friday, Diwali, etc.) each year from the official NSE circular:
https://www.nseindia.com/resources/exchange-communication-holidays

Deliberately NO hardcoded variable dates in code: a wrong guessed date would
silently suppress a real trading day. Editing a text file is safer than code.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_HOLIDAY_FILE = Path(__file__).resolve().parent / "nse_holidays.txt"
_cache: set[date] | None = None


def _load() -> set[date]:
    global _cache
    if _cache is not None:
        return _cache
    holidays: set[date] = set()
    if _HOLIDAY_FILE.exists():
        for raw in _HOLIDAY_FILE.read_text(encoding="utf-8").splitlines():
            token = raw.split("#", 1)[0].strip()   # strip inline comments
            if not token:
                continue
            try:
                holidays.add(date.fromisoformat(token))
            except ValueError:
                logger.warning("nse_holidays.txt: ignoring unparseable line %r", raw)
    else:
        logger.info("market_calendar: %s not found — no holidays loaded", _HOLIDAY_FILE)
    _cache = holidays
    return _cache


def is_trading_holiday(d: date) -> bool:
    """True if `d` is a listed NSE full-day trading holiday."""
    return d in _load()


# NSE closes on ~14–16 weekdays a year. Far fewer listed weekday closures means
# the variable/lunar dates were never added — the failure mode that let the bot
# scan the 2026 Holi/Good Friday/Dussehra closures on the previous session's
# candles as if they were live.
MIN_WEEKDAY_HOLIDAYS_PER_YEAR = 10


def coverage_warning(today: date | None = None) -> str | None:
    """Human-readable warning when the holiday file looks incomplete for the
    current year (or, in December, for next year). None when coverage is OK."""
    today = today or date.today()
    years = [today.year] + ([today.year + 1] if today.month == 12 else [])
    gaps = []
    for y in years:
        n = sum(1 for d in _load() if d.year == y and d.weekday() < 5)
        if n < MIN_WEEKDAY_HOLIDAYS_PER_YEAR:
            gaps.append(f"{y}: {n} weekday holidays listed")
    if not gaps:
        return None
    return ("nse_holidays.txt looks incomplete (" + "; ".join(gaps) + ") — add "
            "the dates from the NSE holiday circular, or scans will run on "
            "stale candles on unlisted holidays")


def reload() -> None:
    """Drop the cache so an edited holiday file is picked up without a restart."""
    global _cache
    _cache = None
