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


def reload() -> None:
    """Drop the cache so an edited holiday file is picked up without a restart."""
    global _cache
    _cache = None
