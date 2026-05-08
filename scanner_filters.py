"""
Universal filters from §5 of STRATEGY.md.
Every setup must pass ALL of these gates before any entry decision.
"""
from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd
import pytz

from analyzer import atr as atr_fn
from scanner_indicators import vwap, vwap_slope_pct

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class FilterResult:
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


def time_window_filter(now: datetime | None = None) -> FilterResult:
    """
    No new entries:
    - Before 09:30 IST (ORB still forming)
    - 12:00–13:30 IST (lunch chop)
    - After 14:30 IST (square-off pressure)
    """
    now = now or datetime.now(IST)
    t = now.time()
    if t < time(9, 30):
        return FilterResult(False, "Before 09:30 — ORB still forming")
    if time(12, 0) <= t < time(13, 30):
        return FilterResult(False, "Lunch window 12:00–13:30")
    if t >= time(14, 30):
        return FilterResult(False, "After 14:30 — no new entries (square-off pressure)")
    return FilterResult(True)


def atr_bounds_filter(df: pd.DataFrame, lo: float = 0.004, hi: float = 0.015) -> FilterResult:
    """ATR(14) on 5-min must be between 0.4% and 1.5% of price."""
    a = atr_fn(df, 14).iloc[-1]
    if pd.isna(a):
        return FilterResult(False, "ATR not yet available")
    last = df["Close"].iloc[-1]
    pct = a / last
    if pct < lo:
        return FilterResult(False, f"ATR {pct*100:.2f}% < {lo*100:.1f}% (no movement)")
    if pct > hi:
        return FilterResult(False, f"ATR {pct*100:.2f}% > {hi*100:.1f}% (chop / news)")
    return FilterResult(True)


def trigger_volume_filter(df: pd.DataFrame, ratio: float = 1.5) -> FilterResult:
    """Trigger candle volume ≥ ratio × avg of last 12 candles' volume."""
    if len(df) < 13:
        return FilterResult(False, "Need 13+ candles for volume baseline")
    cur = df["Volume"].iloc[-1]
    avg = df["Volume"].iloc[-13:-1].mean()
    if avg <= 0:
        return FilterResult(False, "Zero avg-volume baseline")
    if cur < avg * ratio:
        return FilterResult(False,
                            f"Trigger volume {cur:.0f} < {ratio}× avg ({avg*ratio:.0f})")
    return FilterResult(True)


def vwap_slope_filter(df: pd.DataFrame, min_slope_abs: float = 0.02) -> FilterResult:
    """VWAP must be sloping (not flat). min_slope_abs in % per candle."""
    s = vwap_slope_pct(vwap(df))
    if s is None:
        return FilterResult(False, "VWAP slope not computable")
    if abs(s) < min_slope_abs:
        return FilterResult(False, f"VWAP flat ({s:+.3f}%/candle) — chop")
    return FilterResult(True)


def round_number_filter(df: pd.DataFrame, direction: str,
                        threshold: float = 0.003) -> FilterResult:
    """
    Reject if price is within `threshold` of a major round number going
    AGAINST the trade direction.
    For long: round number above is resistance.
    For short: round number below is support.
    """
    last = float(df["Close"].iloc[-1])
    if last < 100:
        bucket = 25
    elif last < 500:
        bucket = 50
    elif last < 1000:
        bucket = 100
    else:
        bucket = 500

    candidates = [
        round(last / bucket) * bucket - bucket,
        round(last / bucket) * bucket,
        round(last / bucket) * bucket + bucket,
    ]

    for lvl in candidates:
        if lvl <= 0:
            continue
        diff_pct = (lvl - last) / last
        if direction == "long" and 0 < diff_pct < threshold:
            return FilterResult(False,
                                f"₹{lvl:.0f} round number {diff_pct*100:.2f}% above (resistance)")
        if direction == "short" and 0 > diff_pct > -threshold:
            return FilterResult(False,
                                f"₹{lvl:.0f} round number {-diff_pct*100:.2f}% below (support)")
    return FilterResult(True)


def apply_universal_filters(df: pd.DataFrame, direction: str = "long",
                            check_time: bool = True) -> FilterResult:
    """
    Run all §5 universal filters. Returns first failure or pass.

    `check_time=False` skips the time-of-day filter — useful for
    backtesting / off-hours testing where you want to validate the
    setup detection without waiting for market hours.
    """
    filters = []
    if check_time:
        filters.append(time_window_filter())
    filters.extend([
        atr_bounds_filter(df),
        trigger_volume_filter(df),
        vwap_slope_filter(df),
        round_number_filter(df, direction),
    ])
    for f in filters:
        if not f.passed:
            return f
    return FilterResult(True)
