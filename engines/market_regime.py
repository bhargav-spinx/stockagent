"""
Market-regime engine (Phase 2): deterministic day-type / volatility / breadth
/ expiry labels.

Why this exists: "does this setup work on trend days but bleed on range
days?" is unanswerable unless every alert carries the day's regime label at
generation time. This engine produces those labels as LOGGED FEATURES — it
never gates. The existing index veto (intraday_score's regime_ok) stays the
only regime-based gate; duplicating it here would double-count the same
information and make attribution murky.

Determinism matters more than sophistication: the labels must be exactly
reproducible from archived candles so the walk-forward harness can recompute
them offline and get byte-identical values. Hence pure threshold rules from
config.CONFIG.regime and no fitted models.

Two entry points:
    classify_regime(...)  pure + total: caller supplies all inputs, nothing
                          is fetched. Unknown/thin input ⇒ None values plus a
                          diagnostics line, never an exception.
    live_snapshot(...)    bot-facing wrapper: fetches NIFTY 5-min and the
                          India VIX daily close (both guarded + TTL-cached),
                          then delegates to classify_regime.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

import data_provider
import market_calendar
from config import CONFIG
from constants import IST
from engines.base import EngineResult
from scanner_indicators import split_sessions

logger = logging.getLogger(__name__)

ENGINE = "market_regime"

# How many calendar days is_expiry_day walks back from the nominal expiry
# looking for a trading day. One full week is enough for any realistic run of
# holidays + weekend; beyond that the calendar data itself is suspect, so we
# stop rather than loop forever.
_EXPIRY_WALKBACK_DAYS = 7


# ---------------------------------------------------------------------------
# expiry
# ---------------------------------------------------------------------------

def is_expiry_day(d: date, cfg=None) -> bool:
    """True if `d` is the weekly index-expiry day.

    Nominal expiry is cfg.expiry_weekday (config, not code — NSE has
    reshuffled the weekday more than once). When the nominal date is an NSE
    holiday, the exchange moves expiry to the PREVIOUS trading day, so the
    real rule is: `d` is expiry iff `d` is the last non-holiday weekday on or
    before its week's nominal expiry date. A date AFTER the nominal expiry in
    the same week can therefore never be expiry.
    """
    cfg = cfg or CONFIG.regime
    # Nominal expiry date within d's Mon-Sun week (may be before or after d).
    nominal = d + timedelta(days=cfg.expiry_weekday - d.weekday())
    candidate = nominal
    for _ in range(_EXPIRY_WALKBACK_DAYS):
        is_weekend = candidate.weekday() >= 5
        try:
            holiday = market_calendar.is_trading_holiday(candidate)
        except Exception:  # calendar file unreadable — fail open to nominal
            holiday = False
        if not is_weekend and not holiday:
            return candidate == d
        candidate -= timedelta(days=1)
    return False


# ---------------------------------------------------------------------------
# pure sub-classifiers (each total: bad input ⇒ None)
# ---------------------------------------------------------------------------

def _to_float(x: Any) -> Optional[float]:
    """Coerce to a plain JSON-safe float (numpy scalars included); None if
    not a finite number."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _vol_state(vix: Optional[float], cfg) -> Optional[str]:
    """VIX bucket. Thresholds are exclusive so a print exactly on the line
    reads 'normal' — the conservative label when the tape is ambiguous."""
    if vix is None:
        return None
    if vix < cfg.vix_low:
        return "low"
    if vix > cfg.vix_high:
        return "high"
    return "normal"


def _breadth_state(frac: Optional[float], cfg) -> Optional[str]:
    """Universe fraction above VWAP → bullish / bearish / mixed."""
    if frac is None:
        return None
    if frac >= cfg.breadth_bullish:
        return "bullish"
    if frac <= cfg.breadth_bearish:
        return "bearish"
    return "mixed"


def _index_bias(idx_trend: Optional[dict],
                diagnostics: Optional[list[str]] = None) -> Optional[str]:
    """Agreement across index trends ({"NIFTY": "bullish", ...}).

    Unanimity is required for a directional label: a bullish NIFTY with a
    bearish BANKNIFTY is exactly the split tape we want flagged as 'mixed',
    not averaged away. Anything that is not a dict (a stray "bullish" string,
    a list) is garbage input and degrades to None + diagnostic — never an
    AttributeError into the signal path."""
    if not idx_trend:
        return None
    if not isinstance(idx_trend, dict):
        if diagnostics is not None:
            diagnostics.append("index_bias: idx_trend is not a dict, ignored")
        return None
    states = [str(v).lower() for v in idx_trend.values()]
    if all(s == "bullish" for s in states):
        return "bullish"
    if all(s == "bearish" for s in states):
        return "bearish"
    return "mixed"


def _classify_day_type(index_df: Optional[pd.DataFrame], cfg,
                       diagnostics: list[str]) -> dict:
    """Day-type label from the 5-min index frame.

    Returns {"day_type", "gap_open_pct", "trend_strength"} — always all three
    keys, Nones when the frame is too thin to say anything honest. 'unknown'
    is a first-class label, not an error: early candles or a data gap should
    log as unknown, not crash the alert path.
    """
    out: dict[str, Any] = {"day_type": "unknown", "gap_open_pct": None,
                           "trend_strength": None}
    if index_df is None or not isinstance(index_df, pd.DataFrame) or index_df.empty:
        diagnostics.append("day_type: no index data")
        return out
    if not {"Open", "High", "Low", "Close"}.issubset(index_df.columns):
        diagnostics.append("day_type: index frame missing OHLC columns")
        return out
    try:
        today_df, priors = split_sessions(index_df)
        priors = [p for p in priors if not p.empty]
        if today_df.empty or not priors:
            diagnostics.append("day_type: need today + >=1 prior session")
            return out

        prior_close = _to_float(priors[-1]["Close"].iloc[-1])
        first_open = _to_float(today_df["Open"].iloc[0])
        last_close = _to_float(today_df["Close"].iloc[-1])
        hi = _to_float(today_df["High"].max())
        lo = _to_float(today_df["Low"].min())
        if None in (prior_close, first_open, last_close, hi, lo) or prior_close == 0:
            diagnostics.append("day_type: non-finite OHLC values")
            return out

        gap_open_pct = (first_open - prior_close) / prior_close * 100.0
        today_range = hi - lo
        # Average prior-session range over the config lookback: the "is today
        # unusually wide?" baseline. Most recent sessions only — old regimes
        # shouldn't dilute the comparison.
        recent = priors[-cfg.range_lookback_days:]
        ranges = [float(p["High"].max() - p["Low"].min()) for p in recent]
        avg_range = sum(ranges) / len(ranges) if ranges else 0.0
        trend_strength = (abs(last_close - first_open) / today_range
                          if today_range > 0 else 0.0)

        out["gap_open_pct"] = round(gap_open_pct, 4)
        out["trend_strength"] = round(trend_strength, 4)

        if abs(gap_open_pct) >= cfg.gap_min_pct:
            # Gap day. Fill fraction uses today's adverse EXTREME (not the
            # last close): the classic gap-fill question is "did price ever
            # trade back to the prior close?", and the extreme answers it
            # even if price bounced afterwards.
            gap = first_open - prior_close
            if gap > 0:     # gap up: fill = travel back DOWN toward prior close
                retrace_frac = (first_open - lo) / gap
            else:           # gap down: fill = travel back UP toward prior close
                retrace_frac = (hi - first_open) / -gap
            retrace_frac = max(0.0, min(1.0, retrace_frac))
            out["day_type"] = ("gap_fill" if retrace_frac >= cfg.gap_fill_frac
                               else "gap_go")
        elif (avg_range > 0
              and today_range >= cfg.trend_day_range_mult * avg_range
              and trend_strength >= cfg.trend_strength_min):
            out["day_type"] = "trend_up" if last_close >= first_open else "trend_down"
        else:
            out["day_type"] = "range"
        return out
    except Exception as e:  # total over garbage input — label, never raise
        diagnostics.append(f"day_type: classification failed ({e})")
        return {"day_type": "unknown", "gap_open_pct": None,
                "trend_strength": None}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def classify_regime(*, index_df: Optional[pd.DataFrame] = None,
                    vix: Optional[float] = None,
                    breadth_above_vwap: Optional[float] = None,
                    idx_trend: Optional[dict] = None,
                    now: Optional[datetime] = None,
                    cfg=None) -> EngineResult:
    """Pure regime classification from caller-supplied inputs.

    Every value degrades independently to None — a missing VIX must not cost
    us the day-type label, and vice versa. ok stays True regardless: regime
    is evidence for later attribution, never a veto (the index veto lives in
    intraday_score and is deliberately not duplicated here).
    """
    cfg = cfg or CONFIG.regime
    diagnostics: list[str] = []

    day = _classify_day_type(index_df, cfg, diagnostics)

    vix_f = _to_float(vix)
    if vix is not None and vix_f is None:
        diagnostics.append("vix: non-numeric value ignored")

    breadth_f = _to_float(breadth_above_vwap)
    if breadth_above_vwap is not None and breadth_f is None:
        diagnostics.append("breadth: non-numeric value ignored")

    # `now` must be a datetime (a date also works — we only need .date()).
    # Garbage falls back to the current IST clock + diagnostic: a bad caller
    # timestamp must not crash the alert path over a boolean expiry flag.
    if isinstance(now, datetime):
        when_date = now.date()
    elif isinstance(now, date):
        when_date = now
    else:
        if now is not None:
            diagnostics.append("now: not a datetime, using current IST time")
        when_date = datetime.now(IST).date()
    values = {
        "day_type": day["day_type"],
        "vol_state": _vol_state(vix_f, cfg),
        "vix": round(vix_f, 2) if vix_f is not None else None,
        "breadth_state": _breadth_state(breadth_f, cfg),
        "breadth_frac": round(breadth_f, 4) if breadth_f is not None else None,
        "expiry_day": bool(is_expiry_day(when_date, cfg)),
        "trend_strength": day["trend_strength"],
        "gap_open_pct": day["gap_open_pct"],
        "index_bias": _index_bias(idx_trend, diagnostics),
    }
    return EngineResult(engine=ENGINE, values=values, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# live snapshot (fetching wrapper) + TTL cache
# ---------------------------------------------------------------------------

# The bot may snapshot the regime for every alert in a scan pass; without a
# cache that's one NIFTY + one VIX fetch per alert against a quota-limited
# provider. TTL comes from config (regime moves slowly relative to a scan).
# Only SUCCESSFUL fetches are cached — caching a transient failure would
# blind the bot for a whole TTL window.
_cache: dict[str, tuple[Any, datetime]] = {}


def _reset_cache() -> None:
    """Test hook: drop the module-level fetch cache."""
    _cache.clear()


def _cached_fetch(key: str, fetch, diagnostics: list[str]) -> Any:
    """Return the cached value for `key` if fresh, else call `fetch()`.
    Any fetch failure ⇒ None + a diagnostics line, never an exception."""
    ttl = timedelta(minutes=CONFIG.regime.cache_ttl_min)
    now = datetime.now(IST)
    hit = _cache.get(key)
    if hit is not None and now - hit[1] < ttl:
        return hit[0]
    try:
        value = fetch()
    except Exception as e:
        logger.warning("market_regime: %s fetch failed: %s", key, e)
        diagnostics.append(f"{key}: fetch failed ({e})")
        return None
    _cache[key] = (value, now)
    return value


def _fetch_vix() -> Optional[float]:
    """Last India VIX daily close.

    Deliberately calls fetch_yfinance DIRECTLY: fetch_data("^INDIAVIX") would
    misroute the symbol down the Angel STOCK path (SYMBOL-EQ scrip lookup,
    which has no VIX and no yfinance fallback for stocks). VIX is regime
    context, so ~15-min-delayed yfinance data is acceptable here.
    """
    df = data_provider.fetch_yfinance("^INDIAVIX", period="1mo", interval="1d")
    return _to_float(df["Close"].iloc[-1])


def live_snapshot(idx_trend: Optional[dict] = None,
                  breadth_above_vwap: Optional[float] = None) -> EngineResult:
    """Bot-facing regime snapshot: fetch NIFTY 5-min + India VIX (guarded,
    TTL-cached), classify, and return. idx_trend / breadth come from the
    caller because the bot already computes them during its scan pass —
    re-deriving them here would double the fetch load for identical numbers.
    """
    diagnostics: list[str] = []
    index_df = _cached_fetch(
        "index_df",
        lambda: data_provider.fetch_data("NIFTY", period="5d", interval="5m"),
        diagnostics,
    )
    vix = _cached_fetch("vix", _fetch_vix, diagnostics)

    result = classify_regime(index_df=index_df, vix=vix,
                             breadth_above_vwap=breadth_above_vwap,
                             idx_trend=idx_trend)
    result.diagnostics = diagnostics + result.diagnostics
    return result
