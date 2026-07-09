"""
Decision-time feature snapshot builders — the Learning Engine's collection half.

Every actionable alert gets the full set of engine-computed values captured AT
THE MOMENT OF THE CALL and persisted to alerts_log.features (JSON, via
subscriptions.log_alert). These are the future training rows: (features @ call,
resolved outcome) pairs. They cannot be reconstructed later by re-fetching
candles — corporate-action adjustments, provider history caps and the repaint
guard all make recomputed values diverge from what the engine actually saw.

Contract:
- Flat dict, JSON-safe scalars only (str / int / float / bool / None).
- None means "engine could not compute it" — an honest unknown, never 0.
- Stable snake_case keys; `_v` (FEATURE_SCHEMA_VERSION) bumps on any breaking
  rename so a future training pipeline can dispatch per version.
- Numpy scalars are coerced to native Python types.

Keep builders total: they must never raise on a well-formed engine output —
a broken snapshot must not take down the alert path (callers also guard).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from constants import IST

FEATURE_SCHEMA_VERSION = 1

# NSE session open — minutes-since-open is a more useful model input than raw
# clock time (setups behave differently at minute 5 vs minute 240).
_SESSION_OPEN_H, _SESSION_OPEN_M = 9, 15


def _num(value: Any) -> float | int | bool | None:
    """Coerce numpy scalars → native; reject non-finite floats to None."""
    if value is None or isinstance(value, bool):
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            return None
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, int):
        return value
    return None


def _round(value: Any, digits: int = 6) -> float | int | bool | None:
    v = _num(value)
    return round(v, digits) if isinstance(v, float) else v


def _time_features(now: datetime | None = None) -> dict:
    now = now or datetime.now(IST)
    if now.tzinfo is None:
        now = IST.localize(now)
    else:
        now = now.astimezone(IST)
    open_min = _SESSION_OPEN_H * 60 + _SESSION_OPEN_M
    return {
        "hour_ist": now.hour,
        "minute_ist": now.minute,
        "min_since_open": (now.hour * 60 + now.minute) - open_min,
        "weekday": now.weekday(),
    }


def _levels_features(entry, stop_loss, target1, target2) -> dict:
    entry = _num(entry)
    out: dict[str, Any] = {"risk_pct": None, "t1_pct": None, "t2_pct": None,
                           "rr1": None, "rr2": None}
    if not entry:
        return out
    sl, t1, t2 = _num(stop_loss), _num(target1), _num(target2)
    risk = abs(entry - sl) if sl is not None else None
    if risk:
        out["risk_pct"] = round(risk / entry * 100, 6)
        if t1 is not None:
            out["t1_pct"] = round(abs(t1 - entry) / entry * 100, 6)
            out["rr1"] = round(abs(t1 - entry) / risk, 6)
        if t2 is not None:
            out["t2_pct"] = round(abs(t2 - entry) / entry * 100, 6)
            out["rr2"] = round(abs(t2 - entry) / risk, 6)
    return out


# Scorecard breakdown display names → stable feature keys.
_BREAKDOWN_KEYS = {
    "Gap Up/Down": "pts_gap",
    "Relative Volume": "pts_rvol",
    "VWAP Confirmation": "pts_vwap",
    "EMA Trend": "pts_ema",
    "ORB Breakout": "pts_orb",
    "Volume Breakout": "pts_volbk",
}


def features_from_scorecard(card, idx_trend: dict | None = None,
                            now: datetime | None = None) -> dict:
    """Snapshot for an intraday_score.ScoreCard (scan / score alerts)."""
    f: dict[str, Any] = {
        "_v": FEATURE_SCHEMA_VERSION,
        "engine": "intraday_score",
        "direction": card.direction,
        "score": _num(card.score),
        "price": _round(card.price),
        "gap_pct": _round(card.gap_pct),
        "rvol": _round(card.rvol),
        "orb_window": _num(card.orb_window),
        "regime_ok": bool(card.regime_ok),
        "event_ok": bool(card.event_ok),
        "delivery_pct": _round(card.delivery_pct),
        "bull_conviction": _num(card.bull_conviction),
        "bear_conviction": _num(card.bear_conviction),
        "n_notes": len(card.notes or []),
        "supertrend_agrees": not any(
            "supertrend disagrees" in n.lower() for n in (card.notes or [])),
        "near_52w_high": any("52-week high" in s for s in (card.context or [])),
        "near_52w_low": any("52-week low" in s for s in (card.context or [])),
    }
    for display, key in _BREAKDOWN_KEYS.items():
        f[key] = _num((card.breakdown or {}).get(display, 0))
    f.update(_levels_features(card.entry, card.stop_loss,
                              card.target1, card.target2))
    for alias, trend in (idx_trend or {}).items():
        f[f"idx_{alias.lower()}"] = str(trend)
    f.update(_time_features(now))
    return f


# analyzer vote display names → stable feature keys ("SMA Crossover" and
# "SMA Trend" are the same voter in two states).
_VOTE_KEYS = {
    "SMA Crossover": "vote_sma",
    "SMA Trend": "vote_sma",
    "RSI": "vote_rsi",
    "MACD": "vote_macd",
    "Bollinger": "vote_bb",
}
_VOTE_VALUE = {"BUY": 1, "HOLD": 0, "SELL": -1}


def features_from_swing_result(result: dict, now: datetime | None = None) -> dict:
    """Snapshot for an analyzer.analyze() result dict (swing / manual alerts)."""
    setup = result.get("trade_setup") or {}
    price = _num(result.get("price"))
    atr_val = _num(setup.get("atr"))
    f: dict[str, Any] = {
        "_v": FEATURE_SCHEMA_VERSION,
        "engine": "analyzer",
        "mode": result.get("mode"),
        "signal": result.get("signal"),
        "price": _round(price),
        "change_pct": _round(result.get("change_pct")),
        "rsi": _round(result.get("rsi")),
        "confidence": _round(result.get("confidence")),
        "atr": _round(atr_val),
        "atr_pct": (round(atr_val / price * 100, 6)
                    if atr_val is not None and price else None),
        "risk_level": setup.get("risk_level"),
        "market_open": bool(result.get("market_open")),
    }
    for name, vote, _reason in result.get("indicators") or []:
        key = _VOTE_KEYS.get(name)
        if key:
            f[key] = _VOTE_VALUE.get(vote)
    # Continuous context features (engines/context_daily, keys ctx_*) — the
    # numeric versions of the votes above; already flat + JSON-safe.
    for k, v in (result.get("context_features") or {}).items():
        f[k] = v
    f.update(_levels_features(setup.get("entry"), setup.get("stop_loss"),
                              setup.get("target1"), setup.get("target2")))
    sh, sl_ = _num(setup.get("swing_high")), _num(setup.get("swing_low"))
    f["swing_high_dist_pct"] = (round((sh - price) / price * 100, 6)
                                if sh is not None and price else None)
    f["swing_low_dist_pct"] = (round((price - sl_) / price * 100, 6)
                               if sl_ is not None and price else None)
    f.update(_time_features(now))
    return f
