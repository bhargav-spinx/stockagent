"""
VWAP engine — VWAP confirmation points (Phase 2.1).

The VWAP series itself stays in scanner_indicators.vwap (single source,
session-reset, shared with the filters and setup detectors); this module owns
the scorer's confirmation semantics. Phase 3 grows it into the full VWAP
state machine: hold / failure / reclaim events, time-above-VWAP fraction,
ATR-normalized distance, slope regimes.
"""
from __future__ import annotations

import pandas as pd

from config import CONFIG
from engines.base import EngineResult
from scanner_indicators import vwap as vwap_series, vwap_slope_pct
from scanner_indicators import today_session


def confirmation_points(price: float, vwap_value: float, is_long: bool,
                        cfg=None) -> int:
    """Full points when price sits on the trade's side of session VWAP,
    else 0 — binary by design (the scorer's original semantics)."""
    cfg = cfg or CONFIG.score
    aligned = (price > vwap_value) if is_long else (price < vwap_value)
    return cfg.vwap_points if aligned else 0


def vwap_state(df: pd.DataFrame, atr_val: float | None = None) -> dict:
    """Session VWAP state machine — the intraday relationship price has with
    VWAP, as features:

      time_above_frac — fraction of today's candles that closed above VWAP
                        (a proxy for who's in control this session).
      reclaim / failure — did price cross BACK above VWAP after being below
                        (reclaim, bullish) or drop below after being above
                        (failure, bearish) on the LAST bar? These transitions
                        matter more than the static above/below.
      dist_atr        — distance from VWAP in ATR units (volatility-normalized,
                        so 'far from VWAP' is comparable across stocks).
      slope_pct       — VWAP slope (trend of the fair-value line itself).
    """
    out = {"above": None, "time_above_frac": None, "reclaim": None,
           "failure": None, "dist_atr": None, "slope_pct": None}
    try:
        vw = vwap_series(df)
        today = today_session(df)
        vw_today = vw.reindex(today.index)
        closes = today["Close"].astype(float)
        rel = (closes > vw_today).dropna()
        if len(rel) == 0:
            return out
        out["above"] = bool(rel.iloc[-1])
        out["time_above_frac"] = round(float(rel.mean()), 4)
        if len(rel) >= 2:
            prev, now = bool(rel.iloc[-2]), bool(rel.iloc[-1])
            out["reclaim"] = (not prev) and now      # crossed up through VWAP
            out["failure"] = prev and (not now)      # dropped below VWAP
        last_vw = float(vw_today.dropna().iloc[-1])
        last_px = float(closes.iloc[-1])
        if atr_val and atr_val > 0:
            out["dist_atr"] = round((last_px - last_vw) / atr_val, 4)
        slope = vwap_slope_pct(vw)
        out["slope_pct"] = round(slope, 6) if slope is not None else None
    except Exception:
        return out
    return out


def evaluate(df: pd.DataFrame, price: float, direction: str,
             atr_val: float | None = None, cfg=None) -> EngineResult:
    """Structured-evidence view: VWAP level, distance, points, and the
    session state machine (hold/reclaim/failure, time-above, slope)."""
    try:
        vw = float(vwap_series(df).iloc[-1])
    except Exception:
        return EngineResult(engine="vwap", values={"vwap": None,
                                                   "dist_pct": None,
                                                   "points": 0},
                            diagnostics=["VWAP not computable"])
    dist = (price - vw) / vw * 100 if vw else None
    values = {
        "vwap": round(vw, 6),
        "dist_pct": round(dist, 6) if dist is not None else None,
        "points": confirmation_points(price, vw, direction == "long", cfg),
    }
    values.update(vwap_state(df, atr_val))
    return EngineResult(engine="vwap", values=values)
