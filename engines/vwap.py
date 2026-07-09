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
from scanner_indicators import vwap as vwap_series


def confirmation_points(price: float, vwap_value: float, is_long: bool,
                        cfg=None) -> int:
    """Full points when price sits on the trade's side of session VWAP,
    else 0 — binary by design (the scorer's original semantics)."""
    cfg = cfg or CONFIG.score
    aligned = (price > vwap_value) if is_long else (price < vwap_value)
    return cfg.vwap_points if aligned else 0


def evaluate(df: pd.DataFrame, price: float, direction: str,
             cfg=None) -> EngineResult:
    """Structured-evidence view: VWAP level, distance and points."""
    try:
        vw = float(vwap_series(df).iloc[-1])
    except Exception:
        return EngineResult(engine="vwap", values={"vwap": None,
                                                   "dist_pct": None,
                                                   "points": 0},
                            diagnostics=["VWAP not computable"])
    dist = (price - vw) / vw * 100 if vw else None
    return EngineResult(
        engine="vwap",
        values={
            "vwap": round(vw, 6),
            "dist_pct": round(dist, 6) if dist is not None else None,
            "points": confirmation_points(price, vw, direction == "long", cfg),
        },
    )
