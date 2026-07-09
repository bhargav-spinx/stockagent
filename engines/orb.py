"""
Opening-Range engine — dual-window ORB breakout detection + points (Phase 2.1).

Extracted verbatim from intraday_score. The "spent move" extension cap is the
part worth keeping forever: a stock 3–5% past its opening range at 2 PM is a
chase, not a breakout — entry would be the extended price with an ATR stop
hung off it. Phase 3 grows this module: false-breakout and retest detection,
opening momentum/strength measures.
"""
from __future__ import annotations

import pandas as pd

from config import CONFIG
from engines.base import EngineResult
from scanner_indicators import orb_levels


def orb_breakout(today_df: pd.DataFrame, price: float, cfg=None):
    """Evaluate every configured ORB window; return the stronger confirmed
    breakout as (window_minutes, direction, strength_pct) or (None, 'none',
    0.0). Breakouts extended beyond cfg.orb_max_extension_pct are ignored
    (spent move)."""
    cfg = cfg or CONFIG.score
    best = (None, "none", 0.0)
    for minutes, n in dict(cfg.orb_windows).items():
        if len(today_df) <= n:          # need candles beyond the ORB to break out
            continue
        hi, lo = orb_levels(today_df, n)
        if price > hi:
            strength = (price - hi) / hi * 100
            if strength <= cfg.orb_max_extension_pct and strength > best[2]:
                best = (minutes, "long", strength)
        elif price < lo:
            strength = (lo - price) / lo * 100
            if strength <= cfg.orb_max_extension_pct and strength > best[2]:
                best = (minutes, "short", strength)
    return best


def orb_points(orb_dir: str, direction: str, strength: float, cfg=None) -> int:
    """Points only when the breakout agrees with the trade direction; the
    strong/weak split rewards a confirmed move over a marginal poke."""
    cfg = cfg or CONFIG.score
    if orb_dir != direction:
        return 0
    return (cfg.orb_points_strong if strength >= cfg.orb_strong_strength_pct
            else cfg.orb_points_weak)


def evaluate(today_df: pd.DataFrame, price: float, direction: str,
             cfg=None) -> EngineResult:
    """Structured-evidence view: window, breakout direction, strength, points."""
    win, odir, strength = orb_breakout(today_df, price, cfg)
    return EngineResult(
        engine="orb",
        values={
            "window_min": win,
            "breakout_dir": odir,
            "strength_pct": round(strength, 6),
            "points": orb_points(odir, direction, strength, cfg),
        },
        diagnostics=([] if win is not None
                     else ["no confirmed ORB breakout (or move spent)"]),
    )
