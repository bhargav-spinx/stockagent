"""
Gap engine — opening-gap measurement + the scorer's gap points (Phase 2.1).

Extracted verbatim from intraday_score so gap intelligence has a dedicated
home to grow in: Phase 3 adds classification (tiny/normal/breakaway/runaway/
exhaustion/fill-candidate) with fill rates measured from the local archive —
never asserted from trading folklore. The legacy 100-point scorer composes
this module; tests/test_golden_parity.py pins that the composition is
byte-identical to the pre-extraction scorer.
"""
from __future__ import annotations

import pandas as pd

from config import CONFIG
from engines.base import EngineResult, tier_points


def gap_pct(today_df: pd.DataFrame, priors: list) -> float | None:
    """Opening gap: today's first Open vs the prior session's last Close, in %.
    None when there is no prior session to gap from."""
    if not priors:
        return None
    prev_close = float(priors[-1]["Close"].iloc[-1])
    today_open = float(today_df["Open"].iloc[0])
    if prev_close == 0:
        return None
    return (today_open - prev_close) / prev_close * 100


def gap_points(gap: float | None, is_long: bool, cfg=None) -> int:
    """Scorer points for the gap, favourable relative to trade direction.
    Unknown gap scores 0 — same semantics as the pre-extraction scorer
    (fav_gap defaulted to 0.0, below every tier)."""
    cfg = cfg or CONFIG.score
    fav = (gap if is_long else -gap) if gap is not None else 0.0
    return tier_points(fav, cfg.gap_tiers)


def gap_retrace_frac(gap: float | None, today_df: pd.DataFrame,
                     priors: list) -> float | None:
    """How much of the opening gap has been retraced toward the prior close,
    0..1 (1 = fully filled). A gap-up that trades back down closes the gap;
    a gap-down that trades back up does likewise. None when gap unknown/zero."""
    if not gap or not priors:
        return None
    prev_close = float(priors[-1]["Close"].iloc[-1])
    today_open = float(today_df["Open"].iloc[0])
    last = float(today_df["Close"].iloc[-1])
    denom = today_open - prev_close
    if denom == 0:
        return None
    # fraction of the open→prev_close distance covered by price so far
    frac = (today_open - last) / denom
    return max(0.0, min(1.0, frac))


def classify_gap(gap: float | None, atr_pct: float | None,
                 rvol: float | None, retrace_frac: float | None = None,
                 cfg=None) -> dict:
    """Classify the gap by SIZE (in ATR units, so it's volatility-relative — a
    1% gap means very different things on a calm vs a wild stock) and CHARACTER
    (a large gap with heavy volume is a breakaway with follow-through; the same
    gap on thin volume is exhaustion that tends to fade). Every field is a
    feature, never a signal — Phase 3 will measure which classes actually
    separate outcomes from the archive before any of this gates a trade."""
    cfg = cfg or CONFIG.gap_class
    out = {"gap_atr": None, "gap_class": None, "fill_candidate": None}
    if gap is None:
        return out
    fill = (retrace_frac is not None and retrace_frac >= cfg.fill_frac)
    out["fill_candidate"] = bool(fill) if retrace_frac is not None else None
    if not atr_pct or atr_pct <= 0:
        return out
    gap_atr = abs(gap) / atr_pct
    out["gap_atr"] = round(gap_atr, 4)
    if gap_atr < cfg.tiny_atr:
        out["gap_class"] = "tiny"
    elif gap_atr < cfg.normal_atr:
        out["gap_class"] = "normal"
    else:
        big = gap_atr >= cfg.large_atr
        if rvol is not None and rvol >= cfg.breakaway_rvol:
            out["gap_class"] = "runaway" if big else "breakaway"
        else:
            out["gap_class"] = "exhaustion"
    return out


def evaluate(today_df: pd.DataFrame, priors: list, direction: str,
             atr_pct: float | None = None, rvol: float | None = None,
             cfg=None) -> EngineResult:
    """Structured-evidence view: measurement + points + (when ATR/RVOL given)
    classification. Never a decision."""
    g = gap_pct(today_df, priors)
    retrace = gap_retrace_frac(g, today_df, priors)
    cls = classify_gap(g, atr_pct, rvol, retrace, cfg)
    return EngineResult(
        engine="gap",
        values={
            "gap_pct": round(g, 6) if g is not None else None,
            "points": gap_points(g, direction == "long", cfg),
            "retrace_frac": round(retrace, 4) if retrace is not None else None,
            **cls,
        },
        diagnostics=([] if g is not None else ["no prior session — gap unknown"]),
    )
