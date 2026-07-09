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


def evaluate(today_df: pd.DataFrame, priors: list, direction: str,
             cfg=None) -> EngineResult:
    """Structured-evidence view: measurement + points, never a decision."""
    g = gap_pct(today_df, priors)
    return EngineResult(
        engine="gap",
        values={
            "gap_pct": round(g, 6) if g is not None else None,
            "points": gap_points(g, direction == "long", cfg),
        },
        diagnostics=([] if g is not None else ["no prior session — gap unknown"]),
    )
