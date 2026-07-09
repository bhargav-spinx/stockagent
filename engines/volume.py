"""
Volume engine — RVOL + volume-breakout measurement and points (Phase 2.1).

Two deliberately different volume reads, both extracted from intraday_score:

- rvol(): time-of-day-matched relative volume — today's cumulative volume vs
  the average cumulative volume of prior sessions AT THE SAME candle count.
  This is the correct construction (a raw ratio vs full prior days would call
  every morning "quiet" and every close "a spike").
- volume-breakout: last-candle volume vs the trailing 12-candle average
  (scanner_indicators.volume_ratio — single source, shared with the filters).

Phase 3 grows this module: per-symbol intraday volume curves from the local
archive (upgrading rvol from cross-session-cumulative to curve-relative),
volume acceleration, and a labeled institutional-activity proxy.
"""
from __future__ import annotations

import statistics

import pandas as pd

from config import CONFIG
from engines.base import EngineResult, tier_points
from scanner_indicators import volume_ratio


def rvol(today_df: pd.DataFrame, priors: list) -> float | None:
    """Time-of-day matched relative volume: today's cumulative volume so far
    vs the average cumulative volume of prior days up to the same candle count.

    Prior days with FEWER than n candles (half-days, muhurat, data gaps) are
    excluded from the baseline — summing a truncated day whole deflates the
    average and inflates today's RVOL, faking a volume-spike signal."""
    if not priors:
        return None
    n = len(today_df)
    today_cum = float(today_df["Volume"].iloc[:n].sum())
    prior_cums = [float(p["Volume"].iloc[:n].sum()) for p in priors if len(p) >= n]
    prior_cums = [c for c in prior_cums if c > 0]
    if not prior_cums:
        return None
    avg = statistics.mean(prior_cums)
    return today_cum / avg if avg > 0 else None


def rvol_points(rvol_value: float | None, cfg=None) -> int:
    cfg = cfg or CONFIG.score
    return tier_points(rvol_value, cfg.rvol_tiers)


def volume_breakout_points(ratio: float | None, cfg=None) -> int:
    cfg = cfg or CONFIG.score
    return tier_points(ratio, cfg.volume_breakout_tiers)


def evaluate(df: pd.DataFrame, today_df: pd.DataFrame, priors: list,
             cfg=None) -> EngineResult:
    """Structured-evidence view: both volume measurements + their points."""
    rv = rvol(today_df, priors)
    vb = volume_ratio(df)
    diags = []
    if rv is None:
        diags.append("RVOL baseline unavailable (no comparable prior sessions)")
    if vb is None:
        diags.append("volume-breakout baseline unavailable (<13 candles)")
    return EngineResult(
        engine="volume",
        values={
            "rvol": round(rv, 6) if rv is not None else None,
            "rvol_points": rvol_points(rv, cfg),
            "breakout_ratio": round(vb, 6) if vb is not None else None,
            "breakout_points": volume_breakout_points(vb, cfg),
        },
        diagnostics=diags,
    )
