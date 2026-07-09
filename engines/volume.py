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


def volume_curve_ratio(today_df: pd.DataFrame, priors: list) -> float | None:
    """Refinement of RVOL that compares TODAY's volume at the current candle
    position to prior days' volume AT THE SAME position (not cumulative). The
    intraday volume curve is U-shaped (heavy at open/close, light midday), so a
    same-position comparison is a cleaner 'unusually active right now?' read
    than a cumulative ratio. None until a stable per-position baseline exists."""
    if not priors:
        return None
    i = len(today_df) - 1                     # current candle position (0-based)
    cur = float(today_df["Volume"].iloc[i])
    prior_at_i = [float(p["Volume"].iloc[i]) for p in priors if len(p) > i]
    prior_at_i = [v for v in prior_at_i if v > 0]
    if not prior_at_i:
        return None
    avg = sum(prior_at_i) / len(prior_at_i)
    return cur / avg if avg > 0 else None


def volume_acceleration(df: pd.DataFrame, window: int = 3) -> float | None:
    """Ratio of the last `window` candles' mean volume to the `window` before
    them — is participation accelerating (>1) or fading (<1)? None if <2·window
    candles or a zero baseline."""
    if len(df) < 2 * window:
        return None
    v = df["Volume"].astype(float)
    recent = float(v.iloc[-window:].mean())
    prior = float(v.iloc[-2 * window:-window].mean())
    return recent / prior if prior > 0 else None


def institutional_proxy(rvol: float | None, delivery_pct: float | None,
                        close_pos: float | None) -> float | None:
    """A 0..1 composite PROXY for institutional participation — NOT a
    measurement (no order-flow/quote feed exists). Blends: relative volume
    (capped), delivery % (accumulation vs intraday churn), and where price
    closed in its range (conviction). Explicitly a heuristic; its value is that
    the feature harness can test whether it separates outcomes at all."""
    parts, weights = [], []
    if rvol is not None:
        parts.append(min(rvol / 3.0, 1.0)); weights.append(0.4)
    if delivery_pct is not None:
        parts.append(min(max(delivery_pct, 0.0) / 100.0, 1.0)); weights.append(0.35)
    if close_pos is not None:
        parts.append(min(max(close_pos, 0.0), 1.0)); weights.append(0.25)
    if not parts:
        return None
    return round(sum(p * w for p, w in zip(parts, weights)) / sum(weights), 4)


def rvol_points(rvol_value: float | None, cfg=None) -> int:
    cfg = cfg or CONFIG.score
    return tier_points(rvol_value, cfg.rvol_tiers)


def volume_breakout_points(ratio: float | None, cfg=None) -> int:
    cfg = cfg or CONFIG.score
    return tier_points(ratio, cfg.volume_breakout_tiers)


def evaluate(df: pd.DataFrame, today_df: pd.DataFrame, priors: list,
             delivery_pct: float | None = None, cfg=None) -> EngineResult:
    """Structured-evidence view: volume measurements, points, and the Phase-3
    refinements (curve-relative RVOL, acceleration, institutional proxy)."""
    rv = rvol(today_df, priors)
    vb = volume_ratio(df)
    curve = volume_curve_ratio(today_df, priors)
    accel = volume_acceleration(df)
    close_pos = None
    try:
        row = today_df.iloc[-1]
        rng = float(row["High"]) - float(row["Low"])
        if rng > 0:
            close_pos = (float(row["Close"]) - float(row["Low"])) / rng
    except Exception:
        close_pos = None
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
            "curve_ratio": round(curve, 6) if curve is not None else None,
            "acceleration": round(accel, 6) if accel is not None else None,
            "close_pos_in_range": round(close_pos, 4) if close_pos is not None else None,
            "institutional_proxy": institutional_proxy(rv, delivery_pct, close_pos),
        },
        diagnostics=diags,
    )
