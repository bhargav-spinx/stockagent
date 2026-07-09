"""
Price Action Engine (Phase 3) — market STRUCTURE as features, never a decision.

Why this exists: the vote/score path reasons about indicators (RSI, MACD, VWAP)
but is blind to the raw geometry a discretionary trader reads first — the
sequence of swing highs and lows, whether the last leg broke structure with or
against the trend, and where the nearest reaction levels sit. This module turns
that geometry into flat, LOGGED numbers so the learning pipeline can later test
whether structure actually predicted outcomes. It emits features only; it does
not gate, size, or rank a trade.

Design constraints (identical discipline to the other Phase-2/3 engines):
- REUSE analyzer.atr — the pivot-prominence and S/R-clustering thresholds are
  expressed in ATR units, and they must be the SAME ATR the risk model uses; a
  private reimplementation drifting by one smoothing choice would make the
  logged structure incomparable to everything else.
- TOTAL over garbage input: a None / malformed / too-short frame yields the
  full key set with None values (n_swings 0) — never an exception into the
  signal path.
- Output is flat and JSON-safe (float / bool / str / None) so it merges
  straight into the alerts_log feature snapshot (features.py).
- No inline tunable literals — every threshold comes from CONFIG.price_action.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from analyzer import atr
from config import CONFIG
from engines.base import EngineResult

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ("High", "Low", "Close")

# Every feature evaluate() emits, in one place so the all-None fallback and the
# populated path can never disagree on the schema (a drifting key set would
# break feature-log column alignment downstream).
FEATURE_KEYS = (
    "trend",
    "last_high",
    "last_low",
    "hh",
    "hl",
    "lh",
    "ll",
    "bos",
    "choch",
    "n_swings",
    "nearest_support",
    "nearest_resistance",
    "dist_support_pct",
    "dist_resistance_pct",
)


def _is_ohlc(df) -> bool:
    """True only for a non-empty DataFrame carrying the OHLC columns we read."""
    return (isinstance(df, pd.DataFrame) and len(df) > 0
            and all(c in df.columns for c in _REQUIRED_COLS))


def _clean(x) -> float | None:
    """Coerce to a JSON-safe rounded float; NaN/inf/numpy scalars/junk → None.

    Every emitted number funnels through here so a numpy scalar can never leak
    into the JSON feature log (json.dumps chokes on np.float64 in some encoders,
    and NaN is not valid JSON at all)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 6)


def _last_atr(df: pd.DataFrame) -> float | None:
    """Latest ATR value as a plain float, or None when undefined (warmup / dead
    ticker). Central so both the pivot filter and the S/R clusterer measure
    thresholds against the identical ATR."""
    try:
        a = atr(df).iloc[-1]
    except Exception:  # noqa: BLE001 — total by contract
        return None
    if pd.isna(a):
        return None
    val = float(a)
    return val if math.isfinite(val) and val > 0 else None


def swing_points(df: pd.DataFrame, n: int | None = None,
                 atr_mult: float | None = None) -> list[dict]:
    """Detect fractal swing pivots over the last ``swing_lookback`` bars.

    A swing HIGH at bar i is a bar whose High is the strict (unique) maximum of
    the window [i-n, i+n] — n bars confirming on each side; a swing LOW is the
    symmetric strict minimum on Low. Requiring a UNIQUE extreme keeps flat tops
    from spawning a cluster of adjacent duplicate pivots. Because a pivot needs
    n bars of confirmation on the right, the most recent n bars can never be
    pivots — this is the intended lag, and it is exactly what lets
    break_of_structure() see a break of already-confirmed structure.

    When ATR is available and atr_mult > 0, a pivot is kept only if the local
    swing amplitude (window high-to-low range) is at least atr_mult × ATR, so a
    trivial one-tick wiggle inside noise is not logged as structure. If ATR is
    undefined (warmup / constant prices) the filter is skipped rather than
    dropping every pivot.

    Returns a CHRONOLOGICAL list of {"idx": int, "type": "high"|"low",
    "price": float}; idx is the row position in the passed frame. Empty list on
    a malformed frame or when there are fewer than 2n+1 bars.
    """
    cfg = CONFIG.price_action
    n = cfg.fractal_n if n is None else int(n)
    atr_mult = cfg.swing_atr_mult if atr_mult is None else float(atr_mult)

    if n < 1 or not _is_ohlc(df) or len(df) < 2 * n + 1:
        return []

    # Scan only the recent structure; older pivots are stale for "current" trend.
    scan = df.iloc[-cfg.swing_lookback:] if len(df) > cfg.swing_lookback else df
    start = len(df) - len(scan)          # map scan positions back to df rows
    m = len(scan)
    if m < 2 * n + 1:
        return []

    highs = scan["High"].to_numpy(dtype=float)
    lows = scan["Low"].to_numpy(dtype=float)

    # ATR threshold in price terms; None ⇒ prominence filter disabled.
    atr_val = _last_atr(df)
    min_move = (atr_val * atr_mult
                if atr_val is not None and atr_mult and atr_mult > 0 else None)

    pivots: list[dict] = []
    for i in range(n, m - n):
        seg_hi = highs[i - n:i + n + 1]
        seg_lo = lows[i - n:i + n + 1]
        if min_move is not None and float(seg_hi.max() - seg_lo.min()) < min_move:
            continue                      # swing too shallow to count
        hi = float(highs[i])
        lo = float(lows[i])
        if hi >= float(seg_hi.max()) and int((seg_hi == highs[i]).sum()) == 1:
            pivots.append({"idx": start + i, "type": "high", "price": hi})
        elif lo <= float(seg_lo.min()) and int((seg_lo == lows[i]).sum()) == 1:
            pivots.append({"idx": start + i, "type": "low", "price": lo})
    return pivots


def structure(df: pd.DataFrame, n: int | None = None,
              atr_mult: float | None = None) -> dict:
    """Classify the latest trend from the swing sequence.

    Uptrend = the last two swing highs are higher (HH) AND the last two swing
    lows are higher (HL); downtrend = lower high (LH) AND lower low (LL);
    anything else with enough pivots is a range (e.g. a broadening HH+LL, or
    equal extremes). Trend is None when there are not yet two highs and two
    lows to compare — an honest "not enough structure" rather than a guess.
    """
    pts = swing_points(df, n, atr_mult)
    highs = [p for p in pts if p["type"] == "high"]
    lows = [p for p in pts if p["type"] == "low"]

    out: dict = {"trend": None, "last_high": None, "last_low": None,
                 "hh": None, "hl": None, "lh": None, "ll": None}
    if highs:
        out["last_high"] = highs[-1]["price"]
    if lows:
        out["last_low"] = lows[-1]["price"]

    hh = hl = lh = ll = None
    if len(highs) >= 2:
        hh = highs[-1]["price"] > highs[-2]["price"]
        lh = highs[-1]["price"] < highs[-2]["price"]
    if len(lows) >= 2:
        hl = lows[-1]["price"] > lows[-2]["price"]
        ll = lows[-1]["price"] < lows[-2]["price"]
    out.update(hh=hh, hl=hl, lh=lh, ll=ll)

    if hh and hl:
        out["trend"] = "up"
    elif lh and ll:
        out["trend"] = "down"
    elif len(highs) >= 2 and len(lows) >= 2:
        out["trend"] = "range"
    return out


def break_of_structure(df: pd.DataFrame, n: int | None = None,
                        atr_mult: float | None = None) -> dict:
    """Detect a Break Of Structure vs a Change Of Character on the latest close.

    Both are the same event mechanically — the last close pushing beyond the
    most recent confirmed swing high (bullish) or swing low (bearish). What
    distinguishes them is DIRECTION RELATIVE TO THE PREVAILING TREND, and that
    distinction is the whole point:

      * BOS  — the break runs WITH the trend (up-trend taking out its last
               swing high). It is continuation; the trend is intact.
      * CHoCH — the break runs AGAINST the trend (an up-trend's close snapping
               below its last swing low). It is the first evidence the trend
               may be over — a regime warning, not a continuation.

    Because confirmed pivots lag by n bars, structure() still reads the PRIOR
    trend while the breaking bar itself has not yet formed a counter-pivot —
    which is exactly why a counter-trend break is detectable as CHoCH here.

    Returns {"bos": "bullish"|"bearish"|None, "choch": bool}. ``bos`` always
    carries the break DIRECTION (bullish/bearish); ``choch`` is True when that
    direction opposes the prevailing trend.
    """
    out: dict = {"bos": None, "choch": False}
    pts = swing_points(df, n, atr_mult)
    if not pts:
        return out

    highs = [p for p in pts if p["type"] == "high"]
    lows = [p for p in pts if p["type"] == "low"]
    last_close = float(df["Close"].iloc[-1])
    trend = structure(df, n, atr_mult)["trend"]

    recent_high = highs[-1] if highs else None
    recent_low = lows[-1] if lows else None
    broke_up = recent_high is not None and last_close > recent_high["price"]
    broke_down = recent_low is not None and last_close < recent_low["price"]

    direction: str | None = None
    if broke_up and broke_down:
        # Degenerate (older high sits below a newer low): resolve toward the
        # more recently formed level so the reported break is the live one.
        direction = ("bullish" if recent_high["idx"] >= recent_low["idx"]
                     else "bearish")
    elif broke_up:
        direction = "bullish"
    elif broke_down:
        direction = "bearish"

    if direction is None:
        return out                        # no break this bar

    out["bos"] = direction
    out["choch"] = bool((trend == "up" and direction == "bearish")
                        or (trend == "down" and direction == "bullish"))
    return out


def support_resistance(df: pd.DataFrame, n: int | None = None,
                       atr_mult: float | None = None) -> dict:
    """Cluster swing-pivot prices into levels and report the nearest to price.

    Pivots that stack within sr_cluster_atr × ATR of one another describe the
    SAME reaction level (a zone re-tested several times), so they are merged to
    their mean — otherwise three touches of "roughly 106" would masquerade as
    three separate levels. Nearest support is the highest merged level at or
    below the last close; nearest resistance the lowest at or above it.

    Distances are signed for intuition: dist_support_pct is how far price sits
    ABOVE support (positive), dist_resistance_pct how far BELOW resistance
    (positive). Returns None values when there are no pivots to cluster.
    """
    out: dict = {"nearest_support": None, "nearest_resistance": None,
                 "dist_support_pct": None, "dist_resistance_pct": None}
    pts = swing_points(df, n, atr_mult)
    if not pts:
        return out

    last_close = float(df["Close"].iloc[-1])
    prices = sorted(float(p["price"]) for p in pts)

    atr_val = _last_atr(df)
    tol = atr_val * CONFIG.price_action.sr_cluster_atr if atr_val else 0.0

    # Single left-to-right merge: extend the current cluster while the next
    # price is within tol of the last one added, else close the cluster.
    levels: list[float] = []
    cluster = [prices[0]]
    for pr in prices[1:]:
        if pr - cluster[-1] <= tol:
            cluster.append(pr)
        else:
            levels.append(sum(cluster) / len(cluster))
            cluster = [pr]
    levels.append(sum(cluster) / len(cluster))

    supports = [lv for lv in levels if lv <= last_close]
    resistances = [lv for lv in levels if lv >= last_close]
    if supports and last_close:
        s = max(supports)
        out["nearest_support"] = s
        out["dist_support_pct"] = (last_close - s) / last_close * 100.0
    if resistances and last_close:
        r = min(resistances)
        out["nearest_resistance"] = r
        out["dist_resistance_pct"] = (r - last_close) / last_close * 100.0
    return out


def _none_values() -> dict:
    """The full feature key set as an honest 'unknown' vector (n_swings 0)."""
    v = {k: None for k in FEATURE_KEYS}
    v["n_swings"] = 0
    return v


def evaluate(df: pd.DataFrame, cfg=None) -> EngineResult:
    """Flatten the structure read into one JSON-safe EngineResult.

    Total by contract: a None / thin / malformed frame returns the full key set
    with None values (n_swings 0) plus a diagnostic — never a raise. This is a
    pure feature engine, so ok stays True (it never gates a signal).
    """
    cfg = cfg or CONFIG.price_action
    values = _none_values()
    diags: list[str] = []
    try:
        if not _is_ohlc(df):
            diags.append("price_action: empty/invalid frame — features None")
            return EngineResult(engine="price_action", values=values,
                                diagnostics=diags)

        pts = swing_points(df, cfg.fractal_n, cfg.swing_atr_mult)
        values["n_swings"] = len(pts)
        if not pts:
            diags.append("price_action: no confirmed swing pivots "
                         "(insufficient/flat history)")
            return EngineResult(engine="price_action", values=values,
                                diagnostics=diags)

        st = structure(df, cfg.fractal_n, cfg.swing_atr_mult)
        bos = break_of_structure(df, cfg.fractal_n, cfg.swing_atr_mult)
        sr = support_resistance(df, cfg.fractal_n, cfg.swing_atr_mult)
        values.update({
            "trend": st["trend"],
            "last_high": _clean(st["last_high"]),
            "last_low": _clean(st["last_low"]),
            "hh": st["hh"], "hl": st["hl"], "lh": st["lh"], "ll": st["ll"],
            "bos": bos["bos"],
            "choch": bool(bos["choch"]),
            "nearest_support": _clean(sr["nearest_support"]),
            "nearest_resistance": _clean(sr["nearest_resistance"]),
            "dist_support_pct": _clean(sr["dist_support_pct"]),
            "dist_resistance_pct": _clean(sr["dist_resistance_pct"]),
        })
    except Exception as e:  # noqa: BLE001 — total by contract
        logger.warning("price_action: evaluation failed (%s) — None features", e)
        values = _none_values()
        diags = [f"price_action: evaluation error ({e})"]
    return EngineResult(engine="price_action", values=values, diagnostics=diags)
