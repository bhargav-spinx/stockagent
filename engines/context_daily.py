"""
Daily Context Engine — step 1: continuous re-expression of the analyzer's
ternary indicator votes.

Why this exists: the swing analyzer collapses each indicator into a
BUY/SELL/HOLD vote, which throws away magnitude — "RSI 51" and "RSI 69" both
become the same bullish vote. The learning pipeline (Phase 2) needs the
underlying NUMBERS so a model can later discover which magnitudes actually
predicted outcomes. This module is purely additive: the vote path remains the
decision rule; these features only get LOGGED alongside each alert.

Design constraints:
- REUSES analyzer.rsi / macd / bollinger_bands / atr rather than reimplementing
  them, so the logged features are computed by byte-identical code to what the
  votes saw (a reimplementation drifting by one smoothing choice would poison
  every downstream comparison).
- TOTAL over garbage input: a short / malformed / None frame yields the full
  key set with None values — never an exception into the signal path.
- Output is flat and JSON-safe (float or None only) so it merges directly into
  the alerts_log feature snapshot (features.py).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from analyzer import MODES, atr, bollinger_bands, macd, rsi
from engines.base import EngineResult

logger = logging.getLogger(__name__)

# ADX period is deliberately NOT taken from the per-mode MODES params: ADX(14)
# is the charting standard on every platform (TradingView / Zerodha) for both
# daily and intraday charts, so we keep one comparable definition across modes
# rather than inventing a mode-specific variant nobody charts.
ADX_PERIOD = 14

# EMA slope is regressed over its last N values. 5 candles ≈ one trading week
# on dailies — long enough to smooth single-bar noise, short enough that the
# slope still describes the CURRENT push rather than the whole leg.
SLOPE_WINDOW = 5

# Return lookbacks are definitional, not tunable: they are baked into the
# feature NAMES (ctx_ret_5 / ctx_ret_20), so changing them silently would
# corrupt the meaning of every previously logged row.
RET_SHORT = 5
RET_LONG = 20

# Every feature this engine emits — kept in one place so the all-None fallback
# and the populated path can never emit different key sets (a drifting schema
# would break feature-log column alignment downstream).
FEATURE_KEYS = (
    "ctx_rsi",
    "ctx_macd_hist_atr",
    "ctx_bb_z",
    "ctx_ema_fast_slope_pct",
    "ctx_adx",
    "ctx_atr_pct",
    "ctx_dist_sma_long_pct",
    "ctx_ret_5",
    "ctx_ret_20",
)

_REQUIRED_COLS = ("High", "Low", "Close")


def _none_features() -> dict:
    """The full key set, all None — the honest 'unknown' vector."""
    return {k: None for k in FEATURE_KEYS}


def _clean(x) -> float | None:
    """Coerce to a JSON-safe rounded float; NaN/inf/np scalars/junk → None.

    Every emitted value funnels through here so numpy scalars can never leak
    into the JSON feature log (json.dumps chokes on np.float64 in some
    serializers, and NaN is not valid JSON at all)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 6)


# ---------- ADX (Wilder) ----------

def _directional_movement(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Raw (unsmoothed) +DM / −DM series — Wilder's selection rule.

    +DM = today's high-advance IF it exceeds the low-decline and is positive,
    else 0; −DM symmetric. Split out from adx() so the selection logic is
    directly unit-testable against a hand-worked example."""
    up = df["High"].diff()          # today's push above yesterday's high
    down = -df["Low"].diff()        # today's push below yesterday's low
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    return plus_dm, minus_dm


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """Wilder ADX — trend-STRENGTH (0..100), direction-agnostic.

    Why we need it: RSI/MACD magnitudes mean different things in a trend vs a
    chop; ADX is the standard regime discriminator that lets a model condition
    on that. Construction matches charting platforms: Wilder RMA smoothing
    (ewm alpha=1/period, adjust=False) for DM, TR and DX — the smoothed TR is
    exactly analyzer.atr, which we reuse. NaN during warmup (min_periods)."""
    plus_dm, minus_dm = _directional_movement(df)
    atr_s = atr(df, period)  # Wilder-smoothed TR — same RMA as the DMs below
    # Guard: a zero/NaN ATR (dead ticker, constant prices) must not divide.
    atr_safe = atr_s.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr_safe
    di_sum = (plus_di + minus_di).replace(0, np.nan)  # guard 0-division
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ---------- Continuous features ----------

def continuous_features(df: pd.DataFrame | None, mode: str = "swing") -> dict:
    """Continuous versions of the analyzer's vote inputs, as a flat dict.

    Total function: a None / malformed / too-short frame returns the full key
    set with None values (never raises) — downstream logging must always get
    a schema-stable row. Indicator params come from analyzer.MODES[mode] so
    the numbers describe the SAME indicators the votes were cast on.
    """
    out = _none_features()
    cfg = MODES.get(mode)
    if cfg is None:
        return out
    if df is None or not isinstance(df, pd.DataFrame):
        return out
    if any(c not in df.columns for c in _REQUIRED_COLS):
        return out
    # One global history floor instead of per-feature dribble: a frame that
    # can't populate the slowest indicator (SMA-long / ADX warmup) would yield
    # a partially-None vector whose gaps shift with mode params — worse for a
    # learner than an honestly all-None row it can drop wholesale.
    min_rows = max(cfg["sma_long"], cfg["bb_period"],
                   cfg["rsi_period"] + 1, RET_LONG + 1, 2 * ADX_PERIOD)
    if len(df) < min_rows:
        return out

    try:
        close = df["Close"]
        last_close = float(close.iloc[-1])

        # RSI — the vote only kept oversold/overbought crossings; the level
        # itself (e.g. 62 vs 48) is the feature.
        rsi_s = rsi(close, cfg["rsi_period"])
        out["ctx_rsi"] = _clean(rsi_s.iloc[-1])

        # MACD histogram normalized by ATR: raw histogram is price-scale
        # dependent (₹3 on a ₹3000 stock ≠ ₹3 on a ₹90 stock); dividing by
        # ATR expresses momentum in units of typical daily movement.
        _, _, hist = macd(close, *cfg["macd"])
        atr_s = atr(df)  # ATR(14) — analyzer default, the charting standard
        atr_last = atr_s.iloc[-1]
        hist_last = hist.iloc[-1]
        if pd.notna(atr_last) and float(atr_last) != 0.0 and pd.notna(hist_last):
            out["ctx_macd_hist_atr"] = _clean(float(hist_last) / float(atr_last))

        # Bollinger z-score: how many population std-devs price sits from the
        # middle band. Derive std from the analyzer's own bands (std_dev=2 ⇒
        # std = (upper − middle)/2) instead of recomputing, so ddof=0 and the
        # window match the vote path exactly.
        upper, middle, _ = bollinger_bands(close, cfg["bb_period"])
        mid_last, up_last = middle.iloc[-1], upper.iloc[-1]
        if pd.notna(mid_last) and pd.notna(up_last):
            std_last = (float(up_last) - float(mid_last)) / 2.0
            if std_last != 0.0:
                out["ctx_bb_z"] = _clean((last_close - float(mid_last)) / std_last)

        # Fast-EMA slope: the vote used SMA-cross (binary); the regression
        # slope of the fast EMA, as % of its level, captures how HARD the
        # short-term trend is pushing per candle.
        ema_fast = close.ewm(span=cfg["sma_short"], adjust=False).mean()
        tail = ema_fast.iloc[-SLOPE_WINDOW:].to_numpy(dtype=float)
        if len(tail) == SLOPE_WINDOW and np.all(np.isfinite(tail)):
            mean = tail.mean()
            if mean != 0.0:
                slope = np.polyfit(np.arange(SLOPE_WINDOW), tail, 1)[0]
                out["ctx_ema_fast_slope_pct"] = _clean(slope / mean * 100.0)

        # ADX(14) — see ADX_PERIOD comment for why it ignores mode params.
        out["ctx_adx"] = _clean(adx(df).iloc[-1])

        # ATR as % of price — volatility in comparable units across symbols.
        if last_close != 0.0 and pd.notna(atr_last):
            out["ctx_atr_pct"] = _clean(float(atr_last) / last_close * 100.0)

        # Distance from the long SMA in % — trend location, continuous form
        # of the analyzer's price-vs-SMA-long vote.
        sma_long = close.rolling(cfg["sma_long"]).mean().iloc[-1]
        if pd.notna(sma_long) and float(sma_long) != 0.0:
            out["ctx_dist_sma_long_pct"] = _clean(
                (last_close - float(sma_long)) / float(sma_long) * 100.0)

        # Plain momentum returns — the simplest features often carry the most
        # signal; log them so the fancy ones must beat them to matter.
        out["ctx_ret_5"] = _clean(close.pct_change(RET_SHORT).iloc[-1] * 100.0)
        out["ctx_ret_20"] = _clean(close.pct_change(RET_LONG).iloc[-1] * 100.0)
    except Exception as e:  # noqa: BLE001 — total by contract
        logger.warning("context_daily: feature computation failed (%s) — "
                       "returning None features", e)
        return _none_features()
    return out


def features_from_analysis(df: pd.DataFrame | None, mode: str = "swing") -> dict:
    """Orchestrator-facing alias for continuous_features.

    Kept as a separate name so the orchestrator's call site reads as intent
    ("features derived from this analysis") and survives any future internal
    refactor of continuous_features without a rename ripple."""
    return continuous_features(df, mode=mode)


def run(df: pd.DataFrame | None, mode: str = "swing") -> EngineResult:
    """EngineResult adapter — the standard Phase-2 engine contract.

    values are already ctx_-prefixed (they merge into the feature snapshot
    as-is), so callers using feature_dict() should pass prefix=""."""
    values = continuous_features(df, mode=mode)
    diags: list[str] = []
    if all(v is None for v in values.values()):
        diags.append(f"context_daily: insufficient/invalid history for "
                     f"mode={mode!r} — all features None")
    return EngineResult(engine="context_daily", values=values,
                        diagnostics=diags)
