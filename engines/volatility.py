"""
Volatility Engine (Phase 3) — emits volatility CONTEXT, never a decision.

Why this exists: the same setup means opposite things in different volatility
regimes. An ORB breakout OUT of a multi-week squeeze — ATR pinned at the low
end of its own range, an NR7 or inside bar on the trigger candle — is a
coiled-spring release; the identical breakout when ATR already sits at the top
of its range is a chase into exhaustion, entered at an extended price with an
ATR stop hung off it. The scorer must not conflate the two. So this module
LOGS the volatility state alongside every alert and lets the Phase-2 learning
pipeline discover how much it matters — it emits features only, it never
rejects or scores.

Design constraints (shared with every Phase-2/3 engine):
- REUSES analyzer.atr (Wilder RMA) instead of reimplementing ATR, so the ATR
  this module ranks is byte-identical to the ATR the risk engine hangs stops
  and targets off. A second ATR definition drifting by one smoothing choice
  would make the logged regime silently disagree with the executed risk.
- TOTAL over garbage input: a None / empty / too-short / malformed frame yields
  the full key set with None values plus a diagnostic — never an exception into
  the signal path (the same fail-open-with-a-note discipline base.py mandates).
- values are flat and JSON-safe (float / bool / str / None) so they merge
  straight into the alerts_log feature snapshot (features.py).
"""
from __future__ import annotations

import logging
import math

import pandas as pd

from analyzer import atr
from config import CONFIG
from engines.base import EngineResult

logger = logging.getLogger(__name__)

# The ATR path needs OHLC's High/Low/Close; the range flags (NR7, inside bar)
# need only High/Low. Kept as module constants so every guard checks the same
# column names rather than re-listing them inline.
_ATR_COLS = ("High", "Low", "Close")
_RANGE_COLS = ("High", "Low")

# Every value evaluate() emits, in one place, so the all-None fallback and the
# populated path can never disagree on the schema — a drifting key set would
# break feature-log column alignment downstream.
VALUE_KEYS = ("atr", "atr_pct", "atr_percentile", "nr7", "inside_bar", "state")


def _clean(x) -> float | None:
    """Coerce to a JSON-safe rounded float; NaN / inf / numpy scalars / junk
    become None.

    Every numeric result funnels through here so a numpy float64 (which some
    JSON serializers choke on) or a NaN (which is not valid JSON at all) can
    never leak into the feature log."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 6)


def _has_cols(df, cols) -> bool:
    """True only for a real, non-empty DataFrame carrying every named column —
    the single input gate the public functions share so their totality
    guarantees are identical rather than subtly different per function."""
    return (isinstance(df, pd.DataFrame) and not df.empty
            and all(c in df.columns for c in cols))


def atr_percentile(df: pd.DataFrame, period: int | None = None,
                   lookback: int | None = None) -> float | None:
    """Percentile rank (0..100] of the CURRENT ATR within its own trailing
    `lookback` history.

    Why a percentile and not raw ATR: raw ATR is in units of price and is
    therefore incomparable across symbols and across time. Its RANK within its
    own recent history is scale-free — it states "volatility is unusually
    high/low FOR THIS STOCK RIGHT NOW", which is exactly the squeeze/expansion
    read the setups condition on.

    Definition: the fraction of the last `lookback` ATR values that are <= the
    current ATR, times 100. The window includes the current value itself, so
    the result lands in (0, 100]. Returns None when there is not enough history
    to form even one Wilder ATR value (fewer than `period` bars) — the honest
    'unknown', never a fabricated number.

    period / lookback default to config.CONFIG.volatility so tunables live in
    one place; callers may override for cross-window comparisons."""
    cfg = CONFIG.volatility
    period = cfg.atr_period if period is None else period
    lookback = cfg.atr_pct_lookback if lookback is None else lookback
    if not _has_cols(df, _ATR_COLS):
        return None
    series = atr(df, period).dropna()           # drop the Wilder warmup NaNs
    if series.empty:
        return None
    window = series.iloc[-lookback:]
    current = float(series.iloc[-1])
    if not math.isfinite(current):
        return None
    le = int((window <= current).sum())         # current <= current is counted
    return le / len(window) * 100.0


def is_nr_n(df: pd.DataFrame, n: int | None = None) -> bool | None:
    """True when the LAST bar's High-Low range is the narrowest of the last
    `n` bars (this is NR7 at the default n=7).

    Why it matters: a narrowest-range bar is range CONTRACTION — the market
    coiling. Arriving after an expansion it frequently marks the pause before
    the next directional push, which is why NR7 is a classic pre-breakout tell.
    Returns None when there are fewer than `n` bars to compare against.

    n defaults to config.CONFIG.volatility.nr_lookback."""
    cfg = CONFIG.volatility
    n = cfg.nr_lookback if n is None else n
    if not _has_cols(df, _RANGE_COLS) or len(df) < n:
        return None
    rng = (df["High"] - df["Low"]).iloc[-n:]
    if rng.isna().any():                        # a NaN range makes min undefined
        return None
    # rng includes the last bar, so 'last <= window min' means last IS the min.
    return bool(float(rng.iloc[-1]) <= float(rng.min()))


def is_inside_bar(df: pd.DataFrame) -> bool | None:
    """True when the last bar is an inside bar: its High <= the prior High AND
    its Low >= the prior Low, so the whole bar nests inside the previous range.

    Why it matters: an inside bar is a one-bar volatility contraction / balance
    — the prior bar's range fully contains this one, so neither buyer nor
    seller extended control. It is the finest-grained squeeze this engine
    tracks. Returns None with fewer than 2 bars (nothing to nest inside)."""
    if not _has_cols(df, _RANGE_COLS) or len(df) < 2:
        return None
    high, prev_high = df["High"].iloc[-1], df["High"].iloc[-2]
    low, prev_low = df["Low"].iloc[-1], df["Low"].iloc[-2]
    if any(pd.isna(v) for v in (high, prev_high, low, prev_low)):
        return None
    return bool(high <= prev_high and low >= prev_low)


def _state_from_percentile(pct: float | None, cfg) -> str | None:
    """Map an ATR percentile to a coarse regime label.

    Split out from evaluate() so the threshold semantics are unit-testable in
    isolation: BOTH ends are INCLUSIVE (<= compression, >= expansion) so a
    value sitting exactly on a boundary is classified into the regime, never
    silently demoted to 'normal'. A None percentile stays None — unknown in,
    unknown out."""
    if pct is None:
        return None
    if pct <= cfg.compression_pctile:
        return "compressed"
    if pct >= cfg.expansion_pctile:
        return "expanded"
    return "normal"


def evaluate(df: pd.DataFrame, cfg=None) -> EngineResult:
    """Structured volatility-context view for the feature log.

    Emits (all flat + JSON-safe, None when unknown): atr (Wilder ATR level),
    atr_pct (that ATR as a % of the last close — the cross-symbol-comparable
    form), atr_percentile (rank within its own history), nr7 and inside_bar
    (the two contraction flags), and state (compressed / normal / expanded).

    Purely descriptive: ok stays True — this engine never gates a signal, it
    only annotates one. Every value left unknown drops a diagnostic so a
    silently-None feature downstream is always traceable to a reason here. TOTAL
    by contract: a None / empty / malformed frame returns the all-None schema
    with a diagnostic instead of raising."""
    cfg = cfg or CONFIG.volatility
    values: dict = {k: None for k in VALUE_KEYS}
    diags: list[str] = []

    if not _has_cols(df, _ATR_COLS):
        diags.append("volatility: empty/invalid frame — all values None")
        return EngineResult(engine="volatility", values=values,
                            diagnostics=diags)

    try:
        atr_series = atr(df, cfg.atr_period).dropna()
        if not atr_series.empty:
            values["atr"] = _clean(atr_series.iloc[-1])
            close_last = df["Close"].iloc[-1]
            if (values["atr"] is not None and pd.notna(close_last)
                    and float(close_last) != 0.0):
                values["atr_pct"] = _clean(
                    values["atr"] / float(close_last) * 100.0)

        values["atr_percentile"] = _clean(
            atr_percentile(df, cfg.atr_period, cfg.atr_pct_lookback))

        nr = is_nr_n(df, cfg.nr_lookback)
        values["nr7"] = None if nr is None else bool(nr)
        ib = is_inside_bar(df)
        values["inside_bar"] = None if ib is None else bool(ib)

        # state is derived from the (already rounded) logged percentile so the
        # label can never disagree with the atr_percentile value beside it.
        values["state"] = _state_from_percentile(values["atr_percentile"], cfg)
    except Exception as e:  # noqa: BLE001 — total by contract
        logger.warning("volatility: computation failed (%s) — all values None",
                       e)
        return EngineResult(
            engine="volatility",
            values={k: None for k in VALUE_KEYS},
            diagnostics=["volatility: computation failed — all values None"])

    for k in VALUE_KEYS:
        if values[k] is None:
            diags.append(f"volatility: {k} unknown")
    return EngineResult(engine="volatility", values=values, diagnostics=diags)
