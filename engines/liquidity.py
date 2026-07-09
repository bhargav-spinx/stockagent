"""
Liquidity engine (Phase 2): proxy-based liquidity floors.

Why proxies: this platform has no quote/depth feed — no bid/ask, no order
book, no ticks. Every metric here is therefore an HONEST PROXY computed from
OHLCV bars, and is named as such rather than pretending to be the real thing:

    median_turnover_cr        median daily Close×Volume in ₹ crore — proxies
                              "how much money actually changes hands daily".
    spread_proxy_pct          median 5-min (High−Low)/Close over the last ~2
                              sessions — proxies the spread+noise band a
                              market order must cross; NOT a quoted spread.
    per_min_traded_value_inr  median 5-min Close×Volume ÷ 5 — proxies the ₹
                              the tape absorbs per minute.
    participation_pct         order value as % of per-minute traded value.
    est_slippage_pct          linear impact model: slippage_k × participation.
                              Crude by design — with no depth data, a linear
                              coefficient is the most defensible shape.

The F&O-universe restriction upstream already excludes most illiquid names;
this engine makes the residual floor explicit and tunable (config.liquidity)
instead of leaving it implicit in universe choice.

Gate semantics (fail-open discipline, see engines/base.py): a KNOWN breach of
a floor rejects (ok=False, first breach wins); an UNKNOWN metric (None) never
rejects — it only adds a diagnostic. Rationale: data gaps are routine (fresh
listings, feed hiccups, off-hours calls) and rejecting on missing data would
suppress signals for reasons unrelated to liquidity. `liquid` is True/False
only when at least one gated metric is known; None when nothing is decidable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIG, LiquidityConfig
from constants import IST
from engines.base import EngineResult

logger = logging.getLogger(__name__)

# Structural sanity floors — deliberately NOT config: below these row counts a
# median is noise regardless of tuning, and making them tunable would invite
# "fixing" data-quality problems by loosening statistics.
MIN_DAILY_ROWS = 5          # fewer daily bars than this ⇒ turnover unknown
MIN_INTRADAY_ROWS = 20      # fewer 5-min bars than this ⇒ intraday proxies unknown
SPREAD_SESSIONS = 2         # spread proxy window: last ~2 IST calendar sessions
CANDLE_MINUTES = 5          # the platform's intraday bar size (5-min candles)
CRORE = 1e7                 # ₹ per crore


def _f(x) -> Optional[float]:
    """Coerce to a plain Python float (numpy scalars are JSON-unsafe in some
    serializers and EngineResult.values promises flat JSON-safe scalars)."""
    return None if x is None else float(x)


def _usable(df, min_rows: int) -> bool:
    """A frame we can compute on at all: a real DataFrame with enough rows."""
    return isinstance(df, pd.DataFrame) and len(df) >= min_rows


def _last_sessions(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Restrict to the last `n` IST calendar dates of a 5-min frame.

    Why: yesterday's dead-lunch candles say little about today's tradeable
    band, but ONE session alone is too thin early in the morning — two
    sessions is the compromise. A non-datetime index (or any surprise) falls
    back to all rows: a slightly stale proxy beats an exception here.
    """
    try:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return df
        idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
        dates = sorted(set(idx.date))
        keep = set(dates[-n:])
        mask = np.fromiter((d in keep for d in idx.date), dtype=bool,
                           count=len(idx))
        return df[mask]
    except Exception:
        return df


def _median_turnover_cr(daily_df, cfg: LiquidityConfig):
    """(median daily turnover in ₹ crore, diagnostic-or-None).

    Median (not mean) so one block-deal day cannot make a dead stock look
    liquid; lookback window from config so it can be walk-forward tuned.
    """
    if not _usable(daily_df, MIN_DAILY_ROWS):
        return None, "turnover unknown — not gated (daily data missing/thin)"
    try:
        tail = daily_df.tail(int(cfg.turnover_lookback_days))
        turnover = (tail["Close"].astype(float)
                    * tail["Volume"].astype(float)).dropna()
        if turnover.empty:
            return None, "turnover unknown — not gated (no usable daily rows)"
        return _f(turnover.median()) / CRORE, None
    except Exception as e:
        logger.debug("liquidity: turnover proxy failed: %s", e)
        return None, f"turnover unknown — not gated ({type(e).__name__})"


def _spread_proxy_pct(intraday_df):
    """(median 5-min (High−Low)/Close %, diagnostic-or-None).

    The candle range is an upper-bound stand-in for spread+microstructure
    noise: any market order realistically pays somewhere inside that band.
    Median over ~2 sessions keeps one news candle from condemning the stock.
    """
    if not _usable(intraday_df, MIN_INTRADAY_ROWS):
        return None, ("spread proxy unknown — not gated "
                      "(intraday data missing/thin)")
    try:
        recent = _last_sessions(intraday_df, SPREAD_SESSIONS)
        close = recent["Close"].astype(float)
        rng = recent["High"].astype(float) - recent["Low"].astype(float)
        ratio = (rng / close.where(close > 0) * 100.0).dropna()
        if ratio.empty:
            return None, ("spread proxy unknown — not gated "
                          "(no usable intraday rows)")
        return _f(ratio.median()), None
    except Exception as e:
        logger.debug("liquidity: spread proxy failed: %s", e)
        return None, f"spread proxy unknown — not gated ({type(e).__name__})"


def _per_min_traded_value(intraday_df):
    """(median ₹ traded per minute, diagnostic-or-None).

    Median 5-min Close×Volume ÷ 5: the denominator for participation. Uses
    the full intraday frame (unlike the spread proxy) because traded value is
    the more stable series and benefits from the larger sample.
    """
    if not _usable(intraday_df, MIN_INTRADAY_ROWS):
        return None, ("per-minute value unknown "
                      "(intraday data missing/thin)")
    try:
        value = (intraday_df["Close"].astype(float)
                 * intraday_df["Volume"].astype(float)).dropna()
        if value.empty:
            return None, "per-minute value unknown (no usable intraday rows)"
        return _f(value.median()) / CANDLE_MINUTES, None
    except Exception as e:
        logger.debug("liquidity: per-minute value failed: %s", e)
        return None, f"per-minute value unknown ({type(e).__name__})"


def assess(symbol: str,
           daily_df: Optional[pd.DataFrame] = None,
           intraday_df: Optional[pd.DataFrame] = None,
           order_value_inr: Optional[float] = None,
           cfg: Optional[LiquidityConfig] = None) -> EngineResult:
    """Assess proxy liquidity for `symbol`; total over garbage input.

    daily_df: daily OHLCV (Close, Volume) — turnover floor.
    intraday_df: 5-min OHLCV — spread proxy + per-minute traded value.
    order_value_inr: intended order size in ₹; enables participation/slippage.

    Hard floors (first breach wins, evaluated turnover → spread →
    participation, cheapest-data-first): each rejects only when the metric is
    KNOWN and in breach. Unknown metrics add a diagnostic and never gate —
    see module docstring for why fail-open is correct here.
    """
    cfg = cfg or CONFIG.liquidity
    diagnostics: list[str] = []

    turnover, note = _median_turnover_cr(daily_df, cfg)
    if note:
        diagnostics.append(note)

    spread, note = _spread_proxy_pct(intraday_df)
    if note:
        diagnostics.append(note)

    per_min, note = _per_min_traded_value(intraday_df)
    if note:
        diagnostics.append(note)

    participation: Optional[float] = None
    est_slippage: Optional[float] = None
    if order_value_inr is None:
        diagnostics.append("participation unknown — not gated (no order value)")
    elif per_min is None or per_min <= 0:
        diagnostics.append("participation unknown — not gated "
                           "(no per-minute traded value)")
    else:
        try:
            order_val = float(order_value_inr)
            if order_val < 0:
                raise ValueError("negative order value")
            participation = order_val / per_min * 100.0
            est_slippage = float(cfg.slippage_k) * participation
        except Exception as e:
            participation = est_slippage = None
            diagnostics.append("participation unknown — not gated "
                               f"({type(e).__name__})")

    # --- hard floors: first KNOWN breach wins ------------------------------
    ok = True
    reject_reason = ""
    if turnover is not None and turnover < cfg.min_median_turnover_cr:
        ok = False
        reject_reason = (f"median turnover ₹{turnover:.2f}cr < "
                         f"₹{cfg.min_median_turnover_cr:g}cr floor")
    if ok and spread is not None and spread > cfg.max_spread_proxy_pct:
        ok = False
        reject_reason = (f"spread proxy {spread:.3f}% > "
                         f"{cfg.max_spread_proxy_pct:g}% ceiling")
    if ok and participation is not None and \
            participation > cfg.max_participation_pct:
        ok = False
        reject_reason = (f"participation {participation:.2f}% of per-minute "
                         f"value > {cfg.max_participation_pct:g}% cap")

    # liquid is decidable once ANY gated metric is known; all-unknown ⇒ None
    # (and ok stays True — unknown must not reject).
    any_known = any(v is not None for v in (turnover, spread, participation))
    liquid: Optional[bool] = (False if not ok else
                              True if any_known else None)

    return EngineResult(
        engine="liquidity",
        values={
            "median_turnover_cr": _f(turnover),
            "spread_proxy_pct": _f(spread),
            "per_min_traded_value_inr": _f(per_min),
            "participation_pct": _f(participation),
            "est_slippage_pct": _f(est_slippage),
            "liquid": liquid,
        },
        ok=ok,
        reject_reason=reject_reason,
        diagnostics=diagnostics,
    )
