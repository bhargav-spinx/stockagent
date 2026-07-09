"""
Relative Strength Engine — stock return vs a benchmark index, as CONTEXT
features (Phase 3).

Why this exists: absolute return says nothing about EDGE. A stock up 2% on a
day the index is up 3% is a laggard; up 2% while the index is flat is genuine
strength. The learning pipeline (features.py) needs the *excess* return —
stock minus index over matched windows — plus a beta estimate, logged
alongside each alert so a model can later test whether relative strength (not
raw momentum) predicted outcomes. Nothing here casts a vote: these values are
recorded as context, never gated on.

Design constraints (shared with the other Phase-2/3 engines):
- PURE + OFFLINE: every function takes pre-fetched DataFrames; the engine never
  fetches. The orchestrator supplies the index frame at call time (config names
  which index via `index_alias`).
- TOTAL over garbage: a None / short / malformed frame yields None values plus
  a diagnostic, never an exception into the signal path.
- Output flat + JSON-safe (float or None only) so it merges straight into the
  alerts_log feature snapshot.
- Tunables (`windows`, `beta_lookback`, `index_alias`) come from
  config.CONFIG.rel_strength — never inline literals.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from config import CONFIG
from engines.base import EngineResult

logger = logging.getLogger(__name__)

_ENGINE = "relative_strength"


def _clean(x) -> float | None:
    """Coerce to a JSON-safe rounded float; NaN/inf/numpy-scalar/junk → None.

    Every value the engine emits funnels through here so a numpy scalar can
    never leak into the JSON feature log (some serializers choke on np.float64)
    and NaN — which is not valid JSON — can never be written as a feature."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 6)


def _close_series(obj) -> pd.Series | None:
    """Return a Close-price Series from a DataFrame (its 'Close' column) or a
    Series passed directly; None for anything else.

    Accepting either shape keeps the pure helpers callable with a bare price
    Series (handy in tests and for peer universes) while the orchestrator
    passes full OHLC frames."""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame) and "Close" in obj.columns:
        return obj["Close"]
    return None


def pct_return(series_or_df, lookback: int) -> float | None:
    """Percent change of Close over the last `lookback` bars:
    (last / ref − 1) × 100, where ref is the Close `lookback` bars before last.

    None when fewer than lookback+1 bars exist (no reference bar to measure
    against) or when the reference price is zero / non-finite. Pure: it reads
    only the trailing window and makes no assumption about the index labels."""
    s = _close_series(series_or_df)
    if s is None or lookback is None or lookback < 1 or len(s) < lookback + 1:
        return None
    try:
        last = float(s.iloc[-1])
        ref = float(s.iloc[-1 - lookback])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(last) or not math.isfinite(ref) or ref == 0.0:
        return None
    return (last / ref - 1.0) * 100.0


def relative_strength(stock_df, index_df, windows=None) -> dict:
    """Excess return of the stock over the index for each lookback window.

    For each window w emits stock_ret_w, index_ret_w and
    rs_w = stock_ret_w − index_ret_w (excess return, in percentage points).
    Any leg that is too short — or a missing index — yields None for that leg
    and therefore None for rs_w.

    Alignment: both frames are sorted ascending and each window uses that
    frame's OWN last w+1 bars, computed independently. We deliberately do NOT
    assume the two frames share an index or trading calendar, so a stock whose
    history differs in length from the index still gets a correct trailing
    return (mismatched calendars are handled by the beta() intersection, not
    here — these are simple point-to-point returns)."""
    if windows is None:
        windows = CONFIG.rel_strength.windows
    stock = _close_series(stock_df)
    index = _close_series(index_df)
    if stock is not None:
        stock = stock.sort_index()
    if index is not None:
        index = index.sort_index()
    out: dict = {}
    for w in windows:
        w = int(w)
        sret = pct_return(stock, w)
        iret = pct_return(index, w)
        rs = (sret - iret) if (sret is not None and iret is not None) else None
        out[f"rs_{w}"] = rs
        out[f"stock_ret_{w}"] = sret
        out[f"index_ret_{w}"] = iret
    return out


def beta(stock_df, index_df, lookback: int | None = None) -> float | None:
    """OLS beta of stock daily returns on index daily returns:
    beta = cov(stock, index) / var(index) over the last `lookback` returns.

    Alignment is by INDEX INTERSECTION (inner join on timestamp), and that is
    the crux worth reading twice: each frame's returns are computed on its own
    consecutive bars first, then only bars present in BOTH calendars are paired.
    A holiday, half-day or listing gap in one series would otherwise line a
    stock move up against an index move from a *different* day and silently
    corrupt the covariance; the intersection makes that conflation impossible.

    None when the overlap is < 2 paired returns or the index variance is zero
    (a constant index carries no beta, and dividing by it would be undefined)."""
    if lookback is None:
        lookback = CONFIG.rel_strength.beta_lookback
    stock = _close_series(stock_df)
    index = _close_series(index_df)
    if stock is None or index is None or lookback is None or lookback < 2:
        return None
    try:
        sret = stock.sort_index().pct_change().rename("s")
        iret = index.sort_index().pct_change().rename("i")
        # join="inner" → index intersection; dropna removes the leading NaN
        # return and any bar missing from either side after alignment.
        joined = pd.concat([sret, iret], axis=1, join="inner").dropna()
        tail = joined.iloc[-int(lookback):]
        if len(tail) < 2:
            return None
        s = tail["s"].to_numpy(dtype=float)
        i = tail["i"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(i))):
            return None
        var_i = float(i.var())          # population variance (ddof=0)
        if not math.isfinite(var_i) or var_i == 0.0:
            return None
        # covariance with the SAME ddof=0 as var above, so the normalization
        # cancels and beta is exact for a clean linear relationship.
        cov = float(np.mean((s - s.mean()) * (i - i.mean())))
        b = cov / var_i
        return float(b) if math.isfinite(b) else None
    except Exception as e:  # noqa: BLE001 — total by contract
        logger.warning("relative_strength.beta failed (%s) — None", e)
        return None


def rs_rank(stock_ret, peer_rets) -> float | None:
    """Percentile rank (0..100) of stock_ret within a list of peer returns:
    100 × (# peers strictly below stock_ret) / (# usable peers).

    Placeholder for a future universe-relative RS: the caller supplies the peer
    universe's returns measured over the same window. None when there are no
    usable peers or stock_ret is missing / non-finite. Non-finite peers are
    dropped rather than allowed to poison the count."""
    if stock_ret is None or peer_rets is None:
        return None
    try:
        x = float(stock_ret)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    peers: list[float] = []
    for p in peer_rets:
        try:
            f = float(p)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            peers.append(f)
    if not peers:
        return None
    below = sum(1 for p in peers if p < x)
    return below / len(peers) * 100.0


def _window_keys(windows) -> list[str]:
    """Ordered value keys for the given windows — the single source of truth so
    the all-None fallback and the populated path can never disagree on schema
    (a drifting key set would break feature-log column alignment downstream)."""
    keys: list[str] = []
    for w in windows:
        w = int(w)
        keys += [f"rs_{w}", f"stock_ret_{w}", f"index_ret_{w}"]
    keys += ["beta", "rs_rank"]
    return keys


def evaluate(stock_df, index_df=None, peers=None, cfg=None) -> EngineResult:
    """EngineResult adapter — flat, JSON-safe RS context for one symbol.

    Emits per-window rs / stock_ret / index_ret, the beta estimate, and (when
    peers are supplied) the peer-relative rs_rank of the stock's PRIMARY-window
    return — the first configured window, since a single peers list can only be
    paired with one lookback and the shortest is the most responsive momentum
    read. Every value is a scalar float or None.

    Totality: a missing/thin stock frame ⇒ all values None + a diagnostic; a
    missing/invalid index ⇒ the RS and index-return legs are None + a
    'no index provided' diagnostic naming the expected alias. Never raises —
    any internal failure collapses to the all-None vector with a note."""
    cfg = cfg or CONFIG.rel_strength
    windows = tuple(int(w) for w in cfg.windows)
    values: dict = {k: None for k in _window_keys(windows)}
    diags: list[str] = []
    try:
        stock = _close_series(stock_df)
        if stock is None or len(stock) == 0:
            diags.append(f"{_ENGINE}: no/invalid stock frame — all values None")
            return EngineResult(engine=_ENGINE, values=values, diagnostics=diags)

        index = _close_series(index_df)
        if index is None or len(index) == 0:
            diags.append(f"{_ENGINE}: no index provided "
                         f"(expected alias={cfg.index_alias!r}) — RS / index "
                         f"returns None")

        for k, v in relative_strength(stock_df, index_df, windows).items():
            values[k] = _clean(v)

        values["beta"] = _clean(beta(stock_df, index_df, cfg.beta_lookback))

        if peers and windows:
            primary = windows[0]        # rank the stock's shortest-window return
            values["rs_rank"] = _clean(
                rs_rank(values.get(f"stock_ret_{primary}"), peers))

        if all(v is None for v in values.values()):
            diags.append(f"{_ENGINE}: insufficient history — all values None")
    except Exception as e:  # noqa: BLE001 — total by contract
        logger.warning("%s.evaluate failed (%s) — all None", _ENGINE, e)
        values = {k: None for k in _window_keys(windows)}
        diags.append(f"{_ENGINE}: computation error — all values None")
    return EngineResult(engine=_ENGINE, values=values, diagnostics=diags)
