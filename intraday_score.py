"""
Intraday scoring engine — the 100-point price / volume / momentum / structure
score from the intraday trading spec.

Component                 Max pts
-------------------------------
Gap Up/Down                  20
Relative Volume              25
VWAP Confirmation            15
EMA Trend (20 > 50)          15
ORB Breakout                 15
Volume Breakout              10
-------------------------------
Total                       100

Rating:  80–100 → Strong Buy/Sell candidate
         60–79  → Watchlist
         < 60   → Avoid

Direction (long / short) is inferred from the alignment of gap, VWAP, EMA and
the ORB breakout. The same machinery scores both sides; bullish vs bearish
probability is derived from the directional score.

NOT scored (context/risk only, shown in the card): Supertrend, market-index
trend (regime_ok gate), 52-week proximity, delivery % and earnings proximity
(event_ok gate) — the last two via market_context (NSE, cached daily). Live
news (Marketaux) enriches only fired alerts in the bot layer, not this scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import CONFIG
from constants import DISCLAIMER, trade_type_tag
from engines import gap as gap_engine
from engines import orb as orb_engine
from engines import volume as volume_engine
from engines import vwap as vwap_engine
from scanner_indicators import (
    vwap, ema, supertrend,
    localize_ist, split_sessions, volume_ratio, trade_levels,
)

# --- Rating thresholds (config.ScoreConfig; names kept for importers) --------
STRONG = CONFIG.score.strong
WATCH = CONFIG.score.watch

# --- ORB windows (5-min candles): 15-min = 3, 30-min = 6 ---------------------
ORB_WINDOWS = dict(CONFIG.score.orb_windows)

# Beyond this % past the ORB level the move is treated as SPENT, not breaking
# out: a stock 3–5% past its opening range at 2 PM is a chase, and entry would
# be the extended current price with an ATR stop hung off it. No points.
ORB_MAX_EXTENSION_PCT = CONFIG.score.orb_max_extension_pct

# Phase 2.1: the component measurements + point rules live in engines/
# (gap / volume / vwap / orb). This module composes them into the legacy
# 100-point card — tests/test_golden_parity.py pins the composition as
# byte-identical to the pre-extraction scorer.


@dataclass
class ScoreCard:
    symbol: str
    price: float
    direction: str                 # "long" | "short" | "none"
    score: int
    rating: str                    # "Strong Buy"/"Strong Sell"/"Watchlist"/"Avoid"
    breakdown: dict                # component -> points
    signals: dict                  # component -> human-readable status
    entry: float | None
    stop_loss: float | None
    target1: float | None
    target2: float | None
    bull_conviction: int   # heuristic 50–90 lean from the score — NOT an empirical probability
    bear_conviction: int
    gap_pct: float | None
    rvol: float | None
    orb_window: int | None         # which window (15/30) was used
    regime_ok: bool = True         # False = trade opposes the market-index regime
    event_ok: bool = True          # False = earnings/results within 2 days (event risk)
    delivery_pct: float | None = None
    context: list[str] = field(default_factory=list)   # supertrend, index, 52w
    notes: list[str] = field(default_factory=list)      # warnings / why-avoid


# ----------------------------------------------------------------------------
# External context inputs — wired to market_context (NSE delivery %, NSE results
# calendar). Both are cached once/day there, so per-symbol use here is cheap.
# Any failure returns None ('unknown') and the scorer ignores it — never breaks.
# (Live news is per-symbol + rate-limited, so it enriches only fired alerts in
# the bot layer, NOT this per-universe scorer.)
# ----------------------------------------------------------------------------
def _delivery_pct(symbol: str):
    """Delivery % from NSE's EOD bhavcopy (cached daily). None = unknown."""
    try:
        import market_context
        return market_context.delivery_pct(symbol)
    except Exception:
        return None


def _days_to_earnings(symbol: str):
    """Calendar days to the next scheduled results date (cached daily).
    0 = today, None = none scheduled / unknown."""
    try:
        import market_context
        return market_context.days_to_earnings(symbol)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _dir_conviction(score: int) -> int:
    """Map score → a 50–90 'conviction' lean in the trade direction.

    This is a monotonic relabel of the score for display only. It is NOT a
    probability and is NOT calibrated against realized outcomes — do not present
    it as a win likelihood. The honest measure of how often a score bucket wins
    is the realized win rate in stats.py (available once trades have resolved)."""
    cs = CONFIG.score
    return int(round(min(cs.conviction_hi,
                         max(cs.conviction_lo,
                             cs.conviction_base + score * cs.conviction_slope))))


# ----------------------------------------------------------------------------
# Market-index trend (context; fetched once and passed into bulk scoring)
# ----------------------------------------------------------------------------
def index_trend(aliases=("NIFTY", "BANKNIFTY")) -> dict:
    """Classify each index as bullish/bearish/flat from price vs VWAP & EMA20
    on 5-min candles. Returns {alias: 'bullish'|'bearish'|'flat'}. Best-effort:
    an index that fails to fetch is omitted."""
    from data_provider import fetch_data

    out = {}
    for alias in aliases:
        try:
            df = fetch_data(alias, period="5d", interval="5m")
            from analyzer import drop_forming_candle
            df = drop_forming_candle(df, "5m")   # live: ignore unclosed candle
            today, _ = split_sessions(df)
            if len(today) < 3:
                continue
            price = float(today["Close"].iloc[-1])
            vw = float(vwap(df).iloc[-1])
            e20 = float(ema(df["Close"], 20).iloc[-1])
            if price > vw and price > e20:
                out[alias] = "bullish"
            elif price < vw and price < e20:
                out[alias] = "bearish"
            else:
                out[alias] = "flat"
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------------
# Core scorer
# ----------------------------------------------------------------------------
def score_stock(df: pd.DataFrame, symbol: str,
                idx_trend: dict | None = None,
                daily_df: pd.DataFrame | None = None,
                skip_external: bool = False) -> ScoreCard:
    """Score one stock from its 5-min DataFrame (period≈5d). `idx_trend` is an
    optional dict from index_trend(); `daily_df` (1y daily) enables the 52-week
    proximity note. `skip_external=True` skips the live NSE delivery %/earnings
    lookups (keeps backtests hermetic/offline). Always returns a ScoreCard
    (rating 'Avoid' on thin data)."""
    df = localize_ist(df)
    price = float(df["Close"].iloc[-1])
    today_df, priors = split_sessions(df)

    def _empty(reason: str) -> ScoreCard:
        return ScoreCard(
            symbol=symbol, price=price, direction="none", score=0,
            rating="Avoid", breakdown={}, signals={},
            entry=None, stop_loss=None, target1=None, target2=None,
            bull_conviction=50, bear_conviction=50, gap_pct=None, rvol=None,
            orb_window=None, notes=[reason],
        )

    if len(today_df) < 4 or not priors:
        return _empty("Insufficient intraday data (need ≥2 sessions, 4+ candles today)")

    # --- Raw measurements (component engines) ---
    gap = gap_engine.gap_pct(today_df, priors)
    rvol = volume_engine.rvol(today_df, priors)
    vw = float(vwap(df).iloc[-1])
    e20 = float(ema(df["Close"], 20).iloc[-1])
    e50 = float(ema(df["Close"], 50).iloc[-1])
    orb_win, orb_dir, orb_strength = orb_engine.orb_breakout(today_df, price)
    volbk = volume_ratio(df)

    # --- Direction vote: gap / VWAP / EMA / ORB ---
    votes = 0
    if gap is not None:
        votes += 1 if gap > 0 else -1
    votes += 1 if price > vw else -1
    votes += 1 if e20 > e50 else -1
    if orb_dir == "long":
        votes += 1
    elif orb_dir == "short":
        votes -= 1

    if votes > 0:
        direction = "long"
    elif votes < 0:
        direction = "short"
    else:
        card = _empty("Sideways / conflicting signals — no entry")
        card.gap_pct, card.rvol = gap, rvol
        return card

    long = direction == "long"

    # --- Component scoring (favourable = aligned with `direction`) ---
    breakdown, signals = {}, {}

    cs = CONFIG.score

    # Gap (20)
    breakdown["Gap Up/Down"] = gap_engine.gap_points(gap, long, cs)
    signals["Gap"] = (f"{'Up' if (gap or 0) > 0 else 'Down'} {abs(gap):.2f}%"
                      if gap is not None else "n/a")

    # Relative Volume (25)
    breakdown["Relative Volume"] = volume_engine.rvol_points(rvol, cs)
    signals["Volume Spike"] = (f"{rvol:.2f}× avg" if rvol is not None else "n/a")

    # VWAP Confirmation (15)
    breakdown["VWAP Confirmation"] = vwap_engine.confirmation_points(
        price, vw, long, cs)
    signals["VWAP"] = "Bullish" if price > vw else "Bearish"

    # EMA Trend 20>50 (15)
    ema_ok = (e20 > e50) if long else (e20 < e50)
    breakdown["EMA Trend"] = cs.ema_points if ema_ok else 0
    signals["EMA Trend"] = "Bullish" if e20 > e50 else "Bearish"

    # ORB Breakout (15)
    breakdown["ORB Breakout"] = orb_engine.orb_points(
        orb_dir, direction, orb_strength, cs)
    signals["ORB Breakout"] = (
        f"Confirmed ({orb_win}m, +{orb_strength:.2f}%)"
        if orb_dir == direction else "None")

    # Volume Breakout (10)
    breakdown["Volume Breakout"] = volume_engine.volume_breakout_points(volbk, cs)
    signals["Volume Breakout"] = (f"{volbk:.2f}× last candle"
                                  if volbk is not None else "n/a")

    score = sum(breakdown.values())
    rating = ("Strong " + ("Buy" if long else "Sell")) if score >= STRONG else (
        "Watchlist" if score >= WATCH else "Avoid")

    # --- Entry / SL / targets (canonical model, shared with scanner_setups) ---
    entry = price
    sl, t1, t2 = trade_levels(entry, direction, df)

    dp = _dir_conviction(score)
    bullish = dp if long else 100 - dp
    bearish = 100 - bullish

    # --- Context (not scored) ---
    context, notes = [], []
    try:
        st_dir = int(supertrend(df)[1].iloc[-1])
        st_word = "bullish" if st_dir == 1 else "bearish"
        context.append(f"Supertrend: {st_word}")
        if (st_dir == 1) != long:
            notes.append("Supertrend disagrees with trade direction")
    except Exception:
        pass

    # Market-index regime gate: block ONLY when every fetched index trends
    # OUTRIGHT AGAINST the trade (true counter-trend). A flat/mixed regime is
    # allowed with a note — flat is "no support", not "opposed", and blocking
    # it would silently suppress every signal in sideways sessions (which is
    # not what "no counter-trend" means). regime_ok=False lets the alert layer
    # suppress; the score itself is unchanged. When idx_trend is empty (index
    # fetch failed), the gate is EXPLICITLY fail-open: annotated here, logged
    # by the autoscan layer.
    regime_ok = True
    idx_trend = idx_trend or {}
    if idx_trend:
        context.append("Index: " + ", ".join(f"{k} {v}" for k, v in idx_trend.items()))
        want = "bullish" if long else "bearish"
        opposite = "bearish" if long else "bullish"
        if all(v == opposite for v in idx_trend.values()):
            regime_ok = False
            notes.append(f"All indices {opposite} — counter-trend setup")
        elif all(v != want for v in idx_trend.values()):
            notes.append("Indices flat/mixed — no regime support")
    else:
        notes.append("Index regime unavailable — regime gate inactive")

    if daily_df is not None and len(daily_df) >= 30:
        hi52 = float(daily_df["High"].tail(cs.w52_lookback_days).max())
        lo52 = float(daily_df["Low"].tail(cs.w52_lookback_days).min())
        if hi52 > 0 and (hi52 - price) / hi52 <= cs.near_52w_band:
            context.append("Near 52-week high")
        elif lo52 > 0 and (price - lo52) / lo52 <= cs.near_52w_band:
            context.append("Near 52-week low")

    # External context (live NSE) — skipped in backtests to stay offline.
    deliv = None
    event_ok = True
    if not skip_external:
        # Delivery % (cached daily NSE bhavcopy) — accumulation vs intraday churn.
        deliv = _delivery_pct(symbol)
        if deliv is not None:
            context.append(f"Delivery: {deliv:.0f}%")
            if deliv >= cs.delivery_strong_pct:
                context.append("strong delivery")
            elif deliv < cs.delivery_low_pct:
                notes.append("Low delivery % — intraday churn, weak conviction")

        # Earnings proximity (cached daily NSE calendar) — event risk. A hit
        # within 2 days flips event_ok so the alert layer can suppress the trade.
        # DELIBERATELY fail-open when the calendar itself didn't load (blocking
        # every alert whenever NSE's flaky API is down is worse) — but say so,
        # instead of silently pretending the symbol is event-clear.
        dte = _days_to_earnings(symbol)
        if dte is not None and 0 <= dte <= cs.earnings_risk_days:
            event_ok = False
            when = "today" if dte == 0 else f"in {dte}d"
            notes.append(f"Earnings {when} — event risk")
        elif dte is None:
            try:
                import market_context
                if not market_context.earnings_data_available():
                    notes.append("Earnings calendar unavailable — event risk unchecked")
            except Exception:
                pass

    return ScoreCard(
        symbol=symbol, price=price, direction=direction, score=score,
        rating=rating, breakdown=breakdown, signals=signals,
        entry=entry, stop_loss=sl, target1=t1, target2=t2,
        bull_conviction=bullish, bear_conviction=bearish,
        gap_pct=gap, rvol=rvol, orb_window=orb_win,
        regime_ok=regime_ok, event_ok=event_ok, delivery_pct=deliv,
        context=context, notes=notes,
    )


# ----------------------------------------------------------------------------
# Telegram formatter — mirrors the spec's Output Example
# ----------------------------------------------------------------------------
def format_scorecard(c: ScoreCard) -> str:
    emoji = {"Strong Buy": "🟢", "Strong Sell": "🔴",
             "Watchlist": "🟡", "Avoid": "⚪️"}.get(c.rating, "⚪️")

    if c.direction == "none":
        return (f"{trade_type_tag('intraday')}\n"
                f"{emoji} *{c.symbol}*  ·  ₹{c.price:,.2f}\n"
                f"*Score: {c.score}/100 — {c.rating}*\n"
                + ("\n".join(f"_{n}_" for n in c.notes) if c.notes else ""))

    lines = [
        trade_type_tag("intraday"),
        f"{emoji} *{c.symbol}*  ·  ₹{c.price:,.2f}",
        f"*Score: {c.score}/100 — {c.rating}*",
        "",
        f"Gap: {c.signals.get('Gap', 'n/a')}",
        f"Volume Spike: {c.signals.get('Volume Spike', 'n/a')}",
        f"VWAP: {c.signals.get('VWAP')}",
        f"EMA Trend: {c.signals.get('EMA Trend')}",
        f"ORB Breakout: {c.signals.get('ORB Breakout')}",
        "",
        f"📊 *Breakdown*",
    ]
    lines += [f"  • {k}: {v} pts" for k, v in c.breakdown.items()]
    lines += [
        "",
        f"🎯 Entry: ₹{c.entry:,.2f}",
        f"🛑 Stop Loss: ₹{c.stop_loss:,.2f}",
        f"🥇 Target 1: ₹{c.target1:,.2f}",
        f"🥈 Target 2: ₹{c.target2:,.2f}",
        "",
        f"Directional lean: {'Long' if c.direction == 'long' else 'Short'} "
        f"({max(c.bull_conviction, c.bear_conviction)}/100 conviction)",
        "_Conviction is a heuristic from the score, not a win probability._",
    ]
    if c.context:
        lines += ["", "_" + "  ·  ".join(c.context) + "_"]
    if c.notes:
        lines += ["⚠️ " + "; ".join(c.notes)]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
