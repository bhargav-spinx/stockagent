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

NOT scored (context only, shown in the card): Supertrend, market-index trend,
and 52-week proximity. Two spec inputs have no data source and are stubbed:
delivery % and news/earnings (see _delivery_pct / _news_trigger).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pandas as pd

from scanner_indicators import (
    vwap, ema, supertrend,
    localize_ist, split_sessions, orb_levels, volume_ratio, trade_levels,
)

# --- Rating thresholds -------------------------------------------------------
STRONG = 80
WATCH = 60

# --- ORB windows (5-min candles): 15-min = 3, 30-min = 6 ---------------------
ORB_WINDOWS = {15: 3, 30: 6}


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
    bullish_prob: int
    bearish_prob: int
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
def _gap_pct(df: pd.DataFrame, today_df: pd.DataFrame, priors: list) -> float | None:
    if not priors:
        return None
    prev_close = float(priors[-1]["Close"].iloc[-1])
    today_open = float(today_df["Open"].iloc[0])
    if prev_close == 0:
        return None
    return (today_open - prev_close) / prev_close * 100


def _rvol(today_df: pd.DataFrame, priors: list) -> float | None:
    """Time-of-day matched relative volume: today's cumulative volume so far
    vs the average cumulative volume of prior days up to the same candle count."""
    if not priors:
        return None
    n = len(today_df)
    today_cum = float(today_df["Volume"].iloc[:n].sum())
    prior_cums = [float(p["Volume"].iloc[:n].sum()) for p in priors if len(p) >= 1]
    prior_cums = [c for c in prior_cums if c > 0]
    if not prior_cums:
        return None
    avg = statistics.mean(prior_cums)
    return today_cum / avg if avg > 0 else None


def _orb_breakout(today_df: pd.DataFrame, price: float):
    """Evaluate both ORB windows; return the stronger confirmed breakout as
    (window_minutes, direction, strength_pct) or (None, 'none', 0.0)."""
    best = (None, "none", 0.0)
    for minutes, n in ORB_WINDOWS.items():
        if len(today_df) <= n:          # need candles beyond the ORB to break out
            continue
        hi, lo = orb_levels(today_df, n)
        if price > hi:
            strength = (price - hi) / hi * 100
            if strength > best[2]:
                best = (minutes, "long", strength)
        elif price < lo:
            strength = (lo - price) / lo * 100
            if strength > best[2]:
                best = (minutes, "short", strength)
    return best


def _dir_prob(score: int) -> int:
    """Map score → probability in the trade direction (heuristic, monotonic).
    Calibrated so 88 → ~82, matching the spec's example."""
    return int(round(min(90, max(50, 45 + score * 0.42))))


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
                daily_df: pd.DataFrame | None = None) -> ScoreCard:
    """Score one stock from its 5-min DataFrame (period≈5d). `idx_trend` is an
    optional dict from index_trend(); `daily_df` (1y daily) enables the 52-week
    proximity note. Always returns a ScoreCard (rating 'Avoid' on thin data)."""
    df = localize_ist(df)
    price = float(df["Close"].iloc[-1])
    today_df, priors = split_sessions(df)

    def _empty(reason: str) -> ScoreCard:
        return ScoreCard(
            symbol=symbol, price=price, direction="none", score=0,
            rating="Avoid", breakdown={}, signals={},
            entry=None, stop_loss=None, target1=None, target2=None,
            bullish_prob=50, bearish_prob=50, gap_pct=None, rvol=None,
            orb_window=None, notes=[reason],
        )

    if len(today_df) < 4 or not priors:
        return _empty("Insufficient intraday data (need ≥2 sessions, 4+ candles today)")

    # --- Raw measurements ---
    gap = _gap_pct(df, today_df, priors)
    rvol = _rvol(today_df, priors)
    vw = float(vwap(df).iloc[-1])
    e20 = float(ema(df["Close"], 20).iloc[-1])
    e50 = float(ema(df["Close"], 50).iloc[-1])
    orb_win, orb_dir, orb_strength = _orb_breakout(today_df, price)
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

    # Gap (20)
    fav_gap = (gap if long else -gap) if gap is not None else 0.0
    if fav_gap >= 2:
        gp = 20
    elif fav_gap >= 1.5:
        gp = 15
    elif fav_gap >= 1.0:
        gp = 10
    elif fav_gap >= 0.5:
        gp = 5
    else:
        gp = 0
    breakdown["Gap Up/Down"] = gp
    signals["Gap"] = (f"{'Up' if (gap or 0) > 0 else 'Down'} {abs(gap):.2f}%"
                      if gap is not None else "n/a")

    # Relative Volume (25)
    if rvol is None:
        rp = 0
    elif rvol >= 2:
        rp = 25
    elif rvol >= 1.5:
        rp = 18
    elif rvol >= 1.2:
        rp = 10
    else:
        rp = 0
    breakdown["Relative Volume"] = rp
    signals["Volume Spike"] = (f"{rvol:.2f}× avg" if rvol is not None else "n/a")

    # VWAP Confirmation (15)
    vwap_ok = (price > vw) if long else (price < vw)
    breakdown["VWAP Confirmation"] = 15 if vwap_ok else 0
    signals["VWAP"] = "Bullish" if price > vw else "Bearish"

    # EMA Trend 20>50 (15)
    ema_ok = (e20 > e50) if long else (e20 < e50)
    breakdown["EMA Trend"] = 15 if ema_ok else 0
    signals["EMA Trend"] = "Bullish" if e20 > e50 else "Bearish"

    # ORB Breakout (15)
    if orb_dir == direction:
        op = 15 if orb_strength >= 0.10 else 8
        signals["ORB Breakout"] = f"Confirmed ({orb_win}m, +{orb_strength:.2f}%)"
    else:
        op = 0
        signals["ORB Breakout"] = "None"
    breakdown["ORB Breakout"] = op

    # Volume Breakout (10)
    if volbk is None:
        vbp = 0
    elif volbk >= 1.5:
        vbp = 10
    elif volbk >= 1.2:
        vbp = 5
    else:
        vbp = 0
    breakdown["Volume Breakout"] = vbp
    signals["Volume Breakout"] = (f"{volbk:.2f}× last candle"
                                  if volbk is not None else "n/a")

    score = sum(breakdown.values())
    rating = ("Strong " + ("Buy" if long else "Sell")) if score >= STRONG else (
        "Watchlist" if score >= WATCH else "Avoid")

    # --- Entry / SL / targets (canonical model, shared with scanner_setups) ---
    entry = price
    sl, t1, t2 = trade_levels(entry, direction, df)

    dp = _dir_prob(score)
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

    # Market-index regime gate: when every fetched index trends AGAINST the
    # trade, the setup is counter-trend. regime_ok=False lets the alert layer
    # suppress it (the score itself is unchanged — only the firing gate cares).
    regime_ok = True
    idx_trend = idx_trend or {}
    if idx_trend:
        context.append("Index: " + ", ".join(f"{k} {v}" for k, v in idx_trend.items()))
        want = "bullish" if long else "bearish"
        if all(v != want for v in idx_trend.values()):
            regime_ok = False
            notes.append(f"Market index not {want} — counter-trend setup")

    if daily_df is not None and len(daily_df) >= 30:
        hi52 = float(daily_df["High"].tail(252).max())
        lo52 = float(daily_df["Low"].tail(252).min())
        if hi52 > 0 and (hi52 - price) / hi52 <= 0.03:
            context.append("Near 52-week high")
        elif lo52 > 0 and (price - lo52) / lo52 <= 0.03:
            context.append("Near 52-week low")

    # Delivery % (cached daily NSE bhavcopy) — accumulation vs intraday churn.
    deliv = _delivery_pct(symbol)
    if deliv is not None:
        context.append(f"Delivery: {deliv:.0f}%")
        if deliv >= 60:
            context.append("strong delivery")
        elif deliv < 25:
            notes.append("Low delivery % — intraday churn, weak conviction")

    # Earnings proximity (cached daily NSE calendar) — event risk. A positive
    # hit within 2 days flips event_ok so the alert layer can suppress the trade.
    event_ok = True
    dte = _days_to_earnings(symbol)
    if dte is not None and 0 <= dte <= 2:
        event_ok = False
        when = "today" if dte == 0 else f"in {dte}d"
        notes.append(f"Earnings {when} — event risk")

    return ScoreCard(
        symbol=symbol, price=price, direction=direction, score=score,
        rating=rating, breakdown=breakdown, signals=signals,
        entry=entry, stop_loss=sl, target1=t1, target2=t2,
        bullish_prob=bullish, bearish_prob=bearish,
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
        return (f"{emoji} *{c.symbol}*  ·  ₹{c.price:,.2f}\n"
                f"*Score: {c.score}/100 — {c.rating}*\n"
                + ("\n".join(f"_{n}_" for n in c.notes) if c.notes else ""))

    lines = [
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
        f"Bullish Probability: {c.bullish_prob}%",
        f"Bearish Probability: {c.bearish_prob}%",
    ]
    if c.context:
        lines += ["", "_" + "  ·  ".join(c.context) + "_"]
    if c.notes:
        lines += ["⚠️ " + "; ".join(c.notes)]
    return "\n".join(lines)
