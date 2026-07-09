"""
End-of-day report.

For every alert logged today (and recent open swing alerts), simulate the
hypothetical outcome assuming the user took the trade at the published
Entry/SL/T1/T2 with the default 50/50 partial-exit rules from STRATEGY.md §7.

Outcome categories (P&L is blended 50/50 from the ACTUAL entry/T1/T2 levels,
never a fixed percentage — the targets are ATR-sized, so their distance varies
per stock; see scanner_indicators.trade_levels):
    t2_hit              — T2 reached → blended 50% at T1 + 50% at T2
    t1_then_squareoff   — T1 hit then session ended → blended 50% at T1 + 50% at last close
    t1_then_breakeven   — T1 hit, then trailing-SL at entry → blended 50% at T1 + 50% at entry
    sl_hit              — SL hit before T1 → full SL distance loss
    time_stop           — 45 min elapsed without T1/SL (intraday only) → exit at last close
    squareoff_no_t1     — no T1, no SL, market closed → exit at last close
    open                — swing alerts not yet resolved
    no_data             — could not fetch post-entry candles

P&L is hypothetical and OPTIMISTIC on fills: it assumes the published SL/T1/T2
fill at their exact price within the touching candle (no gap-through on stops,
no spread, no partial-fill on the second leg). It is gross of brokerage/STT/
slippage — apply ~0.13%/round-trip per STRATEGY.md §8 to estimate net, and treat
real outcomes as somewhat worse than these figures.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time, date, timedelta
from typing import Any

import pandas as pd
import pytz

import subscriptions
from config import CONFIG
from data_provider import fetch_data
from constants import IST

logger = logging.getLogger(__name__)

# STRATEGY.md §8 cost model: ~0.05% brokerage/STT per side + ~0.05% slippage
# entry + ~0.03% slippage exit ≈ 0.13% round-trip on tier-1 NSE stocks.
# Single source of truth — imported by backtest.py and bot.py.
# Value lives in config.CostConfig; the module-level name is kept for importers.
COST_PER_TRADE_PCT = CONFIG.costs.cost_per_trade_pct

_TIME_STOP = timedelta(minutes=CONFIG.costs.intraday_time_stop_min)

INTRADAY_CATEGORIES = {"scan", "manual_intraday"}
# Channel tips are analysed on daily candles (swing horizon), so they resolve
# and report through the swing machinery.
#   channel_tip  — OUR analysis of the tip (the bot's own levels)
#   channel_call — the CHANNEL's own stated entry/target/SL (so the channel,
#                  not the bot, gets an accountable scorecard)
SWING_CATEGORIES = {"swing_auto", "manual_swing", "channel_tip", "channel_call"}

PASS_STATUSES = {"t2_hit", "t1_then_squareoff", "t1_then_breakeven"}
FAIL_STATUSES = {"sl_hit"}
NEUTRAL_STATUSES = {"time_stop", "squareoff_no_t1", "open", "no_data"}


def effective_cost_pct(entry: float | None = None,
                       exit_price: float | None = None,
                       category: str | None = None) -> float:
    """Round-trip cost % for one resolved trade.

    Default (CONFIG.costs.model == "flat") returns the historical flat
    COST_PER_TRADE_PCT — output byte-identical to before. model="statutory"
    itemizes real charges via engines/execution, using DELIVERY rates for
    swing-horizon categories (multi-day holds pay 0.1% STT both sides +
    higher stamp duty, which the intraday-calibrated flat 0.13% understates).
    Falls back to flat on any error — a cost model must never break a report."""
    if CONFIG.costs.model == "flat" or not entry:
        return COST_PER_TRADE_PCT
    try:
        from engines.execution import round_trip_cost_pct
        product = "delivery" if category in SWING_CATEGORIES else "intraday"
        return round_trip_cost_pct(entry, exit_price, product=product)
    except Exception as e:
        logger.warning("statutory cost model failed (%s) — using flat", e)
        return COST_PER_TRADE_PCT


# ---------- outcome resolver ----------

def _parse_iso(ts: str) -> datetime:
    """Parse our stored ISO timestamps (always UTC, naive in DB) into IST-aware."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(IST)


def _signed_pct(entry: float, exit_price: float, direction: str) -> float:
    if direction == "long":
        return (exit_price - entry) / entry * 100
    return (entry - exit_price) / entry * 100


def _blended_pct(entry: float, leg1_exit: float, leg2_exit: float,
                 direction: str) -> float:
    """Blended P&L for the 50/50 partial-exit model: half the position exits at
    `leg1_exit` (T1), half at `leg2_exit` (T2 / trailed-SL / squareoff close).
    BOTH legs are priced from the actual levels — never hardcoded — so the
    result tracks the configured ATR-sized targets instead of an obsolete
    fixed-percent assumption."""
    return (0.5 * _signed_pct(entry, leg1_exit, direction)
            + 0.5 * _signed_pct(entry, leg2_exit, direction))


def resolve_intraday(alert: dict[str, Any],
                     df: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """
    Resolve an intraday alert by checking 5-min candles after generation.

    If `df` is provided, use it (backtest mode — pre-fetched history slice).
    Otherwise fetch the last 5 days from data_provider (live EOD mode).
    """
    symbol = alert["symbol"]
    entry = alert["entry"]
    sl = alert["stop_loss"]
    t1 = alert["target1"]
    t2 = alert["target2"]
    direction = alert["direction"]
    gen_time = _parse_iso(alert["generated_at"]) if isinstance(alert["generated_at"], str) else alert["generated_at"]

    if df is None:
        try:
            df = fetch_data(symbol, period="5d", interval="5m")
        except Exception as e:
            logger.warning("resolve_intraday %s: data fetch failed: %s", symbol, e)
            return {"status": "no_data"}

    if df.index.tz is None:
        df = df.tz_localize(IST)
    elif str(df.index.tz) != str(IST):
        df = df.tz_convert(IST)

    # Evaluate from the candle that was FORMING at alert time (index floored
    # to the 5-min grid). `index > gen_time` would skip it entirely — up to
    # 5 minutes of post-entry SL/T1 action invisible to the resolver. The
    # trigger candle (one grid slot earlier) stays excluded. Slightly
    # optimistic on fills within that first candle — consistent with the
    # module-level fill assumptions.
    candle_start = gen_time.replace(second=0, microsecond=0)
    candle_start -= timedelta(minutes=candle_start.minute % 5)
    post = df[df.index >= candle_start]
    # Restrict to today's session
    post = post[post.index.date == gen_time.date()]
    if len(post) == 0:
        return {"status": "open"}

    t1_hit = False
    t1_hit_time = None

    # Excursion telemetry: best favorable (≥0) / worst adverse (≤0) % reached
    # while the trade was live — candle-extreme granularity, exit candle
    # included. MFE/MAE + duration make each scarce outcome several data
    # points (stop placement, target reach, time-stop calibration).
    mfe = 0.0
    mae = 0.0

    def _outcome(status: str, exit_price: float, ts, pnl: float) -> dict:
        return {
            "status": status,
            "exit_price": exit_price,
            "exit_time": ts,
            "pnl_pct": round(pnl, 2),
            "mfe_pct": round(mfe, 2),
            "mae_pct": round(mae, 2),
            "duration_min": round((ts - gen_time).total_seconds() / 60.0, 1),
        }

    for ts, row in post.iterrows():
        fav = float(row["High"] if direction == "long" else row["Low"])
        adv = float(row["Low"] if direction == "long" else row["High"])
        mfe = max(mfe, _signed_pct(entry, fav, direction))
        mae = min(mae, _signed_pct(entry, adv, direction))

        if direction == "long":
            sl_in = row["Low"] <= sl
            t1_in = row["High"] >= t1
            t2_in = row["High"] >= t2
        else:
            sl_in = row["High"] >= sl
            t1_in = row["Low"] <= t1
            t2_in = row["Low"] <= t2

        if not t1_hit:
            # Conservative: if SL and T1 both touched in same candle, assume SL first
            if sl_in:
                return _outcome("sl_hit", sl, ts, _signed_pct(entry, sl, direction))
            # Time-stop: 45 min elapsed with neither SL nor T1. Checked AFTER
            # SL (a stop breach inside the boundary candle must count as a
            # stop, not a time-stop exit at that candle's close) and BEFORE
            # T1 grants continuation.
            if not t1_in and (ts - gen_time) > _TIME_STOP:
                close = float(row["Close"])
                return _outcome("time_stop", close, ts,
                                _signed_pct(entry, close, direction))
            if t1_in:
                t1_hit = True
                t1_hit_time = ts
                # Check t2 in same candle
                if t2_in:
                    # 50% at T1, 50% at T2 — blended from actual levels.
                    return _outcome("t2_hit", t2, ts,
                                    _blended_pct(entry, t1, t2, direction))
        else:
            # After T1: SL is moved to entry per §7
            if direction == "long":
                trail_sl_in = row["Low"] <= entry
                t2_in_now = row["High"] >= t2
            else:
                trail_sl_in = row["High"] >= entry
                t2_in_now = row["Low"] <= t2

            if t2_in_now:
                return _outcome("t2_hit", t2, ts,
                                _blended_pct(entry, t1, t2, direction))
            if trail_sl_in:
                # 50% booked at T1, 50% trailed out at entry (0%).
                return _outcome("t1_then_breakeven", entry, ts,
                                _blended_pct(entry, t1, entry, direction))

    # End of candles: square off at last close
    last_ts = post.index[-1]
    last_close = float(post["Close"].iloc[-1])
    if t1_hit:
        # 50% booked at T1 + 50% squared off at last close — both from actuals.
        return _outcome("t1_then_squareoff", last_close, last_ts,
                        _blended_pct(entry, t1, last_close, direction))
    return _outcome("squareoff_no_t1", last_close, last_ts,
                    _signed_pct(entry, last_close, direction))


def resolve_swing(alert: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a swing alert using daily candles after generation."""
    symbol = alert["symbol"]
    entry = alert["entry"]
    sl = alert["stop_loss"]
    t1 = alert["target1"]
    t2 = alert["target2"]
    direction = alert["direction"]
    gen_time = _parse_iso(alert["generated_at"])

    try:
        df = fetch_data(symbol, period="3mo", interval="1d")
    except Exception as e:
        logger.warning("resolve_swing %s: data fetch failed: %s", symbol, e)
        return {"status": "no_data"}

    if df.index.tz is None:
        df = df.tz_localize(IST)
    elif str(df.index.tz) != str(IST):
        df = df.tz_convert(IST)

    # Daily bars strictly AFTER the alert date
    post = df[df.index.date > gen_time.date()]
    if len(post) == 0:
        return {"status": "open"}

    t1_hit = False

    # Excursion telemetry — daily-bar granularity for swing alerts.
    mfe = 0.0
    mae = 0.0

    def _outcome(status: str, exit_price: float, ts, pnl: float) -> dict:
        return {
            "status": status,
            "exit_price": exit_price,
            "exit_time": ts,
            "pnl_pct": round(pnl, 2),
            "mfe_pct": round(mfe, 2),
            "mae_pct": round(mae, 2),
            "duration_min": round((ts - gen_time).total_seconds() / 60.0, 1),
        }

    for ts, row in post.iterrows():
        fav = float(row["High"] if direction == "long" else row["Low"])
        adv = float(row["Low"] if direction == "long" else row["High"])
        mfe = max(mfe, _signed_pct(entry, fav, direction))
        mae = min(mae, _signed_pct(entry, adv, direction))

        if direction == "long":
            sl_in = row["Low"] <= sl
            t1_in = row["High"] >= t1
            t2_in = row["High"] >= t2
        else:
            sl_in = row["High"] >= sl
            t1_in = row["Low"] <= t1
            t2_in = row["Low"] <= t2

        if not t1_hit:
            if sl_in:
                return _outcome("sl_hit", sl, ts, _signed_pct(entry, sl, direction))
            if t1_in:
                t1_hit = True
                if t2_in:
                    return _outcome("t2_hit", t2, ts,
                                    _blended_pct(entry, t1, t2, direction))
        else:
            if direction == "long":
                trail_sl_in = row["Low"] <= entry
                t2_in_now = row["High"] >= t2
            else:
                trail_sl_in = row["High"] >= entry
                t2_in_now = row["Low"] <= t2
            if t2_in_now:
                return _outcome("t2_hit", t2, ts,
                                _blended_pct(entry, t1, t2, direction))
            if trail_sl_in:
                return _outcome("t1_then_breakeven", entry, ts,
                                _blended_pct(entry, t1, entry, direction))

    return {"status": "open"}


def resolve_alert(alert: dict[str, Any]) -> dict[str, Any] | None:
    cat = alert["category"]
    if cat in INTRADAY_CATEGORIES:
        return resolve_intraday(alert)
    if cat in SWING_CATEGORIES:
        return resolve_swing(alert)
    logger.warning("resolve_alert: unknown category %s", cat)
    return None


def today_intraday_realized_r() -> list[float]:
    """R multiples for TODAY's intraday alerts, resolved against the current
    session so far.

    The daily-risk budget needs this: intraday alerts are otherwise only
    resolved at EOD (16:20 IST), so a gate reading persisted pnl_pct would see
    nothing during market hours and be inert exactly when it must protect. Here
    each still-open intraday alert is resolved IN-MEMORY against today's 5-min
    candles (mark-to-market: a stop hit earlier today books its −R; a runner
    marks at the current close). Nothing is persisted — EOD resolution stays
    the authoritative record. Best-effort: an alert whose fetch fails is
    skipped, not counted."""
    from engines.risk import realized_r      # local import avoids a cycle

    today = subscriptions.get_alerts_for_date()
    rows: list[dict] = []
    for a in today:
        if a["category"] not in INTRADAY_CATEGORIES:
            continue
        if a.get("pnl_pct") is not None:       # already persisted (rare intraday)
            rows.append(a)
            continue
        try:
            outcome = resolve_intraday(a)
        except Exception:
            outcome = None
        if (outcome and outcome.get("pnl_pct") is not None
                and outcome.get("status") not in ("open", "no_data")):
            rows.append({**a, "pnl_pct": outcome["pnl_pct"]})
    return realized_r(rows)


def resolve_pending(max_age_days: int = 30) -> int:
    """Resolve all open alerts up to max_age_days. Returns count of newly resolved."""
    open_alerts = subscriptions.get_open_alerts(max_age_days=max_age_days)
    resolved = 0
    for alert in open_alerts:
        outcome = resolve_alert(alert)
        # "no_data" is a TRANSIENT fetch failure, not a trade outcome. Never
        # persist it: a saved outcome row permanently removes the alert from
        # get_open_alerts, so one rate-limited fetch would silently erase a
        # real win/loss from /stats forever. Leave it open — the next resolver
        # run retries; truly dead symbols age out via max_age_days.
        if outcome is None or outcome.get("status") in ("open", "no_data"):
            continue
        subscriptions.save_outcome(
            alert["id"],
            status=outcome["status"],
            exit_price=outcome.get("exit_price"),
            exit_time=outcome.get("exit_time"),
            pnl_pct=outcome.get("pnl_pct"),
            mfe_pct=outcome.get("mfe_pct"),
            mae_pct=outcome.get("mae_pct"),
            duration_min=outcome.get("duration_min"),
        )
        resolved += 1
    return resolved


# ---------- report formatting ----------

_STATUS_EMOJI = {
    "t2_hit": "✅",
    "t1_then_squareoff": "✅",
    "t1_then_breakeven": "🟡",
    "sl_hit": "❌",
    "time_stop": "⏸",
    "squareoff_no_t1": "⏸",
    "open": "⏳",
    "no_data": "❓",
}

_STATUS_LABEL = {
    "t2_hit": "T2 hit",
    "t1_then_squareoff": "T1+SqOff",
    "t1_then_breakeven": "T1+BE",
    "sl_hit": "SL hit",
    "time_stop": "TimeStop",
    "squareoff_no_t1": "SqOff",
    "open": "Open",
    "no_data": "NoData",
}


def _classify(status: str | None) -> str:
    if status in PASS_STATUSES:
        return "pass"
    if status in FAIL_STATUSES:
        return "fail"
    return "neutral"


def _format_section(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"*{title}*\n_No alerts._\n"

    passed = sum(1 for r in rows if _classify(r.get("status")) == "pass")
    failed = sum(1 for r in rows if _classify(r.get("status")) == "fail")
    neutral = sum(1 for r in rows if _classify(r.get("status")) == "neutral")

    pnls = [r["pnl_pct"] for r in rows if r.get("pnl_pct") is not None]
    total_pnl = sum(pnls) if pnls else 0.0
    avg_pnl = (total_pnl / len(pnls)) if pnls else 0.0

    header = (
        f"*{title}*\n"
        f"Total: {len(rows)} • ✅ {passed} • ❌ {failed} • ⏸ {neutral}\n"
        f"Σ P&L: {total_pnl:+.2f}% • Avg: {avg_pnl:+.2f}% per trade\n\n"
        "```\n"
        f"{'Time':<5}  {'Symbol':<11}  {'Set':<3}  {'Dir':<5}  "
        f"{'Entry':>9}  {'Exit':>9}  {'P&L%':>7}  Status\n"
    )

    lines = []
    for r in rows:
        gen = _parse_iso(r["generated_at"]).strftime("%H:%M")
        sym_short = r["symbol"].replace(".NS", "").replace(".BO", "")[:11]
        setup = (r.get("setup") or "-")[:3]
        direction = r["direction"][:5]
        entry = r["entry"]
        exit_p = r.get("exit_price")
        exit_str = f"{exit_p:>9.2f}" if exit_p is not None else f"{'-':>9}"
        pnl = r.get("pnl_pct")
        pnl_str = f"{pnl:>+7.2f}" if pnl is not None else f"{'-':>7}"
        status = r.get("status") or "open"
        emoji = _STATUS_EMOJI.get(status, "•")
        label = _STATUS_LABEL.get(status, status)
        lines.append(
            f"{gen:<5}  {sym_short:<11}  {setup:<3}  {direction:<5}  "
            f"{entry:>9.2f}  {exit_str}  {pnl_str}  {emoji} {label}"
        )
    return header + "\n".join(lines) + "\n```\n"


def build_report(user_id: int | None = None,
                trade_date_str: str | None = None,
                today_rejection_stats: dict[str, int] | None = None) -> str:
    """
    Build the EOD report text. If user_id given, only their alerts; else all.
    Resolves any pending alerts before formatting.
    If `today_rejection_stats` is provided AND the day has zero alerts, the
    report includes a "why no alerts today?" breakdown.
    """
    resolved_now = resolve_pending(max_age_days=30)
    if resolved_now:
        logger.info("EOD: resolved %d pending alerts", resolved_now)

    # IST trade date, not the server's UTC date (see subscriptions._ist_today).
    trade_date_str = trade_date_str or datetime.now(IST).date().isoformat()
    all_today = subscriptions.get_alerts_for_date(trade_date_str, user_id=user_id)

    scan_rows = [r for r in all_today if r["category"] == "scan"]
    manual_intraday_rows = [r for r in all_today if r["category"] == "manual_intraday"]
    swing_rows = [r for r in all_today if r["category"] in SWING_CATEGORIES]

    cost_note = (
        f"_Gross of brokerage/STT/slippage (~{COST_PER_TRADE_PCT:.2f}% round-trip)._\n\n"
        if CONFIG.costs.model == "flat" else
        "_Gross figures; net columns apply the itemized statutory cost model._\n\n")
    parts = [
        f"📊 *End-of-Day Report* — {trade_date_str}\n",
        "_Hypothetical paper-trade outcomes assuming default 50/50 partial-exit rules._\n",
        cost_note,
        _format_section("🟦 Intraday auto-scan (/scan_alerts)", scan_rows),
        _format_section("🟧 Manual intraday (/intraday)", manual_intraday_rows),
        _format_section("🟪 Swing (/swing_alerts + /swing)", swing_rows),
    ]

    # Hypothetical exposure at configured sizing (Phase 2, default OFF —
    # rendered only when CONFIG.risk.capital is set).
    if CONFIG.risk.capital:
        exposure = _format_exposure_section()
        if exposure:
            parts.append(exposure)

    # If today had zero alerts and we have rejection stats, explain why
    if not all_today and today_rejection_stats:
        parts.append(_format_empty_day_explanation(today_rejection_stats))

    return "\n".join(parts)


def _format_exposure_section() -> str:
    """What exposure WOULD be if every currently-open alert were taken at the
    configured fixed-fractional size. Signals-only platform — this is a
    hypothetical risk view, not a brokerage position report; say so."""
    try:
        from engines.risk import position_size
        open_alerts = subscriptions.get_open_alerts(max_age_days=30)
        rows: list[str] = []
        total = 0.0
        for a in open_alerts:
            sz = position_size(a["entry"], a["stop_loss"])
            v = sz.values
            if not (sz.ok and v.get("qty")):
                continue
            total += v["position_value_inr"]
            sym = a["symbol"].replace(".NS", "").replace(".BO", "")
            rows.append(f"• {sym} {a['direction']} — {v['qty']} sh "
                        f"≈ ₹{v['position_value_inr']:,.0f}")
        cap = CONFIG.risk.capital
        lines = ["*📐 Hypothetical exposure (configured sizing)*"]
        if not rows:
            lines.append("_No open sized positions._")
        else:
            lines += rows
            lines.append(f"Total ≈ ₹{total:,.0f} ({total / cap * 100:.0f}% of "
                         f"capital) · position limit "
                         f"{CONFIG.risk.max_concurrent_positions}")
            if len(rows) > CONFIG.risk.max_concurrent_positions:
                lines.append("⚠️ Open alerts exceed max_concurrent_positions — "
                             "taking all of them would breach the limit.")
        lines.append("_Hypothetical: sizes assume every open alert was taken._")
        return "\n".join(lines) + "\n"
    except Exception as e:
        logger.warning("exposure section failed: %s", e)
        return ""


# ---------- swing-completion notification ----------

# Statuses at which a swing trade is considered finished (vs. still "open").
SWING_TERMINAL_STATUSES = tuple(PASS_STATUSES | FAIL_STATUSES)

_SWING_HEADLINE = {
    "t2_hit": "🎯 Target hit — full move captured",
    "t1_then_squareoff": "🎯 Target 1 hit, rest squared off",
    "t1_then_breakeven": "🟡 Target 1 hit, trailed out at breakeven",
    "sl_hit": "🛑 Stop-loss hit",
}


def _hold_days(rec: dict[str, Any]) -> int | None:
    """Calendar days a swing trade was held: recommendation date → exit date."""
    exit_t = rec.get("exit_time")
    if not (isinstance(exit_t, str) and exit_t):
        return None
    try:
        return (_parse_iso(exit_t).date() - _parse_iso(rec["generated_at"]).date()).days
    except Exception:
        return None


def swing_record_summary(records: list[dict[str, Any]]) -> str:
    """One-line running performance record from resolved swing alerts."""
    resolved = [r for r in records if r.get("pnl_pct") is not None]
    if not resolved:
        return ""
    wins = sum(1 for r in resolved if _classify(r.get("status")) == "pass")
    losses = sum(1 for r in resolved if _classify(r.get("status")) == "fail")
    neutral = len(resolved) - wins - losses
    gross = sum(r["pnl_pct"] for r in resolved)
    net = gross - sum(effective_cost_pct(r.get("entry"), r.get("exit_price"),
                                         r.get("category")) for r in resolved)
    win_rate = wins / len(resolved) * 100
    rec = f"{wins}W / {losses}L"
    if neutral:
        rec += f" / {neutral}N"
    holds = [d for d in (_hold_days(r) for r in resolved) if d is not None]
    hold_str = f"  ·  avg hold {sum(holds) / len(holds):.0f}d" if holds else ""
    return (f"📈 *Swing record to date:* {rec}  ·  "
            f"win rate {win_rate:.0f}%  ·  net {net:+.2f}%{hold_str}")


def format_swing_completion(rec: dict[str, Any],
                            record_summary: str | None = None) -> str:
    """Per-trade message sent when a swing recommendation finishes."""
    status = rec.get("status")
    emoji = _STATUS_EMOJI.get(status, "•")
    sym = rec["symbol"].replace(".NS", "").replace(".BO", "")
    direction = rec["direction"].upper()
    headline = _SWING_HEADLINE.get(status, _STATUS_LABEL.get(status, status or "Closed"))

    pnl = rec.get("pnl_pct")
    exit_p = rec.get("exit_price")
    gen = _parse_iso(rec["generated_at"]).strftime("%d %b")
    exit_t = rec.get("exit_time")
    exit_d = _parse_iso(exit_t).strftime("%d %b") if isinstance(exit_t, str) and exit_t else "—"
    held = _hold_days(rec)
    held_str = f"   ·   Held: {held}d" if held is not None else ""

    lines = [
        f"{emoji} *Swing trade closed — {sym}*  ({direction})",
        headline,
        "",
        f"Recommended: {gen}   ·   Closed: {exit_d}{held_str}",
        f"Entry: ₹{rec['entry']:.2f}",
    ]
    if exit_p is not None:
        lines.append(f"Exit:  ₹{exit_p:.2f}")
    if pnl is not None:
        cost = effective_cost_pct(rec.get("entry"), rec.get("exit_price"),
                                  rec.get("category"))
        net = pnl - cost
        lines.append(f"Result: *{pnl:+.2f}%*  (net ~{net:+.2f}% after ~{cost:.2f}% costs)")
    if record_summary:
        lines += ["", record_summary]
    lines += ["", "_Hypothetical paper-trade outcome, gross of slippage. "
                  "Educational only — not financial advice._"]
    return "\n".join(lines)


def _format_empty_day_explanation(rejections: dict[str, int]) -> str:
    """Render a 'why no alerts today?' section based on filter rejection stats."""
    total = sum(rejections.values())
    if total == 0:
        return (
            "\n*Why no alerts today?*\n"
            "_No scan cycles completed — check `/angel_status` or VM service health._"
        )
    top = sorted(rejections.items(), key=lambda x: -x[1])[:5]
    lines = [
        "\n*Why no alerts today?*",
        f"_Across {total} candle-checks today, every one was filtered out._",
        "",
        "*Top rejection reasons:*",
    ]
    for reason, count in top:
        pct = (count / total) * 100
        lines.append(f"• {reason}: *{count}* ({pct:.0f}%)")
    lines.append("")
    lines.append(
        "_The strategy correctly skipped dead markets. This is normal in "
        "low-volatility regimes — you'll get pings the moment a real setup fires._"
    )
    return "\n".join(lines)
