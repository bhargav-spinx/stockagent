"""
End-of-day report.

For every alert logged today (and recent open swing alerts), simulate the
hypothetical outcome assuming the user took the trade at the published
Entry/SL/T1/T2 with the default 50/50 partial-exit rules from STRATEGY.md §7.

Outcome categories:
    t2_hit              — T2 reached → +1.5% blended (50% at T1, 50% at T2)
    t1_then_squareoff   — T1 hit then session ended → blended 50% at T1 + 50% at last close
    t1_then_breakeven   — T1 hit, then trailing-SL at entry → +0.5% blended
    sl_hit              — SL hit before T1 → full SL distance loss
    time_stop           — 45 min elapsed without T1/SL (intraday only) → exit at last close
    squareoff_no_t1     — no T1, no SL, market closed → exit at last close
    open                — swing alerts not yet resolved
    no_data             — could not fetch post-entry candles

P&L is hypothetical, gross of brokerage/STT/slippage. Apply ~0.13%/round-trip
per STRATEGY.md §8 to estimate net.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time, date, timedelta
from typing import Any

import pandas as pd
import pytz

import subscriptions
from data_provider import fetch_data
from constants import IST

logger = logging.getLogger(__name__)

# STRATEGY.md §8 cost model: ~0.05% brokerage/STT per side + ~0.05% slippage
# entry + ~0.03% slippage exit ≈ 0.13% round-trip on tier-1 NSE stocks.
# Single source of truth — imported by backtest.py and bot.py.
COST_PER_TRADE_PCT = 0.13

INTRADAY_CATEGORIES = {"scan", "manual_intraday"}
# Channel tips are analysed on daily candles (swing horizon), so they resolve
# and report through the swing machinery.
SWING_CATEGORIES = {"swing_auto", "manual_swing", "channel_tip"}

PASS_STATUSES = {"t2_hit", "t1_then_squareoff", "t1_then_breakeven"}
FAIL_STATUSES = {"sl_hit"}
NEUTRAL_STATUSES = {"time_stop", "squareoff_no_t1", "open", "no_data"}


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

    # Only candles after the trigger candle (which already closed at entry)
    post = df[df.index > gen_time]
    # Restrict to today's session
    post = post[post.index.date == gen_time.date()]
    if len(post) == 0:
        return {"status": "open"}

    t1_hit = False
    t1_hit_time = None

    for ts, row in post.iterrows():
        # Time-stop check: 45 min after generation
        if not t1_hit and (ts - gen_time) > timedelta(minutes=45):
            close = float(row["Close"])
            return {
                "status": "time_stop",
                "exit_price": close,
                "exit_time": ts,
                "pnl_pct": round(_signed_pct(entry, close, direction), 2),
            }

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
                return {
                    "status": "sl_hit",
                    "exit_price": sl,
                    "exit_time": ts,
                    "pnl_pct": round(_signed_pct(entry, sl, direction), 2),
                }
            if t1_in:
                t1_hit = True
                t1_hit_time = ts
                # Check t2 in same candle
                if t2_in:
                    # 50% at T1 (1%), 50% at T2 (2%) → blended 1.5%
                    return {
                        "status": "t2_hit",
                        "exit_price": t2,
                        "exit_time": ts,
                        "pnl_pct": 1.5,
                    }
        else:
            # After T1: SL is moved to entry per §7
            if direction == "long":
                trail_sl_in = row["Low"] <= entry
                t2_in_now = row["High"] >= t2
            else:
                trail_sl_in = row["High"] >= entry
                t2_in_now = row["Low"] <= t2

            if t2_in_now:
                return {
                    "status": "t2_hit",
                    "exit_price": t2,
                    "exit_time": ts,
                    "pnl_pct": 1.5,
                }
            if trail_sl_in:
                # 50% at T1 (1%), 50% at entry (0%) → blended 0.5%
                return {
                    "status": "t1_then_breakeven",
                    "exit_price": entry,
                    "exit_time": ts,
                    "pnl_pct": 0.5,
                }

    # End of candles: square off at last close
    last_ts = post.index[-1]
    last_close = float(post["Close"].iloc[-1])
    if t1_hit:
        # 50% at T1 (1%) + 50% at last close
        remainder_pct = _signed_pct(entry, last_close, direction)
        blended = (1.0 + remainder_pct) / 2
        return {
            "status": "t1_then_squareoff",
            "exit_price": last_close,
            "exit_time": last_ts,
            "pnl_pct": round(blended, 2),
        }
    return {
        "status": "squareoff_no_t1",
        "exit_price": last_close,
        "exit_time": last_ts,
        "pnl_pct": round(_signed_pct(entry, last_close, direction), 2),
    }


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
    for ts, row in post.iterrows():
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
                return {
                    "status": "sl_hit",
                    "exit_price": sl,
                    "exit_time": ts,
                    "pnl_pct": round(_signed_pct(entry, sl, direction), 2),
                }
            if t1_in:
                t1_hit = True
                if t2_in:
                    return {
                        "status": "t2_hit",
                        "exit_price": t2,
                        "exit_time": ts,
                        "pnl_pct": 1.5,
                    }
        else:
            if direction == "long":
                trail_sl_in = row["Low"] <= entry
                t2_in_now = row["High"] >= t2
            else:
                trail_sl_in = row["High"] >= entry
                t2_in_now = row["Low"] <= t2
            if t2_in_now:
                return {"status": "t2_hit", "exit_price": t2, "exit_time": ts, "pnl_pct": 1.5}
            if trail_sl_in:
                return {"status": "t1_then_breakeven", "exit_price": entry,
                        "exit_time": ts, "pnl_pct": 0.5}

    return {"status": "open"}


def resolve_alert(alert: dict[str, Any]) -> dict[str, Any] | None:
    cat = alert["category"]
    if cat in INTRADAY_CATEGORIES:
        return resolve_intraday(alert)
    if cat in SWING_CATEGORIES:
        return resolve_swing(alert)
    logger.warning("resolve_alert: unknown category %s", cat)
    return None


def resolve_pending(max_age_days: int = 30) -> int:
    """Resolve all open alerts up to max_age_days. Returns count of newly resolved."""
    open_alerts = subscriptions.get_open_alerts(max_age_days=max_age_days)
    resolved = 0
    for alert in open_alerts:
        outcome = resolve_alert(alert)
        if outcome is None or outcome.get("status") == "open":
            continue
        subscriptions.save_outcome(
            alert["id"],
            status=outcome["status"],
            exit_price=outcome.get("exit_price"),
            exit_time=outcome.get("exit_time"),
            pnl_pct=outcome.get("pnl_pct"),
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

    trade_date_str = trade_date_str or date.today().isoformat()
    all_today = subscriptions.get_alerts_for_date(trade_date_str, user_id=user_id)

    scan_rows = [r for r in all_today if r["category"] == "scan"]
    manual_intraday_rows = [r for r in all_today if r["category"] == "manual_intraday"]
    swing_rows = [r for r in all_today if r["category"] in SWING_CATEGORIES]

    parts = [
        f"📊 *End-of-Day Report* — {trade_date_str}\n",
        "_Hypothetical paper-trade outcomes assuming default 50/50 partial-exit rules._\n",
        f"_Gross of brokerage/STT/slippage (~{COST_PER_TRADE_PCT:.2f}% round-trip)._\n\n",
        _format_section("🟦 Intraday auto-scan (/scan_alerts)", scan_rows),
        _format_section("🟧 Manual intraday (/intraday)", manual_intraday_rows),
        _format_section("🟪 Swing (/swing_alerts + /swing)", swing_rows),
    ]

    # If today had zero alerts and we have rejection stats, explain why
    if not all_today and today_rejection_stats:
        parts.append(_format_empty_day_explanation(today_rejection_stats))

    return "\n".join(parts)


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
    net = gross - COST_PER_TRADE_PCT * len(resolved)
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
        net = pnl - COST_PER_TRADE_PCT
        lines.append(f"Result: *{pnl:+.2f}%*  (net ~{net:+.2f}% after ~{COST_PER_TRADE_PCT:.2f}% costs)")
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
