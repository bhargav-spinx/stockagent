"""
Realized-performance stats from resolved alert outcomes.

Closes the loop on the recommendations the bot has actually made: it reads
`alert_outcomes` (populated by eod_report.resolve_*) and reports realized
win rate and net P&L sliced by category, by intraday score bucket, and by
channel. Use it to decide whether the gates (e.g. score ≥ 90) are set right.

Win rate is computed over DECISIVE trades only (wins + losses); neutral
outcomes (time-stop, square-off, breakeven) are reported but excluded from
the rate so it isn't flattered by trades that never really resolved.
"""
from __future__ import annotations

from typing import Any

import subscriptions
import eod_report
import riskmetrics
from eod_report import COST_PER_TRADE_PCT, _classify, _hold_days


def _net_returns(rows: list[dict[str, Any]]) -> list[float]:
    """Per-trade NET % returns from resolved rows (gross pnl minus costs)."""
    return [r["pnl_pct"] - COST_PER_TRADE_PCT
            for r in rows if r.get("pnl_pct") is not None]

ALL_CATEGORIES = tuple(eod_report.INTRADAY_CATEGORIES | eod_report.SWING_CATEGORIES)

# Intraday score buckets for the "is the gate right?" breakdown.
_SCORE_BUCKETS = [(95, 101, "95–100"), (90, 95, "90–94"),
                  (80, 90, "80–89"), (0, 80, "<80")]


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a set of resolved alerts into headline numbers."""
    resolved = [r for r in rows if r.get("pnl_pct") is not None]
    n = len(resolved)
    wins = sum(1 for r in resolved if _classify(r.get("status")) == "pass")
    losses = sum(1 for r in resolved if _classify(r.get("status")) == "fail")
    neutral = n - wins - losses
    gross = sum(r["pnl_pct"] for r in resolved)
    net = gross - COST_PER_TRADE_PCT * n
    decisive = wins + losses
    win_rate = (wins / decisive * 100) if decisive else 0.0
    holds = [d for d in (_hold_days(r) for r in resolved) if d is not None]
    avg_hold = (sum(holds) / len(holds)) if holds else None
    return {
        "n": n, "wins": wins, "losses": losses, "neutral": neutral,
        "net": net, "avg": (net / n) if n else 0.0,
        "win_rate": win_rate, "avg_hold": avg_hold,
    }


def _line(label: str, a: dict[str, Any]) -> str:
    if a["n"] == 0:
        return f"{label}: _no resolved trades_"
    rec = f"{a['wins']}W/{a['losses']}L"
    if a["neutral"]:
        rec += f"/{a['neutral']}N"
    hold = f"  ·  hold {a['avg_hold']:.0f}d" if a["avg_hold"] is not None else ""
    return (f"{label}: {rec}  ·  win {a['win_rate']:.0f}%  ·  "
            f"net {a['net']:+.1f}% (avg {a['avg']:+.2f}%){hold}")


def _score_of(row: dict[str, Any]) -> int | None:
    setup = row.get("setup") or ""
    if setup.startswith("score"):
        try:
            return int(setup[5:])
        except ValueError:
            return None
    return None


def build_stats_report() -> str:
    """System-wide realized-performance report across all resolved alerts."""
    # Resolve anything still pending so the numbers are current.
    try:
        eod_report.resolve_pending(max_age_days=60)
    except Exception:
        pass

    rows = subscriptions.get_resolved_alerts(ALL_CATEGORIES)
    if not rows:
        return ("📈 *Performance stats*\n\n_No resolved recommendations yet._\n"
                "Stats appear here once alerts have run their course "
                "(intraday same day, swing/tips over the following days).")

    by_cat = {c: [r for r in rows if r["category"] == c] for c in ALL_CATEGORIES}
    scan_rows = by_cat.get("scan", [])

    parts = [
        "📈 *Performance stats* — realized outcomes\n",
        f"_Net of ~{COST_PER_TRADE_PCT:.2f}% round-trip costs. Win rate over "
        "decisive (W+L) trades only._\n",
        "*Overall*",
        _line("All", _agg(rows)),
        riskmetrics.format_line(_net_returns(rows), "Risk-adj"),
        "_Win rate/P&L alone can't separate skill from luck — if the 95% CI "
        "straddles 0, there is no demonstrable edge yet._",
        "",
        "*By type*",
        _line("🟦 Intraday auto", _agg(scan_rows)),
        _line("🟧 Manual intraday", _agg(by_cat.get("manual_intraday", []))),
        _line("🟪 Swing auto", _agg(by_cat.get("swing_auto", []))),
        _line("📅 Manual swing", _agg(by_cat.get("manual_swing", []))),
        _line("📡 Channel tips (our analysis)", _agg(by_cat.get("channel_tip", []))),
        _line("📞 Channel CALLS (their levels)", _agg(by_cat.get("channel_call", []))),
    ]

    # Intraday by score bucket — is the gate set right?
    if scan_rows:
        parts += ["", "*Intraday by score* (gate calibration)"]
        for lo, hi, label in _SCORE_BUCKETS:
            bucket = [r for r in scan_rows
                      if (s := _score_of(r)) is not None and lo <= s < hi]
            if bucket:
                parts.append(_line(f"score {label}", _agg(bucket)))

    # Channel CALLS by source (the channel's OWN levels) — which sources
    # actually work, judged on what they posted, not on our re-derived signal.
    call_rows = by_cat.get("channel_call", [])
    if call_rows:
        channels = sorted({(r.get("setup") or "?") for r in call_rows})
        parts += ["", "*Channel CALLS by source* (their own entry/target/SL)"]
        for ch in channels:
            parts.append(_line(ch, _agg([r for r in call_rows if r.get("setup") == ch])))

    return "\n".join(parts)
