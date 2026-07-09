"""
Deterministic analysis-and-narrative engine.

Turns the computed output of intraday_score.score_stock() / analyzer.analyze()
into the institutional-style write-up described in the system spec — WITHOUT any
LLM/API call. Because it's templated over engine fields, it structurally cannot:
  - fabricate a level/price/indicator (it only formats provided values),
  - state a probability (conviction is an ordinal bucket, never a %),
  - use guarantee language,
  - or omit the disclaimer.

Sections it can populate come from fields the engine computes today; everything
the engine doesn't produce (options/OI, India VIX, FII-DII, SMC/multi-timeframe)
is reported as "not provided" rather than invented.

FACT  = a value read from the engine.   INTERPRETATION = a rule-based read of it.
"""
from __future__ import annotations

from config import CONFIG
from constants import DISCLAIMER, trade_type_tag

_G = CONFIG.gates


# --- §8 Conviction buckets (strength of confluence, NOT probability) ----------
def conviction_bucket(strength: float | None, gate_passed: bool = True) -> str:
    """Map a 0–100 score (or swing confidence) to a conviction label."""
    if not gate_passed or strength is None:
        return "No-Trade"
    if strength >= _G.conviction_high:
        return "High confluence"
    if strength >= _G.conviction_constructive:
        return "Constructive"
    if strength >= _G.conviction_mixed:
        return "Mixed / needs confirmation"
    return "No-Trade"


def _rr(entry: float, sl: float, target: float) -> tuple[float, str]:
    """Risk:reward with the arithmetic shown."""
    risk = abs(entry - sl)
    reward = abs(target - entry)
    if risk <= 0:
        return 0.0, "risk=0 (invalid SL)"
    rr = reward / risk
    return rr, f"|{target:.2f}−{entry:.2f}| / |{entry:.2f}−{sl:.2f}| = {reward:.2f}/{risk:.2f} = 1:{rr:.2f}"


# --- §9 Validation gates over INTRADAY ScoreCard fields ------------------------
def evaluate_gates_intraday(card) -> tuple[bool, list[str], list[str]]:
    """Returns (passed, failed_gates, soft_flags) using only computed fields.
    Gates the engine can't assess (multi-timeframe, structure, rel-strength) are
    surfaced as soft notes, not silent passes."""
    failed: list[str] = []
    flags: list[str] = []

    if card.direction == "none":
        failed.append("No directional thesis (sideways/conflicting)")
    if card.score < _G.min_intraday_score:
        failed.append(f"Score {card.score} < {_G.min_intraday_score} "
                      "(below actionable floor)")
    if not card.regime_ok:
        failed.append("Index regime hostile (counter-trend)")
    if not card.event_ok:
        failed.append("Unresolved event risk (earnings within window)")

    # R:R gate (T1 must be ≥ 1:2 by design; verify from actual levels).
    # Tolerance: canonical levels are BUILT at exactly 2R (entry + 2·risk), and
    # `(entry + 2·risk) − entry` can land one ULP under 2·risk — without the
    # epsilon this gate rejected its own levels with "R:R 2.00 < 2.0".
    if card.entry and card.stop_loss and card.target1:
        rr1, _ = _rr(card.entry, card.stop_loss, card.target1)
        if rr1 < _G.min_rr - 1e-9:
            failed.append(f"R:R to T1 {rr1:.2f} < {_G.min_rr}")

    for n in (card.notes or []):
        if "supertrend disagrees" in n.lower() or "low delivery" in n.lower():
            flags.append(n)

    # Gates not assessable from current data — declared, not assumed-passed.
    flags.append("Multi-timeframe / market-structure / relative-strength: not provided")
    return (len(failed) == 0, failed, flags)


def _carry_drag(breakdown: dict) -> tuple[str, str]:
    if not breakdown:
        return "not provided", "not provided"
    items = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
    carry = ", ".join(f"{k} {v}" for k, v in items if v > 0) or "none"
    drag = ", ".join(f"{k} {v}" for k, v in items if v == 0) or "none"
    return carry, drag


def narrate_scorecard(card) -> str:
    """Compact institutional narrative for an intraday ScoreCard."""
    if card.direction == "none":
        passed, failed, _ = evaluate_gates_intraday(card)
        return (
            f"{trade_type_tag('intraday')}\n"
            f"📑 *{card.symbol}* · ₹{card.price:,.2f} · _(last closed candle)_\n\n"
            f"*Verdict: NO-TRADE* — {', '.join(failed) or 'no directional thesis'}\n"
            + ("\n".join(f"_{n}_" for n in card.notes) if card.notes else "")
            + f"\n\n{DISCLAIMER}"
        )

    passed, failed, flags = evaluate_gates_intraday(card)
    bias = "Long" if card.direction == "long" else "Short"
    bucket = conviction_bucket(card.score, passed)
    carry, drag = _carry_drag(card.breakdown)

    lines = [
        trade_type_tag("intraday"),
        f"📑 *{card.symbol}* · ₹{card.price:,.2f} · _(last closed candle)_",
        "",
        "*FACT (engine-computed)*",
        f"• Score *{card.score}/100* — {card.rating}; direction *{bias}*",
        f"• Carrying: {carry}",
        f"• Dragging: {drag}",
        f"• Gap: {card.signals.get('Gap', 'n/a')} · RVOL: {card.signals.get('Volume Spike', 'n/a')} · "
        f"VWAP: {card.signals.get('VWAP', 'n/a')} · EMA: {card.signals.get('EMA Trend', 'n/a')}",
    ]
    if card.delivery_pct is not None:
        lines.append(f"• Delivery: {card.delivery_pct:.0f}%")
    if card.context:
        lines.append("• Context: " + "  ·  ".join(card.context))

    lines += [
        "",
        "*INTERPRETATION*",
        f"• Conviction: *{bucket}* (ordinal, not a probability)",
        "• Collinearity note: VWAP / EMA / score derive from one price series — "
        "treat their agreement as one thesis, not independent confirmations.",
        f"• Regime: {'supportive' if card.regime_ok else 'HOSTILE (counter-trend)'}",
    ]
    if flags:
        lines.append("• Flags: " + "; ".join(flags))

    # §9 verdict
    lines += ["", f"*GATES: {'PASS — actionable' if passed else 'NO-TRADE'}*"]
    if not passed:
        lines.append("Failed: " + "; ".join(failed))

    # §7 trade plan (only if gates pass and levels exist)
    if passed and all(v is not None for v in (card.entry, card.stop_loss, card.target1, card.target2)):
        rr1, rr1_math = _rr(card.entry, card.stop_loss, card.target1)
        rr2, _ = _rr(card.entry, card.stop_loss, card.target2)
        lines += [
            "",
            "*TRADE PLAN*",
            f"• Bias: *{bias}* — strongest driver: {carry.split(',')[0]}",
            f"• Entry: ₹{card.entry:,.2f}  ·  SL: ₹{card.stop_loss:,.2f} (ATR-sized)",
            f"• Targets: T1 ₹{card.target1:,.2f} (R:R {rr1_math}) · T2 ₹{card.target2:,.2f} (1:{rr2:.2f})",
            "• Manage: 50% at T1 → trail SL to entry → 50% at T2; 45-min time-stop; EOD square-off",
            "• Sizing: size so the SL distance equals your predefined per-trade risk % "
            "(no quantity/lot/capital figure — that would be personalised advice)",
            f"• Invalidation: 5-min close beyond SL ₹{card.stop_loss:,.2f} "
            f"or loss of VWAP against the {bias.lower()} thesis",
        ]

    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def narrate_swing(result: dict) -> str:
    """Compact institutional narrative for an analyzer.analyze() swing result."""
    sym = result["symbol"]
    sig = result["signal"]
    conf = result.get("confidence")
    setup = result.get("trade_setup") or {}

    # Swing gate: actionable BUY/SELL with a real setup.
    actionable = sig in ("BUY", "SELL") and setup.get("action") in ("BUY", "SELL")
    bucket = conviction_bucket(conf, actionable)

    votes = result.get("indicators", [])
    agree = [name for name, s, _ in votes if s == sig]
    disagree = [f"{name} ({s})" for name, s, _ in votes if s not in (sig, "HOLD")]

    lines = [
        trade_type_tag("swing"),
        f"📑 *{sym}* · ₹{result['price']:,.2f} · _(daily, last closed candle)_",
        "",
        "*FACT (engine-computed)*",
        f"• Vote signal: *{sig}*  ({conf}% of indicators agree)" if conf is not None
        else f"• Vote signal: *{sig}*",
        f"• Agreeing: {', '.join(agree) or 'none'}",
        f"• Conflicting: {', '.join(disagree) or 'none'}",
        f"• RSI: {result.get('rsi', 'n/a')}",
    ]

    lines += [
        "",
        "*INTERPRETATION*",
        f"• Conviction: *{bucket}* (ordinal, not a probability)",
        "• Collinearity: the 4 votes share one daily price series — agreement is "
        "one thesis, not four independent confirmations.",
    ]
    if disagree:
        lines.append("• Internal conflict present — genuine signal to lower conviction.")

    if not actionable:
        lines += ["", "*GATES: NO-TRADE* — vote is HOLD or below the swing standard.",
                  "", DISCLAIMER]
        return "\n".join(lines)

    entry, sl = setup["entry"], setup["stop_loss"]
    t1, t2 = setup["target1"], setup["target2"]
    rr1, rr1_math = _rr(entry, sl, t1)
    rr2, _ = _rr(entry, sl, t2)
    bias = "Long" if sig == "BUY" else "Short"
    lines += [
        "",
        "*GATES: PASS — actionable*",
        "",
        "*TRADE PLAN*",
        f"• Bias: *{bias}*",
        f"• Entry: ₹{entry:,.2f}  ·  SL: ₹{sl:,.2f} ({setup.get('risk_pct', '?')}% , ATR-sized)",
        f"• Targets: T1 ₹{t1:,.2f} (R:R {rr1_math}) · T2 ₹{t2:,.2f} (1:{rr2:.2f})",
        "• Manage: 50% at T1 → trail SL to entry → 50% at T2",
        "• Sizing: size so the SL distance equals your predefined per-trade risk % "
        "(no quantity/lot/capital figure)",
        f"• Invalidation: daily close beyond SL ₹{sl:,.2f}",
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines)
