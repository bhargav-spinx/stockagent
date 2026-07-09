"""
TradeEvidence — the platform's structured output contract (Phase 1.6).

Every engine result converges into this one shape. It is the interface later
phases plug into: the regime/liquidity/risk engines (Phase 2) fill their
fields, and the probability engine (Phase 4) fills `probability_*` — until
then those fields are None BY DESIGN.

Honesty invariant: `probability_of_success` stays None until a calibrated,
walk-forward-validated model exists. Emitting a pseudo-probability derived
from a hand-weighted score would be LESS honest than the current ordinal
conviction buckets — consumers must treat None as "not yet measurable", never
substitute the score for it.

Converters are read-only views over existing engine outputs; they change no
live behavior. Telegram formatting keeps using the existing formatters in
Phase 1; adapters move onto this contract in Phase 2.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import narrative
from features import (
    FEATURE_SCHEMA_VERSION, features_from_scorecard, features_from_swing_result,
)

EVIDENCE_SCHEMA_VERSION = 1

_STAT_CONFIDENCE_NOTE = (
    "No calibrated probability model yet — probability fields are None by "
    "design (Phase 4 gate: calibrated + walk-forward-validated, or nothing). "
    "Score/conviction are heuristics, not win likelihoods; realized per-bucket "
    "win rates live in /stats."
)


@dataclass
class TradeEvidence:
    """One trade thesis, expressed as evidence — never as a bare BUY/SELL."""
    # provenance
    schema_version: int
    engine: str                      # "intraday_score" | "analyzer"
    symbol: str
    generated_at: str                # ISO UTC
    # thesis
    direction: str                   # "long" | "short" | "none"
    score: float | None              # engine score (100-pt / swing confidence)
    confidence_label: str            # ordinal bucket — NOT a probability
    # probabilities — None until the Phase-4 model earns them
    probability_of_success: float | None = None
    probability_of_failure: float | None = None
    probability_of_no_trade: float | None = None
    # context (Phase 2/3 engines will populate what's None today)
    market_regime: dict | None = None        # {"NIFTY": "bullish", ...}
    regime_ok: bool | None = None
    event_ok: bool | None = None
    volatility_state: str | None = None      # Phase 3 volatility engine
    relative_strength: float | None = None   # Phase 3 RS engine
    institutional_activity: dict | None = None   # delivery-% proxy for now
    # levels & risk
    entry: float | None = None
    stop_loss: float | None = None
    target1: float | None = None
    target2: float | None = None
    rr1: float | None = None
    rr2: float | None = None
    expected_return_pct: float | None = None  # needs probabilities → Phase 4
    expected_value_r: float | None = None     # needs probabilities → Phase 4
    position_size: int | None = None          # Phase 2 risk engine
    # evidence body
    features: dict = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistical_confidence: str = _STAT_CONFIDENCE_NOTE

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True,
                          default=str)


def _rr(entry, sl, target) -> float | None:
    try:
        risk = abs(entry - sl)
        return round(abs(target - entry) / risk, 4) if risk else None
    except TypeError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def from_scorecard(card, idx_trend: dict | None = None) -> TradeEvidence:
    """View an intraday_score.ScoreCard as TradeEvidence."""
    reasoning = [f"{k}: {v}" for k, v in (card.signals or {}).items()]
    reasoning += list(card.context or [])
    gate_passed, failed, _flags = narrative.evaluate_gates_intraday(card)
    return TradeEvidence(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        engine="intraday_score",
        symbol=card.symbol,
        generated_at=_now_iso(),
        direction=card.direction,
        score=float(card.score),
        confidence_label=narrative.conviction_bucket(card.score, gate_passed),
        market_regime=dict(idx_trend) if idx_trend else None,
        regime_ok=bool(card.regime_ok),
        event_ok=bool(card.event_ok),
        institutional_activity=(
            {"delivery_pct": card.delivery_pct, "source": "NSE bhavcopy proxy"}
            if card.delivery_pct is not None else None),
        entry=card.entry,
        stop_loss=card.stop_loss,
        target1=card.target1,
        target2=card.target2,
        rr1=_rr(card.entry, card.stop_loss, card.target1),
        rr2=_rr(card.entry, card.stop_loss, card.target2),
        features=features_from_scorecard(card, idx_trend=idx_trend),
        reasoning=reasoning,
        warnings=list(card.notes or []) + failed,
    )


def from_swing_result(result: dict) -> TradeEvidence:
    """View an analyzer.analyze() result dict as TradeEvidence."""
    setup = result.get("trade_setup") or {}
    signal = result.get("signal")
    actionable = signal in ("BUY", "SELL") and setup.get("action") in ("BUY", "SELL")
    direction = ("long" if signal == "BUY" else
                 "short" if signal == "SELL" else "none")
    reasoning = [f"{name}: {vote} — {reason}"
                 for name, vote, reason in (result.get("indicators") or [])]
    warnings: list[str] = []
    if not actionable:
        warnings.append("No actionable setup (HOLD / WAIT)")
    return TradeEvidence(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        engine="analyzer",
        symbol=result.get("symbol", "?"),
        generated_at=_now_iso(),
        direction=direction,
        score=result.get("confidence"),
        confidence_label=narrative.conviction_bucket(
            result.get("confidence"), gate_passed=actionable),
        entry=setup.get("entry") if actionable else None,
        stop_loss=setup.get("stop_loss") if actionable else None,
        target1=setup.get("target1") if actionable else None,
        target2=setup.get("target2") if actionable else None,
        rr1=(_rr(setup.get("entry"), setup.get("stop_loss"), setup.get("target1"))
             if actionable else None),
        rr2=(_rr(setup.get("entry"), setup.get("stop_loss"), setup.get("target2"))
             if actionable else None),
        features=features_from_swing_result(result),
        reasoning=reasoning,
        warnings=warnings,
    )
