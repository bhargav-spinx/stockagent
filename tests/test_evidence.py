"""
TradeEvidence contract (Phase 1.6): probabilities are None until a calibrated
model exists, converters are faithful read-only views, and the object is
JSON-serializable end to end.

    venv/Scripts/python.exe -m pytest tests/test_evidence.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evidence  # noqa: E402
from intraday_score import ScoreCard  # noqa: E402


def _card(**kw) -> ScoreCard:
    defaults = dict(
        symbol="TEST.NS", price=515.86, direction="long", score=83,
        rating="Strong Buy",
        breakdown={"Gap Up/Down": 10, "Relative Volume": 18,
                   "VWAP Confirmation": 15, "EMA Trend": 15,
                   "ORB Breakout": 15, "Volume Breakout": 10},
        signals={"Gap": "Up 1.26%", "VWAP": "Bullish"},
        entry=515.86, stop_loss=511.77, target1=524.06, target2=528.16,
        bull_conviction=80, bear_conviction=20,
        gap_pct=1.26, rvol=1.8, orb_window=15,
        regime_ok=True, event_ok=True, delivery_pct=62.0,
        context=["Supertrend: bullish"], notes=[],
    )
    defaults.update(kw)
    return ScoreCard(**defaults)


def test_probabilities_none_by_design():
    ev = evidence.from_scorecard(_card())
    assert ev.probability_of_success is None
    assert ev.probability_of_failure is None
    assert ev.probability_of_no_trade is None
    assert ev.expected_value_r is None and ev.expected_return_pct is None
    assert ev.position_size is None
    assert "None by design" in ev.statistical_confidence


def test_scorecard_view_is_faithful():
    ev = evidence.from_scorecard(_card(), idx_trend={"NIFTY": "bullish"})
    assert ev.engine == "intraday_score"
    assert ev.direction == "long" and ev.score == 83.0
    assert ev.market_regime == {"NIFTY": "bullish"}
    assert ev.regime_ok is True and ev.event_ok is True
    assert ev.entry == 515.86 and ev.stop_loss == 511.77
    assert ev.rr1 is not None and 1.9 < ev.rr1 < 2.1
    assert ev.institutional_activity["delivery_pct"] == 62.0
    assert "proxy" in ev.institutional_activity["source"]   # labeled as proxy
    assert any("Gap" in r for r in ev.reasoning)
    assert ev.features["engine"] == "intraday_score"
    assert ev.confidence_label != ""                         # ordinal bucket


def test_swing_view_hold_is_not_actionable():
    result = {
        "symbol": "TEST.NS", "mode": "swing", "signal": "HOLD",
        "price": 1616.42, "change_pct": 0.25, "rsi": 61.0, "confidence": 50.0,
        "market_open": False,
        "indicators": [("SMA Trend", "BUY", "above"), ("RSI", "HOLD", "neutral")],
        "trade_setup": {"action": "WAIT", "swing_high": 1622.43,
                        "swing_low": 1555.05, "atr": 19.11},
    }
    ev = evidence.from_swing_result(result)
    assert ev.direction == "none"
    assert ev.entry is None and ev.rr1 is None
    assert ev.confidence_label == "No-Trade"
    assert any("No actionable setup" in w for w in ev.warnings)


def test_json_round_trip():
    ev = evidence.from_scorecard(_card())
    blob = ev.to_json()
    back = json.loads(blob)
    assert back["symbol"] == "TEST.NS"
    assert back["probability_of_success"] is None
    assert back["schema_version"] == evidence.EVIDENCE_SCHEMA_VERSION
