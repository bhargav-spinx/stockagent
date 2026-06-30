"""
Deterministic narrative engine: conviction buckets, validation gates, FACT/
INTERP structure, disclaimer, and the structural guarantees (no probability %,
no guarantee language). CI-safe (needs pandas via intraday_score.ScoreCard).

    venv/Scripts/python.exe tests/test_narrative.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import narrative  # noqa: E402
from intraday_score import ScoreCard  # noqa: E402


def _card(score=88, direction="long", regime_ok=True, event_ok=True):
    return ScoreCard(
        symbol="TEST.NS", price=100.0, direction=direction, score=score,
        rating="Strong Buy" if direction == "long" else "Strong Sell",
        breakdown={"Gap": 10, "Relative Volume": 25, "VWAP Confirmation": 15,
                   "EMA Trend": 15, "ORB Breakout": 8, "Volume Breakout": 10},
        signals={"Gap": "Up 1.5%", "Volume Spike": "2.0x", "VWAP": "Bullish",
                 "EMA Trend": "Bullish"},
        entry=100.0, stop_loss=98.0, target1=104.0, target2=106.0,
        bull_conviction=82, bear_conviction=18, gap_pct=1.5, rvol=2.0,
        orb_window=15, regime_ok=regime_ok, event_ok=event_ok,
        delivery_pct=70.0, context=["Supertrend: bullish"], notes=[],
    )


def test_conviction_buckets():
    assert narrative.conviction_bucket(92) == "High confluence"
    assert narrative.conviction_bucket(80) == "Constructive"
    assert narrative.conviction_bucket(65) == "Mixed / needs confirmation"
    assert narrative.conviction_bucket(50) == "No-Trade"
    assert narrative.conviction_bucket(95, gate_passed=False) == "No-Trade"


def test_passing_card_produces_plan_and_disclaimer():
    out = narrative.narrate_scorecard(_card())
    assert "GATES: PASS" in out
    assert "TRADE PLAN" in out
    assert "Invalidation" in out
    assert "not investment advice" in out.lower()        # disclaimer present
    # structural guarantees
    # no FALSE probability claim (the phrase "not a probability" is fine)
    assert "% probability" not in out.lower() and "probability of" not in out.lower()
    assert "guarantee" not in out.lower()


def test_hostile_regime_is_no_trade():
    out = narrative.narrate_scorecard(_card(regime_ok=False))
    assert "NO-TRADE" in out
    assert "regime hostile" in out.lower()


def test_low_score_no_directional_thesis():
    out = narrative.narrate_scorecard(_card(direction="none"))
    assert "NO-TRADE" in out
    assert "not investment advice" in out.lower()


def test_swing_buy_has_plan_hold_is_no_trade():
    buy = {
        "symbol": "X.NS", "signal": "BUY", "confidence": 75.0, "price": 100.0,
        "rsi": 45.0,
        "indicators": [("SMA Crossover", "BUY", "..."), ("RSI", "HOLD", "..."),
                       ("MACD", "BUY", "..."), ("Bollinger", "HOLD", "...")],
        "trade_setup": {"action": "BUY", "entry": 100.0, "stop_loss": 98.0,
                        "target1": 104.0, "target2": 106.0, "risk_pct": 2.0},
    }
    out = narrative.narrate_swing(buy)
    assert "TRADE PLAN" in out and "Constructive" in out
    assert "not investment advice" in out.lower()
    # no FALSE probability claim (the phrase "not a probability" is fine)
    assert "% probability" not in out.lower() and "probability of" not in out.lower()

    hold = {"symbol": "X.NS", "signal": "HOLD", "confidence": 50.0, "price": 100.0,
            "rsi": 55.0, "indicators": [], "trade_setup": {"action": "WAIT"}}
    assert "NO-TRADE" in narrative.narrate_swing(hold)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1; print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
