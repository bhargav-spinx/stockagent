"""
Trade-type label on every actionable message (Intraday / Swing).
Only the two horizons the engine actually analyses are labelled — Scalping /
Positional / Long-Term / Investment are intentionally NOT produced.
CI-safe (pandas via intraday_score).

    venv/Scripts/python.exe tests/test_trade_type.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants                       # noqa: E402
from intraday_score import format_scorecard  # noqa: E402
import narrative                       # noqa: E402
from analyzer import format_report     # noqa: E402

# reuse the scorecard factory from the narrative tests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_narrative import _card       # noqa: E402


def test_tag_helper_intraday_and_swing():
    assert "Intraday" in constants.trade_type_tag("intraday")
    assert "Swing" in constants.trade_type_tag("swing")
    # unknown mode degrades, never fabricates a horizon
    assert constants.trade_type_tag("") == "🏷 *—*"


def test_scorecard_labeled_intraday():
    assert "Intraday" in format_scorecard(_card())


def test_report_swing_labeled():
    result = {
        "symbol": "X.NS", "mode": "swing", "mode_label": "Swing (daily candles)",
        "price": 100.0, "change_pct": 1.2, "change_label": "vs prev close",
        "signal": "BUY", "confidence": 75.0,
        "indicators": [("RSI", "BUY", "ok")],
        "trade_setup": {}, "timestamp": "2026-06-30 10:00 IST",
    }
    out = format_report(result)
    assert "🏷 *Swing*" in out


def test_narrative_swing_labeled():
    buy = {
        "symbol": "X.NS", "signal": "BUY", "confidence": 75.0, "price": 100.0,
        "rsi": 45.0, "indicators": [("SMA", "BUY", "..."), ("MACD", "BUY", "...")],
        "trade_setup": {"action": "BUY", "entry": 100.0, "stop_loss": 98.0,
                        "target1": 104.0, "target2": 106.0, "risk_pct": 2.0},
    }
    assert "🏷 *Swing*" in narrative.narrate_swing(buy)


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
