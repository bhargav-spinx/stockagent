"""
Wilder-smoothing correctness for RSI and ATR (matches TradingView/Zerodha).
CI-safe: only needs pandas/numpy via analyzer.

    venv/Scripts/python.exe tests/test_indicators.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer  # noqa: E402


def test_rsi_all_gains_pins_to_100():
    s = pd.Series(np.arange(1, 101, dtype=float))   # strictly rising → no losses
    assert analyzer.rsi(s, 14).iloc[-1] > 99.0


def test_rsi_all_losses_pins_to_0():
    s = pd.Series(np.arange(100, 0, -1, dtype=float))  # strictly falling
    assert analyzer.rsi(s, 14).iloc[-1] < 1.0


def test_rsi_warmup_is_nan():
    s = pd.Series(np.arange(1, 10, dtype=float))    # fewer than period
    assert analyzer.rsi(s, 14).isna().all()


def test_atr_wilder_converges_to_constant_true_range():
    # Flat closes, constant 2-wide bars → TR = 2 every bar → ATR → 2.0
    n = 60
    df = pd.DataFrame({"High": [101.0] * n, "Low": [99.0] * n,
                       "Close": [100.0] * n})
    a = analyzer.atr(df, 14).iloc[-1]
    assert abs(a - 2.0) < 0.01, a


def test_atr_warmup_is_nan():
    df = pd.DataFrame({"High": [101.0] * 5, "Low": [99.0] * 5,
                       "Close": [100.0] * 5})
    assert analyzer.atr(df, 14).isna().all()


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
