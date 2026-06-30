"""
Unadjusted-split warning for Angel daily candles. CI-safe (data_provider only;
no network — scrip master loads lazily).

    venv/Scripts/python.exe tests/test_data_adjust.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_provider as dp  # noqa: E402


def _daily(closes):
    idx = pd.date_range("2026-01-05", periods=len(closes), freq="D",
                        tz="Asia/Kolkata")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": [1] * len(closes)}, index=idx)


def _capture(df, symbol, interval):
    """Call the detector with logger.warning captured; return list of calls."""
    calls = []
    orig = dp.logger.warning
    dp.logger.warning = lambda *a, **k: calls.append(a)
    try:
        dp._warn_unadjusted_split(df, symbol, interval)
    finally:
        dp.logger.warning = orig
    return calls


def test_warns_on_unadjusted_1_for_2_split():
    # price halves overnight (1:2 split, ratio 0.5) → should warn
    assert _capture(_daily([100, 100, 50, 50]), "X", "ONE_DAY")


def test_no_warn_on_normal_daily_moves():
    assert not _capture(_daily([100, 103, 101, 104, 99]), "X", "ONE_DAY")


def test_intraday_interval_is_ignored_even_with_big_gap():
    assert not _capture(_daily([100, 50]), "X", "FIVE_MINUTE")


def test_single_row_no_warn():
    assert not _capture(_daily([100]), "X", "ONE_DAY")


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
