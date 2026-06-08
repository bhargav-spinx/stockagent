"""
Validates the walk-forward orchestration (BLOCKER-2) WITHOUT market data:
 - folds split correctly,
 - parameters are optimised on train and applied to the *next* fold only,
 - only test-fold trades count toward the OOS result,
 - the parameter-evaluation counter (multiple-testing context) is right.

Uses synthetic data + monkeypatched fetch/sim so it runs offline/deterministic.
    venv/Scripts/python.exe tests/test_walkforward.py
"""
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest  # noqa: E402
from backtest import Trade, _chunks, _slice_with_context  # noqa: E402
from constants import IST  # noqa: E402


def _synth_df():
    """6 sessions × 8 5-min candles, IST-aware, 6 distinct dates."""
    rows, idx = [], []
    for day in range(6):
        base = pd.Timestamp(2026, 1, 5 + day, 9, 30, tz=IST)
        for c in range(8):
            idx.append(base + pd.Timedelta(minutes=5 * c))
            rows.append({"Open": 100, "High": 100, "Low": 100,
                         "Close": 100, "Volume": 1000})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_chunks_contiguous_and_complete():
    seq = list(range(10))
    cs = _chunks(seq, 3)
    assert sum(cs, []) == seq, cs            # nothing lost / reordered
    assert len(cs) == 3, cs


def test_slice_with_context_no_forward_leak():
    df = _synth_df()
    dates = {pd.Timestamp(2026, 1, 8).date()}   # the 4th session
    sub = _slice_with_context(df, dates, context_days=10)
    # includes history up to and including the target date, nothing after it
    assert max(sub.index.date) == pd.Timestamp(2026, 1, 8).date(), max(sub.index.date)
    assert min(sub.index.date) == pd.Timestamp(2026, 1, 5).date()


def test_walkforward_optimises_train_applies_to_test(monkeypatch=None):
    df = _synth_df()

    # fetch returns our synthetic frame for any symbol
    backtest._fetch_5m = lambda sym, days: (
        (sym if "." in sym else sym + ".NS"), df)

    GOOD = (0.004, 0.015)

    def fake_sim(df_, sym, setup_filter=None, atr_lo=None, atr_hi=None,
                entry_dates=None):
        # GOOD param wins (+1/trade); others lose (-1). One trade per entry date.
        pnl = 1.0 if (atr_lo, atr_hi) == GOOD else -1.0
        trades = []
        for d in sorted(entry_dates or []):
            trades.append(Trade(
                symbol=sym, setup="A", direction="long",
                entry_time=datetime(d.year, d.month, d.day, 10, 0, tzinfo=IST),
                entry=100, stop_loss=98, target1=102, target2=104,
                status="t2_hit", pnl_gross_pct=pnl))
        return trades, len(entry_dates or []), {}

    backtest._simulate_setups = fake_sim

    grid = [GOOD, (0.003, 0.020)]
    res = backtest.walk_forward(
        ["TEST"], days=30, score_mode=False,
        n_folds=3, train_folds=1, grid=grid)

    assert res is not None
    # 3 folds, train_folds=1 -> test folds at index 1 and 2 = 2 test folds
    assert len(res["folds"]) == 2, res["folds"]
    # every fold must have picked the GOOD param
    assert all(fl["param"] == GOOD for fl in res["folds"]), res["folds"]
    # grid(2) × test folds(2) parameter evaluations
    assert res["optimizations"] == 4, res["optimizations"]
    # 2 test folds × 2 dates each = 4 OOS trades, all winners (+1)
    assert len(res["oos_trades"]) == 4, len(res["oos_trades"])
    assert all(t.pnl_gross_pct == 1.0 for t in res["oos_trades"])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
