"""
Known-answer tests for riskmetrics — including the Phase-1 additions
(profit factor, max drawdown, recovery factor, bootstrap CI, monthly buckets).
Stdlib + pytest only.

    venv/Scripts/python.exe -m pytest tests/test_riskmetrics.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import riskmetrics as rm  # noqa: E402


def test_profit_factor_known_answer():
    # wins 2+3=5, losses |-1-2|=3
    assert rm.profit_factor([2, -1, 3, -2]) == pytest.approx(5 / 3)


def test_profit_factor_undefined_cases():
    assert rm.profit_factor([]) is None
    assert rm.profit_factor([1.0, 2.0]) is None      # no losses → undefined, not inf


def test_max_drawdown_known_answer():
    # cum: 1, -1, -2, 1 → peaks: 1,1,1,1 → drawdowns: 0,2,3,0
    assert rm.max_drawdown([1, -2, -1, 3]) == pytest.approx(3.0)
    assert rm.max_drawdown([1, 1, 1]) == pytest.approx(0.0)   # never below peak
    assert rm.max_drawdown([]) is None


def test_recovery_factor():
    assert rm.recovery_factor([1, -2, -1, 3]) == pytest.approx(1.0 / 3.0)
    assert rm.recovery_factor([1, 1]) is None       # zero drawdown → undefined
    assert rm.recovery_factor([]) is None


def test_bootstrap_ci_deterministic_and_sane():
    series = [1.2, -0.4, 0.9, 2.1, -0.8, 1.5, 0.3, -0.2, 1.1, 0.7]
    a = rm.bootstrap_ci95(series, n_boot=500, seed=7)
    b = rm.bootstrap_ci95(series, n_boot=500, seed=7)
    assert a == b                                    # same seed → same interval
    lo, hi = a
    assert lo < rm._mean(series) < hi                # mean inside its own CI
    assert rm.bootstrap_ci95([1.0]) is None          # n<2 → None


def test_bootstrap_tighter_than_wide_for_consistent_series():
    # A strongly one-sided series should have a CI that excludes 0.
    lo, hi = rm.bootstrap_ci95([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.05],
                               n_boot=1000, seed=3)
    assert lo > 0


def test_monthly_returns_buckets():
    rows = [
        ("2026-06-02T10:00:00", 1.0),
        ("2026-06-15T10:00:00", -0.5),
        ("2026-07-01T10:00:00", 2.0),
        ("garbage", 99.0),          # skipped, not guessed
        (None, 5.0),                # skipped
    ]
    out = rm.monthly_returns(rows)
    assert list(out.keys()) == ["2026-06", "2026-07"]
    assert out["2026-06"] == {"n": 2, "net": 0.5}
    assert out["2026-07"] == {"n": 1, "net": 2.0}


def test_summarize_includes_new_keys():
    s = rm.summarize([2, -1, 3, -2])
    assert s["profit_factor"] == pytest.approx(5 / 3)
    assert s["max_drawdown"] is not None
    assert "recovery_factor" in s
    # Degenerate input stays safe.
    s0 = rm.summarize([])
    assert s0["profit_factor"] is None and s0["max_drawdown"] is None


def test_format_line_still_safe_on_small_n():
    assert "too few trades" in rm.format_line([0.5])
