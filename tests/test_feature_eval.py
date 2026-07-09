"""
Feature evaluation harness (Phase 3.1). Known-answer tests on synthetic
(feature, outcome) rows: a perfectly predictive feature scores IC≈+1 with a
monotone quintile ladder; noise scores ~0; the honesty guards (underpowered
flag, empty-data message) fire correctly.

    venv/Scripts/python.exe -m pytest tests/test_feature_eval.py
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_eval as fe  # noqa: E402


def _row(feat_val, pnl, entry=100.0, stop=98.0, extra=None):
    f = {"feat": feat_val}
    if extra:
        f.update(extra)
    return {"features": f, "pnl_pct": pnl, "entry": entry, "stop_loss": stop}


def test_information_coefficient_extremes():
    x = list(range(20))
    assert fe.information_coefficient(x, x) == pytest.approx(1.0)      # perfect +
    assert fe.information_coefficient(x, x[::-1]) == pytest.approx(-1.0)  # perfect −
    assert fe.information_coefficient([1, 1, 1, 1], [1, 2, 3, 4]) is None  # zero var
    assert fe.information_coefficient([1, 2], [1, 2]) is None          # <3


def test_average_ranks_ties():
    # values 10,10,20 → ranks 1.5,1.5,3
    assert fe._average_ranks([10, 10, 20]) == [1.5, 1.5, 3.0]


def test_predictive_feature_scores_high_and_monotone():
    # outcome increases with feature → IC≈+1, monotone quintiles, positive spread
    rows = [_row(i, pnl=(i - 15) * 0.2) for i in range(30)]
    rep = fe.feature_report(rows, outcome_key="pnl_pct")
    sc = rep["features"]["feat"]
    assert sc["n"] == 30
    assert sc["ic"] > 0.95
    assert sc["monotone_quintiles"] is True
    assert sc["top_minus_bottom"] > 0
    assert sc["underpowered"] is False


def test_noise_feature_scores_near_zero():
    # feature and outcome unrelated (deterministic but decorrelated)
    rows = []
    for i in range(40):
        feat = math.sin(i * 1.7)
        pnl = math.cos(i * 0.3)     # different frequency → ~uncorrelated ranks
        rows.append(_row(feat, pnl))
    rep = fe.feature_report(rows, outcome_key="pnl_pct")
    sc = rep["features"]["feat"]
    assert abs(sc["ic"]) < 0.5      # not a strong separator


def test_underpowered_flag_and_report_text():
    rows = [_row(i, pnl=i * 0.1) for i in range(15)]   # < MIN_SAMPLES
    rep = fe.feature_report(rows)
    assert rep["features"]["feat"]["underpowered"] is True
    txt = fe.format_report(rows)
    assert "UNDERPOWERED" in txt


def test_empty_and_no_numeric_features():
    assert "No numeric features" in fe.format_report([])
    # only boolean/string features → excluded from numeric scoring
    rows = [{"features": {"flag": True, "label": "up"}, "pnl_pct": 1.0,
             "entry": 100, "stop_loss": 98} for _ in range(20)]
    rep = fe.feature_report(rows)
    assert rep["features"] == {}


def test_R_outcome_conversion():
    # entry100/stop98 → risk 2% ; pnl 4% → 2R. Feature ranks with R.
    rows = [_row(i, pnl=(i - 10) * 0.4, entry=100.0, stop=98.0) for i in range(20)]
    rep = fe.feature_report(rows, outcome_key="R")
    sc = rep["features"]["feat"]
    assert sc["ic"] > 0.95          # monotone in feat by construction


def test_stability_sign_agreement():
    rows = [_row(i, pnl=(i - 15) * 0.2) for i in range(30)]
    st = fe.stability(rows, "feat", "pnl_pct")
    assert st["ic_first_half"] is not None and st["ic_second_half"] is not None
    assert st["sign_stable"] is True


def test_correlation_matrix_detects_redundancy():
    rows = [_row(i, pnl=0.0, extra={"feat2": i * 2.0 + 1}) for i in range(20)]
    cm = fe.correlation_matrix(rows, ["feat", "feat2"])
    key = "feat|feat2" if "feat|feat2" in cm else "feat2|feat"
    assert cm[key] == pytest.approx(1.0, abs=1e-6)   # perfectly collinear
