"""
Phase 4.2 — logistic model: correctness, determinism, hygiene.

    venv/Scripts/python.exe -m pytest tests/test_prob_model.py
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prob_model import LogisticModel, select_numeric_columns, extract_matrix  # noqa: E402


def _toy(n=400, signal_col="f_sig", seed_stride=7):
    """Deterministic toy set: f_sig separates classes with noise-free margin
    plus a pure-noise feature and a string feature (must be ignored)."""
    rows, y = [], []
    for i in range(n):
        label = 1 if (i * seed_stride) % 10 >= 5 else 0
        sig = 1.0 + 0.5 * label + 0.05 * math.sin(i)     # shifted by class
        noise = math.sin(i * 1.7)
        rows.append({signal_col: sig, "f_noise": noise, "name": "x",
                     "f_bool": (i % 2 == 0)})
        y.append(label)
    return rows, y


def test_column_selection_hygiene():
    rows, _ = _toy(50)
    rows[0]["f_sparse"] = 1.0                 # 2% coverage → dropped
    rows[0]["f_const"] = 5.0
    for r in rows[1:]:
        r["f_const"] = 5.0                    # constant → dropped
        r["f_nan"] = float("nan")             # non-finite → never counted
    cols = select_numeric_columns(rows)
    assert "f_sig" in cols and "f_noise" in cols and "f_bool" in cols
    assert "name" not in cols                 # string
    assert "f_sparse" not in cols and "f_const" not in cols
    assert "f_nan" not in cols
    assert cols == sorted(cols)               # deterministic order


def test_fit_separates_and_is_calibrated_on_train_marginal():
    rows, y = _toy(400)
    m = LogisticModel().fit(rows, y, lam=0.5)
    assert m.converged
    p = m.predict_proba_many(rows)
    # discriminates: mean p among winners >> among losers
    p_win = sum(pi for pi, yi in zip(p, y) if yi) / sum(y)
    p_loss = sum(pi for pi, yi in zip(p, y) if not yi) / (len(y) - sum(y))
    assert p_win - p_loss > 0.3
    # marginal calibration: mean predicted ≈ base rate (IRLS property)
    assert sum(p) / len(p) == pytest.approx(sum(y) / len(y), abs=0.02)
    # the signal feature dominates the noise feature
    coef = dict(m.coefficients())
    assert abs(coef["f_sig"]) > 3 * abs(coef["f_noise"])


def test_regularization_shrinks_toward_base_rate():
    rows, y = _toy(200)
    p_small = LogisticModel().fit(rows, y, lam=0.01).predict_proba(rows[0])
    p_huge = LogisticModel().fit(rows, y, lam=1e6).predict_proba(rows[0])
    base = sum(y) / len(y)
    # huge lambda → prediction pinned to the base rate; small lambda → away from it
    assert abs(p_huge - base) < 0.01
    assert abs(p_small - base) > abs(p_huge - base)


def test_scale_invariance_via_standardization():
    rows, y = _toy(300)
    scaled = [{**r, "f_sig": r["f_sig"] * 1000.0} for r in rows]
    p1 = LogisticModel().fit(rows, y, lam=1.0).predict_proba_many(rows)
    p2 = LogisticModel().fit(scaled, y, lam=1.0).predict_proba_many(scaled)
    assert p1 == pytest.approx(p2, abs=1e-9)


def test_missing_feature_imputes_to_mean_not_zero():
    rows, y = _toy(300)
    m = LogisticModel().fit(rows, y, lam=1.0)
    # a snapshot missing f_sig entirely → imputed at train mean → z gets no
    # contribution from f_sig (standardized 0), NOT raw-zero (which would be
    # far below the ~1.25 train mean and wildly bearish)
    p_missing = m.predict_proba({"f_noise": 0.0, "f_bool": True})
    p_mean = m.predict_proba({"f_sig": sum(r["f_sig"] for r in rows) / len(rows),
                              "f_noise": 0.0, "f_bool": True})
    assert p_missing == pytest.approx(p_mean, abs=1e-6)


def test_determinism_and_json_round_trip():
    rows, y = _toy(250)
    m1 = LogisticModel().fit(rows, y, lam=1.0)
    m2 = LogisticModel().fit(rows, y, lam=1.0)
    assert m1.weights == m2.weights and m1.intercept == m2.intercept

    p = Path(tempfile.gettempdir()) / "prob_model_test.json"
    m1.meta = {"note": "test artifact"}
    m1.save(p)
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        assert blob["kind"] == "logistic_l2_irls"     # JSON, not pickle
        m3 = LogisticModel.load(p)
    finally:
        p.unlink()
    assert m3.predict_proba_many(rows) == pytest.approx(
        m1.predict_proba_many(rows), abs=1e-12)
    assert m3.meta["note"] == "test artifact"


def test_degenerate_inputs_raise_cleanly():
    with pytest.raises(ValueError):
        LogisticModel().fit([], [])
    with pytest.raises(ValueError):
        LogisticModel().fit([{"only_str": "x"}], [1])
    with pytest.raises(ValueError):
        LogisticModel().fit([{"a": 1.0}], [1, 0])
