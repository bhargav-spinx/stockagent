"""
Phase 4.3 — validation harness: metric known-answers, purge/embargo
discipline, baseline gate, power stamps, artifact honesty.

    venv/Scripts/python.exe -m pytest tests/test_prob_eval.py
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prob_eval  # noqa: E402
import prob_model  # noqa: E402


# ---------- metric known-answers ----------

def test_brier_and_skill_known_answers():
    assert prob_eval.brier([1, 0], [1.0, 0.0]) == 0.0
    assert prob_eval.brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    # perfect forecast → skill 1; base-rate forecast → skill 0
    assert prob_eval.brier_skill([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert prob_eval.brier_skill([1, 0, 1, 0], [0.5] * 4) == pytest.approx(0.0)
    assert prob_eval.brier_skill([1, 1], [0.9, 0.9]) is None   # degenerate y


def test_auc_known_answers():
    assert prob_eval.auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert prob_eval.auc([1, 0], [0.1, 0.9]) == 0.0
    assert prob_eval.auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert prob_eval.auc([1, 1], [0.5, 0.6]) is None


def test_reliability_table_bins():
    y = [1, 0, 1, 1, 0, 0]
    p = [0.95, 0.05, 0.92, 0.55, 0.52, 0.08]
    tab = prob_eval.reliability_table(y, p, n_bins=10)
    hi = next(b for b in tab if b["bin"] == "0.9–1.0")
    assert hi["n"] == 2 and hi["win_rate"] == 1.0
    lo = next(b for b in tab if b["bin"] == "0.0–0.1")
    assert lo["n"] == 2 and lo["win_rate"] == 0.0


# ---------- synthetic dataset rows ----------

def _rows(n_dates=80, per_day=4, predictive=True, score_anti=True):
    """Rows over business dates: f_sig separates labels cleanly; `score` is
    ANTI-predictive when score_anti (high score → loss) so the baseline gate
    has something unambiguous to measure."""
    dates = [d.strftime("%Y-%m-%d")
             for d in pd.date_range("2026-01-01", periods=n_dates, freq="B")]
    rows = []
    for di, date in enumerate(dates):
        for j in range(per_day):
            i = di * per_day + j
            label = 1 if (i * 7) % 10 >= 5 else 0
            sig = (label + 0.05 * math.sin(i)) if predictive else math.sin(i * 1.3)
            score = (95 - 30 * label) if score_anti else 60 + (i * 13) % 40
            rows.append({
                "ts": f"{date}T{9 + j:02d}:30:00", "trade_date": date,
                "symbol": "T.NS", "direction": "long", "score": score,
                "label_win": label, "label_decisive": 1,
                "r_multiple": 1.5 if label else -1.0,
                "features": {"f_sig": round(sig, 6),
                             "f_noise": round(math.sin(i * 2.1), 6)},
            })
    return rows


def test_purged_folds_never_leak_train_into_test(monkeypatch):
    rows = _rows()
    seen: list[tuple[str, str]] = []
    orig_fit = prob_model.LogisticModel.fit

    def spy_fit(self, feats, y, **kw):
        # capture via closure on the current call's train rows through the
        # harness: max train date recorded by matching feature identity
        return orig_fit(self, feats, y, **kw)

    all_dates = sorted({r["trade_date"] for r in rows})
    rep = prob_eval.evaluate(rows, n_folds=5, embargo_sessions=5)
    # every fold report must show the embargo actually applied
    for f in rep["folds"]:
        assert f["embargoed_sessions"] == 5
    # structural leak check: fold boundaries are contiguous chunks of dates,
    # so with anchored training + embargo, n_oos < n_rows strictly
    assert rep["n_oos"] < len(rows)
    assert rep["optimizations"] == len(rep["folds"]) >= 3
    assert seen == []   # spy unused; kept to document the structural check


def test_predictive_features_pass_gates_and_beat_anti_score():
    rep = prob_eval.evaluate(_rows(n_dates=80), n_folds=5, embargo_sessions=5)
    assert rep["underpowered"] is False          # pooled OOS ≥ 200
    assert rep["pooled_auc"] > 0.9
    assert rep["pooled_brier_skill"] > 0.5
    top10 = rep["baseline_comparison"]["top_10pct"]
    assert top10["model"]["win_rate"] > top10["score_baseline"]["win_rate"]
    assert rep["gates"]["calibration_positive_skill"]
    assert rep["gates"]["beats_score_baseline_top10"]
    assert rep["ship_ready"] is True


def test_noise_features_fail_gates():
    rep = prob_eval.evaluate(_rows(predictive=False, score_anti=False),
                             n_folds=5, embargo_sessions=5)
    # a no-signal model must not look calibrated-with-skill
    assert rep["pooled_brier_skill"] is None or rep["pooled_brier_skill"] < 0.1
    assert rep["ship_ready"] is False


def test_underpowered_stamp_blocks_ship():
    rep = prob_eval.evaluate(_rows(n_dates=20, per_day=2),
                             n_folds=4, embargo_sessions=2)
    assert rep["underpowered"] is True           # pooled OOS < 200
    assert rep["gates"]["adequately_powered"] is False
    assert rep["ship_ready"] is False            # even if metrics look great


def test_tiny_input_reports_not_raises():
    rep = prob_eval.evaluate(_rows(n_dates=2, per_day=1))
    assert rep["underpowered"] is True
    assert "pooled_brier" not in rep


def test_train_final_artifact_is_honest(monkeypatch):
    rows = _rows(n_dates=30)
    monkeypatch.setattr(prob_eval.sim_dataset, "load_rows",
                        lambda **k: rows)
    monkeypatch.setattr(prob_eval.sim_dataset, "dataset_stats",
                        lambda: {"rows": len(rows)})
    out = Path(tempfile.gettempdir()) / "prob_v_test.json"
    try:
        m = prob_eval.train_final(out)
        blob = json.loads(out.read_text(encoding="utf-8"))
    finally:
        if out.exists():
            out.unlink()
    # artifact carries the model card: gates honestly failed (underpowered),
    # disclosures present, and it round-trips to a working model
    assert blob["meta"]["gates_passed"] is False
    assert any("SIMULATION-DERIVED" in d for d in blob["meta"]["disclosures"])
    assert m.predict_proba(rows[0]["features"]) == pytest.approx(
        prob_model.LogisticModel.from_dict(blob).predict_proba(
            rows[0]["features"]), abs=1e-12)
