"""
Phase 4.4 — shadow mode: write-only telemetry, totally silent when absent.

    venv/Scripts/python.exe -m pytest tests/test_shadow.py
"""
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shadow  # noqa: E402
from prob_model import LogisticModel  # noqa: E402


def _artifact(tmp_name: str, gates_passed: bool = False,
              flip: bool = False) -> Path:
    rows, y = [], []
    for i in range(200):
        label = 1 if (i * 7) % 10 >= 5 else 0
        val = float(label if not flip else 1 - label) + 0.05 * math.sin(i)
        rows.append({"f_sig": val})
        y.append(label)
    m = LogisticModel().fit(rows, y, lam=0.5)
    m.meta = {"gates_passed": gates_passed}
    p = Path(tempfile.gettempdir()) / tmp_name
    m.save(p)
    return p


@pytest.fixture(autouse=True)
def _fresh_cache():
    shadow._reset_cache()
    yield
    shadow._reset_cache()


def test_missing_artifact_is_silent(monkeypatch):
    monkeypatch.setattr(shadow, "MODEL_PATH",
                        Path(tempfile.gettempdir()) / "no_such_model.json")
    assert shadow.shadow_probability({"f_sig": 1.0}) == {}
    assert shadow.shadow_probability(None) == {}
    assert shadow.shadow_probability({}) == {}


def test_corrupt_artifact_is_silent(monkeypatch):
    p = Path(tempfile.gettempdir()) / "shadow_corrupt.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(shadow, "MODEL_PATH", p)
    try:
        assert shadow.shadow_probability({"f_sig": 1.0}) == {}
    finally:
        p.unlink()


def test_probability_logged_with_honest_gate_flag(monkeypatch):
    p = _artifact("shadow_ok.json", gates_passed=False)
    monkeypatch.setattr(shadow, "MODEL_PATH", p)
    try:
        out_hi = shadow.shadow_probability({"f_sig": 1.0})
        out_lo = shadow.shadow_probability({"f_sig": 0.0})
    finally:
        p.unlink()
    assert 0.0 <= out_lo["shadow_p_win"] < out_hi["shadow_p_win"] <= 1.0
    assert out_hi["shadow_gates_passed"] is False       # honesty flag rides along
    assert out_hi["shadow_model"]                        # provenance stamp
    # keys are feature-safe scalars (mergeable into the snapshot JSON)
    assert all(isinstance(v, (int, float, bool, str))
               for v in out_hi.values())


def test_artifact_replacement_reloads_via_mtime(monkeypatch):
    p = _artifact("shadow_swap.json")
    monkeypatch.setattr(shadow, "MODEL_PATH", p)
    try:
        p1 = shadow.shadow_probability({"f_sig": 1.0})["shadow_p_win"]
        # retrain with flipped signal → replacing the file must flip predictions
        flipped = _artifact("shadow_swap.json", flip=True)
        os.utime(flipped, (time.time() + 5, time.time() + 5))   # force new mtime
        p2 = shadow.shadow_probability({"f_sig": 1.0})["shadow_p_win"]
    finally:
        p.unlink()
    assert p1 > 0.5 > p2                                # reload actually happened


def test_never_leaks_into_messages():
    """The contract 'never printed' is enforced by construction: bot.py merges
    shadow keys into the DB snapshot only. Guard the construction — no message
    formatter may reference shadow keys."""
    root = Path(__file__).resolve().parent.parent
    for fname in ("bot.py", "narrative.py", "intraday_score.py", "eod_report.py"):
        src = (root / fname).read_text(encoding="utf-8")
        for line in src.splitlines():
            if "shadow_p_win" in line and any(
                    tok in line for tok in ("msg", "reply_text", "format", "print(")):
                raise AssertionError(
                    f"{fname}: shadow probability appears in a message path: "
                    f"{line.strip()}")
