"""
Shadow-mode probability (Phase 4.4).

If a trained artifact exists at models/prob_v1.json, every fired scan alert
gets the model's P(win) logged INTO ITS FEATURE SNAPSHOT — and nowhere else.
Not in the Telegram message, not in the narrative, not in any gate. The point
is to accrue a live record of (predicted probability, realized outcome) pairs
so the model's calibration can be judged against reality before anyone is
allowed to see a probability.

Ship path (unchanged): probabilities surface to users ONLY after
prob_eval's gates pass on adequately-powered data AND the live shadow record
agrees with the sim-trained calibration. Until then this module is
write-only telemetry.

TOTAL: no artifact / corrupt artifact / scoring error ⇒ {} — the alert path
must never notice this module exists.
"""
from __future__ import annotations

import logging
from pathlib import Path

from prob_model import LogisticModel

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "prob_v1.json"

_cached: tuple[float, LogisticModel | None] | None = None   # (mtime, model)


def _load() -> LogisticModel | None:
    """mtime-cached artifact load; retrains land by just replacing the file."""
    global _cached
    try:
        mtime = MODEL_PATH.stat().st_mtime
    except OSError:
        _cached = None
        return None
    if _cached is not None and _cached[0] == mtime:
        return _cached[1]
    try:
        model = LogisticModel.load(MODEL_PATH)
        logger.info("shadow: loaded %s (n_train=%d, gates_passed=%s)",
                    MODEL_PATH.name, model.n_train,
                    model.meta.get("gates_passed"))
    except Exception as e:
        logger.warning("shadow: artifact unreadable (%s) — shadow mode off", e)
        model = None
    _cached = (mtime, model)
    return model


def _reset_cache() -> None:
    """Test hook."""
    global _cached
    _cached = None


def shadow_probability(features: dict | None) -> dict:
    """P(win) for one decision-time snapshot, as mergeable feature keys.
    {} when no model is available — caller merges nothing."""
    if not features:
        return {}
    model = _load()
    if model is None:
        return {}
    try:
        p = model.predict_proba(features)
    except Exception as e:
        logger.warning("shadow: scoring failed: %s", e)
        return {}
    return {
        "shadow_p_win": round(p, 4),
        "shadow_model": model.trained_at[:19] or "unknown",
        "shadow_gates_passed": bool(model.meta.get("gates_passed", False)),
    }
