"""
Probability model (Phase 4.2) — L2-regularized logistic regression, fitted by
IRLS (Newton), numpy-only.

Why hand-rolled instead of scikit-learn: the entire platform is deliberately
dependency-light and auditable; logistic-via-IRLS is ~60 lines of linear
algebra, deterministic to the bit, and its artifact serializes to plain JSON
(no pickle — pickles are unauditable and a code-execution risk on load). The
API is sklearn-shaped (fit / predict_proba) so LightGBM/XGBoost can drop in
later WHEN the sample size justifies them — thousands of decisive rows, not
hundreds.

HONESTY CONTRACT:
- This model's outputs become user-visible probabilities ONLY after the
  prob_eval gates pass (calibration + beats the 100-pt score OOS). Until
  then they live in shadow mode: logged as features, never printed.
- Trained on SIMULATION-DERIVED rows (sim_dataset) with optimistic fills and
  missing external context. The model card (saved inside the artifact's meta)
  must carry that disclosure with every artifact.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MODEL_VERSION = 1

# Feature hygiene defaults: a column must be present-and-numeric in this
# fraction of training rows to be used at all (sparser columns would be
# mostly imputation, i.e. noise the model can overfit through).
MIN_COVERAGE = 0.8


def select_numeric_columns(feature_dicts: Sequence[dict],
                           min_coverage: float = MIN_COVERAGE) -> list[str]:
    """Deterministic (sorted) list of usable feature columns: numeric or bool,
    covered in ≥ min_coverage of rows, non-constant. Strings and bookkeeping
    keys are excluded — the model sees numbers only."""
    if not feature_dicts:
        return []
    skip = {"_v", "sim", "engine"}
    counts: dict[str, int] = {}
    values: dict[str, set] = {}
    for f in feature_dicts:
        for k, v in (f or {}).items():
            # shadow_* = the model's OWN earlier output logged on live alerts;
            # training on it would be a feedback loop, not a feature.
            if (k in skip or k.startswith("shadow_") or isinstance(v, str)
                    or v is None):
                continue
            if isinstance(v, bool) or isinstance(v, (int, float)):
                fv = float(v)
                if fv != fv or abs(fv) == float("inf"):
                    continue
                counts[k] = counts.get(k, 0) + 1
                if len(values.setdefault(k, set())) < 3:
                    values[k].add(fv)
    n = len(feature_dicts)
    return sorted(k for k, c in counts.items()
                  if c / n >= min_coverage and len(values.get(k, set())) > 1)


# ---------- model-input transform (versioned; shared by fit AND inference) ----
#
# v1 (2026-10). The first corrected (v2-label) fit put its largest weights on
# artefacts of how the snapshot is laid out, not on setup quality:
#   - RAW RUPEE LEVELS (price, swing high/low, S/R levels, ATR in ₹) are not
#     comparable across a ₹50 and a ₹5,000 stock or across a split — `price`
#     (−0.31) and `p3_pa_last_low` (+0.26) were a near-cancelling collinear
#     pair fitting price bands. Their scale-free twins (atr_pct, risk_pct,
#     dist_atr, dist_*_pct) stay.
#   - bull_conviction / bear_conviction are exact complements and a monotone
#     relabel of score × direction (bull = dp if long else 100 − dp), i.e. a
#     hidden "is this a long" flag split across two columns.
#   - DIRECTIONAL features were raw-signed while longs and shorts are pooled:
#     "far above VWAP" is extension for a long but the wrong side for a short,
#     and a linear model cannot learn that interaction. They are re-expressed
#     relative to the TRADE (dir_* : + = favourable side), with an explicit
#     is_long so any long/short drift is visible instead of smuggled in.
# The version is stored in the artifact; artifacts without it (transform 0)
# keep reading raw features exactly as before.
FEATURE_TRANSFORM_VERSION = 1

_RAW_LEVEL_KEYS = frozenset({
    "price", "p3_pa_last_high", "p3_pa_last_low",
    "p3_pa_nearest_support", "p3_pa_nearest_resistance", "p3_vol_atr",
})
_REDUNDANT_KEYS = frozenset({"bull_conviction", "bear_conviction"})
# Continuous, sign-carrying features: multiplied by the trade sign.
_SIGNED_CONTINUOUS = re.compile(
    r"^(gap_pct|p3_vwap_dist_atr|p3_vwap_slope_pct|"
    r"p3_rs_(rs|stock_ret|index_ret)_\d+)$")
_DIRECTIONAL_OTHER = frozenset({
    "p3_vwap_above", "p3_vwap_time_above_frac", "p3_vwap_reclaim",
    "p3_vwap_failure", "p3_pa_hh", "p3_pa_hl", "p3_pa_lh", "p3_pa_ll",
    "p3_pa_bos", "p3_pa_dist_support_pct", "p3_pa_dist_resistance_pct",
})


def _is_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and abs(v) != float("inf"))


def _flag(v) -> int | None:
    """Boolean-ish feature → 0/1; None stays None (missing ≠ False)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    return int(bool(v)) if _is_num(v) else None


def model_features(features: dict | None,
                   version: int = FEATURE_TRANSFORM_VERSION) -> dict:
    """Decision-time snapshot → model input (see the v1 rationale above).
    Pure and total: never raises, never mutates the caller's dict."""
    f = dict(features or {})
    if version < 1:
        return f
    for k in _RAW_LEVEL_KEYS | _REDUNDANT_KEYS:
        f.pop(k, None)
    d = str(f.get("direction") or "").strip().lower()
    sign = 1 if d in ("long", "buy") else -1 if d in ("short", "sell") else 0
    directional = [k for k in f
                   if _SIGNED_CONTINUOUS.match(k) or k in _DIRECTIONAL_OTHER]
    raw = {k: f.pop(k) for k in directional}
    if not sign:
        return f          # can't orient them → drop, never mis-sign
    long = sign > 0
    f["is_long"] = long

    for k, v in raw.items():
        if _SIGNED_CONTINUOUS.match(k) and _is_num(v):
            f[f"dir_{k}"] = v * sign

    above = _flag(raw.get("p3_vwap_above"))
    if above is not None:
        f["dir_vwap_side"] = above if long else 1 - above
    frac = raw.get("p3_vwap_time_above_frac")
    if _is_num(frac):
        f["dir_vwap_time_with"] = frac if long else 1.0 - frac
    rec, fail = _flag(raw.get("p3_vwap_reclaim")), _flag(raw.get("p3_vwap_failure"))
    if rec is not None and fail is not None:
        f["dir_vwap_flip_with"] = rec if long else fail
        f["dir_vwap_flip_against"] = fail if long else rec

    hh, hl, lh, ll = (_flag(raw.get(f"p3_pa_{k}")) for k in ("hh", "hl", "lh", "ll"))
    if None not in (hh, hl, lh, ll):
        bull, bear = hh + hl, lh + ll
        f["dir_pa_structure_with"] = bull if long else bear
        f["dir_pa_structure_against"] = bear if long else bull
    bos = raw.get("p3_pa_bos")
    if isinstance(bos, str) and bos in ("bullish", "bearish"):
        f["dir_pa_bos"] = 1 if (bos == "bullish") == long else -1
    elif "p3_pa_bos" in raw:
        f["dir_pa_bos"] = 0
    ds, dr = raw.get("p3_pa_dist_support_pct"), raw.get("p3_pa_dist_resistance_pct")
    room, backstop = (dr, ds) if long else (ds, dr)
    if _is_num(room):
        f["dir_pa_room_pct"] = room         # distance to the level in the way
    if _is_num(backstop):
        f["dir_pa_backstop_pct"] = backstop  # distance to the level behind
    return f


def extract_matrix(feature_dicts: Sequence[dict],
                   columns: Sequence[str]) -> np.ndarray:
    """Rows × columns matrix; missing/non-numeric entries become NaN (the
    model standardizes and mean-imputes them — explicitly, not silently)."""
    X = np.full((len(feature_dicts), len(columns)), np.nan)
    for i, f in enumerate(feature_dicts):
        f = f or {}
        for j, col in enumerate(columns):
            v = f.get(col)
            if isinstance(v, bool):
                X[i, j] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                X[i, j] = float(v)
    return X


@dataclass
class LogisticModel:
    """L2 logistic regression with internal standardization + mean-imputation.

    Artifact fields are all JSON-safe; `meta` is the model card (training
    provenance, sample sizes, disclosures)."""
    columns: list[str] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    intercept: float = 0.0
    lam: float = 1.0
    n_train: int = 0
    trained_at: str = ""
    converged: bool = False
    meta: dict = field(default_factory=dict)
    # Model-input transform version (see model_features). 0 = raw snapshot
    # features, the pre-2026-10 behaviour; loaded artifacts keep theirs.
    transform: int = FEATURE_TRANSFORM_VERSION

    # ---------- fitting ----------

    def fit(self, feature_dicts: Sequence[dict], y: Sequence[int],
            lam: float = 1.0, columns: Sequence[str] | None = None,
            max_iter: int = 100, tol: float = 1e-8) -> "LogisticModel":
        """IRLS with L2 penalty (intercept unpenalized). Deterministic."""
        y_arr = np.asarray(list(y), dtype=float)
        if len(feature_dicts) != len(y_arr):
            raise ValueError("X/y length mismatch")
        feature_dicts = [model_features(f, self.transform) for f in feature_dicts]
        if len(y_arr) == 0:
            raise ValueError("empty training set")
        self.columns = list(columns) if columns is not None else \
            select_numeric_columns(feature_dicts)
        if not self.columns:
            raise ValueError("no usable numeric feature columns")
        self.lam = float(lam)

        X = extract_matrix(feature_dicts, self.columns)
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        col_std = X.std(axis=0, ddof=0)
        col_std = np.where(col_std <= 0, 1.0, col_std)
        Xs = (X - col_mean) / col_std

        n, d = Xs.shape
        Xb = np.hstack([np.ones((n, 1)), Xs])
        w = np.zeros(d + 1)
        penalty = np.full(d + 1, self.lam)
        penalty[0] = 0.0                      # never shrink the base rate

        self.converged = False
        for _ in range(max_iter):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
            W = np.clip(p * (1.0 - p), 1e-10, None)
            H = Xb.T @ (Xb * W[:, None]) + np.diag(penalty)
            g = Xb.T @ (y_arr - p) - penalty * w
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, g, rcond=None)[0]
            w = w + delta
            if float(np.max(np.abs(delta))) < tol:
                self.converged = True
                break

        self.intercept = float(w[0])
        self.weights = [float(v) for v in w[1:]]
        self.mean = [float(v) for v in col_mean]
        self.std = [float(v) for v in col_std]
        self.n_train = int(n)
        self.trained_at = datetime.now(timezone.utc).isoformat()
        return self

    # ---------- inference ----------

    def _standardize_one(self, features: dict) -> np.ndarray:
        features = model_features(features, self.transform)
        x = np.zeros(len(self.columns))
        for j, col in enumerate(self.columns):
            v = (features or {}).get(col)
            if isinstance(v, bool):
                fv = 1.0 if v else 0.0
            elif isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                fv = float(v)
            else:
                fv = self.mean[j]             # explicit mean-imputation
            x[j] = (fv - self.mean[j]) / self.std[j]
        return x

    def predict_proba(self, features: dict) -> float:
        """P(win) for one decision-time feature snapshot."""
        z = self.intercept + float(np.dot(self._standardize_one(features),
                                          np.asarray(self.weights)))
        return float(1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z)))))

    def predict_proba_many(self, feature_dicts: Sequence[dict]) -> list[float]:
        return [self.predict_proba(f) for f in feature_dicts]

    def coefficients(self) -> list[tuple[str, float]]:
        """(column, standardized weight), largest |weight| first — the honest
        'importance' read for a linear model."""
        pairs = list(zip(self.columns, self.weights))
        return sorted(pairs, key=lambda kv: -abs(kv[1]))

    # ---------- artifact (JSON, never pickle) ----------

    def to_dict(self) -> dict:
        return {
            "model_version": MODEL_VERSION,
            "kind": "logistic_l2_irls",
            "columns": self.columns,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "intercept": self.intercept,
            "lam": self.lam,
            "n_train": self.n_train,
            "trained_at": self.trained_at,
            "converged": self.converged,
            "feature_transform": self.transform,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True),
                              encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        m = cls()
        m.columns = list(d["columns"])
        m.mean = [float(v) for v in d["mean"]]
        m.std = [float(v) for v in d["std"]]
        m.weights = [float(v) for v in d["weights"]]
        m.intercept = float(d["intercept"])
        m.lam = float(d.get("lam", 1.0))
        m.n_train = int(d.get("n_train", 0))
        m.trained_at = d.get("trained_at", "")
        m.converged = bool(d.get("converged", False))
        m.meta = dict(d.get("meta", {}))
        m.transform = int(d.get("feature_transform", 0))   # absent = legacy raw
        return m

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
