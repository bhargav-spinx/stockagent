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
            if k in skip or isinstance(v, str) or v is None:
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

    # ---------- fitting ----------

    def fit(self, feature_dicts: Sequence[dict], y: Sequence[int],
            lam: float = 1.0, columns: Sequence[str] | None = None,
            max_iter: int = 100, tol: float = 1e-8) -> "LogisticModel":
        """IRLS with L2 penalty (intercept unpenalized). Deterministic."""
        y_arr = np.asarray(list(y), dtype=float)
        if len(feature_dicts) != len(y_arr):
            raise ValueError("X/y length mismatch")
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
        return m

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
