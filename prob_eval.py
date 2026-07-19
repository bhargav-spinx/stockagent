"""
Probability-model validation harness (Phase 4.3) — the referee that decides
whether model probabilities may ever leave shadow mode.

Runs purged walk-forward CV over the simulated-trade dataset and reports:
  • CALIBRATION  — reliability table + Brier score vs the base-rate Brier
                   (skill score). A model that says 60% must win ~60% of the
                   time, or its probabilities don't ship. Headline metric.
  • DISCRIMINATION — AUC (rank-based).
  • BASELINE GATE — the model must BEAT ranking by the existing 100-pt score
                   on the same OOS rows (top-K realized win rate / mean R by
                   model-p vs by score). If it can't out-rank the hand-built
                   score, the score stays in production and the model stays
                   research — a likely and acceptable outcome.
  • POWER        — everything is stamped UNDERPOWERED below a minimum pooled
                   OOS sample; underpowered numbers are provisional, period.

Purging: folds are contiguous in time (anchored train → later test), and an
EMBARGO of N sessions is dropped from the train side before each test fold —
features carry multi-day lookbacks (ATR percentile, RS windows, 52-week), so
adjacent sessions leak; a random shuffle split would be flatly dishonest here.

CLI:
    python prob_eval.py evaluate [--folds 5 --embargo 5 --lam 1.0
                                  --decisive-only] [--min-score 60]
    python prob_eval.py train --out models/prob_v1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import sim_dataset
from prob_model import LogisticModel, select_numeric_columns

logger = logging.getLogger(__name__)

MIN_POOLED_OOS = 200        # below this, every number is provisional
DEFAULT_FOLDS = 5
DEFAULT_EMBARGO_SESSIONS = 5
DEFAULT_LAM = 1.0
TOP_FRACS = (0.1, 0.2)      # top-K slices compared against the score baseline

# Gate floors — conservative luck-margins, not tuning knobs. A pure-noise
# model scores ~0 skill / ~0.5 AUC / coin-flip baseline comparisons, and can
# clear a bare ">" by luck; these floors make "passes the gate" mean something.
MIN_BRIER_SKILL = 0.02      # calibration must beat base-rate by a margin
MIN_AUC = 0.55              # must actually rank winners above losers
MIN_BASELINE_EDGE = 0.02    # must beat the score's top-10% win rate by ≥2pts


# ---------- metrics (numpy-only, deterministic) ----------

def brier(y: Sequence[int], p: Sequence[float]) -> float:
    y_arr, p_arr = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p_arr - y_arr) ** 2))


def brier_skill(y: Sequence[int], p: Sequence[float]) -> float | None:
    """1 − BS/BS_base where BS_base predicts the (in-sample OOS) base rate.
    >0 = better than knowing only the base rate; ≤0 = the probabilities add
    nothing. None on degenerate y."""
    y_arr = np.asarray(y, float)
    base = float(y_arr.mean())
    bs_base = float(np.mean((base - y_arr) ** 2))
    if bs_base == 0:
        return None
    return 1.0 - brier(y, p) / bs_base


def _rank_avg_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks over ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def auc(y: Sequence[int], p: Sequence[float]) -> float | None:
    """Mann–Whitney AUC with tie handling. None when y is one-class."""
    y_arr, p_arr = np.asarray(y, float), np.asarray(p, float)
    n1 = int(y_arr.sum())
    n0 = len(y_arr) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = _rank_avg_ties(p_arr)
    return float((r[y_arr == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability_table(y: Sequence[int], p: Sequence[float],
                      n_bins: int = 10) -> list[dict]:
    """Equal-width probability bins → (range, n, mean_p, win_rate). Empty bins
    are omitted; with small n use fewer bins upstream."""
    y_arr, p_arr = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_arr >= lo) & (p_arr < hi) if hi < 1.0 else \
            (p_arr >= lo) & (p_arr <= 1.0)
        if not m.any():
            continue
        out.append({"bin": f"{lo:.1f}–{hi:.1f}", "n": int(m.sum()),
                    "mean_p": round(float(p_arr[m].mean()), 4),
                    "win_rate": round(float(y_arr[m].mean()), 4)})
    return out


def _top_k_stats(rows: list[dict], key_values: Sequence[float],
                 frac: float) -> dict:
    """Realized win rate / mean R of the top `frac` rows ranked by key_values
    (desc). Deterministic tie-break by row ts."""
    k = max(1, int(len(rows) * frac))
    order = sorted(range(len(rows)),
                   key=lambda i: (-float(key_values[i]), rows[i]["ts"]))
    top = [rows[i] for i in order[:k]]
    wins = sum(r["label_win"] for r in top)
    rs = [r["r_multiple"] for r in top if r["r_multiple"] is not None]
    return {"k": k, "win_rate": round(wins / k, 4),
            "mean_r": round(float(np.mean(rs)), 4) if rs else None}


# ---------- purged walk-forward ----------

def _date_folds(rows: list[dict], n_folds: int) -> list[list[str]]:
    dates = sorted({r["trade_date"] for r in rows})
    n_folds = max(2, min(n_folds, len(dates)))
    size = len(dates) / n_folds
    return [dates[round(i * size):round((i + 1) * size)] for i in range(n_folds)]


def evaluate(rows: list[dict] | None = None, *, n_folds: int = DEFAULT_FOLDS,
             embargo_sessions: int = DEFAULT_EMBARGO_SESSIONS,
             lam: float = DEFAULT_LAM, decisive_only: bool = False,
             min_score: int | None = None) -> dict[str, Any]:
    """Purged anchored walk-forward. Returns the full report dict; the CLI
    pretty-prints it. Never raises on thin data — reports UNDERPOWERED."""
    if rows is None:
        rows = sim_dataset.load_rows(min_score=min_score,
                                     decisive_only=decisive_only)
    if len(rows) < 10:
        return {"underpowered": True, "n_rows": len(rows),
                "note": "fewer than 10 rows — generate the dataset first "
                        "(python sim_dataset.py generate …)"}

    all_dates = sorted({r["trade_date"] for r in rows})
    folds = _date_folds(rows, n_folds)

    oos_y: list[int] = []
    oos_p: list[float] = []
    oos_rows: list[dict] = []
    fold_reports = []
    optimizations = 0

    for f_idx in range(1, len(folds)):
        test_dates = set(folds[f_idx])
        test_start_pos = all_dates.index(folds[f_idx][0])
        embargo_cut = max(0, test_start_pos - embargo_sessions)
        train_dates = set(all_dates[:embargo_cut])
        train = [r for r in rows if r["trade_date"] in train_dates]
        test = [r for r in rows if r["trade_date"] in test_dates]
        if len(train) < 20 or len(test) < 5:
            continue
        cols = select_numeric_columns([r["features"] for r in train])
        if not cols:
            continue
        model = LogisticModel().fit([r["features"] for r in train],
                                    [r["label_win"] for r in train],
                                    lam=lam, columns=cols)
        optimizations += 1
        p = model.predict_proba_many([r["features"] for r in test])
        y = [r["label_win"] for r in test]
        oos_y.extend(y)
        oos_p.extend(p)
        oos_rows.extend(test)
        fold_reports.append({
            "fold": f_idx, "n_train": len(train), "n_test": len(test),
            "embargoed_sessions": test_start_pos - embargo_cut,
            "auc": auc(y, p), "brier_skill": brier_skill(y, p),
        })

    if not oos_rows:
        return {"underpowered": True, "n_rows": len(rows),
                "note": "no evaluable folds (too few dates/rows per fold)"}

    underpowered = len(oos_rows) < MIN_POOLED_OOS
    base_rate = float(np.mean(oos_y))

    baseline = {}
    for frac in TOP_FRACS:
        by_model = _top_k_stats(oos_rows, oos_p, frac)
        by_score = _top_k_stats(oos_rows,
                                [float(r["score"]) for r in oos_rows], frac)
        baseline[f"top_{int(frac * 100)}pct"] = {
            "model": by_model, "score_baseline": by_score,
            "model_beats_score": (by_model["win_rate"]
                                  >= by_score["win_rate"] + MIN_BASELINE_EDGE),
        }

    pooled_bss = brier_skill(oos_y, oos_p)
    pooled_auc = auc(oos_y, oos_p)
    n_bins = 10 if len(oos_rows) >= 200 else 5
    gates = {
        "calibration_positive_skill": bool(
            pooled_bss is not None and pooled_bss >= MIN_BRIER_SKILL),
        "discriminates": bool(pooled_auc is not None and pooled_auc >= MIN_AUC),
        "beats_score_baseline_top10": bool(
            baseline.get("top_10pct", {}).get("model_beats_score", False)),
        "adequately_powered": not underpowered,
    }
    return {
        "underpowered": underpowered,
        "n_rows": len(rows),
        "n_oos": len(oos_rows),
        "oos_base_rate": round(base_rate, 4),
        "pooled_brier": round(brier(oos_y, oos_p), 5),
        "pooled_brier_skill": (round(pooled_bss, 4)
                               if pooled_bss is not None else None),
        "pooled_auc": round(pooled_auc, 4) if pooled_auc is not None else None,
        "reliability": reliability_table(oos_y, oos_p, n_bins=n_bins),
        "baseline_comparison": baseline,
        "folds": fold_reports,
        "optimizations": optimizations,
        "gates": gates,
        "ship_ready": all(gates.values()),
    }


# ---------- final artifact ----------

def train_final(out_path: str | Path, *, lam: float = DEFAULT_LAM,
                decisive_only: bool = False,
                min_score: int | None = None) -> LogisticModel:
    """Fit on ALL rows and save the JSON artifact for SHADOW MODE. The eval
    report is embedded as the model card; gates_passed records honestly
    whether this artifact is cleared to ever surface probabilities (it is not,
    until evaluate() says ship_ready on adequate data)."""
    rows = sim_dataset.load_rows(min_score=min_score,
                                 decisive_only=decisive_only)
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} rows — not enough to fit anything "
                         "worth saving. Generate the dataset first.")
    report = evaluate(rows, lam=lam)
    model = LogisticModel().fit([r["features"] for r in rows],
                                [r["label_win"] for r in rows], lam=lam)
    model.meta = {
        "dataset": sim_dataset.dataset_stats(),
        "eval": {k: report.get(k) for k in
                 ("n_oos", "oos_base_rate", "pooled_brier",
                  "pooled_brier_skill", "pooled_auc", "gates",
                  "ship_ready", "underpowered")},
        "gates_passed": bool(report.get("ship_ready")),
        "disclosures": [
            "trained on SIMULATION-DERIVED trades (optimistic fills; no "
            "delivery/earnings/regime context)",
            "shadow-mode only until prob_eval gates pass on adequately "
            "powered data AND live calibration agrees",
        ],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return model


# ---------- CLI ----------

def _print_report(rep: dict) -> None:
    if rep.get("underpowered") and "pooled_brier" not in rep:
        print(f"UNDERPOWERED: {rep.get('note', '')} (rows={rep.get('n_rows')})")
        return
    stamp = "⚠ UNDERPOWERED — provisional numbers" if rep["underpowered"] else ""
    print(f"PROBABILITY-MODEL EVALUATION (purged walk-forward)  {stamp}")
    print(f"rows={rep['n_rows']}  pooled OOS={rep['n_oos']}  "
          f"base rate={rep['oos_base_rate']:.2%}")
    print(f"Brier {rep['pooled_brier']}  skill {rep['pooled_brier_skill']}  "
          f"AUC {rep['pooled_auc']}  ({rep['optimizations']} fits)")
    print("\nReliability (predicted → realized):")
    for b in rep["reliability"]:
        print(f"  {b['bin']}: n={b['n']:<5} mean_p={b['mean_p']:.2f} "
              f"→ win {b['win_rate']:.2%}")
    print("\nBaseline (model ranking vs 100-pt score ranking, same OOS rows):")
    for k, v in rep["baseline_comparison"].items():
        m, s = v["model"], v["score_baseline"]
        verdict = "BEATS" if v["model_beats_score"] else "does NOT beat"
        print(f"  {k}: model win {m['win_rate']:.2%} (R {m['mean_r']}) vs "
              f"score win {s['win_rate']:.2%} (R {s['mean_r']}) — {verdict}")
    print("\nGates:", json.dumps(rep["gates"]))
    print("SHIP-READY" if rep["ship_ready"] else
          "NOT ship-ready — probabilities stay in shadow mode.")


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ev.add_argument("--embargo", type=int, default=DEFAULT_EMBARGO_SESSIONS)
    ev.add_argument("--lam", type=float, default=DEFAULT_LAM)
    ev.add_argument("--decisive-only", action="store_true")
    ev.add_argument("--min-score", type=int, default=None)
    tr = sub.add_parser("train")
    tr.add_argument("--out", default="models/prob_v1.json")
    tr.add_argument("--lam", type=float, default=DEFAULT_LAM)
    tr.add_argument("--decisive-only", action="store_true")
    tr.add_argument("--min-score", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "evaluate":
        _print_report(evaluate(n_folds=args.folds,
                               embargo_sessions=args.embargo, lam=args.lam,
                               decisive_only=args.decisive_only,
                               min_score=args.min_score))
    else:
        m = train_final(args.out, lam=args.lam,
                        decisive_only=args.decisive_only,
                        min_score=args.min_score)
        print(f"artifact: {args.out}  (n_train={m.n_train}, "
              f"gates_passed={m.meta['gates_passed']})")
        print("Top standardized coefficients:")
        for col, w in m.coefficients()[:10]:
            print(f"  {col:<28} {w:+.4f}")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    _cli()
