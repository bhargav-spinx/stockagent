"""
Feature evaluation harness (Phase 3.1) — the referee for every feature engine.

The discipline this enforces: no feature may influence live decisions until it
demonstrably SEPARATES OUTCOMES on real, resolved trades. This module reads the
(features @ decision-time, resolved outcome) pairs that Phase 1 collects
(subscriptions.get_training_rows) and, per feature, reports:

  • Information coefficient — Spearman rank correlation between the feature
    value and the trade outcome. |IC| is the headline: does the feature rank
    trades in outcome order at all?
  • Quintile table — mean outcome and hit-rate in each fifth of the feature's
    range. Monotone quintiles are what a genuinely predictive feature looks
    like; a good IC with non-monotone quintiles is usually an artefact.
  • Top-minus-bottom spread — the mean-outcome gap between the best and worst
    quintile, the practically-useful effect size.
  • Stability — IC recomputed on the first vs. second time-half. A feature that
    flips sign across halves is noise dressed up as signal.
  • Correlation matrix — pairwise feature correlation, to catch redundancy
    before it inflates a future model.

HONESTY GUARDS baked in: everything is reported WITH its sample size, and
`MIN_SAMPLES` gates the verdict — below it the harness says "underpowered"
rather than pretending an IC on n=6 means anything. Stdlib-only, deterministic;
safe to run any time (it just has little to say until trades resolve).

    python feature_eval.py                 # report over all resolved+featured trades
    python feature_eval.py --outcome R     # use R-multiples instead of pnl_pct
"""
from __future__ import annotations

import math
from typing import Sequence

# Below this many resolved trades, per-feature statistics are noise. We still
# print them (transparency) but stamp the verdict "underpowered".
MIN_SAMPLES = 20
# Per-feature: need at least this many non-null (value, outcome) pairs to score.
MIN_FEATURE_PAIRS = 12


# ---------------------------------------------------------------------------
# Rank / correlation primitives (stdlib, tie-aware).
# ---------------------------------------------------------------------------

def _average_ranks(xs: Sequence[float]) -> list[float]:
    """1-based average ranks, ties share the mean of their rank span."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # mean of ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def information_coefficient(values: Sequence[float],
                            outcomes: Sequence[float]) -> float | None:
    """Spearman rank correlation (Pearson on average ranks). None when <3
    pairs or either side has zero variance."""
    if len(values) != len(outcomes) or len(values) < 3:
        return None
    return _pearson(_average_ranks(values), _average_ranks(outcomes))


# ---------------------------------------------------------------------------
# Per-feature evaluation.
# ---------------------------------------------------------------------------

def quintile_table(values: Sequence[float], outcomes: Sequence[float],
                   q: int = 5) -> list[dict]:
    """Split the (value, outcome) pairs into `q` equal-count buckets ordered by
    value; per bucket report n, mean outcome, and hit-rate (outcome > 0)."""
    pairs = sorted(zip(values, outcomes), key=lambda p: p[0])
    n = len(pairs)
    if n < q:
        q = max(1, n)
    out = []
    for b in range(q):
        lo = round(b * n / q)
        hi = round((b + 1) * n / q)
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        outs = [o for _v, o in chunk]
        out.append({
            "bucket": b + 1,
            "n": len(chunk),
            "value_lo": round(chunk[0][0], 6),
            "value_hi": round(chunk[-1][0], 6),
            "mean_outcome": round(sum(outs) / len(outs), 4),
            "hit_rate": round(sum(1 for o in outs if o > 0) / len(outs), 4),
        })
    return out


def evaluate_feature(values: Sequence[float],
                     outcomes: Sequence[float]) -> dict:
    """Full single-feature scorecard. Assumes value/outcome already paired and
    non-null (see feature_report which does the extraction)."""
    n = len(values)
    ic = information_coefficient(values, outcomes)
    quints = quintile_table(values, outcomes)
    spread = None
    if len(quints) >= 2:
        spread = round(quints[-1]["mean_outcome"] - quints[0]["mean_outcome"], 4)
    # Monotone if bucket mean outcomes are (weakly) sorted one way.
    means = [b["mean_outcome"] for b in quints]
    monotone = (all(a <= b for a, b in zip(means, means[1:]))
                or all(a >= b for a, b in zip(means, means[1:]))) if len(means) >= 2 else None
    return {
        "n": n,
        "ic": round(ic, 4) if ic is not None else None,
        "top_minus_bottom": spread,
        "monotone_quintiles": monotone,
        "quintiles": quints,
        "underpowered": n < MIN_SAMPLES,
    }


# ---------------------------------------------------------------------------
# Row plumbing — extract (feature, outcome) series from training rows.
# ---------------------------------------------------------------------------

def _row_outcome(row: dict, outcome_key: str) -> float | None:
    """Map a training row to a scalar outcome. 'R' converts pnl_pct to an
    R-multiple via the alert's own risk (|entry−stop|/entry); other keys read
    the column directly (pnl_pct / mfe_pct / mae_pct)."""
    if outcome_key == "R":
        pnl = row.get("pnl_pct")
        entry, stop = row.get("entry"), row.get("stop_loss")
        try:
            risk_pct = abs(entry - stop) / entry * 100
        except (TypeError, ZeroDivisionError):
            return None
        if pnl is None or risk_pct <= 0:
            return None
        return pnl / risk_pct
    v = row.get(outcome_key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def numeric_feature_keys(rows: list[dict], min_pairs: int = MIN_FEATURE_PAIRS) -> list[str]:
    """Feature keys that are numeric in at least `min_pairs` rows. Booleans and
    strings (e.g. regime labels, direction) are excluded — they need a different
    (categorical) treatment, deferred."""
    counts: dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("features") or {}).items():
            if _is_number(v):
                counts[k] = counts.get(k, 0) + 1
    return sorted(k for k, c in counts.items() if c >= min_pairs)


def feature_report(rows: list[dict], outcome_key: str = "pnl_pct",
                   min_pairs: int = MIN_FEATURE_PAIRS) -> dict:
    """Per-feature scorecards over the given training rows, sorted by |IC|
    descending. Returns {"n_rows", "outcome", "features": {name: scorecard}}."""
    scored: dict[str, dict] = {}
    for key in numeric_feature_keys(rows, min_pairs):
        vals, outs = [], []
        for r in rows:
            v = (r.get("features") or {}).get(key)
            o = _row_outcome(r, outcome_key)
            if _is_number(v) and o is not None:
                vals.append(float(v))
                outs.append(o)
        if len(vals) >= min_pairs:
            scored[key] = evaluate_feature(vals, outs)
    ordered = dict(sorted(
        scored.items(),
        key=lambda kv: abs(kv[1]["ic"]) if kv[1]["ic"] is not None else -1.0,
        reverse=True))
    return {"n_rows": len(rows), "outcome": outcome_key, "features": ordered}


def stability(rows: list[dict], feature_key: str,
              outcome_key: str = "pnl_pct") -> dict:
    """IC on the first vs. second time-half (rows assumed oldest-first, as
    get_training_rows returns them). sign_stable flags agreement."""
    pairs = []
    for r in rows:
        v = (r.get("features") or {}).get(feature_key)
        o = _row_outcome(r, outcome_key)
        if _is_number(v) and o is not None:
            pairs.append((float(v), o))
    mid = len(pairs) // 2
    first = pairs[:mid]
    second = pairs[mid:]
    ic1 = information_coefficient([v for v, _ in first], [o for _, o in first]) if len(first) >= 3 else None
    ic2 = information_coefficient([v for v, _ in second], [o for _, o in second]) if len(second) >= 3 else None
    sign_stable = (ic1 is not None and ic2 is not None
                   and (ic1 >= 0) == (ic2 >= 0))
    return {"ic_first_half": round(ic1, 4) if ic1 is not None else None,
            "ic_second_half": round(ic2, 4) if ic2 is not None else None,
            "sign_stable": sign_stable}


def correlation_matrix(rows: list[dict], feature_keys: list[str]) -> dict:
    """Pairwise Pearson correlation between features over rows where BOTH are
    present. Flat dict keyed 'a|b' → r (upper triangle). Redundancy radar."""
    out: dict[str, float] = {}
    for i, a in enumerate(feature_keys):
        for b in feature_keys[i + 1:]:
            xa, xb = [], []
            for r in rows:
                f = r.get("features") or {}
                va, vb = f.get(a), f.get(b)
                if _is_number(va) and _is_number(vb):
                    xa.append(float(va))
                    xb.append(float(vb))
            r_ab = _pearson(xa, xb) if len(xa) >= 3 else None
            if r_ab is not None:
                out[f"{a}|{b}"] = round(r_ab, 3)
    return out


# ---------------------------------------------------------------------------
# Deterministic text report.
# ---------------------------------------------------------------------------

def format_report(rows: list[dict], outcome_key: str = "pnl_pct") -> str:
    rep = feature_report(rows, outcome_key)
    n = rep["n_rows"]
    lines = [
        "FEATURE EVALUATION — realized (feature, outcome) pairs",
        f"rows={n}  outcome={outcome_key}",
    ]
    if n < MIN_SAMPLES:
        lines.append(f"⚠ UNDERPOWERED: {n} resolved+featured trades (< {MIN_SAMPLES}). "
                     "Every number below is provisional — let data accumulate.")
    if not rep["features"]:
        lines.append("No numeric features with enough resolved pairs yet. "
                     "This is expected until the deployed bot logs & resolves "
                     "trades carrying feature snapshots.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"{'feature':<26} {'n':>4} {'IC':>7} {'top-bot':>8} {'mono':>5}")
    lines.append("-" * 54)
    for name, sc in rep["features"].items():
        ic = f"{sc['ic']:+.3f}" if sc["ic"] is not None else "  n/a"
        tb = f"{sc['top_minus_bottom']:+.3f}" if sc["top_minus_bottom"] is not None else "   n/a"
        mono = "yes" if sc["monotone_quintiles"] else ("no" if sc["monotone_quintiles"] is False else "n/a")
        flag = " *underpowered" if sc["underpowered"] else ""
        lines.append(f"{name:<26} {sc['n']:>4} {ic:>7} {tb:>8} {mono:>5}{flag}")

    lines.append("")
    lines.append("IC = Spearman rank corr(feature, outcome). top-bot = mean "
                 "outcome of top quintile − bottom. Neither is an edge on its "
                 "own — read alongside stability & OOS backtests.")
    return "\n".join(lines)


def run_from_db(categories: tuple[str, ...] | None = None,
                outcome: str = "pnl_pct") -> str:
    """Pull resolved+featured trades from the DB and render the report."""
    import subscriptions
    rows = subscriptions.get_training_rows(categories)
    return format_report(rows, outcome_key=outcome)


if __name__ == "__main__":
    import argparse
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Feature evaluation harness")
    ap.add_argument("--outcome", default="pnl_pct",
                    choices=["pnl_pct", "R", "mfe_pct", "mae_pct"],
                    help="outcome variable to correlate features against")
    ap.add_argument("--category", nargs="*", default=None,
                    help="restrict to these alert categories")
    args = ap.parse_args()
    cats = tuple(args.category) if args.category else None
    print(run_from_db(cats, args.outcome))
