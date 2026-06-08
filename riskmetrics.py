"""
Risk-adjusted performance metrics, shared by backtest.py and stats.py.

All functions operate on a list of PER-TRADE net % returns (already cost-
adjusted by the caller). They are deliberately dependency-light (stdlib only)
and make no annualisation assumption — a per-trade Sharpe is reported as
mean/std of the trade return series, clearly labelled as such, because these
trades do not occur at a fixed frequency.

Why this module exists: win-rate and summed P&L cannot tell skill from luck.
A small sample of lucky wins and a genuinely positive edge look identical on a
win-rate line. Sharpe/Sortino, a confidence interval on the mean, and a t-stat
against zero are the minimum needed to say "this might be real" — and even then
only an OUT-OF-SAMPLE result counts (see backtest.walk_forward).
"""
from __future__ import annotations

import math
from typing import Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def sharpe(returns: Sequence[float]) -> float | None:
    """Per-trade Sharpe = mean / stdev of the trade return series.
    NOT annualised. None when <2 trades or zero dispersion."""
    if len(returns) < 2:
        return None
    sd = _std(returns)
    if sd == 0:
        return None
    return _mean(returns) / sd


def sortino(returns: Sequence[float]) -> float | None:
    """Per-trade Sortino = mean / downside deviation (negative returns only)."""
    if len(returns) < 2:
        return None
    downside = [min(0.0, r) for r in returns]
    dd = math.sqrt(sum(d * d for d in downside) / len(returns))
    if dd == 0:
        return None
    return _mean(returns) / dd


def mean_ci95(returns: Sequence[float]) -> tuple[float, float] | None:
    """95% CI on the mean per-trade return via normal approx (mean ± 1.96·SE).
    If the interval straddles 0, the mean return is not distinguishable from 0
    at ~95% confidence — i.e. no demonstrable edge yet. None when <2 trades."""
    n = len(returns)
    if n < 2:
        return None
    se = _std(returns) / math.sqrt(n)
    m = _mean(returns)
    return (m - 1.96 * se, m + 1.96 * se)


def t_stat(returns: Sequence[float]) -> float | None:
    """One-sample t-stat of mean return against 0. |t| ≳ 2 is the usual rough
    bar for 'probably not noise' — but beware multiple-testing if many
    strategies/params were tried (see BLOCKER-2)."""
    n = len(returns)
    if n < 2:
        return None
    se = _std(returns) / math.sqrt(n)
    if se == 0:
        return None
    return _mean(returns) / se


def summarize(returns: Sequence[float]) -> dict:
    """Bundle the headline risk metrics for a return series."""
    ci = mean_ci95(returns)
    return {
        "n": len(returns),
        "mean": _mean(returns),
        "std": _std(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "t_stat": t_stat(returns),
        "ci95_lo": ci[0] if ci else None,
        "ci95_hi": ci[1] if ci else None,
        "edge_distinguishable_from_noise": bool(ci and (ci[0] > 0 or ci[1] < 0)),
    }


def format_line(returns: Sequence[float], label: str = "Risk") -> str:
    """One-line human summary, safe on empty/degenerate input."""
    s = summarize(returns)
    if s["n"] < 2:
        return f"{label}: n={s['n']} — too few trades for risk metrics"

    def _f(v, fmt="{:+.2f}"):
        return fmt.format(v) if v is not None else "n/a"

    verdict = ("mean ≠ 0 at 95%" if s["edge_distinguishable_from_noise"]
               else "CI straddles 0 — not distinguishable from noise")
    return (
        f"{label}: Sharpe {_f(s['sharpe'])} · Sortino {_f(s['sortino'])} · "
        f"t {_f(s['t_stat'])} · mean {_f(s['mean'])}% "
        f"(95% CI {_f(s['ci95_lo'])}…{_f(s['ci95_hi'])}%) — {verdict}"
    )
