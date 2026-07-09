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
import random
from collections import defaultdict
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


def profit_factor(returns: Sequence[float]) -> float | None:
    """Gross profit ÷ gross loss. None when there are no losing trades (the
    ratio is undefined, and 'infinite profit factor' on a tiny sample is
    exactly the kind of number that misleads) or no trades at all."""
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = -sum(r for r in returns if r < 0)
    if not returns or gross_loss == 0:
        return None
    return gross_profit / gross_loss


def max_drawdown(returns: Sequence[float]) -> float | None:
    """Maximum peak-to-trough drawdown of the ADDITIVE cumulative %-return
    equity curve (same convention as backtest.py: equal-weight, one unit per
    trade, no compounding). Returned as a positive magnitude in percentage
    points. None on an empty series."""
    if not returns:
        return None
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for r in returns:
        equity += r
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dd


def recovery_factor(returns: Sequence[float]) -> float | None:
    """Total net return ÷ max drawdown, per-series. This is the honest,
    annualisation-free cousin of Calmar (a true Calmar needs an annualised
    return, and these trades have no fixed frequency — see module docstring).
    None when drawdown is zero (no losing streak yet — not evidence of skill)."""
    dd = max_drawdown(returns)
    if not returns or dd is None or dd == 0:
        return None
    return sum(returns) / dd


def bootstrap_ci95(returns: Sequence[float], n_boot: int = 2000,
                   seed: int = 42) -> tuple[float, float] | None:
    """Percentile-bootstrap 95% CI on the mean per-trade return.

    Preferred over the normal-approx `mean_ci95` at the small n this project
    actually operates at (n < 30), where trade returns are fat-tailed and the
    normal approximation is optimistic. Deterministic for a given seed.
    None when <2 trades."""
    n = len(returns)
    if n < 2:
        return None
    rng = random.Random(seed)
    means = sorted(
        _mean(rng.choices(returns, k=n)) for _ in range(n_boot)
    )
    lo_idx = int(0.025 * n_boot)
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return (means[lo_idx], means[hi_idx])


def monthly_returns(dated_returns: Sequence[tuple[str, float]]) -> dict[str, dict]:
    """Aggregate (iso_date_or_datetime_str, net_return_pct) pairs by calendar
    month. Returns {"YYYY-MM": {"n": trades, "net": summed %}} sorted by month.
    Rows with unparseable dates are skipped, not guessed."""
    buckets: dict[str, dict] = defaultdict(lambda: {"n": 0, "net": 0.0})
    for ts, ret in dated_returns:
        if not isinstance(ts, str) or len(ts) < 7:
            continue
        month = ts[:7]
        if not (month[:4].isdigit() and month[5:7].isdigit() and month[4] == "-"):
            continue
        buckets[month]["n"] += 1
        buckets[month]["net"] += ret
    return dict(sorted(buckets.items()))


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
        "profit_factor": profit_factor(returns),
        "max_drawdown": max_drawdown(returns),
        "recovery_factor": recovery_factor(returns),
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
