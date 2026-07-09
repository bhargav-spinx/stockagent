"""
Risk engine (Phase 2): fixed-fractional position sizing + daily risk budget.

WHY THIS EXISTS: every alert ships entry/stop/target levels but no quantity —
the human sizes each trade ad hoc, which is exactly where retail accounts
bleed. This module turns a configured capital base into a deterministic
quantity (risk a fixed % of capital per trade, capped by a maximum position
value) and expresses the day's realized outcomes in R units so signaling can
stop after a configured amount of daily bleed instead of revenge-alerting
into a bad tape.

DEFAULT-OFF BY DESIGN: with CONFIG.risk.capital = None (the shipped default)
every sizing call returns {"sizing_enabled": False} and callers change
nothing — the module is safe to wire into the live path without altering
Phase-1 behavior. Sizing only activates when the operator explicitly sets
capital via a config override.

WHY R UNITS for the daily budget: one R is the rupee amount risked on a
single trade (entry→stop distance × qty), so "a −2R day" means the same
thing at any capital level and across differently-sized positions — it is
the only scale on which outcomes of heterogeneous trades are comparable, and
it survives capital changes without re-tuning the cap.

Everything here is pure and in-memory. The DailyRiskLedger is deliberately
NOT persisted: the caller rebuilds it from today's resolved alert rows on
each decision, so a process restart cannot silently forget the morning's
losses. max_concurrent_positions lives in the config section but is enforced
by the caller (it owns the open-position list). No I/O, no network.
"""
from __future__ import annotations

import math
from typing import Any

import config
from engines.base import EngineResult

# Stable value schema for feature snapshots: reject paths emit the same keys
# (as None) so alerts_log columns don't appear/disappear per outcome.
_VALUE_KEYS = ("qty", "risk_per_share", "risk_amount_inr",
               "position_value_inr", "position_pct_of_capital", "capped_by")


def _num(x: Any) -> float | None:
    """Coerce to a finite float, else None — the totality guard that keeps
    garbage input (None, strings, NaN, inf) out of the arithmetic below."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _none_values() -> dict:
    """Full sizing schema with None values (sizing configured but unusable)."""
    vals: dict = {"sizing_enabled": True}
    vals.update({k: None for k in _VALUE_KEYS})
    return vals


def position_size(entry: Any, stop_loss: Any,
                  cfg: "config.RiskEngineConfig | None" = None) -> EngineResult:
    """Fixed-fractional position size for one trade.

    qty = min( floor(risk budget / risk-per-share),          # risk sizing
               floor(capital × max_position_value% / entry) ) # concentration cap

    The concentration cap exists because tight stops explode the risk-sized
    quantity: a 0.3% stop at 1% risk would otherwise put >300% of capital in
    one name. When the cap binds, the ACTUAL rupees at risk (qty ×
    risk-per-share) drop below the budget — values report the actual number,
    never the budget, because that is what the ledger must later be debited.

    Total over garbage: bad prices return ok=False with a reject_reason, and
    capital=None returns the default-off {"sizing_enabled": False} — never an
    exception into the signal path.
    """
    cfg = cfg if cfg is not None else config.CONFIG.risk
    res = EngineResult(engine="risk")

    # Default-off: the whole engine is a no-op until capital is configured.
    if cfg.capital is None:
        res.values = {"sizing_enabled": False}
        res.diagnostics.append("position sizing disabled (risk.capital not set)")
        return res

    capital = _num(cfg.capital)
    if capital is None or capital <= 0:
        res.values = _none_values()
        res.ok = False
        res.reject_reason = "invalid risk config: capital must be positive"
        res.diagnostics.append(res.reject_reason)
        return res

    e = _num(entry)
    s = _num(stop_loss)
    if e is None or s is None:
        res.values = _none_values()
        res.ok = False
        res.reject_reason = "invalid entry/stop: missing or non-numeric"
        res.diagnostics.append(res.reject_reason)
        return res
    if e <= 0 or s <= 0:
        res.values = _none_values()
        res.ok = False
        res.reject_reason = "invalid entry/stop: prices must be positive"
        res.diagnostics.append(res.reject_reason)
        return res
    if e == s:
        res.values = _none_values()
        res.ok = False
        res.reject_reason = "invalid entry/stop: stop equals entry (zero risk per share)"
        res.diagnostics.append(res.reject_reason)
        return res

    risk_per_share = abs(e - s)
    risk_budget_inr = capital * float(cfg.risk_per_trade_pct) / 100.0
    qty_risk = int(math.floor(risk_budget_inr / risk_per_share))
    qty_cap = int(math.floor(capital * float(cfg.max_position_value_pct) / 100.0 / e))
    qty = min(qty_risk, qty_cap)

    if qty <= 0:
        res.values = _none_values()
        res.values["risk_per_share"] = float(risk_per_share)
        res.ok = False
        # Attribute the zero correctly: the position-value cap and the risk
        # budget can each independently floor qty to 0. Blaming "risk per
        # share" when the concentration cap bound (a pricey stock vs a small
        # max_position_value_pct) is factually wrong and self-contradictory.
        if qty_cap <= 0 and qty_risk > 0:
            res.reject_reason = (
                f"entry {e:.2f} exceeds max position value "
                f"({cfg.max_position_value_pct:.0f}% of capital) — 0 affordable shares")
        elif qty_risk <= 0:
            res.reject_reason = "risk per share exceeds budget"
        else:
            res.reject_reason = "position sizing floored to 0 shares"
        res.diagnostics.append(
            f"risk/share {risk_per_share:.2f}, budget {risk_budget_inr:.2f} INR "
            f"-> qty_risk {qty_risk}; value cap -> qty_cap {qty_cap}; qty 0")
        return res

    # Tie goes to "risk": capped_by names the binding constraint, and when
    # both floors agree the risk budget is fully deployed.
    capped_by = "position_value" if qty_cap < qty_risk else "risk"
    risk_amount_inr = qty * risk_per_share      # ACTUAL rupees at risk (post-cap)
    position_value_inr = qty * e

    res.values = {
        "sizing_enabled": True,
        "qty": int(qty),
        "risk_per_share": float(risk_per_share),
        "risk_amount_inr": float(risk_amount_inr),
        "position_value_inr": float(position_value_inr),
        "position_pct_of_capital": float(position_value_inr / capital * 100.0),
        "capped_by": capped_by,
    }
    res.diagnostics.append(
        f"qty {qty} (capped by {capped_by}): position {position_value_inr:.0f} INR "
        f"({position_value_inr / capital * 100.0:.1f}% of capital), "
        f"risk {risk_amount_inr:.0f} INR "
        f"({risk_amount_inr / capital * 100.0:.2f}% of capital)")
    return res


class DailyRiskLedger:
    """In-memory tally of today's realized outcomes in R units.

    WHY in-memory and caller-rebuilt: persisting a running counter invites
    drift (missed decrements, double counts, stale state after a restart).
    Instead the caller reconstructs the ledger from today's RESOLVED alert
    rows (via realized_r) at each decision point — the resolved outcomes in
    the DB are the single source of truth, and this class is just arithmetic
    over them.

    The gate uses net R, not gross losses: a +2R winner earns back budget
    because the purpose of the cap is to bound the DAY'S drawdown, not to
    punish trade count. r_lost is exposed separately for reporting.
    """

    def __init__(self, max_daily_risk_r: float | None = None) -> None:
        self.max_daily_risk_r = (
            None if max_daily_risk_r is None else float(max_daily_risk_r))
        self._rs: list[float] = []

    def add_r(self, r: float) -> None:
        """Record one resolved trade's R multiple; non-finite garbage is
        dropped silently (totality — a bad row must not poison the gate)."""
        v = _num(r)
        if v is not None:
            self._rs.append(v)

    @property
    def net_r(self) -> float:
        """Signed sum of today's R outcomes."""
        return float(sum(self._rs))

    @property
    def r_lost(self) -> float:
        """Gross loss in R: |sum of negative outcomes| — reporting only."""
        return float(abs(sum(v for v in self._rs if v < 0)))

    def allows_new_trade(self) -> tuple[bool, str]:
        """(allowed, reason). Blocks only when a cap is configured AND the
        day's net R has bled through it — cap disabled means always allowed,
        preserving the module's default-off contract."""
        if self.max_daily_risk_r is None:
            return True, "daily risk cap disabled"
        if self.net_r <= -self.max_daily_risk_r:
            return False, (f"daily risk budget exhausted: net {self.net_r:+.2f}R "
                           f"<= -{self.max_daily_risk_r:g}R cap")
        return True, (f"net {self.net_r:+.2f}R within "
                      f"-{self.max_daily_risk_r:g}R daily cap")


def realized_r(rows: "list[dict] | None") -> list[float]:
    """Map resolved alert rows to R multiples: r = pnl_pct / risk_pct where
    risk_pct = |entry − stop| / entry × 100.

    This is the bridge from the alerts_log schema (which stores percentage
    P&L) to the ledger's R scale. Rows that can't be mapped honestly —
    unresolved (pnl_pct None), missing prices, or degenerate risk (stop at
    entry, entry ≤ 0) — are SKIPPED rather than defaulted to 0R: a fabricated
    zero would quietly dilute the daily budget. Pure; never raises.
    """
    out: list[float] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pnl = _num(row.get("pnl_pct"))
        e = _num(row.get("entry"))
        s = _num(row.get("stop_loss"))
        if pnl is None or e is None or s is None or e <= 0:
            continue
        risk_pct = abs(e - s) / e * 100.0
        if risk_pct <= 0:            # stop == entry: undefined R, div-by-zero
            continue
        out.append(float(pnl / risk_pct))
    return out


def expected_value_r(p_win: float | None, rr: float | None,
                     p_loss: float | None = None) -> float | None:
    """Expected value of a trade in R: EV = p_win × rr − p_loss × 1.

    p_loss defaults to (1 − p_win) — the binary win/lose simplification.

    Returns None when p_win (or rr) is None, ON PURPOSE: win probabilities do
    not exist anywhere in this platform until the Phase-4 calibrated model
    ships, and returning None forces callers to render "EV unknown" instead
    of a fabricated number. Never invent a probability to fill this in.
    """
    pw = _num(p_win)
    r = _num(rr)
    if pw is None or r is None:
        return None
    pl = (1.0 - pw) if p_loss is None else _num(p_loss)
    if pl is None:
        return None
    return float(pw * r - pl)
