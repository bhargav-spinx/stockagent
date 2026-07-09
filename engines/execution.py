"""
Itemized Indian equity trading costs (Phase 2, opt-in "statutory" model).

WHY this module exists: the platform's historical cost model is a single flat
0.13% round-trip estimate (CONFIG.costs.cost_per_trade_pct) — fine for coarse
EOD stats, but it hides where the money actually goes and mis-prices delivery
trades (10× the STT, 5× the stamp duty) and large orders (where the ₹20
brokerage cap makes % brokerage collapse). This engine itemizes the real
charge stack — brokerage, STT, exchange transaction charge, SEBI fee, stamp
duty, GST — at the published rates held in CONFIG.costs, so net P&L can be
audited line by line.

Two kinds of numbers live here, deliberately kept on separate lines:

  * statutory charges → statutory_total_inr — EXACT, from published rate
    cards; wrong only if the Budget changes a rate and config isn't updated.
  * slippage → slippage_inr — an ESTIMATE (slippage_pct_per_side). It is
    never folded into statutory_total_inr, so consumers can always trust the
    exact part in isolation.

The flat model stays the DEFAULT (CONFIG.costs.model == "flat", behavior-
identical to Phase 1). effective_cost_pct() is the single dispatch hook the
resolver / backtest / eod_report call, so switching to "statutory" is a
config flip, not a code change.

Deterministic and offline: pure arithmetic over config rates — no network,
no market data, no clock.
"""
from __future__ import annotations

import math

from config import CONFIG, CostConfig
from engines.base import EngineResult

# Every key trade_costs() ever emits — kept in one place so the failure path
# returns the same (all-None) shape as the success path and downstream
# feature snapshots stay column-stable.
_VALUE_KEYS = (
    "brokerage_inr", "stt_inr", "exchange_inr", "sebi_inr", "stamp_inr",
    "gst_inr", "statutory_total_inr", "slippage_inr", "total_inr",
    "total_pct", "gross_pnl_inr", "net_pnl_inr",
)


def _as_float(x) -> float | None:
    """Best-effort finite float, else None. Exists so numpy scalars, ints and
    Decimals all normalize to JSON-safe plain floats, and so garbage (None,
    strings, NaN, inf) becomes a soft failure instead of an exception —
    engines must be total over whatever the signal path hands them."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _reject(reason: str) -> EngineResult:
    """All-None values + diagnostics — never an exception into the signal path."""
    return EngineResult(
        engine="execution",
        values={k: None for k in _VALUE_KEYS},
        ok=False,
        reject_reason=reason,
        diagnostics=[f"execution: {reason}"],
    )


def _itemize(buy_notional: float, sell_notional: float, product: str,
             cfg: CostConfig, cap_brokerage: bool) -> dict[str, float]:
    """Shared rate arithmetic for both the ₹ path (trade_costs, brokerage
    capped) and the % path (round_trip_cost_pct, uncapped — see there for
    why). Single source of truth so the two paths cannot drift apart.

    Rate attribution (published Indian equity structure):
      brokerage   — % per side, optionally capped at brokerage_cap_inr/order
      STT         — intraday: SELL side only; delivery: BOTH sides
      exchange    — both sides;  SEBI fee — both sides
      stamp duty  — BUY side only (intraday vs delivery rates differ)
      GST         — on brokerage + exchange + SEBI (NOT on STT/stamp)
      slippage    — both sides, estimate; reported separately
    """
    def _brokerage_side(notional: float) -> float:
        raw = notional * cfg.brokerage_pct / 100.0
        return min(raw, cfg.brokerage_cap_inr) if cap_brokerage else raw

    both = buy_notional + sell_notional
    brokerage = _brokerage_side(buy_notional) + _brokerage_side(sell_notional)
    if product == "delivery":
        stt = both * cfg.stt_delivery_pct / 100.0
        stamp = buy_notional * cfg.stamp_duty_delivery_buy_pct / 100.0
    else:  # intraday
        stt = sell_notional * cfg.stt_intraday_sell_pct / 100.0
        stamp = buy_notional * cfg.stamp_duty_buy_pct / 100.0
    exchange = both * cfg.exchange_txn_pct / 100.0
    sebi = both * cfg.sebi_fee_pct / 100.0
    gst = (brokerage + exchange + sebi) * cfg.gst_pct / 100.0
    statutory = brokerage + stt + exchange + sebi + stamp + gst
    slippage = both * cfg.slippage_pct_per_side / 100.0
    return {
        "brokerage_inr": brokerage, "stt_inr": stt, "exchange_inr": exchange,
        "sebi_inr": sebi, "stamp_inr": stamp, "gst_inr": gst,
        "statutory_total_inr": statutory, "slippage_inr": slippage,
        "total_inr": statutory + slippage,
    }


def trade_costs(entry_price, exit_price, qty, direction: str = "long",
                product: str = "intraday", cfg: CostConfig | None = None) -> EngineResult:
    """Full ₹ itemization of a round trip, plus gross/net P&L.

    Notional attribution follows trade mechanics, not label symmetry: a short
    SELLS at entry and BUYS to cover at exit, so intraday STT (sell-side-only)
    lands on the ENTRY notional for shorts but the EXIT notional for longs —
    getting this wrong biases short net P&L. total_pct is expressed against
    the entry notional (entry_price × qty) for both directions so it is
    comparable with percentage-space consumers.

    Total over garbage input: bad prices/qty/direction/product ⇒ ok=False
    with all-None values and a diagnostics line, never an exception.
    """
    cfg = cfg or CONFIG.costs
    entry = _as_float(entry_price)
    exit_ = _as_float(exit_price)
    q = _as_float(qty)
    if entry is None or exit_ is None or q is None or entry <= 0 or exit_ <= 0 or q <= 0:
        return _reject(
            f"invalid inputs: entry={entry_price!r} exit={exit_price!r} qty={qty!r} "
            "(need positive finite numbers)")
    # str() before .lower(): pandas rows hand us float NaN for missing cells,
    # which is truthy — (nan or "").lower() would raise AttributeError.
    direction = str(direction or "").lower()
    if direction not in ("long", "short"):
        return _reject(f"unknown direction {direction!r} (expected 'long'/'short')")
    product = str(product or "").lower()
    if product not in ("intraday", "delivery"):
        return _reject(f"unknown product {product!r} (expected 'intraday'/'delivery')")

    if direction == "long":
        buy_notional, sell_notional = entry * q, exit_ * q
        gross = (exit_ - entry) * q
    else:  # short: sell at entry, buy back (cover) at exit
        buy_notional, sell_notional = exit_ * q, entry * q
        gross = (entry - exit_) * q

    items = _itemize(buy_notional, sell_notional, product, cfg, cap_brokerage=True)
    entry_notional = entry * q
    total = items["total_inr"]
    values = {k: float(v) for k, v in items.items()}
    values["total_pct"] = float(total / entry_notional * 100.0)
    values["gross_pnl_inr"] = float(gross)
    values["net_pnl_inr"] = float(gross - total)
    return EngineResult(
        engine="execution",
        values=values,
        diagnostics=[
            f"execution: {product} {direction} qty={q:g} — statutory "
            f"₹{items['statutory_total_inr']:.2f} exact + "
            f"₹{items['slippage_inr']:.2f} slippage est = ₹{total:.2f} "
            f"({values['total_pct']:.4f}% of entry notional)"
        ],
    )


def round_trip_cost_pct(entry_price, exit_price=None, product: str = "intraday",
                        cfg: CostConfig | None = None) -> float:
    """Notional-free round-trip cost as a % of entry price, for
    percentage-space consumers (eod_report / backtest) that track returns,
    not rupees. Same formula as trade_costs with qty=1 and NO brokerage cap:
    percentage space has no notional, so the ₹20/order cap cannot be
    expressed — leaving brokerage uncapped slightly OVERSTATES costs for
    large orders, which is the conservative direction for evaluating an edge.

    exit defaults to entry (a pure cost quote at one price level). Direction
    is ignored: at qty=1 only the intraday STT side differs between long and
    short, and with entry ≈ exit that difference is noise. Bad entry price
    fails open to the flat estimate rather than raising.
    """
    cfg = cfg or CONFIG.costs
    entry = _as_float(entry_price)
    if entry is None or entry <= 0:
        return float(cfg.cost_per_trade_pct)   # fail-open: flat estimate
    exit_ = _as_float(exit_price)
    if exit_ is None or exit_ <= 0:
        exit_ = entry
    # str() guards non-string truthy garbage (e.g. float NaN from a pandas
    # row) — anything unrecognized falls back to intraday, never raises.
    product = "delivery" if str(product or "").lower() == "delivery" else "intraday"
    items = _itemize(entry, exit_, product, cfg, cap_brokerage=False)
    return float(items["total_inr"] / entry * 100.0)


def effective_cost_pct(entry_price=None, exit_price=None, product: str = "intraday",
                       cfg: CostConfig | None = None) -> float:
    """The single cost hook the resolver/backtest call. Dispatches on
    cfg.model so the statutory itemization is a pure config opt-in: with the
    default model=="flat" (or when no entry price is available to anchor the
    statutory math) this returns exactly cfg.cost_per_trade_pct — Phase-1
    behavior, byte-identical."""
    cfg = cfg or CONFIG.costs
    entry = _as_float(entry_price)
    if getattr(cfg, "model", "flat") == "statutory" and entry is not None and entry > 0:
        return round_trip_cost_pct(entry, exit_price, product, cfg)
    return float(cfg.cost_per_trade_pct)
