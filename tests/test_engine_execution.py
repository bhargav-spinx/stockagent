"""
engines/execution.py: known-answer itemization at the published rates pinned
in config.CostConfig defaults (brokerage 0.03%/side cap ₹20, STT intraday
0.025% sell / delivery 0.1% both, exchange 0.00297%/side, SEBI 0.0001%/side,
stamp 0.003% buy / 0.015% delivery buy, GST 18%, slippage 0.03%/side).
Every expected value is derived by hand in the comments.

    venv/Scripts/python.exe -m pytest tests/test_engine_execution.py -q
"""
import dataclasses
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG  # noqa: E402
from engines import execution  # noqa: E402

APPROX = dict(abs=1e-9)


def test_long_intraday_full_itemization_under_cap():
    # entry 500, exit 505, qty 100, long intraday:
    #   buy notional  = 500*100 = 50_000 ; sell notional = 505*100 = 50_500
    #   brokerage: buy 50_000*0.03% = 15.00 (<20, NOT capped)
    #              sell 50_500*0.03% = 15.15 (<20, NOT capped)   → 30.15
    #   STT (intraday, sell only): 50_500*0.025% = 12.625
    #   exchange: (50_000+50_500)*0.00297% = 100_500*0.0000297 = 2.98485
    #   SEBI: 100_500*0.0001% = 0.1005
    #   stamp (buy only, intraday): 50_000*0.003% = 1.5
    #   GST: 18% of (30.15+2.98485+0.1005) = 0.18*33.23535 = 5.982363
    #   statutory = 30.15+12.625+2.98485+0.1005+1.5+5.982363 = 53.342713
    #   slippage: 100_500*0.03% = 30.15  (estimate, separate line)
    #   total = 53.342713+30.15 = 83.492713
    #   total_pct = 83.492713/50_000*100 = 0.166985426
    #   gross = (505-500)*100 = 500 ; net = 500-83.492713 = 416.507287
    r = execution.trade_costs(500, 505, 100, direction="long", product="intraday")
    assert r.ok and r.engine == "execution"
    v = r.values
    assert v["brokerage_inr"] == pytest.approx(30.15, **APPROX)
    assert v["stt_inr"] == pytest.approx(12.625, **APPROX)
    assert v["exchange_inr"] == pytest.approx(2.98485, **APPROX)
    assert v["sebi_inr"] == pytest.approx(0.1005, **APPROX)
    assert v["stamp_inr"] == pytest.approx(1.5, **APPROX)
    assert v["gst_inr"] == pytest.approx(5.982363, **APPROX)
    assert v["statutory_total_inr"] == pytest.approx(53.342713, **APPROX)
    assert v["slippage_inr"] == pytest.approx(30.15, **APPROX)
    assert v["total_inr"] == pytest.approx(83.492713, **APPROX)
    assert v["total_pct"] == pytest.approx(0.166985426, **APPROX)
    assert v["gross_pnl_inr"] == pytest.approx(500.0, **APPROX)
    assert v["net_pnl_inr"] == pytest.approx(416.507287, **APPROX)
    # values are flat JSON-safe plain floats (no numpy leakage)
    json.dumps(v)
    assert all(type(x) is float for x in v.values())
    assert r.diagnostics  # human-readable context line present


def test_brokerage_cap_engages_at_larger_qty():
    # entry 500, exit 505, qty 200, long intraday:
    #   buy 100_000*0.03% = 30 → capped at 20 ; sell 101_000*0.03% = 30.3 → 20
    #   brokerage = 40 (uncapped would have been 60.3)
    #   STT: 101_000*0.025% = 25.25
    #   exchange: 201_000*0.00297% = 5.9697 ; SEBI: 201_000*0.0001% = 0.201
    #   stamp: 100_000*0.003% = 3.0
    #   GST: 0.18*(40+5.9697+0.201) = 0.18*46.1707 = 8.310726
    #   statutory = 40+25.25+5.9697+0.201+3.0+8.310726 = 82.731426
    #   slippage: 201_000*0.03% = 60.3 ; total = 143.031426
    #   total_pct = 143.031426/100_000*100 = 0.143031426
    r = execution.trade_costs(500, 505, 200, direction="long", product="intraday")
    assert r.ok
    v = r.values
    assert v["brokerage_inr"] == pytest.approx(40.0, **APPROX)
    assert v["statutory_total_inr"] == pytest.approx(82.731426, **APPROX)
    assert v["total_inr"] == pytest.approx(143.031426, **APPROX)
    assert v["total_pct"] == pytest.approx(0.143031426, **APPROX)


def test_short_intraday_stt_lands_on_entry_side():
    # Short sells at ENTRY, covers (buys) at EXIT — so intraday sell-side STT
    # is charged on the entry notional. entry 500, exit 495, qty 100:
    #   sell notional (entry) = 50_000 ; buy notional (cover) = 49_500
    #   STT: 50_000*0.025% = 12.5  (NOT 49_500*0.025% = 12.375)
    #   brokerage: 49_500*0.03% = 14.85 + 50_000*0.03% = 15.00 → 29.85
    #   exchange: 99_500*0.00297% = 2.95515 ; SEBI: 99_500*0.0001% = 0.0995
    #   stamp (buy = cover side): 49_500*0.003% = 1.485
    #   GST: 0.18*(29.85+2.95515+0.0995) = 0.18*32.90465 = 5.922837
    #   statutory = 29.85+12.5+2.95515+0.0995+1.485+5.922837 = 52.812487
    #   slippage: 99_500*0.03% = 29.85 ; total = 82.662487
    #   gross = (500-495)*100 = 500 ; net = 417.337513
    #   total_pct = 82.662487/50_000*100 = 0.165324974 (entry notional basis)
    r = execution.trade_costs(500, 495, 100, direction="short", product="intraday")
    assert r.ok
    v = r.values
    assert v["stt_inr"] == pytest.approx(12.5, **APPROX)
    assert v["brokerage_inr"] == pytest.approx(29.85, **APPROX)
    assert v["stamp_inr"] == pytest.approx(1.485, **APPROX)
    assert v["statutory_total_inr"] == pytest.approx(52.812487, **APPROX)
    assert v["total_inr"] == pytest.approx(82.662487, **APPROX)
    assert v["gross_pnl_inr"] == pytest.approx(500.0, **APPROX)
    assert v["net_pnl_inr"] == pytest.approx(417.337513, **APPROX)
    assert v["total_pct"] == pytest.approx(0.165324974, **APPROX)


def test_delivery_both_side_stt_and_higher_stamp():
    # entry 500, exit 505, qty 100, long DELIVERY:
    #   STT (both sides): (50_000+50_500)*0.1% = 100_500*0.001 = 100.5
    #   stamp (buy, delivery rate): 50_000*0.015% = 7.5
    #   brokerage/exchange/SEBI/GST identical to the intraday case
    #   (GST excludes STT and stamp): 30.15 / 2.98485 / 0.1005 / 5.982363
    #   statutory = 30.15+100.5+2.98485+0.1005+7.5+5.982363 = 147.217713
    #   slippage 30.15 ; total = 177.367713
    r = execution.trade_costs(500, 505, 100, direction="long", product="delivery")
    assert r.ok
    v = r.values
    assert v["stt_inr"] == pytest.approx(100.5, **APPROX)
    assert v["stamp_inr"] == pytest.approx(7.5, **APPROX)
    assert v["gst_inr"] == pytest.approx(5.982363, **APPROX)
    assert v["statutory_total_inr"] == pytest.approx(147.217713, **APPROX)
    assert v["total_inr"] == pytest.approx(177.367713, **APPROX)


def test_guards_are_total_never_raise():
    # qty<=0, price<=0, None, garbage strings, bad direction/product:
    # ok=False, all values None (column-stable shape), diagnostics present.
    bad = [
        execution.trade_costs(500, 505, 0),
        execution.trade_costs(500, 505, -10),
        execution.trade_costs(0, 505, 100),
        execution.trade_costs(500, -1, 100),
        execution.trade_costs(None, 505, 100),
        execution.trade_costs("garbage", 505, 100),
        execution.trade_costs(float("nan"), 505, 100),
        execution.trade_costs(500, 505, 100, direction="sideways"),
        execution.trade_costs(500, 505, 100, product="options"),
        # non-string truthy garbage (pandas missing cell → float NaN):
        # must reject, never AttributeError from .lower()
        execution.trade_costs(500, 505, 100, direction=float("nan")),
        execution.trade_costs(500, 505, 100, product=float("nan")),
    ]
    for r in bad:
        assert r.ok is False and r.reject_reason
        assert set(r.values) and all(x is None for x in r.values.values())
        assert r.diagnostics
    # feature snapshot prefixing works on the failure shape too
    assert "execution_total_pct" in bad[0].feature_dict()


def test_round_trip_pct_matches_trade_costs_under_cap():
    # With qty=100 no brokerage side caps, so the % model must agree with the
    # ₹ model's total_pct exactly (same arithmetic, qty scales out).
    rt = execution.round_trip_cost_pct(500, 505, product="intraday")
    tc = execution.trade_costs(500, 505, 100).values["total_pct"]
    assert rt == pytest.approx(tc, abs=1e-12)
    assert rt == pytest.approx(0.166985426, **APPROX)
    # No-cap is the conservative direction: once qty=200 caps brokerage in
    # the ₹ model (total_pct 0.143031426), the % model overstates.
    capped = execution.trade_costs(500, 505, 200).values["total_pct"]
    assert rt > capped


def test_round_trip_pct_exit_defaults_to_entry():
    # round_trip_cost_pct(500) → buy=sell=500, qty=1, uncapped:
    #   brokerage: 1000*0.03% = 0.30 ; STT: 500*0.025% = 0.125
    #   exchange: 1000*0.00297% = 0.0297 ; SEBI: 1000*0.0001% = 0.001
    #   stamp: 500*0.003% = 0.015
    #   GST: 0.18*(0.30+0.0297+0.001) = 0.059526
    #   statutory = 0.530226 ; slippage: 1000*0.03% = 0.30 ; total = 0.830226
    #   pct = 0.830226/500*100 = 0.1660452
    assert execution.round_trip_cost_pct(500) == pytest.approx(0.1660452, **APPROX)
    # bad entry fails open to the flat estimate rather than raising
    assert execution.round_trip_cost_pct(None) == CONFIG.costs.cost_per_trade_pct
    assert execution.round_trip_cost_pct(-5) == CONFIG.costs.cost_per_trade_pct
    # NaN product (pandas missing cell) falls back to intraday, never raises —
    # and effective_cost_pct (the resolver/backtest hook) stays total too.
    assert execution.round_trip_cost_pct(500, product=float("nan")) == pytest.approx(
        0.1660452, **APPROX)
    stat = dataclasses.replace(CONFIG.costs, model="statutory")
    assert execution.effective_cost_pct(
        500, 500, product=float("nan"), cfg=stat) == pytest.approx(0.1660452, **APPROX)


def test_effective_cost_pct_dispatch():
    # Default model is "flat" → exactly the Phase-1 constant, no arithmetic.
    assert CONFIG.costs.model == "flat"
    assert execution.effective_cost_pct(500, 505) == CONFIG.costs.cost_per_trade_pct
    assert execution.effective_cost_pct() == CONFIG.costs.cost_per_trade_pct
    # Statutory opt-in via config, not code:
    stat = dataclasses.replace(CONFIG.costs, model="statutory")
    got = execution.effective_cost_pct(500, 505, cfg=stat)
    assert got == pytest.approx(0.166985426, **APPROX)
    assert got == pytest.approx(
        execution.round_trip_cost_pct(500, 505, cfg=stat), abs=1e-12)
    # Statutory without an entry price to anchor it → flat fallback.
    assert execution.effective_cost_pct(cfg=stat) == stat.cost_per_trade_pct


def test_numpy_scalars_coerce_to_plain_floats():
    # The signal path hands numpy scalars around; values must come out as
    # plain floats (JSON-safe) with identical arithmetic.
    r = execution.trade_costs(np.float64(500), np.float64(505), np.int64(100))
    assert r.ok
    assert type(r.values["total_inr"]) is float
    assert r.values["total_inr"] == pytest.approx(83.492713, **APPROX)
