"""
Risk engine: fixed-fractional sizing arithmetic (hand-computed known answers),
default-off contract, totality over garbage input, the daily R-budget ledger,
row→R mapping, and EV-in-R with honest None propagation.

    venv/Scripts/python.exe -m pytest tests/test_engine_risk.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RiskEngineConfig  # noqa: E402
from engines.risk import (  # noqa: E402
    DailyRiskLedger,
    expected_value_r,
    position_size,
    realized_r,
)

CFG = RiskEngineConfig(capital=1_000_000, risk_per_trade_pct=1.0,
                       max_position_value_pct=25.0)


# ---------------------------------------------------------------- sizing

def test_position_size_capped_by_position_value():
    # capital 1,000,000 @ 1% risk → budget 10,000₹
    # entry 100, stop 98 → risk/share 2 → qty_risk = floor(10000/2) = 5000
    # position would be 5000×100 = 500,000 = 50% of capital
    # cap: floor(1,000,000×25%/100) = 2500 → qty = min(5000, 2500) = 2500
    # actual risk = 2500×2 = 5000₹ = 0.5% of capital (NOT the 10,000 budget)
    r = position_size(100, 98, CFG)
    assert r.ok
    assert r.values["sizing_enabled"] is True
    assert r.values["qty"] == 2500
    assert isinstance(r.values["qty"], int)
    assert r.values["risk_per_share"] == 2.0
    assert r.values["risk_amount_inr"] == 5000.0
    assert r.values["position_value_inr"] == 250_000.0
    assert r.values["position_pct_of_capital"] == 25.0
    assert r.values["capped_by"] == "position_value"
    assert r.diagnostics  # human-readable context always present


def test_position_size_uncapped_risk_binds():
    # entry 100, stop 90 → risk/share 10 → qty_risk = floor(10000/10) = 1000
    # cap qty = floor(250,000/100) = 2500 → qty = min(1000, 2500) = 1000
    # actual risk = 1000×10 = 10,000₹ (full budget), position 100,000 = 10%
    r = position_size(100, 90, CFG)
    assert r.ok
    assert r.values["qty"] == 1000
    assert r.values["risk_per_share"] == 10.0
    assert r.values["risk_amount_inr"] == 10_000.0
    assert r.values["position_value_inr"] == 100_000.0
    assert r.values["position_pct_of_capital"] == 10.0
    assert r.values["capped_by"] == "risk"


def test_position_size_qty_zero_rejected():
    # capital 100,000 @ 1% → budget 1000₹; entry 5000, stop 3000 →
    # risk/share 2000 > budget → qty_risk = floor(1000/2000) = 0 → reject
    cfg = RiskEngineConfig(capital=100_000, risk_per_trade_pct=1.0,
                           max_position_value_pct=25.0)
    r = position_size(5000, 3000, cfg)
    assert not r.ok
    assert r.reject_reason == "risk per share exceeds budget"
    assert r.values["sizing_enabled"] is True
    assert r.values["qty"] is None


def test_position_size_zero_by_concentration_cap_attributed_correctly():
    # entry 30000, stop 29900 → risk/share 100; capital 100k @ 1% → budget
    # 1000 → qty_risk = floor(1000/100) = 10 (budget is FINE). But value cap:
    # floor(100k×25%/30000) = floor(25000/30000) = 0 → qty 0. The rejection
    # must blame the concentration cap, NOT the risk budget.
    cfg = RiskEngineConfig(capital=100_000, risk_per_trade_pct=1.0,
                           max_position_value_pct=25.0)
    r = position_size(30_000, 29_900, cfg)
    assert not r.ok
    assert "risk per share exceeds budget" not in r.reject_reason
    assert "max position value" in r.reject_reason.lower()
    assert r.values["qty"] is None


def test_position_size_invalid_inputs():
    for entry, stop in [(None, 98), (100, None), (0, 98), (-5, 98),
                        (100, 100), (float("nan"), 98), (100, 0)]:
        r = position_size(entry, stop, CFG)
        assert not r.ok, (entry, stop)
        assert r.reject_reason
        assert r.values["qty"] is None
        assert r.diagnostics


def test_position_size_disabled_by_default():
    # Shipped default: capital=None → sizing off, ok=True, nothing else
    # in values — callers wiring this in change no live behavior.
    assert RiskEngineConfig().capital is None
    r = position_size(100, 98, RiskEngineConfig())
    assert r.ok
    assert r.values == {"sizing_enabled": False}
    assert r.diagnostics


def test_values_are_json_safe():
    import json
    for res in (position_size(100, 98, CFG),
                position_size(None, None, CFG),
                position_size(100, 98, RiskEngineConfig())):
        json.dumps(res.values)  # must not raise (no numpy scalars etc.)


# ---------------------------------------------------------------- ledger

def test_ledger_net_r_gate():
    led = DailyRiskLedger(max_daily_risk_r=2)
    for r in (1.5, -1.0, -1.0, -1.0):
        led.add_r(r)
    # net = 1.5 − 1 − 1 − 1 = −1.5 > −2 → still allowed
    assert led.net_r == pytest.approx(-1.5)
    assert led.r_lost == pytest.approx(3.0)   # |−1 −1 −1|
    allowed, reason = led.allows_new_trade()
    assert allowed and reason
    led.add_r(-1.0)
    # net = −2.5 ≤ −2 → blocked
    assert led.net_r == pytest.approx(-2.5)
    allowed, reason = led.allows_new_trade()
    assert not allowed
    assert "exhausted" in reason


def test_ledger_cap_disabled_always_allows():
    led = DailyRiskLedger()          # max_daily_risk_r=None
    led.add_r(-50.0)
    allowed, reason = led.allows_new_trade()
    assert allowed and "disabled" in reason


def test_ledger_ignores_garbage():
    led = DailyRiskLedger(max_daily_risk_r=2)
    led.add_r(None)                  # type: ignore[arg-type]
    led.add_r(float("nan"))
    led.add_r("junk")                # type: ignore[arg-type]
    assert led.net_r == 0.0 and led.r_lost == 0.0
    assert led.allows_new_trade()[0]


# ------------------------------------------------------------ realized_r

def test_realized_r_mapping_and_guards():
    rows = [
        # risk_pct = |100−98|/100×100 = 2% → r = 4/2 = +2R
        {"entry": 100, "stop_loss": 98, "pnl_pct": 4.0},
        # r = −2/2 = −1R
        {"entry": 100, "stop_loss": 98, "pnl_pct": -2.0},
        {"entry": 100, "stop_loss": 98, "pnl_pct": None},   # unresolved: skip
        {"entry": 100, "stop_loss": 100, "pnl_pct": 1.0},   # zero risk: skip
        {"entry": None, "stop_loss": 98, "pnl_pct": 1.0},   # no entry: skip
        {"entry": 0, "stop_loss": 98, "pnl_pct": 1.0},      # div-by-0: skip
        {"pnl_pct": 1.0},                                   # missing keys: skip
        "not-a-dict",                                       # garbage row: skip
    ]
    assert realized_r(rows) == pytest.approx([2.0, -1.0])
    assert realized_r([]) == []
    assert realized_r(None) == []


# -------------------------------------------------------- expected_value_r

def test_expected_value_r_known_answers():
    # 0.4×2 − 0.6×1 = 0.8 − 0.6 = 0.2
    assert expected_value_r(0.4, 2.0) == pytest.approx(0.2)
    # 0.5×1 − 0.5×1 = 0
    assert expected_value_r(0.5, 1.0) == pytest.approx(0.0)
    # explicit p_loss: 0.4×2 − 0.5×1 = 0.3
    assert expected_value_r(0.4, 2.0, p_loss=0.5) == pytest.approx(0.3)


def test_expected_value_r_none_propagation():
    # No Phase-4 model yet → no probabilities → EV must be None, not a guess.
    assert expected_value_r(None, 2.0) is None
    assert expected_value_r(0.5, None) is None
    assert expected_value_r(None, None) is None
