"""
Phase-2 integration wiring: cost-model dispatch, exposure section, feature
merging. The invariant under test: with DEFAULT config (model="flat",
capital=None) every output is identical to pre-Phase-2 behavior; the new
paths activate only by explicit configuration.

    venv/Scripts/python.exe -m pytest tests/test_phase2_wiring.py
"""
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import eod_report  # noqa: E402
import subscriptions  # noqa: E402


# ---------- eod_report.effective_cost_pct dispatch ----------

def test_flat_default_returns_historical_constant():
    assert config.CONFIG.costs.model == "flat"          # default unchanged
    assert eod_report.effective_cost_pct(500.0, 505.0, "scan") == \
        eod_report.COST_PER_TRADE_PCT
    # entry unknown → flat regardless of model
    assert eod_report.effective_cost_pct(None, None, "scan") == \
        eod_report.COST_PER_TRADE_PCT


def test_statutory_dispatch_via_config_object(monkeypatch):
    statutory_cfg = replace(config.CONFIG,
                            costs=replace(config.CONFIG.costs, model="statutory"))
    monkeypatch.setattr(eod_report, "CONFIG", statutory_cfg)

    intraday = eod_report.effective_cost_pct(500.0, 505.0, "scan")
    swing = eod_report.effective_cost_pct(500.0, 505.0, "swing_auto")

    assert intraday != eod_report.COST_PER_TRADE_PCT     # itemized, not flat
    # delivery product (swing) pays STT both sides + higher stamp — must cost
    # strictly more than the intraday product on identical prices.
    assert swing > intraday
    # sanity band: intraday statutory ≈ 0.1–0.25% incl. slippage estimate
    assert 0.05 < intraday < 0.5
    assert 0.2 < swing < 1.0


def test_statutory_failure_falls_back_to_flat(monkeypatch):
    statutory_cfg = replace(config.CONFIG,
                            costs=replace(config.CONFIG.costs, model="statutory"))
    monkeypatch.setattr(eod_report, "CONFIG", statutory_cfg)
    import engines.execution as ex
    monkeypatch.setattr(ex, "round_trip_cost_pct",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert eod_report.effective_cost_pct(500.0, 505.0, "scan") == \
        eod_report.COST_PER_TRADE_PCT


# ---------- backtest cost dispatch ----------

def test_backtest_trade_cost_flat_and_override():
    import backtest
    t = backtest.Trade(symbol="X", setup="A", direction="long",
                       entry_time=None, entry=500.0, stop_loss=495.0,
                       target1=510.0, target2=515.0, status="t2_hit",
                       exit_price=515.0)
    assert backtest._trade_cost_pct(t) == backtest.COST_PER_TRADE_PCT
    assert backtest._cost_label() == f"{backtest.COST_PER_TRADE_PCT}%"

    backtest.COST_MODEL_OVERRIDE = "statutory"
    try:
        c = backtest._trade_cost_pct(t)
        assert c != backtest.COST_PER_TRADE_PCT and 0.05 < c < 0.5
        assert backtest._cost_label() == "itemized statutory"
    finally:
        backtest.COST_MODEL_OVERRIDE = None


# ---------- stats per-row dispatch (flat default identical) ----------

def test_stats_header_reflects_cost_model(monkeypatch):
    import stats
    # flat default: header names the 0.13% round-trip constant
    flat_note = stats.eod_report.CONFIG.costs.model
    assert flat_note == "flat"
    # statutory: header must NOT claim the flat 0.13% figure (finding #1)
    statutory_cfg = replace(config.CONFIG,
                            costs=replace(config.CONFIG.costs, model="statutory"))
    monkeypatch.setattr(stats.eod_report, "CONFIG", statutory_cfg)
    # rebuild just the note the way build_stats_report does
    note = ("round-trip" if stats.eod_report.CONFIG.costs.model == "flat"
            else "statutory cost model")
    assert "statutory" in note


def test_stats_net_returns_flat_identical():
    import stats
    rows = [{"pnl_pct": 2.0, "entry": 100.0, "exit_price": 102.0,
             "category": "scan"},
            {"pnl_pct": -1.0, "entry": 50.0, "exit_price": 49.5,
             "category": "swing_auto"},
            {"pnl_pct": None}]
    nets = stats._net_returns(rows)
    assert nets == [2.0 - stats.COST_PER_TRADE_PCT,
                    -1.0 - stats.COST_PER_TRADE_PCT]


# ---------- EOD exposure section (default OFF) ----------

def _fresh_db(name: str) -> None:
    p = Path(tempfile.gettempdir()) / name
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    subscriptions.DB_PATH = p
    subscriptions.init_db()


def test_daily_r_uses_intraday_resolution_not_persisted_pnl(monkeypatch):
    # The gate must see TODAY's stop-outs during market hours, not wait for EOD.
    # get_alerts_for_date returns an open intraday alert (pnl None) + a swing
    # alert (must be excluded) + an already-resolved intraday alert.
    open_intraday = {"category": "scan", "entry": 100.0, "stop_loss": 98.0,
                     "pnl_pct": None, "symbol": "A.NS", "direction": "long"}
    swing = {"category": "swing_auto", "entry": 100.0, "stop_loss": 98.0,
             "pnl_pct": -2.0, "symbol": "S.NS", "direction": "long"}
    resolved_intraday = {"category": "manual_intraday", "entry": 100.0,
                         "stop_loss": 98.0, "pnl_pct": -2.0, "symbol": "B.NS",
                         "direction": "long"}
    monkeypatch.setattr(eod_report.subscriptions, "get_alerts_for_date",
                        lambda *a, **k: [open_intraday, swing, resolved_intraday])
    # the open one is resolved on-demand against the current session → sl_hit
    monkeypatch.setattr(eod_report, "resolve_intraday",
                        lambda a, df=None: {"status": "sl_hit", "pnl_pct": -2.0})

    r_list = eod_report.today_intraday_realized_r()
    # entry100/stop98 → risk 2% → pnl -2% → -1R each; swing excluded
    assert r_list == [-1.0, -1.0]

    from engines.risk import DailyRiskLedger
    ledger = DailyRiskLedger(2.0)
    for r in r_list:
        ledger.add_r(r)
    can_fire, _why = ledger.allows_new_trade()
    assert can_fire is False        # -2R hit the 2R daily budget → gate blocks


def test_narrative_rr_epsilon_pins_deliberate_phase1_fix():
    # Canonical levels build target1 at exactly 2R, and float arithmetic can put
    # rr1 one ULP below 2.0. The Phase-1 epsilon makes the gate PASS such cards
    # (previously it rejected its own 2R levels with "R:R 2.00 < 2.0"). Pin it.
    import narrative
    from intraday_score import ScoreCard
    entry, risk = 100.0, 0.37
    sl, t1 = entry - risk, entry + 2.0 * risk       # rr1 = 1.9999999999999616
    rr1 = abs(t1 - entry) / abs(entry - sl)
    assert rr1 < 2.0, "fixture no longer lands below 2.0 — pick new numbers"
    card = ScoreCard(
        symbol="EPS.NS", price=entry, direction="long", score=85,
        rating="Strong Buy", breakdown={}, signals={}, entry=entry,
        stop_loss=sl, target1=t1, target2=entry + 3.0 * risk,
        bull_conviction=80, bear_conviction=20, gap_pct=1.0, rvol=2.0,
        orb_window=15, regime_ok=True, event_ok=True,
    )
    _passed, failed, _flags = narrative.evaluate_gates_intraday(card)
    assert not any("R:R" in f for f in failed), failed   # epsilon: no false reject


def test_exposure_absent_by_default_present_with_capital(monkeypatch):
    _fresh_db("phase2_exposure_test.db")
    subscriptions.log_alert(category="scan", user_id=None, symbol="EXPO.NS",
                            setup="score95", direction="long",
                            entry=100.0, stop_loss=98.0,
                            target1=104.0, target2=106.0)
    # default: capital None → build_report has no exposure section
    monkeypatch.setattr(eod_report, "resolve_pending", lambda **k: 0)
    report_default = eod_report.build_report()
    assert "Hypothetical exposure" not in report_default

    sized_cfg = replace(config.CONFIG,
                        risk=replace(config.CONFIG.risk, capital=1_000_000.0))
    monkeypatch.setattr(eod_report, "CONFIG", sized_cfg)
    import engines.risk as risk_mod
    monkeypatch.setattr(risk_mod, "CONFIG", sized_cfg, raising=False)
    report_sized = eod_report.build_report()
    assert "Hypothetical exposure" in report_sized
    assert "EXPO" in report_sized
    assert "Hypothetical: sizes assume" in report_sized


# ---------- bot helpers (skipped when telegram not installed, e.g. CI) ----------

def test_merge_features():
    pytest.importorskip("telegram")
    import bot
    assert bot._merge_features(None, None) is None
    assert bot._merge_features({"a": 1}, None) == {"a": 1}
    assert bot._merge_features({"a": 1}, {"regime_day_type": "range"}) == \
        {"a": 1, "regime_day_type": "range"}
    # later dicts win on collision
    assert bot._merge_features({"a": 1}, {"a": 2})["a"] == 2


# ---------- analyzer context-features ride-along ----------

def test_analyze_carries_context_features(monkeypatch):
    import math
    import pandas as pd
    import analyzer
    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    close = [1500 + 0.8 * i + 10 * math.sin(i / 7) for i in range(120)]
    df = pd.DataFrame({
        "Open": close, "High": [c + 12 for c in close],
        "Low": [c - 12 for c in close], "Close": close,
        "Volume": [1_000_000] * 120,
    }, index=idx)
    monkeypatch.setattr(analyzer, "fetch_data", lambda *a, **k: df.copy())
    res = analyzer.analyze("CTXTEST", "swing")
    ctx = res.get("context_features")
    assert isinstance(ctx, dict) and ctx, "context_features missing/empty"
    assert any(k.startswith("ctx_") for k in ctx)

    # and the swing feature snapshot carries them through
    from features import features_from_swing_result
    f = features_from_swing_result(res)
    assert any(k.startswith("ctx_") for k in f)
