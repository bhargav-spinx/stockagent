"""
Regression pins for the 2026-10 audit fixes. Each test names the defect it
guards; all are offline (synthetic frames, mocked SDK, temp DBs).
"""
import asyncio
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eod_report  # noqa: E402
from constants import IST  # noqa: E402


def _df5(rows, start="2026-01-05 09:30"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame([{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}
                         for (o, h, l, c) in rows], index=idx)


def _dfd(rows, start="2026-01-06"):
    idx = pd.bdate_range(start, periods=len(rows), tz="Asia/Kolkata")
    return pd.DataFrame([{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}
                         for (o, h, l, c) in rows], index=idx)


def _alert(direction="long", entry=100.0, sl=98.0, t1=104.0, t2=106.0, gen=None):
    return {"symbol": "T.NS", "entry": entry, "stop_loss": sl, "target1": t1,
            "target2": t2, "direction": direction, "generated_at": gen}


AFTER_CLOSE = IST.localize(datetime(2026, 1, 5, 17, 0))
MID_SESSION = IST.localize(datetime(2026, 1, 5, 10, 20))


# ---------------------------------------------------------------- resolver --

def test_buy_direction_resolves_as_long():
    # A BUY on a quiet day used to be resolved as a SHORT: "sl_hit +3%".
    df = _df5([(100, 100.5, 99.5, 100)] * 6)
    out = eod_report.resolve_intraday(
        _alert(direction="BUY", sl=97, t1=106, t2=109, gen=df.index[0]),
        df=df, now=AFTER_CLOSE)
    assert out["status"] == "squareoff_no_t1"
    assert abs(out["pnl_pct"]) < 0.01


def test_unknown_direction_raises():
    df = _df5([(100, 100, 100, 100)] * 3)
    with pytest.raises(ValueError):
        eod_report.resolve_intraday(_alert(direction="HOLD", gen=df.index[0]), df=df)


def test_midsession_squareoff_is_not_final():
    # /today at 10:20 used to PERSIST squareoff_no_t1 at the current candle.
    df = _df5([(100, 100.5, 99.5, 100.3)] * 4)
    a = _alert(gen=df.index[0])
    assert eod_report.resolve_intraday(a, df=df, now=MID_SESSION) == {"status": "open"}
    mtm = eod_report.resolve_intraday(a, df=df, now=MID_SESSION, mark_to_market=True)
    assert mtm["status"] == "squareoff_no_t1"
    # Final outcomes (SL) are still returned mid-session.
    df_sl = _df5([(100, 100, 100, 100), (100, 100, 97, 97.5)])
    assert eod_report.resolve_intraday(_alert(gen=df_sl.index[0]), df=df_sl,
                                       now=MID_SESSION)["status"] == "sl_hit"


def test_gap_through_stop_fills_at_open():
    df = _df5([(100, 100, 100, 100), (96, 96.5, 95, 95.5)])   # opens below SL 98
    out = eod_report.resolve_intraday(_alert(gen=df.index[0]), df=df, now=AFTER_CLOSE)
    assert out["status"] == "sl_hit" and out["exit_price"] == 96
    assert abs(out["pnl_pct"] - (-4.0)) < 0.01


def test_post_t1_bar_spanning_breakeven_and_t2_books_breakeven():
    # Conservative ordering after T1, mirroring SL-first before T1.
    df = _df5([(100, 100, 100, 100),
               (101, 104.5, 100.5, 104),     # T1 104 hit
               (103, 107, 99.5, 101)])       # touches entry AND T2 106
    out = eod_report.resolve_intraday(_alert(gen=df.index[0]), df=df, now=AFTER_CLOSE)
    assert out["status"] == "t1_then_breakeven"


def test_swing_time_stop_closes_stalled_trade():
    n = eod_report.SWING_TIME_STOP_SESSIONS
    df = _dfd([(100, 101, 99, 100.5)] * (n + 5))
    gen = IST.localize(datetime(2026, 1, 5, 15, 45))
    out = eod_report.resolve_swing(_alert(sl=95, t1=110, t2=115, gen=gen), df=df,
                                   now=IST.localize(datetime(2026, 6, 1, 17, 0)))
    assert out["status"] == "time_stop"
    assert out["exit_time"] == df.index[n - 1]


def test_swing_gap_through_stop_fills_at_open():
    df = _dfd([(90, 91, 89, 90)])
    gen = IST.localize(datetime(2026, 1, 5, 15, 45))
    out = eod_report.resolve_swing(_alert(sl=97, t1=106, t2=109, gen=gen), df=df,
                                   now=IST.localize(datetime(2026, 2, 1, 17, 0)))
    assert out["status"] == "sl_hit" and abs(out["pnl_pct"] - (-10.0)) < 0.01


def test_replay_must_stamp_candle_close_not_start():
    # Trigger bar (09:30) has a deep wick BEFORE its close (= entry). Stamped at
    # its start, the resolver re-reads that wick as an instant stop.
    import backtest
    df = _df5([(100, 100.5, 97.0, 100), (100, 104.5, 99.8, 104), (104, 106.5, 103, 106)])
    bad = eod_report.resolve_intraday(_alert(gen=df.index[0]), df=df, now=AFTER_CLOSE)
    good = eod_report.resolve_intraday(
        _alert(gen=df.index[0] + backtest.BAR), df=df, now=AFTER_CLOSE)
    assert bad["status"] == "sl_hit"
    assert good["status"] == "t2_hit"


def test_live_window_starts_bound_to_five_sessions():
    import backtest
    idx = pd.DatetimeIndex([])
    for d in pd.bdate_range("2026-01-05", periods=8):
        idx = idx.append(pd.date_range(d + pd.Timedelta("9h15min"), periods=3,
                                       freq="5min"))
    df = pd.DataFrame({"Close": range(len(idx))}, index=idx.tz_localize(IST))
    starts = backtest.live_window_starts(df)
    last_window = df.iloc[starts[-1]:]
    assert len(set(last_window.index.date)) == backtest.LIVE_WINDOW_SESSIONS


def test_daily_r_budget_ignores_manual_rows(monkeypatch):
    rows = [{"category": "manual_intraday", "entry": 100.0, "stop_loss": 98.0,
             "pnl_pct": 2.0, "symbol": "M.NS", "direction": "long"}]
    monkeypatch.setattr(eod_report.subscriptions, "get_alerts_for_date",
                        lambda *a, **k: rows)
    assert eod_report.today_intraday_realized_r() == []


# -------------------------------------------------------------- indicators --

def test_supertrend_turns_bearish_in_decline():
    from scanner_indicators import supertrend
    idx = pd.date_range("2026-10-19 09:15", periods=300, freq="5min", tz="Asia/Kolkata")
    c = np.r_[np.linspace(1000, 760, 150), np.linspace(760, 900, 150)]
    df = pd.DataFrame({"Open": c, "High": c + 2, "Low": c - 2, "Close": c,
                       "Volume": 1000}, index=idx)
    _, d = supertrend(df)
    assert d.iloc[149] == -1 and d.iloc[-1] == 1


def test_vwap_slope_ignores_previous_session():
    from scanner_indicators import vwap_slope_pct
    y = pd.Series([100.0] * 10,
                  index=pd.date_range("2026-01-05 14:40", periods=10, freq="5min",
                                      tz="Asia/Kolkata"))
    t = pd.Series([101.0] * 3,
                  index=pd.date_range("2026-01-06 09:15", periods=3, freq="5min",
                                      tz="Asia/Kolkata"))
    assert vwap_slope_pct(pd.concat([y, t])) is None       # <6 candles today
    t6 = pd.Series([101.0] * 6,
                   index=pd.date_range("2026-01-06 09:15", periods=6, freq="5min",
                                       tz="Asia/Kolkata"))
    assert vwap_slope_pct(pd.concat([y, t6])) == 0          # flat today = flat


def test_neutral_prefilter_skips_round_number_gate():
    import scanner_filters as sf
    df = _df5([(998.5, 999, 998, 998.5)] * 30)
    rn_long = sf.round_number_filter(df, "long")
    assert not rn_long.passed                                # ₹1000 just above
    # direction=None must not apply the directional gate
    calls = []
    orig = sf.round_number_filter
    try:
        sf.round_number_filter = lambda d, direction: calls.append(direction) or orig(d, direction)
        sf.apply_universal_filters(df, direction=None, check_time=False)
    finally:
        sf.round_number_filter = orig
    assert calls == []


def test_stale_reason():
    from scanner_indicators import stale_reason
    df = _df5([(1, 1, 1, 1)] * 3, start="2026-10-23 15:15")   # Friday
    monday = IST.localize(datetime(2026, 10, 26, 10, 0))
    assert "no candles today" in stale_reason(df, now=monday)
    fresh = _df5([(1, 1, 1, 1)] * 3, start="2026-10-26 09:45")
    assert stale_reason(fresh, now=monday) is None


def test_index_trend_works_without_volume(monkeypatch):
    # Index candles have zero volume → VWAP NaN → every index was "flat".
    import data_provider
    import intraday_score
    now = datetime.now(IST).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=5 * 40)
    idx = pd.date_range(start, periods=40, freq="5min")
    c = np.linspace(100, 90, 40)
    df = pd.DataFrame({"Open": c, "High": c + .1, "Low": c - .1, "Close": c,
                       "Volume": 0}, index=idx)
    df.index = df.index.tz_localize(IST) if df.index.tz is None else df.index
    monkeypatch.setattr(data_provider, "fetch_data", lambda *a, **k: df)
    monkeypatch.setattr(intraday_score, "split_sessions",
                        lambda d: (d, []))
    out = intraday_score.index_trend(("NIFTY",))
    assert out == {"NIFTY": "bearish"}


# ------------------------------------------------------------ data provider --

def test_window_start_counts_trading_sessions():
    import data_provider as dp
    mon = IST.localize(datetime(2026, 10, 26, 11, 0))      # Tue 10-20 = Dussehra
    start = dp._window_start(mon, "5d", "5m")
    assert start.date() == date(2026, 10, 19) and start.hour == 0
    assert dp.period_days("30d") == 30 and dp.period_days("2mo") == 60
    with pytest.raises(ValueError):
        dp.period_days("max")


def test_status_false_auth_error_triggers_relogin(monkeypatch):
    import data_provider as dp

    class Sess:
        calls = 0

        def getCandleData(self, params):
            Sess.calls += 1
            if Sess.calls == 1:
                return {"status": False, "errorcode": "AG8001", "message": "Invalid Token"}
            return {"status": True, "data": [["2026-01-05T09:15:00+05:30", 1, 1, 1, 1, 1]]}

    resets = []
    monkeypatch.setattr(dp, "_get_angel_session", lambda: Sess())
    monkeypatch.setattr(dp, "_reset_angel_session", lambda: resets.append(1))
    resp = dp._get_candle_data({}, "X")
    assert resp["status"] and resets == [1]


def test_login_cooldown_fails_fast(monkeypatch):
    import data_provider as dp

    def boom():
        raise dp.AngelLoginCooldown("Angel login on cooldown (50s left)")
    monkeypatch.setattr(dp, "_get_angel_session", boom)
    monkeypatch.setattr(dp.time, "sleep", lambda s: pytest.fail("must not back off"))
    with pytest.raises(ValueError):
        dp._get_candle_data({}, "X")


def test_bse_symbols_resolve_without_eq_suffix(monkeypatch):
    import data_provider as dp
    master = [{"exch_seg": "BSE", "symbol": "RELIANCE", "token": "500325",
               "instrumenttype": ""},
              {"exch_seg": "NSE", "symbol": "RELIANCE-EQ", "token": "2885",
               "instrumenttype": ""}]
    monkeypatch.setattr(dp, "_load_scrip_master", lambda: master)
    assert dp._resolve_token("RELIANCE.BO")[0] == "500325"
    assert dp._resolve_token("RELIANCE")[0] == "2885"


def test_sdk_logging_is_silenced():
    import logzero
    import data_provider as dp
    dp._silence_sdk_logging()
    assert logzero.logger.disabled is True


def test_bse_is_a_stock_not_sensex_and_finnifty_is_an_index():
    import indices
    assert indices.get_index_info("BSE") is None
    assert indices.get_index_info("FINNIFTY")["angel_token"] == "99926037"


# ------------------------------------------------------- calendar / metrics --

def test_holiday_coverage_warning():
    import market_calendar as mc
    assert mc.coverage_warning(date(2026, 6, 1)) is None
    assert "2027" in (mc.coverage_warning(date(2026, 12, 15)) or "")
    assert mc.is_trading_holiday(date(2026, 11, 10))       # Balipratipada


def test_ci_uses_student_t():
    import riskmetrics as rm
    lo, hi = rm.mean_ci95([0.5, 1.5, 0.2, 1.8, 0.5])
    m = 0.9
    assert hi - m > 0.8          # t(4)=2.776 width, not z=1.96
    assert rm.t_crit_975(4) == 2.776 and rm.t_crit_975(5000) == 1.96


def test_top_k_ties_are_fractional():
    import prob_eval
    rows = [{"ts": f"t{i}", "label_win": w, "r_multiple": None}
            for i, w in enumerate([1, 0, 0, 0])]
    # all scores tied: top-1 of 4 → expected win rate 0.25 regardless of order
    assert prob_eval._top_k_stats(rows, [5, 5, 5, 5], 0.25)["win_rate"] == 0.25


# ------------------------------------------------------------------ storage --

def _fresh_db(name):
    import subscriptions
    p = Path(tempfile.gettempdir()) / name
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    subscriptions.DB_PATH = p
    subscriptions.init_db()
    return subscriptions


def test_log_alert_normalises_and_validates():
    s = _fresh_db("audit_log_alert.db")
    aid = s.log_alert(category="manual_swing", user_id=1, symbol="X.NS", setup=None,
                      direction="BUY", entry=100, stop_loss=97, target1=104, target2=106)
    with s._conn() as c:
        assert c.execute("SELECT direction FROM alerts_log WHERE id=?",
                         (aid,)).fetchone()[0] == "long"
    bad = s.log_alert(category="channel_call", user_id=1, symbol="X.NS", setup="@c",
                      direction="BUY", entry=1247, stop_loss=1260, target1=1230,
                      target2=1230)
    assert bad is None


def test_save_outcome_keeps_first_and_notified_flag():
    s = _fresh_db("audit_outcome.db")
    aid = s.log_alert(category="swing_auto", user_id=1, symbol="X.NS", setup=None,
                      direction="long", entry=100, stop_loss=97, target1=104, target2=106)
    s.save_outcome(aid, status="t2_hit", pnl_pct=5.0)
    with s._conn() as c:
        c.execute("UPDATE alert_outcomes SET notified=1 WHERE alert_id=?", (aid,))
    s.save_outcome(aid, status="t2_hit", pnl_pct=5.0)        # overlapping resolver
    with s._conn() as c:
        assert c.execute("SELECT notified FROM alert_outcomes WHERE alert_id=?",
                         (aid,)).fetchone()[0] == 1


def test_stats_dedupe_collapses_per_subscriber_copies():
    import stats
    base = {"category": "swing_auto", "symbol": "X.NS", "direction": "long",
            "trade_date": "2026-01-05", "entry": 100.0}
    assert len(stats._dedupe([dict(base, user_id=1), dict(base, user_id=2)])) == 1


# --------------------------------------------------------------- tip parser --

def test_tip_target_at_is_not_entry_and_commas_parse():
    import tip_parser as t
    k = {"TCS", "RELIANCE", "BSE"}
    r = t.extract_tip("BUY TCS cmp 3500 target at 3600 sl at 3450", k)
    assert (r["entry"], r["target"], r["sl"]) == (3500, 3600, 3450)
    r = t.extract_tip("BUY RELIANCE @ 1,247.50 SL 1,230 TGT 1,280", k)
    assert (r["entry"], r["sl"], r["target"]) == (1247.5, 1230, 1280)
    assert t.extract_tip("BSE: BUY RELIANCE at 1250 tgt 1290 sl 1230", k)["symbol"] == "RELIANCE"


# ------------------------------------------------- model-input transform v1 --

def test_model_features_drops_levels_and_orients_direction():
    import prob_model as pm
    base = {"price": 2500.0, "p3_pa_last_low": 2490.0, "p3_vol_atr": 12.0,
            "bull_conviction": 80, "bear_conviction": 20, "score": 75,
            "p3_vwap_dist_atr": 1.5, "p3_vwap_above": True,
            "p3_vwap_time_above_frac": 0.8, "gap_pct": 0.6,
            "p3_pa_hh": 1, "p3_pa_hl": 1, "p3_pa_lh": 0, "p3_pa_ll": 0,
            "p3_pa_bos": "bullish", "p3_pa_dist_support_pct": 0.4,
            "p3_pa_dist_resistance_pct": 1.2, "p3_rs_stock_ret_20": 2.0}
    lg = pm.model_features(dict(base, direction="long"))
    sh = pm.model_features(dict(base, direction="short"))
    for k in ("price", "p3_pa_last_low", "p3_vol_atr", "bull_conviction",
              "bear_conviction", "p3_vwap_dist_atr", "gap_pct"):
        assert k not in lg and k not in sh
    assert lg["is_long"] is True and sh["is_long"] is False
    assert lg["dir_p3_vwap_dist_atr"] == 1.5 and sh["dir_p3_vwap_dist_atr"] == -1.5
    assert lg["dir_vwap_side"] == 1 and sh["dir_vwap_side"] == 0
    assert abs(sh["dir_vwap_time_with"] - 0.2) < 1e-9
    assert lg["dir_pa_structure_with"] == 2 and sh["dir_pa_structure_with"] == 0
    assert lg["dir_pa_bos"] == 1 and sh["dir_pa_bos"] == -1
    assert lg["dir_pa_room_pct"] == 1.2 and sh["dir_pa_room_pct"] == 0.4
    assert lg["score"] == 75                       # non-directional kept
    # No direction → directional keys dropped, never mis-signed
    nd = pm.model_features(base)
    assert "is_long" not in nd and not any(k.startswith("dir_") for k in nd)


def test_legacy_artifact_keeps_raw_features():
    import prob_model as pm
    m = pm.LogisticModel.from_dict({"columns": ["price"], "mean": [100.0],
                                    "std": [10.0], "weights": [1.0],
                                    "intercept": 0.0})
    assert m.transform == 0                         # absent key = legacy
    hi = m.predict_proba({"price": 130.0, "direction": "long"})
    lo = m.predict_proba({"price": 70.0, "direction": "long"})
    assert hi > lo                                  # raw price still read


def test_fit_and_inference_share_the_transform(tmp_path):
    import random
    import prob_model as pm
    rnd = random.Random(3)
    rows, ys = [], []
    for _ in range(400):
        d = rnd.choice(["long", "short"])
        dist = rnd.uniform(-3, 3)
        oriented = dist if d == "long" else -dist
        rows.append({"direction": d, "p3_vwap_dist_atr": dist,
                     "price": rnd.uniform(50, 5000)})
        ys.append(int(rnd.random() < (0.35 if oriented < 0 else 0.1)))
    m = pm.LogisticModel().fit(rows, ys, lam=1.0)
    assert "price" not in m.columns and "dir_p3_vwap_dist_atr" in m.columns
    p = tmp_path / "m.json"
    m.save(p)
    m2 = pm.LogisticModel.load(p)
    assert m2.transform == pm.FEATURE_TRANSFORM_VERSION
    snap = {"direction": "short", "p3_vwap_dist_atr": 2.0, "price": 999.0}
    assert abs(m.predict_proba(snap) - m2.predict_proba(snap)) < 1e-12
    # The synthetic truth depends on the TRADE-ORIENTED distance, so a short
    # far below VWAP (oriented +2.5) must score like an extended long would —
    # lower than a short with oriented distance −0.5. Raw-signed pooling
    # could not learn this.
    ext = m.predict_proba({"direction": "short", "p3_vwap_dist_atr": -2.5})
    near = m.predict_proba({"direction": "short", "p3_vwap_dist_atr": 0.5})
    assert ext < near


def test_forward_test_scores_only_unseen_sessions(tmp_path, monkeypatch):
    import random
    import prob_eval
    import prob_model as pm
    rnd = random.Random(5)

    def row(day):
        d = rnd.choice(["long", "short"])
        x = rnd.uniform(-2, 2)
        return {"symbol": f"S{rnd.randint(0, 20)}", "ts": f"{day}T10:00",
                "trade_date": day, "score": 60 + 5 * rnd.randint(0, 6),
                "direction": d, "r_multiple": None,
                "label_win": int(rnd.random() < (0.3 if x > 0 else 0.1)),
                "features": {"direction": d, "p3_vwap_dist_atr": x if d == "long" else -x}}
    old = [row(f"2026-07-{d:02d}") for d in range(1, 29) for _ in range(10)]
    new = [row(f"2026-08-{d:02d}") for d in range(1, 29) for _ in range(10)]
    m = pm.LogisticModel().fit([r["features"] for r in old],
                               [r["label_win"] for r in old])
    m.meta = {"dataset": {"date_range": ["2026-07-01", "2026-07-28"]}}
    path = tmp_path / "frozen.json"
    m.save(path)
    monkeypatch.setattr(prob_eval.sim_dataset, "load_rows",
                        lambda **kw: old + new)
    rep = prob_eval.forward_test(path)
    assert rep["since"] == "2026-07-28" and rep["n_rows"] == len(new)
    assert rep["dates"][0] >= "2026-08-01"
    assert rep["auc"] > 0.55                    # synthetic signal carries over
    monkeypatch.setattr(prob_eval.sim_dataset, "load_rows", lambda **kw: old)
    assert "note" in prob_eval.forward_test(path)   # nothing unseen yet
