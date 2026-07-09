"""
Phase-1 Learning-Engine data collection: decision-time feature snapshots.

Covers:
- schema migration on a pre-existing (old-schema) DB — columns appear, no data loss
- log_alert(score=, features=) round-trip through get_alert_features
- get_training_rows joins features with outcomes (incl. MFE/MAE columns)
- feature builders produce flat, JSON-safe dicts for both engines

    venv/Scripts/python.exe -m pytest tests/test_features_logging.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subscriptions  # noqa: E402
from features import (  # noqa: E402
    features_from_scorecard, features_from_swing_result, FEATURE_SCHEMA_VERSION,
)
from intraday_score import ScoreCard  # noqa: E402


def _fresh_db(name: str) -> None:
    p = Path(tempfile.gettempdir()) / name
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    subscriptions.DB_PATH = p
    subscriptions.init_db()


def _log(symbol="TEST.NS", **kw) -> int:
    defaults = dict(category="scan", user_id=None, symbol=symbol,
                    setup="score93", direction="long",
                    entry=100.0, stop_loss=98.0, target1=104.0, target2=106.0)
    defaults.update(kw)
    return subscriptions.log_alert(**defaults)


def test_migration_adds_columns_to_old_schema_db():
    """A DB created with the PRE-Phase-1 DDL gains the new columns on init_db
    and old rows survive with NULLs in them."""
    p = Path(tempfile.gettempdir()) / "features_migration_test.db"
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    import sqlite3
    conn = sqlite3.connect(p)
    conn.execute("""
        CREATE TABLE alerts_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at DATETIME NOT NULL, trade_date DATE NOT NULL,
            category TEXT NOT NULL, user_id INTEGER, symbol TEXT NOT NULL,
            setup TEXT, direction TEXT NOT NULL, entry REAL NOT NULL,
            stop_loss REAL NOT NULL, target1 REAL NOT NULL, target2 REAL NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE alert_outcomes (
            alert_id INTEGER PRIMARY KEY, status TEXT NOT NULL,
            exit_price REAL, exit_time DATETIME, pnl_pct REAL,
            resolved_at DATETIME NOT NULL
        )""")
    conn.execute(
        "INSERT INTO alerts_log (generated_at, trade_date, category, user_id, "
        "symbol, setup, direction, entry, stop_loss, target1, target2) "
        "VALUES ('2026-01-01T00:00:00', '2026-01-01', 'scan', 1, 'OLD.NS', "
        "'score91', 'long', 100, 98, 104, 106)")
    conn.commit()
    conn.close()

    subscriptions.DB_PATH = p
    subscriptions.init_db()   # migration

    with subscriptions._conn() as c:
        al_cols = [r[1] for r in c.execute("PRAGMA table_info(alerts_log)")]
        oc_cols = [r[1] for r in c.execute("PRAGMA table_info(alert_outcomes)")]
        old = c.execute("SELECT symbol, score, features FROM alerts_log").fetchone()
    assert "score" in al_cols and "features" in al_cols
    assert {"mfe_pct", "mae_pct", "duration_min", "notified"} <= set(oc_cols)
    assert old == ("OLD.NS", None, None)   # old row intact, new cols NULL


def test_log_alert_feature_roundtrip():
    _fresh_db("features_roundtrip_test.db")
    feats = {"_v": FEATURE_SCHEMA_VERSION, "engine": "intraday_score",
             "score": 93, "rvol": 2.13, "regime_ok": True, "gap_pct": None}
    aid = _log(score=93.0, features=feats)

    back = subscriptions.get_alert_features(aid)
    assert back == feats                       # exact round-trip incl. None

    # score landed in the numeric column
    with subscriptions._conn() as c:
        row = c.execute("SELECT score, setup FROM alerts_log WHERE id=?",
                        (aid,)).fetchone()
    assert row[0] == 93.0
    assert row[1] == "score93"                 # legacy string still written

    # absent features → None, not an exception
    aid2 = _log(symbol="NOFEAT.NS")
    assert subscriptions.get_alert_features(aid2) is None


def test_training_rows_join_features_and_outcome():
    _fresh_db("features_training_test.db")
    aid = _log(score=95.0, features={"score": 95, "rvol": 1.7})
    _log(symbol="UNRESOLVED.NS", features={"score": 91})   # no outcome → excluded
    _log(symbol="NOFEATS.NS")                              # no features → excluded
    subscriptions.save_outcome(aid, status="t2_hit", exit_price=106.0,
                               pnl_pct=7.5, mfe_pct=9.0, mae_pct=-1.0,
                               duration_min=35.0)

    rows = subscriptions.get_training_rows()
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "TEST.NS"
    assert r["features"] == {"score": 95, "rvol": 1.7}
    assert r["status"] == "t2_hit"
    assert r["mfe_pct"] == 9.0 and r["mae_pct"] == -1.0
    assert r["duration_min"] == 35.0
    assert r["score"] == 95.0

    assert subscriptions.get_training_rows(("swing_auto",)) == []   # category filter


def _dummy_card(**kw) -> ScoreCard:
    defaults = dict(
        symbol="TEST.NS", price=515.86, direction="long", score=83,
        rating="Strong Buy",
        breakdown={"Gap Up/Down": 10, "Relative Volume": 18,
                   "VWAP Confirmation": 15, "EMA Trend": 15,
                   "ORB Breakout": 15, "Volume Breakout": 10},
        signals={}, entry=515.86, stop_loss=511.77, target1=524.06,
        target2=528.16, bull_conviction=80, bear_conviction=20,
        gap_pct=1.26, rvol=1.8, orb_window=15,
        regime_ok=True, event_ok=True, delivery_pct=48.0,
        context=["Supertrend: bullish", "Near 52-week high"],
        notes=[],
    )
    defaults.update(kw)
    return ScoreCard(**defaults)


def test_scorecard_features_flat_and_json_safe():
    f = features_from_scorecard(_dummy_card(), idx_trend={"NIFTY": "bullish"})
    assert f["_v"] == FEATURE_SCHEMA_VERSION
    assert f["engine"] == "intraday_score"
    assert f["score"] == 83 and f["direction"] == "long"
    assert f["pts_gap"] == 10 and f["pts_rvol"] == 18 and f["pts_orb"] == 15
    assert f["idx_nifty"] == "bullish"
    assert f["near_52w_high"] is True and f["near_52w_low"] is False
    assert f["supertrend_agrees"] is True
    assert f["risk_pct"] is not None and f["rr1"] is not None
    assert "min_since_open" in f and "weekday" in f
    json.dumps(f)                                    # JSON-safe throughout
    assert all(not isinstance(v, (dict, list)) for v in f.values())   # flat


def test_swing_features_votes_and_levels():
    result = {
        "symbol": "TEST.NS", "mode": "swing", "signal": "BUY",
        "price": 1616.42, "change_pct": 0.25, "rsi": 61.2, "confidence": 75.0,
        "market_open": False,
        "indicators": [("SMA Crossover", "BUY", "x"), ("RSI", "HOLD", "x"),
                       ("MACD", "BUY", "x"), ("Bollinger", "SELL", "x")],
        "trade_setup": {"action": "BUY", "entry": 1616.42, "stop_loss": 1587.75,
                        "target1": 1659.41, "target2": 1702.4, "atr": 19.11,
                        "risk_level": "Medium", "risk_pct": 1.77,
                        "swing_high": 1622.43, "swing_low": 1555.05},
    }
    f = features_from_swing_result(result)
    assert f["engine"] == "analyzer" and f["mode"] == "swing"
    assert f["vote_sma"] == 1 and f["vote_rsi"] == 0
    assert f["vote_macd"] == 1 and f["vote_bb"] == -1
    assert f["atr_pct"] is not None and f["rr1"] is not None
    assert f["swing_high_dist_pct"] is not None
    json.dumps(f)
    assert all(not isinstance(v, (dict, list)) for v in f.values())


def test_builders_never_choke_on_sparse_input():
    # direction-none card with everything empty
    card = _dummy_card(direction="none", entry=None, stop_loss=None,
                       target1=None, target2=None, gap_pct=None, rvol=None,
                       orb_window=None, breakdown={}, context=[], notes=["thin"])
    f = features_from_scorecard(card)
    assert f["risk_pct"] is None and f["pts_gap"] == 0
    json.dumps(f)

    f2 = features_from_swing_result({"signal": "HOLD", "trade_setup": {}})
    assert f2["price"] is None and f2["rr1"] is None
    json.dumps(f2)
