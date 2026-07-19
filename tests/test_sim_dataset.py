"""
Phase 4.1 — simulated-trade dataset generator.

Verifies on a synthetic archive that generate_symbol():
- fires through the SAME entry-window/score/resolver path as live,
- writes label fields consistent with the resolver status,
- attaches a parseable decision-time feature snapshot (sim=1, p3_* keys),
- is idempotent (PK upsert),
- respects the --min-score sampling floor and --days limit,
- never leaks the SAME-DAY daily bar into the 52-week context (no lookahead).

    venv/Scripts/python.exe -m pytest tests/test_sim_dataset.py
"""
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_archive  # noqa: E402
import sim_dataset  # noqa: E402
import intraday_score  # noqa: E402
import eod_report  # noqa: E402


# ---------- synthetic archive fixtures (borrowed shape from golden parity) ----

def _session_ts(day: datetime, n: int):
    start = day.replace(hour=9, minute=15)
    return [start + timedelta(minutes=5 * i) for i in range(n)]


def _five_min_df() -> pd.DataFrame:
    """Two sessions: day 1 calm around 500 (75 candles), day 2 gaps up ~1.2%
    with a tight ORB, rising drift and a volume ramp — engineered so the score
    engine finds a long with score ≥ 60 somewhere inside the entry window."""
    rows, idx = [], []
    for i, ts in enumerate(_session_ts(datetime(2025, 6, 3), 75)):
        o = 500.0 + 0.05 * i + 0.6 * math.sin(i / 4)
        c = o + 0.05 + 0.25 * math.sin(i / 3 + 1)
        hi = max(o, c) + 1.2 + 0.4 * math.sin(i / 5)
        lo = min(o, c) - 1.2 - 0.4 * math.cos(i / 6)
        v = 100_000 + 15_000 * math.sin(i / 7) ** 2 + 300 * i
        rows.append((o, hi, lo, c, v)); idx.append(ts)
    for i, ts in enumerate(_session_ts(datetime(2025, 6, 4), 60)):
        if i < 3:
            o = 510.0 + 0.10 * i
            c = o + 0.15
            hi = max(o, c) + 0.9
            lo = min(o, c) - 0.9
            v = 220_000.0 - 8_000 * i
        else:
            o = 510.5 + 0.22 * (i - 3) + 0.9 * math.sin(i / 3)
            c = o + 0.10 + 1.15 * math.sin(i * 1.3)
            hi = max(o, c) + 1.1 + 0.3 * math.sin(i / 4)
            lo = min(o, c) - 1.1 - 0.3 * math.cos(i / 5)
            v = 130_000 + 3_000 * i + 15_000 * math.sin(i / 6) ** 2
        rows.append((o, hi, lo, c, v)); idx.append(ts)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(idx))


def _daily_df() -> pd.DataFrame:
    rows, idx = [], []
    day = datetime(2025, 1, 1)
    made, i = 0, 0
    while made < 90:
        d = day + timedelta(days=i); i += 1
        if d.weekday() >= 5:
            continue
        t = made
        c = 480.0 + 0.4 * t + 6.0 * math.sin(t / 9)
        rows.append((c - 1, c + 6, c - 6, c, 1_000_000))
        idx.append(d.replace(hour=15, minute=30))
        made += 1
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(idx))


@pytest.fixture()
def temp_archive(monkeypatch):
    p = Path(tempfile.gettempdir()) / "sim_dataset_test.db"
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    monkeypatch.setattr(data_archive, "ARCHIVE_DB_PATH", p)
    data_archive.init_archive()
    data_archive.store_dataframe("SIM.NS", "5m", _five_min_df(), source="test")
    data_archive.store_dataframe("SIM.NS", "1d", _daily_df(), source="test")
    return p


def test_generate_writes_consistent_labeled_rows(temp_archive):
    n = sim_dataset.generate_symbol("SIM.NS", min_score=60, run_id="t1")
    assert n >= 1

    rows = sim_dataset.load_rows()
    assert len(rows) == n
    r = rows[0]
    assert r["symbol"] == "SIM.NS"
    assert r["direction"] in ("long", "short")
    assert r["score"] >= 60
    # label consistency with the resolver's PASS/FAIL taxonomy
    if r["status"] in eod_report.PASS_STATUSES:
        assert r["label_win"] == 1 and r["label_decisive"] == 1
    elif r["status"] in eod_report.FAIL_STATUSES:
        assert r["label_win"] == 0 and r["label_decisive"] == 1
    else:
        assert r["label_win"] == 0 and r["label_decisive"] == 0
    # decision must sit inside the live entry window (09:30–14:30, no lunch)
    ts = datetime.fromisoformat(r["ts"])
    assert (9, 30) <= (ts.hour, ts.minute) < (14, 30)
    assert not ((12, 0) <= (ts.hour, ts.minute) < (13, 30))
    # feature snapshot: parsed, marked sim, carries base + rich keys
    f = r["features"]
    assert f["sim"] == 1
    assert f["engine"] == "intraday_score"
    assert f["score"] == r["score"]
    assert any(k.startswith("p3_") for k in f)
    json.dumps(f)


def test_generate_is_idempotent(temp_archive):
    n1 = sim_dataset.generate_symbol("SIM.NS", min_score=60, run_id="a")
    n2 = sim_dataset.generate_symbol("SIM.NS", min_score=60, run_id="b")
    assert n1 == n2
    assert len(sim_dataset.load_rows()) == n1     # PK upsert, no duplicates


def test_min_score_floor_filters(temp_archive):
    assert sim_dataset.generate_symbol("SIM.NS", min_score=101) == 0
    assert sim_dataset.load_rows() == []


def test_days_limit_restricts_sessions(temp_archive):
    # last 1 session only → same as full run here (only day 2 can fire, day 1
    # has no prior session), but the restriction path must not error and the
    # allowed-set must hold.
    n = sim_dataset.generate_symbol("SIM.NS", min_score=60, days=1, run_id="d")
    rows = sim_dataset.load_rows()
    assert len(rows) == n
    assert all(r["trade_date"] == "2025-06-04" for r in rows)


def test_daily_context_has_no_same_day_lookahead(temp_archive, monkeypatch):
    seen: list = []
    orig = intraday_score.score_stock

    def spy(df, sym, **kw):
        seen.append(kw.get("daily_df"))
        return orig(df, sym, **kw)

    monkeypatch.setattr(intraday_score, "score_stock", spy)
    monkeypatch.setattr(sim_dataset.intraday_score, "score_stock", spy)
    sim_dataset.generate_symbol("SIM.NS", min_score=0, run_id="look")
    called_with = [d for d in seen if d is not None]
    assert called_with, "expected daily context to be passed"
    for daily in called_with:
        assert max(daily.index.date) < datetime(2025, 6, 4).date()


def _seven_session_df() -> pd.DataFrame:
    """7 sessions of the day-2 template at rising bases — enough history that
    an unwindowed replay would hand the scorer >5 sessions."""
    rows, idx = [], []
    for s in range(7):
        base = 500.0 + 8.0 * s
        day = datetime(2025, 6, 2) + timedelta(days=s if s < 4 else s + 2)  # skip weekend
        for i, ts in enumerate(_session_ts(day, 75)):
            o = base + 0.10 * i + 0.6 * math.sin(i / 4)
            c = o + 0.08 + 0.9 * math.sin(i * 1.3)
            hi = max(o, c) + 1.1
            lo = min(o, c) - 1.1
            v = 120_000 + 2_000 * i + (60_000 if i % 11 == 0 else 0)
            rows.append((o, hi, lo, c, v)); idx.append(ts)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(idx))


def test_scorer_never_sees_more_than_live_window(temp_archive, monkeypatch):
    """FIDELITY PIN: live alerts are computed over a period='5d' fetch, so the
    replay must feed score_stock ≤ LIVE_WINDOW_SESSIONS sessions ending at the
    replayed day — a wider frame would train on features (RVOL baselines etc.)
    that live inference can never reproduce."""
    data_archive.store_dataframe("WIDE.NS", "5m", _seven_session_df(),
                                 source="test")
    captured: list = []
    orig = intraday_score.score_stock

    def spy(df, sym, **kw):
        captured.append(df)
        return orig(df, sym, **kw)

    monkeypatch.setattr(sim_dataset.intraday_score, "score_stock", spy)
    sim_dataset.generate_symbol("WIDE.NS", min_score=0, run_id="win")
    assert captured, "expected the replay to score candles"
    for frame in captured:
        dates = sorted(set(frame.index.date))
        assert len(dates) <= sim_dataset.LIVE_WINDOW_SESSIONS
    # and later replayed days really do use the LATEST window, not the head
    last_frame_dates = sorted(set(captured[-1].index.date))
    assert last_frame_dates[-1] >= datetime(2025, 6, 9).date()


def test_load_rows_time_ordered_and_filters(temp_archive):
    sim_dataset.generate_symbol("SIM.NS", min_score=60, run_id="o")
    rows = sim_dataset.load_rows()
    ts_list = [r["ts"] for r in rows]
    assert ts_list == sorted(ts_list)
    for r in sim_dataset.load_rows(decisive_only=True):
        assert r["label_decisive"] == 1
    assert all(r["score"] >= 80 for r in sim_dataset.load_rows(min_score=80))
