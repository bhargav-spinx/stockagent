"""
Local candle archive (Phase 1.3): round-trip fidelity, idempotent upsert,
range slicing, timezone convention (naive = IST), the never-raise fetch hook,
the fetch_data auto-archive wiring, and backtest --local mode.

Offline — no network.

    venv/Scripts/python.exe -m pytest tests/test_archive.py
"""
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_archive  # noqa: E402
from constants import IST  # noqa: E402


@pytest.fixture()
def fresh_archive(monkeypatch):
    p = Path(tempfile.gettempdir()) / "archive_test.db"
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    monkeypatch.setattr(data_archive, "ARCHIVE_DB_PATH", p)
    return p


def _df5(n=6, start="2026-06-01 09:15", tz="Asia/Kolkata"):
    idx = pd.date_range(start, periods=n, freq="5min", tz=tz)
    return pd.DataFrame({
        "Open": [100.0 + i for i in range(n)],
        "High": [101.0 + i for i in range(n)],
        "Low": [99.0 + i for i in range(n)],
        "Close": [100.5 + i for i in range(n)],
        "Volume": [1000.0 * (i + 1) for i in range(n)],
    }, index=idx)


def test_store_load_round_trip(fresh_archive):
    df = _df5()
    n = data_archive.store_dataframe("TEST.NS", "5m", df, source="unit")
    assert n == len(df)

    back = data_archive.load("TEST.NS", "5m")
    assert list(back.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(back.index.tz) == str(IST)
    pd.testing.assert_frame_equal(
        back, df.astype(float), check_freq=False)


def test_upsert_is_idempotent_and_settles_forming_candle(fresh_archive):
    df = _df5()
    data_archive.store_dataframe("TEST.NS", "5m", df)
    data_archive.store_dataframe("TEST.NS", "5m", df)      # same rows again
    assert data_archive.coverage("TEST.NS")[0]["rows"] == len(df)

    # A re-fetch with a settled last candle overwrites the forming snapshot.
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("Close")] = 999.0
    data_archive.store_dataframe("TEST.NS", "5m", df2)
    back = data_archive.load("TEST.NS", "5m")
    assert len(back) == len(df)
    assert back["Close"].iloc[-1] == 999.0


def test_range_slicing_and_naive_input(fresh_archive):
    naive = _df5(tz=None)
    naive.index = naive.index.tz_localize(None) if naive.index.tz else naive.index
    data_archive.store_dataframe("N.NS", "5m", naive)      # naive → assumed IST

    back = data_archive.load(
        "N.NS", "5m",
        start=datetime(2026, 6, 1, 9, 25),                 # naive bound → IST
        end=datetime(2026, 6, 1, 9, 35),
    )
    assert len(back) == 3                                  # 09:25, 09:30, 09:35
    assert back.index[0].strftime("%H:%M") == "09:25"


def test_empty_and_missing_column_inputs(fresh_archive):
    assert data_archive.store_dataframe("X.NS", "5m", None) == 0
    assert data_archive.store_dataframe("X.NS", "5m", pd.DataFrame()) == 0
    bad = pd.DataFrame({"Close": [1.0]},
                       index=pd.date_range("2026-06-01", periods=1, tz="UTC"))
    assert data_archive.store_dataframe("X.NS", "5m", bad) == 0   # missing cols
    empty = data_archive.load("X.NS", "5m")
    assert empty.empty and list(empty.columns) == data_archive._COLUMNS


def test_archive_fetch_never_raises_and_honors_disable(fresh_archive, monkeypatch):
    data_archive.archive_fetch("Y.NS", "5m", "not a dataframe", "unit")  # no raise

    monkeypatch.setenv("ARCHIVE_DISABLED", "true")
    data_archive.archive_fetch("Y.NS", "5m", _df5(), "unit")
    assert data_archive.coverage("Y.NS") == []             # nothing written

    monkeypatch.delenv("ARCHIVE_DISABLED")
    data_archive.archive_fetch("Y.NS", "5m", _df5(), "unit")
    assert data_archive.coverage("Y.NS")[0]["rows"] == 6


def test_fetch_data_auto_archives(fresh_archive, monkeypatch):
    import data_provider
    fixture = _df5()
    monkeypatch.setattr(data_provider, "_fetch_routed",
                        lambda symbol, period, interval: (fixture, "unit"))
    out = data_provider.fetch_data("HOOK.NS", period="5d", interval="5m")
    assert out is fixture                                  # fetch result untouched
    assert data_archive.coverage("HOOK.NS", "5m")[0]["rows"] == len(fixture)


def test_backtest_local_mode_reads_archive(fresh_archive, monkeypatch):
    import backtest
    df = _df5(n=40)
    data_archive.store_dataframe("LOCAL.NS", "5m", df)
    backtest.set_local_mode(True)
    try:
        # days window must reach back to the fixture's (fixed) 2026-06-01 dates
        sym, got = backtest._fetch_5m("LOCAL", days=365)
        assert sym == "LOCAL.NS"
        assert len(got) == 40
        # empty archive symbol degrades to an empty frame, not an exception
        sym2, got2 = backtest._fetch_5m("MISSING", days=365)
        assert len(got2) == 0
    finally:
        backtest.set_local_mode(False)
