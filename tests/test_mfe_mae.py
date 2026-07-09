"""
Excursion telemetry (Phase 1): resolvers report MFE / MAE / duration alongside
the outcome, computed from candle extremes INCLUDING the exit candle.

Known-answer paths on synthetic candles, both directions, plus the guarantee
that the pre-existing status/pnl behavior is untouched (additive keys only —
tests/test_resolver.py continues to pin those).

    venv/Scripts/python.exe -m pytest tests/test_mfe_mae.py
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eod_report  # noqa: E402


def _df(rows, start="2026-01-05 09:30"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}
         for (o, h, l, c) in rows],
        index=idx,
    )


def _alert(direction="long", entry=100.0, sl=98.0, t1=106.0, t2=109.0, gen=None):
    return {
        "symbol": "TEST.NS", "entry": entry, "stop_loss": sl,
        "target1": t1, "target2": t2, "direction": direction,
        "generated_at": gen,
    }


def test_long_t2_hit_excursions():
    df = _df([
        (100, 100, 100, 100),    # gen candle: no excursion
        (100, 103, 99, 102),     # dip to 99 (MAE −1%), high 103 (MFE +3%)
        (102, 110, 101, 108),    # tags T1+T2 → exit; high 110 → MFE +10%
    ])
    out = eod_report.resolve_intraday(_alert(gen=df.index[0]), df=df)
    assert out["status"] == "t2_hit"
    assert out["pnl_pct"] == pytest.approx(7.5)          # unchanged blend
    assert out["mfe_pct"] == pytest.approx(10.0)
    assert out["mae_pct"] == pytest.approx(-1.0)
    assert out["duration_min"] == pytest.approx(10.0)


def test_long_sl_hit_excursions():
    df = _df([
        (100, 100, 100, 100),
        (100, 101, 97.5, 98),    # high 101 → MFE +1%, low 97.5 → MAE −2.5%
    ])
    out = eod_report.resolve_intraday(_alert(gen=df.index[0]), df=df)
    assert out["status"] == "sl_hit"
    assert out["pnl_pct"] == pytest.approx(-2.0)
    assert out["mfe_pct"] == pytest.approx(1.0)
    assert out["mae_pct"] == pytest.approx(-2.5)         # beyond SL distance
    assert out["duration_min"] == pytest.approx(5.0)


def test_short_direction_symmetry():
    # short: entry 100, SL 102, T1 94, T2 91
    df = _df([
        (100, 100, 100, 100),
        (100, 101.5, 93, 94),    # low 93 → MFE +7%; high 101.5 → MAE −1.5%; T1 in
        (94, 95, 90, 91),        # low 90 → MFE +10%; T2 in → exit
    ])
    out = eod_report.resolve_intraday(
        _alert(direction="short", sl=102.0, t1=94.0, t2=91.0, gen=df.index[0]),
        df=df)
    assert out["status"] == "t2_hit"
    assert out["pnl_pct"] == pytest.approx(7.5)          # (6+9)/2
    assert out["mfe_pct"] == pytest.approx(10.0)
    assert out["mae_pct"] == pytest.approx(-1.5)
    assert out["duration_min"] == pytest.approx(10.0)


def test_squareoff_paths_carry_excursions():
    # neither SL nor T1 within the session → squareoff_no_t1 at last close
    df = _df([
        (100, 100, 100, 100),
        (100, 102, 99.5, 101),
        (101, 102.5, 100.5, 101.5),
    ])
    out = eod_report.resolve_intraday(_alert(gen=df.index[0]), df=df)
    assert out["status"] == "squareoff_no_t1"
    assert out["mfe_pct"] == pytest.approx(2.5)
    assert out["mae_pct"] == pytest.approx(-0.5)
    assert out["duration_min"] == pytest.approx(10.0)


def test_swing_resolver_reports_excursions(monkeypatch):
    # daily bars: dip then rally through T1+T2
    idx = pd.date_range("2026-01-06", periods=3, freq="B", tz="Asia/Kolkata")
    df = pd.DataFrame(
        [{"Open": 100, "High": 101, "Low": 96.5, "Close": 99, "Volume": 1},   # skipped (alert date)
         {"Open": 99, "High": 104, "Low": 98.5, "Close": 103, "Volume": 1},
         {"Open": 103, "High": 111, "Low": 102, "Close": 110, "Volume": 1}],
        index=idx,
    )
    monkeypatch.setattr(eod_report, "fetch_data",
                        lambda *a, **k: df)
    alert = _alert(gen=None)
    alert["generated_at"] = "2026-01-06T05:00:00"        # UTC ≈ 10:30 IST that day
    out = eod_report.resolve_swing(alert)
    assert out["status"] == "t2_hit"
    # bars AFTER the alert date only: day2 low 98.5 → MAE −1.5; day3 high 111 → MFE +11
    assert out["mfe_pct"] == pytest.approx(11.0)
    assert out["mae_pct"] == pytest.approx(-1.5)
    assert out["duration_min"] > 0


def test_open_and_no_data_have_no_excursion_keys():
    df = _df([(100, 100, 100, 100)])
    # gen time AFTER the only candle's session → post is empty → "open"
    alert = _alert(gen=df.index[0] + pd.Timedelta(days=1))
    out = eod_report.resolve_intraday(alert, df=df)
    assert out["status"] == "open"
    assert "mfe_pct" not in out
