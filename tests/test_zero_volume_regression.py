"""
Regression: zero-volume candles must not crash the indicator/scoring path.

THE INCIDENT (2026-07-20): scanner_indicators.vwap used
`cum_v.replace(0, pd.NA)`. Real archived 5-min data contains no-trade bars, so
cumulative volume is 0 at session start → pd.NA → object dtype → pandas 3
raises "float() argument must be a string or a real number, not 'NAType'" from
.astype(float). Every score_stock call on archived candles died; sim_dataset
swallowed 1008 failures per symbol at DEBUG and reported a serene "0 rows"
across all 190 symbols. The same landmine sat under the LIVE scanner for any
symbol whose session opened untraded.

Synthetic fixtures elsewhere always have volume > 0, which is exactly why the
whole suite stayed green through it — hence this file.

    venv/Scripts/python.exe -m pytest tests/test_zero_volume_regression.py
"""
import logging
import math
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import intraday_score  # noqa: E402
import sim_dataset  # noqa: E402
from scanner_indicators import vwap  # noqa: E402


def _frame(volumes, start="2026-06-15 09:15"):
    n = len(volumes)
    idx = pd.date_range(start, periods=n, freq="5min", tz="Asia/Kolkata")
    base = [100.0 + 0.1 * i for i in range(n)]
    return pd.DataFrame({
        "Open": base,
        "High": [b + 0.5 for b in base],
        "Low": [b - 0.5 for b in base],
        "Close": [b + 0.1 for b in base],
        "Volume": [float(v) for v in volumes],
    }, index=idx)


def test_vwap_survives_leading_zero_volume():
    """The exact production shape: session opens with no-trade bars."""
    s = vwap(_frame([0, 0, 100, 200, 300]))
    assert str(s.dtype) == "float64"           # never object/NA dtype
    assert math.isnan(s.iloc[0]) and math.isnan(s.iloc[1])   # undefined, not a crash
    assert s.iloc[2] == pytest.approx(100.2, abs=0.5)        # real once traded
    assert s.iloc[-1] > 0


def test_vwap_all_zero_volume_is_all_nan_not_error():
    s = vwap(_frame([0, 0, 0, 0]))
    assert str(s.dtype) == "float64"
    assert s.isna().all()


def test_vwap_unchanged_when_no_zero_volume():
    """Guards the fix itself: normal data must produce identical numbers to
    the classic typical-price VWAP definition."""
    df = _frame([100, 200, 300, 400])
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    expected = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    assert vwap(df).tolist() == pytest.approx(expected.tolist())


def _two_session_frame_with_dead_open() -> pd.DataFrame:
    """Two sessions, each opening with zero-volume bars — the archive shape
    that killed 1008 candles per symbol."""
    rows, idx = [], []
    for s, day in enumerate((datetime(2026, 6, 15), datetime(2026, 6, 16))):
        start = day.replace(hour=9, minute=15)
        for i in range(60):
            base = 500.0 + 4.0 * s + 0.15 * i + 0.8 * math.sin(i * 1.3)
            rows.append((base, base + 1.2, base - 1.2, base + 0.2,
                         0.0 if i < 3 else 120_000 + 900 * i))
            idx.append(start + timedelta(minutes=5 * i))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(idx))


def test_score_stock_no_longer_raises_on_zero_volume_open():
    df = _two_session_frame_with_dead_open()
    card = intraday_score.score_stock(df, "ZEROVOL.NS", skip_external=True)
    assert card is not None
    assert isinstance(card.score, int)          # scored, not crashed
    assert card.rating in ("Strong Buy", "Strong Sell", "Watchlist", "Avoid")


def test_generate_symbol_reports_score_errors_loudly(monkeypatch, caplog):
    """Observability half of the fix: a systematic scoring crash must SHOUT,
    never present as a calm '0 rows'."""
    monkeypatch.setattr(
        sim_dataset.intraday_score, "score_stock",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(sim_dataset.data_archive, "load",
                        lambda sym, interval: (
                            _two_session_frame_with_dead_open()
                            if interval == "5m" else pd.DataFrame()))
    monkeypatch.setattr(sim_dataset, "_init_table", lambda: None)

    with caplog.at_level(logging.WARNING):
        n = sim_dataset.generate_symbol("BOOM.NS", min_score=0)
    assert n == 0
    text = caplog.text
    assert "score_stock failed" in text
    assert "failed to score" in text and "INCOMPLETE" in text
