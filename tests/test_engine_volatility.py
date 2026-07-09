"""
Volatility Engine: ATR-percentile ranking/bounds/windowing, NR7 and inside-bar
detection, the compressed/normal/expanded state thresholds (inclusive on both
boundaries), and totality + JSON-safety of evaluate().

All frames are built deterministically from explicit High-Low RANGES (no RNG,
no network). The `_ohlc_ranges` helper centers every bar on a constant close,
which makes the True Range of bar i EXACTLY ranges[i] (H-L dominates the two
close-gap terms), so the Wilder ATR is a pure function of the ranges we choose
— every percentile below is therefore hand-predictable and stable.

    venv/Scripts/python.exe -m pytest tests/test_engine_volatility.py -q
"""
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG  # noqa: E402
from engines import volatility as vol  # noqa: E402


# ---------- deterministic synthetic frames (no RNG) ----------

def _ohlc_ranges(ranges, base=100.0):
    """OHLC where bar i has High-Low == ranges[i], centered on a constant
    close. TR reduces to exactly ranges[i], so ATR is a clean function of the
    range sequence — the whole point, so percentiles are predictable."""
    highs = [base + r / 2.0 for r in ranges]
    lows = [base - r / 2.0 for r in ranges]
    closes = [base] * len(ranges)
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows,
                         "Close": closes, "Volume": [1e5] * len(ranges)})


def _df(highs, lows, closes=None):
    """Explicit High/Low frame for the bar-shape tests (inside bar / NR)."""
    closes = closes if closes is not None else list(lows)
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows,
                         "Close": closes, "Volume": [1e5] * len(highs)})


# Verified via the ATR pipeline (see calibration): a final spike drives the
# last ATR to the top of its window -> percentile 100.
_SPIKE = _ohlc_ranges([1.0] * 59 + [50.0])            # pct == 100.0 -> expanded
# Oscillating ranges leave the last ATR mid-ranked -> a genuine 'normal'.
_CALM_SINE = _ohlc_ranges([6.0 + 4.0 * math.sin(i / 3.0) for i in range(60)])  # ~42.55
# High constant range then a shrinking tail -> last ATR is the minimum.
_COMPRESSION = _ohlc_ranges([10.0] * 55 + [8.0, 6.0, 4.0, 2.0, 1.0])  # ~2.13
# High block then a low block: the recent (short) window ranks the current ATR
# higher than the full window, which still contains the old high-vol regime.
_LOOKBACK = _ohlc_ranges([100.0] * 30 + [1.0] * 30)


# ---------- atr_percentile: bounds, ranking, windowing, totality ----------

def test_atr_percentile_bounds():
    p = vol.atr_percentile(_CALM_SINE)
    assert p is not None
    assert 0.0 <= p <= 100.0


def test_atr_percentile_final_spike_ranks_high():
    spike = vol.atr_percentile(_SPIKE)
    calm = vol.atr_percentile(_CALM_SINE)
    # A one-bar volatility explosion puts the current ATR at the top of its
    # own history -> ranks near the ceiling, and strictly above a non-spiking
    # series' current rank.
    assert spike >= 90.0
    assert spike > calm


def test_atr_percentile_compression_ranks_low():
    # A shrinking-range tail leaves the current ATR the smallest in its window.
    assert vol.atr_percentile(_COMPRESSION) < CONFIG.volatility.compression_pctile


def test_atr_percentile_lookback_windows_the_history():
    # Same frame, two windows: the short window sees only the recent low-vol
    # block (current ranks relatively higher within it); the full window still
    # holds the old high-vol block that dwarfs the current ATR. Proves the
    # lookback argument actually bounds the comparison set.
    full = vol.atr_percentile(_LOOKBACK, lookback=60)
    short = vol.atr_percentile(_LOOKBACK, lookback=5)
    assert full is not None and short is not None
    assert short > full
    assert 0.0 <= full <= 100.0 and 0.0 <= short <= 100.0


def test_atr_percentile_insufficient_or_invalid_is_none():
    assert vol.atr_percentile(_ohlc_ranges([2.0] * 5)) is None   # < atr_period
    assert vol.atr_percentile(None) is None
    assert vol.atr_percentile(pd.DataFrame()) is None
    # Missing the Close column the ATR path requires.
    assert vol.atr_percentile(pd.DataFrame({"High": [1.0] * 30,
                                            "Low": [0.0] * 30})) is None


# ---------- NR7 / NR-n ----------

def test_nr_fires_on_narrowest_last_bar():
    # Nine wide bars then a tiny one: the last range is strictly the narrowest
    # of the trailing seven -> NR7 fires.
    assert vol.is_nr_n(_ohlc_ranges([5.0] * 9 + [1.0])) is True


def test_nr_does_not_fire_when_last_bar_is_not_narrowest():
    # Last range (7.0) is the widest of the trailing seven -> not an NR7.
    assert vol.is_nr_n(_ohlc_ranges([5.0] * 9 + [7.0])) is False


def test_nr_respects_the_n_window():
    # Last bar (range 2) is the narrowest of the last THREE (9, 8, 2) but not
    # of the last SEVEN (four bars of range 1 sit below it).
    df = _ohlc_ranges([1.0, 1.0, 1.0, 1.0, 9.0, 8.0, 2.0])
    assert vol.is_nr_n(df, 3) is True
    assert vol.is_nr_n(df, 7) is False


def test_nr_none_when_too_few_bars():
    assert vol.is_nr_n(_ohlc_ranges([1.0] * 5), 7) is None
    assert vol.is_nr_n(None) is None
    assert vol.is_nr_n(pd.DataFrame()) is None


# ---------- inside bar ----------

def test_inside_bar_true_when_nested():
    # last H=105<=110 and L=95>=90 -> fully nested inside the prior range.
    assert vol.is_inside_bar(_df([110.0, 105.0], [90.0, 95.0])) is True


def test_inside_bar_equal_bounds_is_inclusive_true():
    # Identical high and low as the prior bar still nests (<= / >= inclusive).
    assert vol.is_inside_bar(_df([110.0, 110.0], [90.0, 90.0])) is True


def test_inside_bar_false_when_high_breaks_out():
    assert vol.is_inside_bar(_df([110.0, 112.0], [90.0, 95.0])) is False


def test_inside_bar_false_when_low_breaks_out():
    assert vol.is_inside_bar(_df([110.0, 108.0], [90.0, 88.0])) is False


def test_inside_bar_none_with_fewer_than_two_bars():
    assert vol.is_inside_bar(_df([110.0], [90.0])) is None
    assert vol.is_inside_bar(None) is None


# ---------- state thresholds (inclusive on both boundaries) ----------

def test_state_thresholds_at_boundaries():
    cfg = CONFIG.volatility
    lo, hi = cfg.compression_pctile, cfg.expansion_pctile   # 25.0 / 75.0
    # Exactly on the compression boundary -> compressed (<= is inclusive).
    assert vol._state_from_percentile(lo, cfg) == "compressed"
    assert vol._state_from_percentile(lo - 1e-6, cfg) == "compressed"
    # Just above it, and just below the expansion boundary -> normal.
    assert vol._state_from_percentile(lo + 1e-6, cfg) == "normal"
    assert vol._state_from_percentile(50.0, cfg) == "normal"
    assert vol._state_from_percentile(hi - 1e-6, cfg) == "normal"
    # Exactly on / above the expansion boundary -> expanded (>= is inclusive).
    assert vol._state_from_percentile(hi, cfg) == "expanded"
    assert vol._state_from_percentile(hi + 1e-6, cfg) == "expanded"
    # Unknown percentile stays unknown.
    assert vol._state_from_percentile(None, cfg) is None


def test_evaluate_state_matches_regime_end_to_end():
    # The three regimes surface through the full evaluate() pipeline.
    assert vol.evaluate(_COMPRESSION).values["state"] == "compressed"
    assert vol.evaluate(_CALM_SINE).values["state"] == "normal"
    assert vol.evaluate(_SPIKE).values["state"] == "expanded"


# ---------- evaluate: contract, totality, JSON-safety ----------

def test_evaluate_contract_and_schema_when_populated():
    res = vol.evaluate(_CALM_SINE)
    assert res.engine == "volatility"
    assert res.ok is True                       # a feature engine never gates
    assert set(res.values) == set(vol.VALUE_KEYS)
    # A 60-bar frame populates everything -> no 'unknown' diagnostics.
    assert res.diagnostics == []
    v = res.values
    assert isinstance(v["atr"], float)
    assert isinstance(v["atr_pct"], float)
    assert isinstance(v["atr_percentile"], float)
    assert isinstance(v["nr7"], bool)
    assert isinstance(v["inside_bar"], bool)
    assert v["state"] in ("compressed", "normal", "expanded")


def test_evaluate_is_total_on_thin_and_invalid_frames():
    # None, empty, and a single-row frame must all yield the full all-None
    # schema with a diagnostic and NEVER raise.
    for bad in (None, pd.DataFrame(), _ohlc_ranges([2.0])):
        res = vol.evaluate(bad)
        assert res.engine == "volatility"
        assert res.ok is True
        assert set(res.values) == set(vol.VALUE_KEYS)
        assert all(val is None for val in res.values.values())
        assert len(res.diagnostics) >= 1


def test_evaluate_values_are_json_safe():
    # values must serialize and hold only JSON-safe scalars. Numerics are
    # float, the two contraction flags are bool, and `state` is the single
    # documented string label; None marks any unknown.
    for frame in (_CALM_SINE, _COMPRESSION, _SPIKE, pd.DataFrame(), None):
        res = vol.evaluate(frame)
        json.dumps(res.values)                  # must not raise
        for k, val in res.values.items():
            assert val is None or isinstance(val, (bool, float, str)), \
                f"{k}: {type(val)}"
