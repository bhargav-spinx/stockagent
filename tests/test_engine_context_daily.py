"""
Daily Context Engine: ADX construction sanity (bounds, warmup, trend vs chop,
Wilder DM selection) and continuous_features totality / known answers.

    venv/Scripts/python.exe -m pytest tests/test_engine_context_daily.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines import context_daily as cd  # noqa: E402


# ---------- deterministic synthetic frames (no RNG) ----------

def _ohlc(closes, spread=0.5):
    """OHLC frame with High/Low a fixed band around close — deterministic."""
    c = pd.Series([float(x) for x in closes])
    return pd.DataFrame({
        "Open": c, "High": c + spread, "Low": c - spread,
        "Close": c, "Volume": 100000.0,
    })


def _trending(n=60):
    """Monotonic +1/bar ramp: every bar advances the high, none breaks low."""
    return _ohlc([100.0 + i for i in range(n)])


def _oscillating(n=60):
    """Direction flips every couple of bars via sin — no persistent trend."""
    return _ohlc([100.0 + 3.0 * math.sin(i * math.pi / 2) for i in range(n)])


def _wavy(n=80):
    """Gently varying series so RSI(9) and RSI(14) genuinely differ."""
    return _ohlc([100.0 + 5.0 * math.sin(i / 3.0) + 0.1 * i for i in range(n)])


# ---------- ADX ----------

def test_adx_bounds_and_warmup():
    a = cd.adx(_trending(60), period=14)
    # Warmup: DM/TR smoothing uses min_periods=14, so nothing before index 13,
    # and the DX smoothing needs 14 more valid points on top of that.
    assert a.iloc[:14].isna().all()
    valid = a.dropna()
    assert len(valid) > 0
    assert ((valid >= 0) & (valid <= 100)).all()


def test_adx_trend_vs_chop():
    trend = cd.adx(_trending(60)).iloc[-1]
    chop = cd.adx(_oscillating(60)).iloc[-1]
    # Ramp: +DM=1 every bar, -DM=0 -> DX=100 -> ADX ~100.
    # Sin flip-flop: +DI ~ -DI -> DX small -> ADX small.
    assert trend > 90
    assert trend > chop + 25


def test_directional_movement_hand_example():
    # bar0: H=100 L=90
    # bar1: H=105 L=95  -> up=+5, down=-5  -> +DM=5 (up>down, up>0), -DM=0
    # bar2: H=104 L=88  -> up=-1, down=+7  -> -DM=7 (down>up, down>0), +DM=0
    # bar3: H=106 L=93  -> up=+2, down=-5  -> +DM=2, -DM=0
    df = pd.DataFrame({
        "High": [100.0, 105.0, 104.0, 106.0],
        "Low": [90.0, 95.0, 88.0, 93.0],
        "Close": [95.0, 100.0, 96.0, 99.0],
    })
    plus_dm, minus_dm = cd._directional_movement(df)
    assert plus_dm.iloc[0] == 0.0 and minus_dm.iloc[0] == 0.0  # NaN diff -> 0
    assert plus_dm.iloc[1] == 5.0 and minus_dm.iloc[1] == 0.0
    assert plus_dm.iloc[2] == 0.0 and minus_dm.iloc[2] == 7.0
    assert plus_dm.iloc[3] == 2.0 and minus_dm.iloc[3] == 0.0


def test_adx_constant_prices_is_nan_not_crash():
    # Constant closes, constant band: DMs are all 0 -> DI sum 0 -> guarded NaN.
    a = cd.adx(_ohlc([100.0] * 60))
    assert a.isna().all()


# ---------- continuous_features: totality ----------

def test_short_df_all_none():
    feats = cd.continuous_features(_trending(10), mode="swing")
    assert set(feats) == set(cd.FEATURE_KEYS)
    assert all(v is None for v in feats.values())


def test_garbage_inputs_all_none():
    # None frame, empty frame, missing OHLC columns, unknown mode — each must
    # return the schema-stable all-None dict, never raise.
    assert all(v is None for v in cd.continuous_features(None).values())
    assert all(v is None for v in cd.continuous_features(pd.DataFrame()).values())
    assert all(v is None for v in
               cd.continuous_features(pd.DataFrame({"Close": [1.0] * 60})).values())
    assert all(v is None for v in
               cd.continuous_features(_trending(60), mode="no_such_mode").values())


def test_run_engine_result_contract():
    res = cd.run(None, mode="swing")
    assert res.engine == "context_daily"
    assert res.ok is True
    assert set(res.values) == set(cd.FEATURE_KEYS)
    assert all(v is None for v in res.values.values())
    assert len(res.diagnostics) == 1  # the all-None case must leave a note


# ---------- continuous_features: known answers ----------

def test_atr_pct_known_answer_constant_range():
    # Close constant at 100, High=101, Low=99 every bar:
    #   TR = max(H-L, |H-prevC|, |L-prevC|) = max(2, 1, 1) = 2 each bar
    #   Wilder RMA of a constant 2 stays exactly 2
    #   atr_pct = 2 / 100 * 100 = 2.0
    df = _ohlc([100.0] * 60, spread=1.0)
    feats = cd.continuous_features(df, mode="swing")
    assert feats["ctx_atr_pct"] == 2.0
    # Constant closes: rolling std = 0 -> bb_z guarded to None, not inf.
    assert feats["ctx_bb_z"] is None
    # (close - SMA50)/SMA50 = (100-100)/100 = 0
    assert feats["ctx_dist_sma_long_pct"] == 0.0
    # pct_change over 5 and 20 candles of a constant = 0
    assert feats["ctx_ret_5"] == 0.0 and feats["ctx_ret_20"] == 0.0


def test_bb_z_sign_matches_close_vs_sma():
    up = cd.continuous_features(_trending(60), mode="swing")
    down = cd.continuous_features(_ohlc([160.0 - i for i in range(60)]),
                                  mode="swing")
    # Rising ramp: last close sits above the 20-bar middle band -> z > 0;
    # falling ramp is the mirror image -> z < 0.
    assert up["ctx_bb_z"] > 0
    assert down["ctx_bb_z"] < 0
    # Same logic must show in the SMA-long distance and the EMA slope.
    assert up["ctx_dist_sma_long_pct"] > 0 > down["ctx_dist_sma_long_pct"]
    assert up["ctx_ema_fast_slope_pct"] > 0 > down["ctx_ema_fast_slope_pct"]


def test_json_safe_and_types():
    feats = cd.continuous_features(_wavy(80), mode="swing")
    json.dumps(feats)  # must not raise (no numpy scalars, no NaN)
    for k, v in feats.items():
        assert v is None or isinstance(v, float), f"{k}: {type(v)}"
    # A wavy 80-bar frame is long enough that everything should populate.
    assert all(v is not None for v in feats.values())


def test_mode_params_change_values():
    # swing uses rsi_period=14, intraday uses 9 (analyzer.MODES): on the same
    # frame the two RSI levels must differ, proving mode params are honored.
    df = _wavy(80)
    swing = cd.continuous_features(df, mode="swing")
    intra = cd.continuous_features(df, mode="intraday")
    assert swing["ctx_rsi"] is not None and intra["ctx_rsi"] is not None
    assert swing["ctx_rsi"] != intra["ctx_rsi"]
    # ADX(14) deliberately ignores mode params -> identical across modes.
    assert swing["ctx_adx"] == intra["ctx_adx"]


def test_alias_matches():
    df = _wavy(80)
    assert cd.features_from_analysis(df, "swing") == \
        cd.continuous_features(df, mode="swing")
