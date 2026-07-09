"""
Config module: defaults must equal the historical literals (belt-and-braces on
top of the golden parity test), JSON overlays must apply partially, and broken
override files must fall back to defaults instead of crashing.

    venv/Scripts/python.exe -m pytest tests/test_config.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def test_defaults_match_historical_literals():
    c = config.Config()
    assert c.score.strong == 80 and c.score.watch == 60
    assert dict(c.score.orb_windows) == {15: 3, 30: 6}
    assert c.score.orb_max_extension_pct == 2.0
    assert c.score.gap_tiers == ((2.0, 20), (1.5, 15), (1.0, 10), (0.5, 5))
    assert c.score.rvol_tiers == ((2.0, 25), (1.5, 18), (1.2, 10))
    assert c.score.volume_breakout_tiers == ((1.5, 10), (1.2, 5))
    assert c.filters.atr_pct_lo == 0.004 and c.filters.atr_pct_hi == 0.015
    assert c.filters.trigger_volume_ratio == 1.5
    assert c.filters.vwap_min_slope_abs == 0.02
    assert c.filters.round_number_threshold == 0.003
    assert c.filters.entry_open == (9, 30) and c.filters.entry_close == (14, 30)
    assert c.setup_a.max_orb_range_pct == 0.012
    assert c.setup_a.min_confluences == 3
    assert (c.setup_a.rsi_long_lo, c.setup_a.rsi_long_hi) == (55.0, 70.0)
    assert (c.setup_a.rsi_short_lo, c.setup_a.rsi_short_hi) == (30.0, 45.0)
    assert c.intraday_risk.atr_period == 14
    assert c.intraday_risk.atr_sl_mult == 1.5
    assert c.intraday_risk.stop_pct_fallback == 0.01
    assert (c.intraday_risk.rr_t1, c.intraday_risk.rr_t2) == (2.0, 3.0)
    assert c.swing_setup.sl_atr_mult_swing == 1.5
    assert c.swing_setup.sl_atr_mult_intraday == 1.0
    assert (c.swing_setup.tgt1_mult, c.swing_setup.tgt2_mult) == (1.5, 3.0)
    assert c.swing_setup.swing_stop_buffer == 0.005
    assert (c.swing_setup.rsi_oversold, c.swing_setup.rsi_overbought) == (30.0, 70.0)
    assert c.swing_setup.min_votes == 2
    assert c.gates.min_intraday_score == 60 and c.gates.min_rr == 2.0
    assert c.gates.autoscan_min_score == 90
    assert c.costs.cost_per_trade_pct == 0.13
    assert c.costs.intraday_time_stop_min == 45


def test_partial_json_overlay():
    overrides = {
        "score": {"strong": 85, "gap_tiers": [[3.0, 20], [1.0, 5]]},
        "filters": {"atr_pct_lo": 0.003},
        "bogus_section": {"x": 1},
        "costs": {"unknown_key": 99},
    }
    p = Path(tempfile.gettempdir()) / "stockagent_cfg_test.json"
    p.write_text(json.dumps(overrides), encoding="utf-8")
    try:
        c = config.load_config(p)
    finally:
        p.unlink()

    assert c.score.strong == 85                        # overridden
    assert c.score.watch == 60                         # untouched default
    assert c.score.gap_tiers == ((3.0, 20), (1.0, 5))  # list→tuple coercion
    assert c.filters.atr_pct_lo == 0.003
    assert c.filters.atr_pct_hi == 0.015
    assert c.costs.cost_per_trade_pct == 0.13          # unknown key ignored


def test_broken_override_file_falls_back_to_defaults():
    p = Path(tempfile.gettempdir()) / "stockagent_cfg_broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    try:
        c = config.load_config(p)
    finally:
        p.unlink()
    assert c == config.Config()

    # Missing file likewise.
    c2 = config.load_config(Path(tempfile.gettempdir()) / "does_not_exist_xyz.json")
    assert c2 == config.Config()
