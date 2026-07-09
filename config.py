"""
Central strategy configuration.

Every tunable threshold of the signal path lives here as a frozen dataclass
field. Defaults are EXACTLY the literals that were previously hardcoded across
intraday_score / scanner_filters / scanner_setups / scanner_indicators /
analyzer / eod_report / narrative — pinned by tests/test_golden_parity.py, so
a transcription typo cannot silently change live behavior.

Why this module exists: "optimize from historical data" is impossible while
thresholds are inline literals. Anything here can be grid-searched by the
walk-forward harness or overridden per-run, and the config in force can be
logged next to results for experiment provenance.

Overrides (optional): set STOCKAGENT_CONFIG=/path/to/overrides.json with a
partial nested mapping, e.g. {"score": {"strong": 85}, "filters":
{"atr_pct_lo": 0.003}}. Unknown sections/keys are warned about and ignored;
lists are coerced to tuples (JSON has no tuples). No file / no env var means
pure defaults — identical to pre-extraction behavior.

Dependency-free (stdlib only) and imported at module load by the engine
modules; treat CONFIG as immutable for the life of the process.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "STOCKAGENT_CONFIG"


@dataclass(frozen=True)
class ScoreConfig:
    """intraday_score.py — the 100-point scorecard."""
    strong: int = 80                    # rating floor: "Strong Buy/Sell"
    watch: int = 60                     # rating floor: "Watchlist"
    # (minutes, n_5min_candles) opening-range windows evaluated for breakout
    orb_windows: tuple[tuple[int, int], ...] = ((15, 3), (30, 6))
    orb_max_extension_pct: float = 2.0  # beyond this % past ORB = spent move
    # (favourable-% threshold, points), descending — first match wins
    gap_tiers: tuple[tuple[float, int], ...] = ((2.0, 20), (1.5, 15), (1.0, 10), (0.5, 5))
    rvol_tiers: tuple[tuple[float, int], ...] = ((2.0, 25), (1.5, 18), (1.2, 10))
    volume_breakout_tiers: tuple[tuple[float, int], ...] = ((1.5, 10), (1.2, 5))
    vwap_points: int = 15
    ema_points: int = 15
    orb_strong_strength_pct: float = 0.10   # breakout % for full ORB points
    orb_points_strong: int = 15
    orb_points_weak: int = 8
    # display-only conviction relabel: clamp(base + slope·score, lo, hi)
    conviction_base: float = 45.0
    conviction_slope: float = 0.42
    conviction_lo: int = 50
    conviction_hi: int = 90
    delivery_strong_pct: float = 60.0
    delivery_low_pct: float = 25.0
    earnings_risk_days: int = 2         # event_ok=False within this many days
    near_52w_band: float = 0.03         # "near 52-week high/low" proximity
    w52_lookback_days: int = 252


@dataclass(frozen=True)
class FilterConfig:
    """scanner_filters.py — §5 universal gates."""
    atr_pct_lo: float = 0.004
    atr_pct_hi: float = 0.015
    trigger_volume_ratio: float = 1.5
    vwap_min_slope_abs: float = 0.02    # % per candle
    round_number_threshold: float = 0.003
    # session times as (hour, minute) IST
    entry_open: tuple[int, int] = (9, 30)
    entry_close: tuple[int, int] = (14, 30)
    lunch_start: tuple[int, int] = (12, 0)
    lunch_end: tuple[int, int] = (13, 30)


@dataclass(frozen=True)
class SetupAConfig:
    """scanner_setups.py — Setup A (ORB)."""
    orb_candles: int = 3                # first 15 min = three 5-min candles
    max_orb_range_pct: float = 0.012    # skip if ORB range wider than this
    rsi_long_lo: float = 55.0
    rsi_long_hi: float = 70.0
    rsi_short_lo: float = 30.0
    rsi_short_hi: float = 45.0
    min_confluences: int = 3


@dataclass(frozen=True)
class IntradayRiskConfig:
    """scanner_indicators.trade_levels — canonical intraday risk model."""
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    stop_pct_fallback: float = 0.01
    rr_t1: float = 2.0
    rr_t2: float = 3.0


@dataclass(frozen=True)
class SwingSetupConfig:
    """analyzer.build_trade_setup + vote thresholds — swing risk model."""
    lookback_swing: int = 20
    lookback_intraday: int = 30
    sl_atr_mult_swing: float = 1.5
    sl_atr_mult_intraday: float = 1.0
    tgt1_mult: float = 1.5
    tgt2_mult: float = 3.0
    swing_stop_buffer: float = 0.005    # SL 0.5% beyond the swing extreme
    breakout_buffer: float = 0.002      # HOLD watch level 0.2% past extreme
    breakout_sl_buffer: float = 0.01    # HOLD watch SL 1% inside extreme
    risk_low_pct: float = 1.5
    risk_medium_pct: float = 3.5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    min_votes: int = 2                  # same-way votes needed for non-HOLD


@dataclass(frozen=True)
class GateConfig:
    """narrative.py gates + alert-layer floors."""
    min_intraday_score: int = 60        # narrative actionability floor
    min_rr: float = 2.0                 # narrative R:R-to-T1 gate
    autoscan_min_score: int = 90        # bot auto-scan alert floor
    conviction_high: int = 90
    conviction_constructive: int = 75
    conviction_mixed: int = 60


@dataclass(frozen=True)
class CostConfig:
    """eod_report.py + engines/execution.py — cost simulation.

    model="flat" keeps the historical 0.13% round-trip estimate (default,
    behavior-identical). model="statutory" itemizes real Indian equity
    charges via engines/execution.py — statutory rates below are the
    published 2025-26 intraday figures; update on Budget changes."""
    cost_per_trade_pct: float = 0.13    # STRATEGY.md §8 round-trip estimate
    intraday_time_stop_min: int = 45
    model: str = "flat"                 # "flat" | "statutory"
    brokerage_pct: float = 0.03         # discount-broker % per side…
    brokerage_cap_inr: float = 20.0     # …capped at ₹20/order
    stt_intraday_sell_pct: float = 0.025    # STT: intraday, sell side only
    stt_delivery_pct: float = 0.1           # STT: delivery, both sides
    exchange_txn_pct: float = 0.00297       # NSE transaction charge per side
    sebi_fee_pct: float = 0.0001            # ₹10/crore per side
    stamp_duty_buy_pct: float = 0.003       # intraday, buy side only
    stamp_duty_delivery_buy_pct: float = 0.015
    gst_pct: float = 18.0               # on brokerage + txn + SEBI fees
    slippage_pct_per_side: float = 0.03


@dataclass(frozen=True)
class RegimeConfig:
    """engines/market_regime.py — deterministic day-type/vol/breadth labels."""
    vix_low: float = 12.0               # IndiaVIX below → low-vol state
    vix_high: float = 17.0              # above → high-vol state
    trend_day_range_mult: float = 1.3   # today range ≥ mult × 10d avg → trend day
    trend_strength_min: float = 0.6     # |net move|/range floor for trend label
    gap_min_pct: float = 0.3            # smaller opening gaps aren't "gap days"
    gap_fill_frac: float = 0.5          # ≥ this fraction retraced → gap_fill
    breadth_bullish: float = 0.60       # universe fraction above VWAP
    breadth_bearish: float = 0.40
    range_lookback_days: int = 10
    expiry_weekday: int = 3             # NIFTY weekly expiry weekday (0=Mon);
                                        # NSE reshuffles this — config, not code
    cache_ttl_min: int = 15             # live snapshot (VIX etc.) cache


@dataclass(frozen=True)
class LiquidityConfig:
    """engines/liquidity.py — proxy-based liquidity floors (no quote feed)."""
    min_median_turnover_cr: float = 5.0     # ₹ crore/day, 20d median floor
    turnover_lookback_days: int = 20
    max_spread_proxy_pct: float = 0.35      # median 5-min (H−L)/C ceiling, %
    max_participation_pct: float = 5.0      # order ≤ this % of per-min volume
    slippage_k: float = 0.05                # est. slippage % per 1% participation


@dataclass(frozen=True)
class VolatilityConfig:
    """engines/volatility.py — ATR percentile, NR7, inside day, squeeze."""
    atr_period: int = 14
    atr_pct_lookback: int = 60          # sessions for the ATR-percentile window
    nr_lookback: int = 7                # NR7 window
    compression_pctile: float = 25.0    # ATR%ile below → compressed (squeeze)
    expansion_pctile: float = 75.0      # ATR%ile above → expanded


@dataclass(frozen=True)
class PriceActionConfig:
    """engines/price_action.py — swing structure, BOS/CHoCH, S/R."""
    fractal_n: int = 2                  # bars each side for a fractal pivot
    swing_atr_mult: float = 0.5         # min pivot move in ATR to count as swing
    swing_lookback: int = 60            # bars scanned for structure
    sr_cluster_atr: float = 0.25        # S/R levels within this ATR frac merge


@dataclass(frozen=True)
class RelStrengthConfig:
    """engines/relative_strength.py — return vs index over windows."""
    index_alias: str = "NIFTY"
    windows: tuple[int, ...] = (5, 20)  # daily-candle lookbacks for RS
    beta_lookback: int = 20             # bars for the beta estimate


@dataclass(frozen=True)
class GapClassConfig:
    """engines/gap.py classification — gap size measured in ATR units."""
    tiny_atr: float = 0.5               # |gap|/ATR below → "tiny"
    normal_atr: float = 1.5             # below → "normal", above → large
    large_atr: float = 3.0             # above → "runaway/exhaustion" range
    breakaway_rvol: float = 1.5         # large gap + this RVOL → breakaway
    fill_frac: float = 0.5              # gap retraced ≥ this → fill candidate


@dataclass(frozen=True)
class RiskEngineConfig:
    """engines/risk.py — position sizing + daily risk budget.

    capital=None (default) disables sizing entirely: alerts and messages are
    byte-identical to Phase-1 behavior. Set capital to activate."""
    capital: float | None = None            # trading capital, ₹
    risk_per_trade_pct: float = 1.0         # % of capital risked per trade
    max_position_value_pct: float = 25.0    # single position ≤ % of capital
    max_concurrent_positions: int = 5
    max_daily_risk_r: float | None = None   # stop signaling after this many R
                                            # lost today; None = cap disabled


@dataclass(frozen=True)
class Config:
    score: ScoreConfig = field(default_factory=ScoreConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    setup_a: SetupAConfig = field(default_factory=SetupAConfig)
    intraday_risk: IntradayRiskConfig = field(default_factory=IntradayRiskConfig)
    swing_setup: SwingSetupConfig = field(default_factory=SwingSetupConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    risk: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    price_action: PriceActionConfig = field(default_factory=PriceActionConfig)
    rel_strength: RelStrengthConfig = field(default_factory=RelStrengthConfig)
    gap_class: GapClassConfig = field(default_factory=GapClassConfig)


def _coerce(value: Any) -> Any:
    """JSON lists → tuples (recursively), everything else unchanged."""
    if isinstance(value, list):
        return tuple(_coerce(v) for v in value)
    return value


def _overlay(section_default: Any, overrides: dict[str, Any], name: str) -> Any:
    known = {f.name for f in fields(section_default)}
    clean: dict[str, Any] = {}
    for key, val in overrides.items():
        if key not in known:
            logger.warning("config: unknown key %s.%s ignored", name, key)
            continue
        clean[key] = _coerce(val)
    return replace(section_default, **clean) if clean else section_default


def load_config(path: str | Path | None = None) -> Config:
    """Build the Config: defaults, optionally overlaid with a partial JSON
    file from `path` or $STOCKAGENT_CONFIG. Any load error falls back to pure
    defaults (loudly) — a broken overrides file must not take the bot down."""
    cfg = Config()
    path = path or os.getenv(ENV_VAR)
    if not path:
        return cfg
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("top level must be a JSON object")
    except Exception as e:
        logger.error("config: cannot load overrides from %s (%s) — "
                     "using defaults", path, e)
        return cfg

    sections = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    updates: dict[str, Any] = {}
    for name, overrides in raw.items():
        if name not in sections:
            logger.warning("config: unknown section %r ignored", name)
            continue
        if not isinstance(overrides, dict):
            logger.warning("config: section %r is not an object — ignored", name)
            continue
        updates[name] = _overlay(sections[name], overrides, name)
    loaded = replace(cfg, **updates) if updates else cfg
    logger.info("config: loaded overrides from %s", path)
    return loaded


CONFIG = load_config()
