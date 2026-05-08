"""
Setup A/B/C detection from §6 of STRATEGY.md.

Phase A: Setup A (Opening Range Breakout) implemented.
Phase B (next): Setup B (VWAP Pullback) and Setup C (Range Reversal).
"""
from dataclasses import dataclass, field

import pandas as pd
import pytz

from analyzer import atr as atr_fn, rsi as rsi_fn
from scanner_indicators import vwap, ema, swing_low, swing_high

IST = pytz.timezone("Asia/Kolkata")

# Per-leg targets and SL distance, from §7 of STRATEGY.md
T1_PCT = 0.01   # 1%
T2_PCT = 0.02   # 2%
ATR_SL_MULT = 1.5


@dataclass
class Signal:
    symbol: str
    setup: str            # "A" | "B" | "C"
    direction: str        # "long" | "short"
    entry: float
    stop_loss: float
    target1: float
    target2: float
    confluences: list[str] = field(default_factory=list)
    notes: str = ""


def _today_session(df: pd.DataFrame) -> pd.DataFrame:
    """Return only candles from today's IST session (last unique date)."""
    if df.index.tz is None:
        df = df.tz_localize(IST)
    elif str(df.index.tz) != str(IST):
        df = df.tz_convert(IST)
    last_date = df.index[-1].date()
    return df[df.index.date == last_date]


def _compute_sl(df: pd.DataFrame, entry: float, direction: str) -> float:
    """
    SL per §7: max(swing-based, ATR-based) for long, min for short.
    Whichever is tighter (less distance to entry) wins.
    """
    atr_val = float(atr_fn(df, 14).iloc[-1])
    if direction == "long":
        sl_swing = swing_low(df, lookback=5)
        sl_atr = entry - ATR_SL_MULT * atr_val
        return max(sl_swing, sl_atr)  # tighter = closer to entry = larger value
    else:
        sl_swing = swing_high(df, lookback=5)
        sl_atr = entry + ATR_SL_MULT * atr_val
        return min(sl_swing, sl_atr)


def detect_setup_a(df: pd.DataFrame, symbol: str) -> Signal | None:
    """
    Setup A: Opening Range Breakout.

    - ORB = first 15 min after market open = first three 5-min candles (09:15–09:30)
    - Long trigger:  last candle CLOSES above ORB high
    - Short trigger: last candle CLOSES below ORB low
    - Confluences (≥3 required):
        1. Price above (long) / below (short) VWAP
        2. EMA9 > EMA20 (long) / EMA9 < EMA20 (short)
        3. RSI(14) in 55–70 (long) / 30–45 (short)
    - Skip if ORB range > 1.2% of stock price.
    """
    today = _today_session(df)
    if len(today) < 4:  # need ORB (3 candles) + at least one trigger candle
        return None

    orb = today.iloc[:3]
    orb_high = float(orb["High"].max())
    orb_low = float(orb["Low"].min())
    orb_range_pct = (orb_high - orb_low) / orb_low

    if orb_range_pct > 0.012:
        return None  # ORB too wide — already extended

    last_candle = today.iloc[-1]
    last_price = float(last_candle["Close"])

    vwap_val = float(vwap(df).iloc[-1])
    ema9 = float(ema(df["Close"], 9).iloc[-1])
    ema20 = float(ema(df["Close"], 20).iloc[-1])
    rsi_val = float(rsi_fn(df["Close"], 14).iloc[-1])

    notes = (f"ORB {orb_low:.2f}–{orb_high:.2f} "
             f"({orb_range_pct*100:.2f}% range)")

    # Long trigger: closes above ORB high
    if last_price > orb_high:
        confluences = []
        if last_price > vwap_val:
            confluences.append(f"Price > VWAP ({vwap_val:.2f})")
        if ema9 > ema20:
            confluences.append(f"EMA9 > EMA20 ({ema9:.2f} > {ema20:.2f})")
        if 55 <= rsi_val <= 70:
            confluences.append(f"RSI {rsi_val:.0f} in 55–70 zone")

        if len(confluences) >= 3:
            sl = _compute_sl(df, last_price, "long")
            return Signal(
                symbol=symbol, setup="A", direction="long",
                entry=last_price,
                stop_loss=sl,
                target1=last_price * (1 + T1_PCT),
                target2=last_price * (1 + T2_PCT),
                confluences=confluences,
                notes=notes,
            )

    # Short trigger: closes below ORB low
    if last_price < orb_low:
        confluences = []
        if last_price < vwap_val:
            confluences.append(f"Price < VWAP ({vwap_val:.2f})")
        if ema9 < ema20:
            confluences.append(f"EMA9 < EMA20 ({ema9:.2f} < {ema20:.2f})")
        if 30 <= rsi_val <= 45:
            confluences.append(f"RSI {rsi_val:.0f} in 30–45 zone")

        if len(confluences) >= 3:
            sl = _compute_sl(df, last_price, "short")
            return Signal(
                symbol=symbol, setup="A", direction="short",
                entry=last_price,
                stop_loss=sl,
                target1=last_price * (1 - T1_PCT),
                target2=last_price * (1 - T2_PCT),
                confluences=confluences,
                notes=notes,
            )

    return None


def detect_setup_b(df: pd.DataFrame, symbol: str) -> Signal | None:
    """Setup B: VWAP Pullback Continuation. TODO Phase B."""
    return None


def detect_setup_c(df: pd.DataFrame, symbol: str) -> Signal | None:
    """Setup C: Range Reversal at Key Level. TODO Phase B."""
    return None


ALL_DETECTORS = [detect_setup_a, detect_setup_b, detect_setup_c]
