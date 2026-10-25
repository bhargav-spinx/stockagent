"""
Intraday-specific indicators + shared intraday utilities used by the scanner,
the setup detectors, and the scoring engine.
Reuses analyzer.py for atr/rsi/macd/bollinger.
"""
from datetime import datetime, timedelta

import pandas as pd

from config import CONFIG
from constants import IST

# Canonical intraday risk model (spec §Risk Management):
# volatility-sized stop = ATR_SL_MULT × ATR, targets locked to R:R 1:2 / 1:3.
# Single source of truth — both scanner_setups and intraday_score use it.
# Values come from config.IntradayRiskConfig; the module-level names are kept
# so existing importers keep working.
ATR_PERIOD = CONFIG.intraday_risk.atr_period
ATR_SL_MULT = CONFIG.intraday_risk.atr_sl_mult
STOP_PCT_FALLBACK = CONFIG.intraday_risk.stop_pct_fallback
RR_T1 = CONFIG.intraday_risk.rr_t1
RR_T2 = CONFIG.intraday_risk.rr_t2


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP. Resets at each session boundary so today's VWAP
    is not contaminated by yesterday's volume.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]
    by_date = df.index.date
    cum_pv = (typical * vol).groupby(by_date).cumsum()
    cum_v = vol.groupby(by_date).cumsum()
    # Zero cumulative volume (a session opening with no-trade bars — common in
    # real archived data) must become NaN, NOT pd.NA: pd.NA poisons the series
    # into object dtype and pandas 3 then raises
    # "float() argument must be ... not 'NAType'" out of .astype(float).
    # That crash took down every score_stock call on archived candles and,
    # latently, any live scan of a symbol whose session opened untraded.
    # .where keeps the series float end-to-end; NaN divides propagate cleanly.
    return (cum_pv / cum_v.where(cum_v != 0)).astype(float)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def vwap_slope_pct(vwap_series: pd.Series, lookback: int = 6) -> float | None:
    """
    Linear-regression slope of VWAP over the last N candles, expressed
    as percent of mean VWAP per candle. Used by §5 "VWAP slope" filter:
    - Positive  → uptrend
    - Negative  → downtrend
    - Near zero → flat (chop, no trade)

    Computed on TODAY's VWAP only: VWAP resets each session, so a regression
    spanning the overnight reset (the 09:35/09:40 scans, exactly when ORB
    setups trigger) measured the jump from yesterday's VWAP — a dead-flat
    session after a 1% gap read as a 0.2%/candle "trend" and passed the
    not-flat filter. Returns None until today has `lookback` values.
    """
    idx = vwap_series.index
    if len(vwap_series) and isinstance(idx, pd.DatetimeIndex):
        vwap_series = vwap_series[idx.date == idx[-1].date()]
    recent = vwap_series.tail(lookback).dropna()
    if len(recent) < lookback:
        return None
    x = list(range(len(recent)))
    y = recent.tolist()
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxy = sum(a * b for a, b in zip(x, y))
    sxx = sum(a * a for a in x)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    mean_v = sum(y) / n
    if mean_v == 0:
        return None
    return (slope / mean_v) * 100


def supertrend(df: pd.DataFrame, period: int = 10,
               multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator. Returns (line, direction) where direction is
    +1 (bullish / price above the line) or -1 (bearish / below).

    Standard final-band recursion (TradingView ta.supertrend):
      final_upper = basic_upper if basic_upper < prev_final_upper or
                    prev_close > prev_final_upper, else prev_final_upper
      final_lower = basic_lower if basic_lower > prev_final_lower or
                    prev_close < prev_final_lower, else prev_final_lower
    Direction flips when the close crosses the CURRENT final band on the
    active side. (The previous version applied max/min exactly in the reset
    case and never ratcheted otherwise, so direction almost never left its
    initial +1 — "bullish" at the end of a steady −26% decline.)
    """
    from analyzer import atr as atr_fn

    atr = atr_fn(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = (hl2 + multiplier * atr).to_numpy(dtype=float)
    lower = (hl2 - multiplier * atr).to_numpy(dtype=float)

    close = df["Close"].to_numpy(dtype=float)
    n = len(df)
    fu = upper.copy()
    fl = lower.copy()
    line = [float("nan")] * n
    direction = [1] * n

    for i in range(1, n):
        if upper[i] != upper[i] or fu[i - 1] != fu[i - 1]:   # NaN ATR warm-up
            direction[i] = direction[i - 1]
            continue
        fu[i] = (upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1])
                 else fu[i - 1])
        fl[i] = (lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1])
                 else fl[i - 1])
        if direction[i - 1] == -1:            # line was the upper band
            direction[i] = 1 if close[i] > fu[i] else -1
        else:                                 # line was the lower band
            direction[i] = -1 if close[i] < fl[i] else 1
        line[i] = fl[i] if direction[i] == 1 else fu[i]

    return (pd.Series(line, index=df.index),
            pd.Series(direction, index=df.index))


def swing_low(df: pd.DataFrame, lookback: int = 5) -> float:
    """Lowest low of the last `lookback` candles before the trigger candle."""
    if len(df) < lookback + 1:
        return float(df["Low"].min())
    return float(df["Low"].iloc[-(lookback + 1):-1].min())


def swing_high(df: pd.DataFrame, lookback: int = 5) -> float:
    """Highest high of the last `lookback` candles before the trigger candle."""
    if len(df) < lookback + 1:
        return float(df["High"].max())
    return float(df["High"].iloc[-(lookback + 1):-1].max())


# --- Shared intraday utilities (single source for setups + scoring) ----------

def localize_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is IST-aware."""
    if df.index.tz is None:
        return df.tz_localize(IST)
    if str(df.index.tz) != str(IST):
        return df.tz_convert(IST)
    return df


# Max age of the last CLOSED candle before a live signal is refused. Generous
# enough for the ~15-min-delayed yfinance dev path; tight enough to reject the
# real failure modes: a holiday missing from nse_holidays.txt, a lagging feed,
# a halted stock — all of which otherwise score YESTERDAY's session ("today"
# is just the last date in the data) and stamp it with today's date.
MAX_CANDLE_LAG = timedelta(minutes=30)


def stale_reason(df: pd.DataFrame, interval_min: int = 5,
                 now: datetime | None = None) -> str | None:
    """Why `df`'s last candle is too old to act on at `now`, or None if fresh.
    For LIVE signal paths only (backtests replay history by definition)."""
    if df is None or len(df) == 0:
        return "no candles"
    now = now or datetime.now(IST)
    last = localize_ist(df).index[-1].to_pydatetime()
    if last.date() != now.date():
        return f"no candles today yet (last {last:%d %b %H:%M})"
    lag = now - (last + timedelta(minutes=interval_min))
    if lag > MAX_CANDLE_LAG:
        return f"stale data (last candle closed {int(lag.total_seconds() // 60)} min ago)"
    return None


def today_session(df: pd.DataFrame) -> pd.DataFrame:
    """Return only candles from today's IST session (the last unique date)."""
    df = localize_ist(df)
    last_date = df.index[-1].date()
    return df[df.index.date == last_date]


def split_sessions(df: pd.DataFrame):
    """Split into (today_df, [prior_day_dfs...]) by IST calendar date."""
    df = localize_ist(df)
    dates = sorted(set(df.index.date))
    today = dates[-1]
    today_df = df[df.index.date == today]
    priors = [df[df.index.date == d] for d in dates[:-1]]
    return today_df, priors


def orb_levels(today_df: pd.DataFrame, n_candles: int) -> tuple[float, float]:
    """Opening-range (high, low) over the first `n_candles` of the session."""
    orb = today_df.iloc[:n_candles]
    return float(orb["High"].max()), float(orb["Low"].min())


def volume_ratio(df: pd.DataFrame) -> float | None:
    """Last candle volume ÷ average of the prior 12 candles.
    Returns None when there is no usable baseline (<13 candles or zero avg).

    When today's session already has 13+ candles, the baseline is restricted
    to TODAY only: a cross-session window in the first hour would compare the
    morning candle against yesterday's late-afternoon volumes (elevated by the
    close), skewing the ratio. Early in the session (<13 candles today) the
    cross-session window is kept as a documented compromise — no same-day
    baseline exists yet."""
    if len(df) < 13:
        return None
    today = today_session(df)
    base = today if len(today) >= 13 else df
    cur = float(base["Volume"].iloc[-1])
    avg = float(base["Volume"].iloc[-13:-1].mean())
    return cur / avg if avg > 0 else None


def trade_levels(entry: float, direction: str, df: pd.DataFrame,
                 atr_mult: float = ATR_SL_MULT, atr_period: int = ATR_PERIOD,
                 rr1: float = RR_T1, rr2: float = RR_T2):
    """Canonical INTRADAY (stop_loss, target1, target2) for an entry.

    NOTE: this is one of TWO deliberate risk models in the project — do not
    unify them casually:
      • INTRADAY (here): stop = 1.5×ATR(14) on 5-min; targets at 2R / 3R.
      • SWING (analyzer.build_trade_setup): stop = tighter of 1.5×ATR(daily)
        vs 0.5% beyond the 20-day swing extreme; targets at 1.5R / 3R.
    Different horizons price risk differently; each model is the single source
    of truth for its horizon.

    Stop distance (`risk`) = atr_mult × ATR(atr_period), so it adapts to each
    stock's volatility; targets are locked to R:R multiples of that risk
    (1:2 / 1:3 by default) — satisfying the spec's minimum 1:2. Falls back to a
    fixed STOP_PCT_FALLBACK stop when ATR can't be computed (thin data).
    Used by both entry engines so a stock yields identical levels everywhere."""
    from analyzer import atr as atr_fn

    risk = None
    try:
        a = float(atr_fn(df, atr_period).iloc[-1])
        if a == a and a > 0:        # not NaN, positive
            risk = atr_mult * a
    except Exception:
        risk = None
    if risk is None or risk <= 0:
        risk = entry * STOP_PCT_FALLBACK

    if direction == "long":
        return entry - risk, entry + rr1 * risk, entry + rr2 * risk
    return entry + risk, entry - rr1 * risk, entry - rr2 * risk
