"""
Intraday-specific indicators + shared intraday utilities used by the scanner,
the setup detectors, and the scoring engine.
Reuses analyzer.py for atr/rsi/macd/bollinger.
"""
import pandas as pd

from constants import IST

# Canonical intraday risk model (spec §Risk Management):
# volatility-sized stop = ATR_SL_MULT × ATR, targets locked to R:R 1:2 / 1:3.
# Single source of truth — both scanner_setups and intraday_score use it.
ATR_PERIOD = 14
ATR_SL_MULT = 1.5
STOP_PCT_FALLBACK = 0.01   # used only when ATR is unavailable (thin data)
RR_T1 = 2.0
RR_T2 = 3.0


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
    return (cum_pv / cum_v.replace(0, pd.NA)).astype(float)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def vwap_slope_pct(vwap_series: pd.Series, lookback: int = 6) -> float | None:
    """
    Linear-regression slope of VWAP over the last N candles, expressed
    as percent of mean VWAP per candle. Used by §5 "VWAP slope" filter:
    - Positive  → uptrend
    - Negative  → downtrend
    - Near zero → flat (chop, no trade)
    """
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

    Standard ATR-band implementation with band-locking carry-forward.
    """
    from analyzer import atr as atr_fn

    atr = atr_fn(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    close = df["Close"].to_numpy()
    up = upper.to_numpy()
    lo = lower.to_numpy()
    n = len(df)
    line = [float("nan")] * n
    direction = [1] * n

    for i in range(1, n):
        # Tighten bands: the band only moves in the trend's favour.
        if close[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])
        if close[i - 1] < lo[i - 1]:
            lo[i] = min(lo[i], lo[i - 1])

        prev_dir = direction[i - 1]
        if close[i] > up[i - 1]:
            direction[i] = 1
        elif close[i] < lo[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_dir
        line[i] = lo[i] if direction[i] == 1 else up[i]

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
    Returns None when there is no usable baseline (<13 candles or zero avg)."""
    if len(df) < 13:
        return None
    cur = float(df["Volume"].iloc[-1])
    avg = float(df["Volume"].iloc[-13:-1].mean())
    return cur / avg if avg > 0 else None


def trade_levels(entry: float, direction: str, df: pd.DataFrame,
                 atr_mult: float = ATR_SL_MULT, atr_period: int = ATR_PERIOD,
                 rr1: float = RR_T1, rr2: float = RR_T2):
    """Canonical (stop_loss, target1, target2) for an entry.

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
