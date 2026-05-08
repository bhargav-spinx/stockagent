"""
Intraday-specific indicators used by the scanner.
Reuses analyzer.py for atr/rsi/macd/bollinger.
"""
import pandas as pd


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
