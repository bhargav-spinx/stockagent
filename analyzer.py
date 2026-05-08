"""
Indian Stock Analyzer
Pulls NSE/BSE data and runs technical analysis to generate buy/sell/hold signals.
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time

from data_provider import fetch_data, get_provider_name


# Per-mode indicator config. Swing = daily candles, slow indicators.
# Intraday = 5-min candles, faster indicators tuned to a single session.
MODES = {
    "swing": {
        "period": "6mo",
        "interval": "1d",
        "sma_short": 20,
        "sma_long": 50,
        "rsi_period": 14,
        "macd": (12, 26, 9),
        "bb_period": 20,
        "label": "Swing / Positional (daily candles)",
        "unit": "day",
    },
    "intraday": {
        "period": "5d",
        "interval": "5m",
        "sma_short": 9,
        "sma_long": 21,
        "rsi_period": 9,
        "macd": (5, 13, 6),
        "bb_period": 20,
        "label": "Intraday (5-min candles)",
        "unit": "candle",
    },
}


def normalize_symbol(symbol: str) -> str:
    """
    Convert user input to yfinance-compatible NSE ticker.
    'RELIANCE' -> 'RELIANCE.NS', 'TCS.BO' stays as is.
    """
    symbol = symbol.upper().strip()
    if "." not in symbol:
        symbol = f"{symbol}.NS"  # default to NSE
    return symbol


def is_nse_open() -> bool:
    """NSE trading hours: 09:15-15:30 IST, Mon-Fri."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)


# ---------- Technical Indicators ----------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — standard volatility measure used to size stops."""
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift()
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_trade_setup(df: pd.DataFrame, signal: str, last_price: float,
                      atr_val: float, mode: str) -> dict:
    """
    Generate entry / stop-loss / targets / risk level from ATR + recent swings.
    Returns levels for BUY/SELL; for HOLD returns breakout watch levels.
    """
    lookback = 20 if mode == "swing" else 30
    recent = df.tail(lookback)
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())

    # Tighter stops for intraday (single session) vs swing (multi-day noise)
    sl_mult = 1.5 if mode == "swing" else 1.0
    tgt1_mult, tgt2_mult = 1.5, 3.0

    setup = {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "atr": round(atr_val, 2),
    }

    if signal == "BUY":
        entry = last_price
        sl_atr = entry - atr_val * sl_mult
        sl_swing = swing_low * 0.995  # 0.5% below recent swing low
        stop_loss = max(sl_atr, sl_swing)  # whichever is closer (less risk)
        risk = entry - stop_loss
        target1 = entry + risk * tgt1_mult
        target2 = entry + risk * tgt2_mult
        rr1 = tgt1_mult
    elif signal == "SELL":
        entry = last_price
        sl_atr = entry + atr_val * sl_mult
        sl_swing = swing_high * 1.005
        stop_loss = min(sl_atr, sl_swing)
        risk = stop_loss - entry
        target1 = entry - risk * tgt1_mult
        target2 = entry - risk * tgt2_mult
        rr1 = tgt1_mult
    else:  # HOLD — give breakout watch levels instead
        setup["action"] = "WAIT"
        setup["buy_breakout"] = round(swing_high * 1.002, 2)
        setup["buy_breakout_sl"] = round(swing_high * 0.99, 2)
        setup["sell_breakdown"] = round(swing_low * 0.998, 2)
        setup["sell_breakdown_sl"] = round(swing_low * 1.01, 2)
        return setup

    risk_pct = (risk / entry) * 100
    if risk_pct < 1.5:
        risk_level = "Low"
    elif risk_pct < 3.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    setup.update({
        "action": signal,
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "risk_per_share": round(risk, 2),
        "risk_pct": round(risk_pct, 2),
        "rr_ratio": rr1,
        "risk_level": risk_level,
    })
    return setup


# ---------- Signal Engine ----------

def analyze(symbol: str, mode: str = "swing") -> dict:
    """
    Run a multi-indicator analysis and return a structured recommendation.
    mode = "swing" (daily candles, positional) or "intraday" (5-min candles).
    Each indicator votes BUY/SELL/HOLD; final signal is the majority.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use 'swing' or 'intraday'.")
    cfg = MODES[mode]

    sym = normalize_symbol(symbol)
    df = fetch_data(sym, period=cfg["period"], interval=cfg["interval"])
    close = df["Close"]

    if len(close) < cfg["sma_long"] + 2:
        raise ValueError(
            f"Not enough data for {mode} analysis on {sym} "
            f"(need {cfg['sma_long'] + 2} candles, got {len(close)})."
        )

    sma_short = close.rolling(cfg["sma_short"]).mean()
    sma_long = close.rolling(cfg["sma_long"]).mean()
    rsi_val = rsi(close, period=cfg["rsi_period"]).iloc[-1]
    macd_line, signal_line, hist = macd(close, *cfg["macd"])
    upper_bb, _, lower_bb = bollinger_bands(close, period=cfg["bb_period"])
    atr_val = atr(df, period=cfg["rsi_period"]).iloc[-1]

    last_price = close.iloc[-1]

    if mode == "intraday":
        # Compare to today's session open, not the prior 5-min candle
        last_date = df.index[-1].date()
        today = df[df.index.date == last_date]
        baseline = today["Open"].iloc[0] if len(today) else close.iloc[-2]
        change_label = "from open"
    else:
        baseline = close.iloc[-2]
        change_label = "vs prev close"
    change_pct = ((last_price - baseline) / baseline) * 100

    unit = cfg["unit"]
    signals = []

    # 1. SMA crossover
    if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] <= sma_long.iloc[-2]:
        signals.append(("SMA Crossover", "BUY",
                        f"Bullish cross — {cfg['sma_short']}-{unit} above {cfg['sma_long']}-{unit}"))
    elif sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] >= sma_long.iloc[-2]:
        signals.append(("SMA Crossover", "SELL",
                        f"Bearish cross — {cfg['sma_short']}-{unit} below {cfg['sma_long']}-{unit}"))
    elif sma_short.iloc[-1] > sma_long.iloc[-1]:
        signals.append(("SMA Trend", "BUY", "Short-term trend above long-term"))
    else:
        signals.append(("SMA Trend", "SELL", "Short-term trend below long-term"))

    # 2. RSI
    if rsi_val < 30:
        signals.append(("RSI", "BUY", f"Oversold (RSI={rsi_val:.1f})"))
    elif rsi_val > 70:
        signals.append(("RSI", "SELL", f"Overbought (RSI={rsi_val:.1f})"))
    else:
        signals.append(("RSI", "HOLD", f"Neutral (RSI={rsi_val:.1f})"))

    # 3. MACD
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        signals.append(("MACD", "BUY", "Bullish crossover"))
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        signals.append(("MACD", "SELL", "Bearish crossover"))
    elif hist.iloc[-1] > 0:
        signals.append(("MACD", "BUY", "Positive momentum"))
    else:
        signals.append(("MACD", "SELL", "Negative momentum"))

    # 4. Bollinger Bands
    if last_price < lower_bb.iloc[-1]:
        signals.append(("Bollinger", "BUY", "Price below lower band"))
    elif last_price > upper_bb.iloc[-1]:
        signals.append(("Bollinger", "SELL", "Price above upper band"))
    else:
        signals.append(("Bollinger", "HOLD", "Price within bands"))

    # Aggregate vote
    votes = [s[1] for s in signals]
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")

    if buy_count > sell_count and buy_count >= 2:
        final = "BUY"
        confidence = (buy_count / len(votes)) * 100
    elif sell_count > buy_count and sell_count >= 2:
        final = "SELL"
        confidence = (sell_count / len(votes)) * 100
    else:
        final = "HOLD"
        confidence = 50.0

    setup = build_trade_setup(df, final, float(last_price), float(atr_val), mode)

    ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))
    return {
        "symbol": sym,
        "mode": mode,
        "mode_label": cfg["label"],
        "price": round(last_price, 2),
        "change_pct": round(change_pct, 2),
        "change_label": change_label,
        "rsi": round(rsi_val, 2),
        "signal": final,
        "confidence": round(confidence, 1),
        "indicators": signals,
        "trade_setup": setup,
        "market_open": is_nse_open(),
        "timestamp": ist_now.strftime("%Y-%m-%d %H:%M IST"),
    }


RISK_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


def _format_trade_setup(setup: dict, signal: str) -> list[str]:
    """Build the Trade Setup block for the report."""
    if not setup:
        return []

    lines = ["", "*🎯 Trade Setup:*"]

    if setup.get("action") == "WAIT":
        # HOLD case — show breakout watch levels
        lines.append(f"⏸ Action: *WAIT* — no clear entry")
        lines.append(f"🟢 Buy breakout above ₹{setup['buy_breakout']} (SL ₹{setup['buy_breakout_sl']})")
        lines.append(f"🔴 Sell breakdown below ₹{setup['sell_breakdown']} (SL ₹{setup['sell_breakdown_sl']})")
        lines.append(f"📊 Recent range: ₹{setup['swing_low']} – ₹{setup['swing_high']}")
        lines.append(f"📏 ATR (volatility): ₹{setup['atr']}")
        return lines

    is_buy = setup["action"] == "BUY"
    direction = "Long" if is_buy else "Short"
    entry_emoji = "🟢" if is_buy else "🔴"
    risk_emoji = RISK_EMOJI.get(setup["risk_level"], "🟡")

    t1_pct = ((setup["target1"] - setup["entry"]) / setup["entry"]) * 100
    t2_pct = ((setup["target2"] - setup["entry"]) / setup["entry"]) * 100
    sl_pct = ((setup["stop_loss"] - setup["entry"]) / setup["entry"]) * 100

    lines.append(f"{entry_emoji} Direction: *{direction}* ({setup['action']})")
    lines.append(f"🎯 Entry: ₹{setup['entry']}")
    lines.append(f"🛑 Stop-Loss: ₹{setup['stop_loss']} ({sl_pct:+.2f}%)")
    lines.append(f"🥇 Target 1: ₹{setup['target1']} ({t1_pct:+.2f}%) — R:R 1:{setup['rr_ratio']}")
    lines.append(f"🥈 Target 2: ₹{setup['target2']} ({t2_pct:+.2f}%) — R:R 1:3.0")
    lines.append(f"{risk_emoji} Risk Level: *{setup['risk_level']}* "
                 f"(₹{setup['risk_per_share']}/share, {setup['risk_pct']}%)")
    lines.append(f"📏 ATR (volatility): ₹{setup['atr']}")
    lines.append(f"📊 Recent range: ₹{setup['swing_low']} – ₹{setup['swing_high']}")
    return lines


def format_report(result: dict) -> str:
    """Format the analysis result as a Telegram-friendly message."""
    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    arrow = "📈" if result["change_pct"] >= 0 else "📉"

    lines = [
        f"*{result['symbol']}*  _({result.get('mode_label', '')})_",
        f"💰 ₹{result['price']}  {arrow} {result['change_pct']:+.2f}% _{result.get('change_label', '')}_",
        "",
        f"{emoji[result['signal']]} *Signal: {result['signal']}*  ({result['confidence']}% confidence)",
        "",
        "*Indicator breakdown:*",
    ]
    for name, sig, reason in result["indicators"]:
        lines.append(f"{emoji[sig]} {name}: {reason}")

    lines.extend(_format_trade_setup(result.get("trade_setup", {}), result["signal"]))

    lines.append("")
    if result.get("mode") == "intraday" and not result.get("market_open"):
        lines.append("⚠️ _NSE is closed — intraday data is stale._")
    lines.append(f"_Updated: {result['timestamp']} • Source: {get_provider_name()}_")
    lines.append("_⚠️ Educational only. Not financial advice._")
    return "\n".join(lines)
