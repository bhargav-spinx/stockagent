"""Shared constants. Kept dependency-free (only stdlib + pytz) so any module
can import it without risk of an import cycle."""
import pytz

# Single source of truth for the market timezone used across the project.
IST = pytz.timezone("Asia/Kolkata")

# Swing-signal standard — one bar applied everywhere: the daily swing scan
# (bot.py) and the channel-tip "standard gate" (telethon_listener.py).
SWING_MIN_CONFIDENCE = 90   # drop swing signals below this (near-unanimous indicators)
SWING_MAX_ALERTS = 5        # cap on swing alerts sent per run

# Single disclaimer string appended to every actionable (BUY/SELL + levels)
# output. Keep it on every call: this is a personal, educational tool — not
# SEBI-registered research/advice, and signals are hypothetical (lagging
# indicators, gross of costs/slippage).
DISCLAIMER = ("⚠️ _Educational & personal-use only. Not investment advice; "
              "not SEBI-registered research. Signals can be wrong — verify "
              "independently before risking capital._")

# Trade-type label shown at the top of every actionable message. The engine only
# analyses two horizons honestly — Intraday (5-min, same session) and Swing
# (daily, multi-day). Scalping / Positional / Long-Term / Investment are NOT
# produced (the bot doesn't analyse those timeframes), so they are deliberately
# absent rather than fabricated. Extend this map if such modes are ever added.
TRADE_TYPE_LABELS = {
    "intraday": ("Intraday", "5-min candles · same-session, square off ~3:15 PM"),
    "swing":    ("Swing", "daily candles · multi-day to a few weeks"),
}


def trade_type_tag(mode: str) -> str:
    """Crisp trade-type header line for a message, e.g. '🏷 *Intraday* — _…_'."""
    name, desc = TRADE_TYPE_LABELS.get((mode or "").lower(), (mode.title() if mode else "—", ""))
    return f"🏷 *{name}*" + (f" — _{desc}_" if desc else "")
