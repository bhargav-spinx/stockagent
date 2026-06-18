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
