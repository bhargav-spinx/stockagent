"""Shared constants. Kept dependency-free (only stdlib + pytz) so any module
can import it without risk of an import cycle."""
import pytz

# Single source of truth for the market timezone used across the project.
IST = pytz.timezone("Asia/Kolkata")

# Swing-signal standard — one bar applied everywhere: the daily swing scan
# (bot.py) and the channel-tip "standard gate" (telethon_listener.py).
SWING_MIN_CONFIDENCE = 90   # drop swing signals below this (near-unanimous indicators)
SWING_MAX_ALERTS = 5        # cap on swing alerts sent per run
