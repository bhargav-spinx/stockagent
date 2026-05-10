"""
NSE index constituent lists used by the scanner and swing-alert loops
when no per-user watchlist is provided.

Lists are bundled as static Python data — they change only on quarterly
NSE rebalances. Update manually when rebalances happen, or wire in a
nightly fetch from NSE constituent CSVs (Phase 2).

Sources (current as of mid-2026):
- NIFTY 50:        nsearchives.nseindia.com/content/indices/ind_nifty50list.csv
- NIFTY Next 50:   nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv
- NIFTY Bank:      nsearchives.nseindia.com/content/indices/ind_niftybanklist.csv
"""

# ---------- NIFTY 50 ----------
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# ---------- NIFTY Next 50 ----------
NIFTY_NEXT_50 = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
    "BAJAJHLDNG", "BANKBARODA", "BERGEPAINT", "BOSCHLTD", "CANBK",
    "CGPOWER", "CHOLAFIN", "DABUR", "DIVISLAB", "DLF",
    "DMART", "GAIL", "GODREJCP", "HAL", "HAVELLS",
    "HINDPETRO", "ICICIGI", "ICICIPRULI", "IOC", "INDIGO",
    "IRFC", "JINDALSTEL", "JIOFIN", "LICI", "LODHA",
    "MOTHERSON", "NAUKRI", "PFC", "PIDILITIND", "PNB",
    "RECLTD", "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM",
    "TVSMOTOR", "UNITDSPR", "VBL", "VEDL", "ZOMATO",
    "ZYDUSLIFE", "INDHOTEL", "MARICO", "MUTHOOTFIN", "POLYCAB",
]

# ---------- NIFTY Bank ----------
NIFTY_BANK = [
    "AXISBANK", "BANKBARODA", "CANBK", "FEDERALBNK", "HDFCBANK",
    "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB",
    "SBIN", "AUBANK",
]


def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------- Scanner universes ----------

# Intraday: NIFTY 100 + Bank NIFTY (deduped). All F&O liquid, tight spreads.
INTRADAY_UNIVERSE = _dedup_keep_order(NIFTY_50 + NIFTY_NEXT_50 + NIFTY_BANK)

# Swing: same default for now. Could be widened to NIFTY 200 later if Angel
# rate limits are not hit.
SWING_UNIVERSE = INTRADAY_UNIVERSE

# Original tier-1 watchlist — kept for backward compatibility / quick scans.
# Used by `/scan` (no args) by default; `/scan_alerts` uses INTRADAY_UNIVERSE.
TIER1_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "INFY", "TCS", "LT", "BAJFINANCE",
    "MARUTI", "TATAMOTORS", "ITC", "HINDUNILVR",
]
