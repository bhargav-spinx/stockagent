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

# Static fallback used when the Angel scrip master can't be loaded
# (e.g. offline / TLS-blocked dev machine): NIFTY 100 + Bank NIFTY, deduped.
_STATIC_DEFAULT = _dedup_keep_order(NIFTY_50 + NIFTY_NEXT_50 + NIFTY_BANK)


def _build_from_scrip_master():
    """Derive (all_nse_equities, fno_stocks) as bare NSE symbols from the
    Angel scrip master.

    - all_nse_equities: every NSE `*-EQ` symbol minus iNAV ETF trackers
      (~2.5k names; includes ETFs/InvITs since the master can't tag them).
    - fno_stocks: the subset that also has stock futures in NFO — the liquid,
      tight-spread names (~210). This is the only liquidity signal the master
      carries, so it stands in for "NIFTY 500" (which isn't derivable here).

    Returns (None, None) if the master can't be loaded, so callers fall back
    to the static NIFTY 100 + Bank list.
    """
    try:
        from data_provider import _load_scrip_master
        rows = _load_scrip_master()
    except Exception:
        return None, None

    equities, seen = [], set()
    for r in rows:
        if r.get("exch_seg") != "NSE":
            continue
        sym = r.get("symbol", "")
        if not sym.endswith("-EQ"):
            continue
        base = sym[:-3]                      # round-trips via _resolve_token
        if "INAV" in base or base in seen:   # drop ETF iNAV trackers / dupes
            continue
        seen.add(base)
        equities.append(base)

    fno_underlyings = {
        r.get("name", "")
        for r in rows
        if r.get("exch_seg") == "NFO"
        and r.get("instrumenttype") == "FUTSTK"
        and "TEST" not in r.get("name", "")
    }
    fno = [b for b in equities if b in fno_underlyings]

    # Guard against a malformed/empty master yielding a uselessly small list.
    if len(equities) < 100 or len(fno) < 50:
        return None, None
    return equities, fno


_ALL_NSE_EQUITIES, _FNO_STOCKS = _build_from_scrip_master()

# Intraday / scalp: liquid F&O stocks (tight spreads). ~210 names, scans in
# ~105s — fits the 5-min autoscan loop. Falls back to NIFTY 100 + Bank.
INTRADAY_UNIVERSE = _FNO_STOCKS or _STATIC_DEFAULT

# Swing: every NSE-listed equity (once-daily EOD batch, time is not the cap).
# Falls back to NIFTY 100 + Bank when the master is unavailable.
SWING_UNIVERSE = _ALL_NSE_EQUITIES or _STATIC_DEFAULT

# Original tier-1 watchlist — kept for backward compatibility / quick scans.
# Used by `/scan` (no args) by default; `/scan_alerts` uses INTRADAY_UNIVERSE.
TIER1_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "INFY", "TCS", "LT", "BAJFINANCE",
    "MARUTI", "TATAMOTORS", "ITC", "HINDUNILVR",
]
