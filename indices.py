"""
Major Indian stock indices for market-context display + direct analysis.

Each primary index has:
- yfinance ticker  (fallback data source)
- Angel One token + exchange (primary data source when ANGEL_API_KEY set)
- Display name

The bot routes index fetches through Angel SmartAPI by token (see
data_provider._fetch_angel_index) when available, falling back to yfinance.
"""

# Primary indices: alias → metadata
PRIMARY = {
    "NIFTY": {
        "yf": "^NSEI",
        "angel_exchange": "NSE",
        "angel_token": "99926000",
        "angel_symbol": "Nifty 50",
        "display": "NIFTY 50",
    },
    "BANKNIFTY": {
        "yf": "^NSEBANK",
        "angel_exchange": "NSE",
        "angel_token": "99926009",
        "angel_symbol": "Nifty Bank",
        "display": "Bank NIFTY",
    },
    "SENSEX": {
        "yf": "^BSESN",
        "angel_exchange": "BSE",
        "angel_token": "99919000",
        "angel_symbol": "SENSEX",
        "display": "SENSEX",
    },
}

# Friendly alias → primary key. Case + spaces + hyphens stripped on lookup.
ALIASES = {
    "NIFTY": "NIFTY",
    "NIFTY50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
    "SENSEX": "SENSEX",
    "BSE": "SENSEX",
}

PRIMARY_INDICES: list[str] = ["NIFTY", "BANKNIFTY", "SENSEX"]


def _normalize(symbol: str) -> str:
    """Strip whitespace, hyphens, .NS/.BO suffix; uppercase."""
    s = symbol.upper().strip().replace(".NS", "").replace(".BO", "")
    return s.replace(" ", "").replace("-", "")


def get_index_info(symbol: str) -> dict | None:
    """Return the metadata dict for an index alias, or None if not an index."""
    norm = _normalize(symbol)
    if norm.startswith("^"):
        for info in PRIMARY.values():
            if info["yf"] == norm or info["yf"].upper() == norm:
                return info
        return None
    key = ALIASES.get(norm)
    return PRIMARY.get(key) if key else None


def resolve_index_alias(symbol: str) -> str | None:
    """
    If `symbol` is a known index alias, return the yfinance ticker
    (e.g. '^NSEI'). Otherwise None. Kept for backward compatibility with
    analyzer.normalize_symbol — yfinance is still the canonical ticker
    representation passed through the system.
    """
    info = get_index_info(symbol)
    return info["yf"] if info else None


def display_name(symbol: str) -> str:
    """Human-readable name for an index symbol."""
    info = get_index_info(symbol)
    return info["display"] if info else symbol


# Backward-compat: legacy code expects INDICES dict with (yf_ticker, display_name)
INDICES: dict[str, tuple[str, str]] = {
    alias: (PRIMARY[primary]["yf"], PRIMARY[primary]["display"])
    for alias, primary in ALIASES.items()
}
