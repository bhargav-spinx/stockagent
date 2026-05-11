"""
Major Indian stock indices used for market-context display + direct analysis.

User-facing aliases (NIFTY, BANKNIFTY, SENSEX) → yfinance tickers.
Angel SmartAPI also supports indices, but their scrip master uses a
different exchange segment ("NSE" with no `-EQ` suffix) — we route
indices through yfinance to keep ticker resolution simple.
"""

# Friendly alias → (yfinance symbol, display name)
INDICES: dict[str, tuple[str, str]] = {
    "NIFTY":      ("^NSEI",    "NIFTY 50"),
    "NIFTY50":    ("^NSEI",    "NIFTY 50"),
    "BANKNIFTY":  ("^NSEBANK", "Bank NIFTY"),
    "NIFTYBANK":  ("^NSEBANK", "Bank NIFTY"),
    "SENSEX":     ("^BSESN",   "SENSEX"),
    "BSE":        ("^BSESN",   "SENSEX"),
}

# Primary three for the /index snapshot
PRIMARY_INDICES: list[str] = ["NIFTY", "BANKNIFTY", "SENSEX"]


def resolve_index_alias(symbol: str) -> str | None:
    """
    If `symbol` is a known index alias (NIFTY, BANKNIFTY, SENSEX),
    return the yfinance ticker (e.g. '^NSEI'). Otherwise None.
    Case-insensitive; strips '.NS' suffix if present.
    """
    s = symbol.upper().strip().replace(".NS", "").replace(".BO", "")
    s = s.replace(" ", "").replace("-", "")
    entry = INDICES.get(s)
    return entry[0] if entry else None


def display_name(symbol: str) -> str:
    """Human-readable name for an index symbol (alias or yfinance ticker)."""
    s = symbol.upper().strip().replace(".NS", "").replace(".BO", "")
    s = s.replace(" ", "").replace("-", "")
    if s in INDICES:
        return INDICES[s][1]
    # Reverse lookup by yfinance ticker
    for _, (yf_sym, name) in INDICES.items():
        if yf_sym == symbol:
            return name
    return symbol
