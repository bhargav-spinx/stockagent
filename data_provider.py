"""
Market data provider.

Uses Angel One SmartAPI when ANGEL_API_KEY is configured (realtime broker feed).
Falls back to yfinance (free, ~15 min delayed) when Angel creds are not set,
so the bot keeps working during development.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# ---------- Angel One adapter ----------

_angel_session = None
_scrip_master = None
SCRIP_MASTER_PATH = Path(__file__).parent / "angel_scrip_master.json"
SCRIP_MASTER_URL = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
)

ANGEL_INTERVALS = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}

PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
    "6mo": 180, "1y": 365, "2y": 730,
}


def _angel_login():
    """Authenticate to Angel One and return a SmartConnect session."""
    from SmartApi import SmartConnect
    import pyotp

    api_key = os.getenv("ANGEL_API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_CODE")
    password = os.getenv("ANGEL_PASSWORD")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET")

    missing = [k for k, v in {
        "ANGEL_API_KEY": api_key,
        "ANGEL_CLIENT_CODE": client_code,
        "ANGEL_PASSWORD": password,
        "ANGEL_TOTP_SECRET": totp_secret,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Angel One creds missing: {', '.join(missing)}")

    # disable_ssl=True is a local-dev workaround for corp TLS inspection.
    # SmartConnect's _postRequest passes verify=not self.disable_ssl directly,
    # overriding any requests.Session-level patch.
    disable_ssl = os.getenv("ANGEL_DISABLE_SSL", "").lower() in ("1", "true", "yes")
    smart = SmartConnect(api_key=api_key, disable_ssl=disable_ssl)
    totp = pyotp.TOTP(totp_secret).now()
    resp = smart.generateSession(client_code, password, totp)
    if not resp.get("status"):
        raise RuntimeError(f"Angel login failed: {resp.get('message')}")

    logger.info("Angel One session established for client %s", client_code)
    return smart


def _get_angel_session():
    """Lazy-init / reuse the Angel session."""
    global _angel_session
    if _angel_session is None:
        _angel_session = _angel_login()
    return _angel_session


def _reset_angel_session():
    global _angel_session
    _angel_session = None


def force_angel_login() -> dict:
    """
    Force a fresh Angel One login, replacing any existing session.
    Returns a minimal status dict.
    """
    global _angel_session
    _angel_session = None
    _angel_session = _angel_login()
    return {
        "ok": True,
        "client_code": os.getenv("ANGEL_CLIENT_CODE"),
    }


def angel_session_active() -> bool:
    return _angel_session is not None


def _load_scrip_master():
    """Load Angel's symbol→token map. Cached on disk; refreshed once per day."""
    global _scrip_master
    if _scrip_master is not None:
        return _scrip_master

    needs_download = True
    if SCRIP_MASTER_PATH.exists():
        age_hours = (time.time() - SCRIP_MASTER_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            needs_download = False

    if needs_download:
        import requests
        logger.info("Downloading Angel scrip master (~10 MB)...")
        r = requests.get(SCRIP_MASTER_URL, timeout=60)
        r.raise_for_status()
        SCRIP_MASTER_PATH.write_text(r.text, encoding="utf-8")

    with open(SCRIP_MASTER_PATH, "r", encoding="utf-8") as f:
        _scrip_master = json.load(f)
    return _scrip_master


def _resolve_token(symbol: str):
    """Map ticker like 'RELIANCE' or 'TCS.BO' to Angel (token, exchange, tradingsymbol)."""
    sym = symbol.upper().strip()
    if sym.endswith(".BO"):
        exch, base = "BSE", sym[:-3]
    else:
        exch, base = "NSE", sym.replace(".NS", "")

    target = f"{base}-EQ"
    for s in _load_scrip_master():
        if s.get("exch_seg") == exch and s.get("symbol") == target:
            return s["token"], exch, s["symbol"]
    raise ValueError(f"'{symbol}' not found in Angel scrip master (looked for {target} on {exch})")


def fetch_angel(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV from Angel One. Returns DataFrame indexed by timestamp."""
    angel_interval = ANGEL_INTERVALS.get(interval)
    if not angel_interval:
        raise ValueError(f"Interval '{interval}' not supported by Angel One. "
                         f"Supported: {', '.join(ANGEL_INTERVALS)}")
    days = PERIOD_DAYS.get(period, 180)

    token, exch, _ = _resolve_token(symbol)

    now = datetime.now(IST)
    params = {
        "exchange": exch,
        "symboltoken": token,
        "interval": angel_interval,
        "fromdate": (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M"),
        "todate": now.strftime("%Y-%m-%d %H:%M"),
    }

    def call():
        smart = _get_angel_session()
        return smart.getCandleData(params)

    try:
        resp = call()
    except Exception as e:
        # Session may have expired — retry once with fresh login
        logger.warning("Angel call failed (%s); re-authenticating", e)
        _reset_angel_session()
        resp = call()

    if not resp.get("status") or not resp.get("data"):
        raise ValueError(f"No Angel data for {symbol}: {resp.get('message')}")

    df = pd.DataFrame(resp["data"],
                      columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.index = df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize(IST)
    return df


# ---------- yfinance fallback ----------

def fetch_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No yfinance data for {symbol}")
    return df


# ---------- Public API ----------

def get_provider_name() -> str:
    return "Angel One" if os.getenv("ANGEL_API_KEY") else "Yahoo Finance (delayed)"


def fetch_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data. Routes to Angel One when credentials are configured,
    yfinance otherwise. Indices (symbols starting with `^`) always go to
    yfinance because Angel's scrip master only has equities.
    """
    if symbol.startswith("^"):
        return fetch_yfinance(symbol, period, interval)
    if os.getenv("ANGEL_API_KEY"):
        return fetch_angel(symbol, period, interval)
    return fetch_yfinance(symbol, period, interval)
