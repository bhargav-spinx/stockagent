"""
Market data provider.

Uses Angel One SmartAPI when ANGEL_API_KEY is configured (realtime broker feed).
yfinance (free, ~15 min delayed) is used when Angel creds are NOT set, so the
bot keeps working during development.

Fallback semantics — deliberate, not an accident:
- INDICES fall back to yfinance when the Angel fetch fails (regime context is
  better delayed than absent).
- STOCKS do NOT fall back when Angel is configured: silently mixing 15-min
  delayed candles into intraday signal generation would produce entries/stops
  computed on stale prices — worse than an honest error. A stock fetch failure
  surfaces as an error and the symbol is skipped for that pass.
"""

import os
import json
import time
import random
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from constants import IST

logger = logging.getLogger(__name__)

# ---------- Angel One adapter ----------

_angel_session = None
_scrip_master = None
_scrip_master_date = None   # IST date the in-memory master was loaded on
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


# Login serialization + storm brake. fetch paths run on asyncio worker THREADS
# (to_thread), so without a lock two threads can race _get_angel_session() into
# concurrent generateSession calls. And without a cooldown, a network outage
# mid-scan re-attempts login per symbol per retry — hundreds of TOTP logins in
# minutes, which Angel rate-limits and can temp-lock the account.
_login_lock = threading.Lock()
_last_login_failure = 0.0            # time.monotonic() of the last failed login
LOGIN_COOLDOWN_SEC = 60


def _get_angel_session():
    """Lazy-init / reuse the Angel session. Thread-safe; failed logins arm a
    cooldown so an outage can't turn into a login storm."""
    global _angel_session, _last_login_failure
    if _angel_session is not None:
        return _angel_session
    with _login_lock:
        if _angel_session is not None:      # another thread won the race
            return _angel_session
        since_fail = time.monotonic() - _last_login_failure
        if since_fail < LOGIN_COOLDOWN_SEC:
            raise RuntimeError(
                f"Angel login on cooldown ({LOGIN_COOLDOWN_SEC - since_fail:.0f}s "
                "left) after a recent failure — not retrying yet")
        try:
            _angel_session = _angel_login()
        except Exception:
            _last_login_failure = time.monotonic()
            raise
    return _angel_session


def _reset_angel_session():
    global _angel_session
    _angel_session = None


# Only errors that look like a dead/expired session justify re-login. A
# connection error, DNS failure or 5xx does NOT — re-authing on those turns an
# ordinary network blip into a per-symbol login storm.
_AUTH_ERROR_MARKERS = ("token", "session", "login", "auth", "unauthor",
                       "invalid credentials", "ag8001", "ab8050", "ab8051")


def _looks_like_auth_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(m in s for m in _AUTH_ERROR_MARKERS)


def force_angel_login() -> dict:
    """
    Force a fresh Angel One login, replacing any existing session.
    Deliberately bypasses the failure cooldown (explicit operator action via
    /angel_login) and clears it on success. Returns a minimal status dict.
    """
    global _angel_session, _last_login_failure
    with _login_lock:
        _angel_session = None
        _angel_session = _angel_login()
        _last_login_failure = 0.0
    return {
        "ok": True,
        "client_code": os.getenv("ANGEL_CLIENT_CODE"),
    }


def angel_session_active() -> bool:
    return _angel_session is not None


# Angel throttles the historical/candle API hard (per-second + daily caps).
# A rate-limited call comes back as status:false with an "exceeding access
# rate" message rather than raising — the old code re-authed only on
# exceptions, so throttled responses failed instantly and every symbol in a
# scan tick errored at once. Retry rate-limited responses with backoff.
CANDLE_MAX_RETRIES = 3
CANDLE_BACKOFF_BASE_SEC = 1.0
_RATE_LIMIT_MARKERS = ("access rate", "rate limit", "too many requests")


def _is_rate_limited_text(s: str) -> bool:
    """Rate-limit detection on a raw message string. Needed because the SDK
    THROWS on a throttled call (the response body is plaintext 'Access denied
    because of exceeding access rate', which fails its JSON parse) — so the
    rate limit surfaces as an exception, not a status:false response."""
    s = s.lower()
    return any(m in s for m in _RATE_LIMIT_MARKERS)


def _is_rate_limited(resp: dict) -> bool:
    code = str(resp.get("errorcode", "")).lower()
    return _is_rate_limited_text(str(resp.get("message", ""))) or code == "ab1004"


def _get_candle_data(params: dict, label: str) -> dict:
    """Fetch candles with backoff-on-rate-limit and auth-aware re-login.

    Returns the raw Angel response (status truthy, data present) or raises
    ValueError with the provider's message. Rate-limited responses are retried
    with exponential backoff. Re-login happens ONLY for auth-shaped errors —
    a network blip must not trigger a login per symbol per attempt (login
    storm → Angel lockout); those just back off and retry the same session.
    """
    last_msg = ""
    for attempt in range(CANDLE_MAX_RETRIES):
        try:
            resp = _get_angel_session().getCandleData(params)
        except Exception as e:
            last_msg = str(e)
            if _looks_like_auth_error(e):
                # Session expired/revoked — drop it; next attempt re-logins
                # (subject to the login lock + cooldown).
                logger.warning("Angel auth error for %s (%s); dropping session",
                               label, e)
                _reset_angel_session()
            elif _is_rate_limited_text(last_msg):
                # Throttled call surfacing as an exception (see
                # _is_rate_limited_text). Don't burn the remaining retries
                # against a quota block — one attempt is enough to know.
                logger.warning("Angel rate-limited (via exception) on %s — "
                               "not retrying against a quota block", label)
                raise ValueError(f"No Angel data for {label}: rate-limited: {last_msg}")
            else:
                logger.warning("Angel call failed for %s (%s); will retry "
                               "same session", label, e)
            if attempt < CANDLE_MAX_RETRIES - 1:
                wait = CANDLE_BACKOFF_BASE_SEC * (2 ** attempt)
                time.sleep(wait + random.uniform(0, wait * 0.5))
                continue
            raise ValueError(f"No Angel data for {label}: {last_msg}")

        if resp.get("status") and resp.get("data"):
            return resp

        last_msg = str(resp.get("message"))
        if _is_rate_limited(resp) and attempt < CANDLE_MAX_RETRIES - 1:
            # Jittered exponential backoff — desynchronizes retries if scans
            # ever run concurrently (e.g. manual /scan during the auto loop).
            wait = CANDLE_BACKOFF_BASE_SEC * (2 ** attempt)
            wait += random.uniform(0, wait * 0.5)
            logger.warning("Angel rate-limited on %s (%s) — backing off %.1fs "
                           "(attempt %d/%d)", label, last_msg, wait,
                           attempt + 1, CANDLE_MAX_RETRIES)
            time.sleep(wait)
            continue
        # Non-retryable (bad symbol, no data, permission) — stop retrying.
        break

    raise ValueError(f"No Angel data for {label}: {last_msg}")


def _load_scrip_master():
    """Load Angel's symbol→token map. Cached on disk (re-downloaded when >24h
    old) AND in memory — the in-memory copy is invalidated on IST date change
    so a long-running process picks up new listings / token remaps daily
    instead of holding the first day's map until restart."""
    global _scrip_master, _scrip_master_date
    today = datetime.now(IST).date()
    if _scrip_master is not None and _scrip_master_date == today:
        return _scrip_master
    _scrip_master_date = today

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


def _fetch_angel_by_token(exchange: str, token: str, period: str, interval: str,
                           label: str, *,
                           start: datetime | None = None,
                           end: datetime | None = None) -> pd.DataFrame:
    """
    Lower-level Angel fetch by explicit token + exchange.
    Used for indices (which aren't in the standard SYMBOL-EQ format).
    Explicit start/end override the period-derived window (paged backfill).
    """
    angel_interval = ANGEL_INTERVALS.get(interval)
    if not angel_interval:
        raise ValueError(f"Interval '{interval}' not supported by Angel One. "
                         f"Supported: {', '.join(ANGEL_INTERVALS)}")

    now = datetime.now(IST)
    if start is None:
        start = now - timedelta(days=PERIOD_DAYS.get(period, 180))
    end = end or now
    params = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": angel_interval,
        "fromdate": start.strftime("%Y-%m-%d %H:%M"),
        "todate": end.strftime("%Y-%m-%d %H:%M"),
    }

    resp = _get_candle_data(params, label)

    df = pd.DataFrame(resp["data"],
                      columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.index = df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize(IST)
    return df


def fetch_angel_index(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch index OHLCV from Angel using token lookup in indices.py."""
    from indices import get_index_info
    info = get_index_info(symbol)
    if not info:
        raise ValueError(f"'{symbol}' is not a recognized index")
    return _fetch_angel_by_token(
        exchange=info["angel_exchange"],
        token=info["angel_token"],
        period=period,
        interval=interval,
        label=info["display"],
    )


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

    resp = _get_candle_data(params, symbol)

    df = pd.DataFrame(resp["data"],
                      columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.index = df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize(IST)
    _warn_unadjusted_split(df, symbol, angel_interval)
    return df


def _warn_unadjusted_split(df: pd.DataFrame, symbol: str, angel_interval: str) -> None:
    """Angel daily candles are NOT split/bonus-adjusted. A corporate action inside
    the window leaves a price cliff that corrupts SMA/RSI/Bollinger/ATR and trade
    levels. We can't back-adjust here (no action data in the candle API), but we
    warn so the operator knows the window is suspect. Conservative threshold
    (>40% adjacent-close gap) sits well beyond NSE circuit limits, so genuine
    single-day moves won't trip it."""
    if angel_interval != "ONE_DAY" or len(df) < 2:
        return
    close = df["Close"].astype(float)
    ratio = close / close.shift(1)
    bad = ratio[(ratio < 0.6) | (ratio > 1.66)]
    if len(bad):
        when = ", ".join(str(i.date()) for i in bad.index[:3])
        logger.warning(
            "%s: suspicious >40%% daily gap (%s) — likely an UNADJUSTED split/bonus "
            "in Angel data; swing indicators/levels for this window may be wrong.",
            symbol, when)


# ---------- yfinance fallback ----------

def fetch_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    # Pin auto_adjust=True explicitly: yfinance's default flipped across versions,
    # and unadjusted ('Close' = raw traded price) puts a split/bonus discontinuity
    # into the series that corrupts SMA/RSI/Bollinger/ATR and the trade levels.
    # auto_adjust=True back-adjusts OHLC for splits + dividends — correct for TA.
    df = yf.Ticker(symbol).history(
        period=period, interval=interval, auto_adjust=True, actions=False,
    )
    if df.empty:
        raise ValueError(f"No yfinance data for {symbol}")
    return df


# ---------- Public API ----------

def get_provider_name() -> str:
    return "Angel One" if os.getenv("ANGEL_API_KEY") else "Yahoo Finance (delayed)"


def _fetch_routed(symbol: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    """Provider routing (see fetch_data docstring). Returns (df, source)."""
    from indices import get_index_info
    idx_info = get_index_info(symbol)

    if idx_info is not None:
        if os.getenv("ANGEL_API_KEY"):
            try:
                return fetch_angel_index(symbol, period, interval), "angel"
            except Exception as e:
                logger.warning(
                    "Angel index fetch failed for %s (%s) — falling back to yfinance",
                    symbol, e,
                )
        return fetch_yfinance(idx_info["yf"], period, interval), "yfinance"

    if os.getenv("ANGEL_API_KEY"):
        return fetch_angel(symbol, period, interval), "angel"
    return fetch_yfinance(symbol, period, interval), "yfinance"


def fetch_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV. Routes:
    - Indices (^... or known aliases) → Angel by token, yfinance fallback
    - Stocks → Angel by SYMBOL-EQ lookup; NO yfinance fallback when Angel is
      configured (see module docstring — delayed data must not silently feed
      intraday signals). yfinance is used only when Angel creds are absent.

    Every successful fetch is appended to the local candle archive
    (data_archive) — provider history caps make candles perishable; the
    archive makes them compound. The write is fire-and-forget: it can never
    fail the fetch, and ARCHIVE_DISABLED=true turns it off.
    """
    df, source = _fetch_routed(symbol, period, interval)
    try:
        import data_archive
        data_archive.archive_fetch(symbol, interval, df, source)
    except Exception as e:
        logger.debug("candle archive skipped for %s: %s", symbol, e)
    return df


def fetch_range(symbol: str, start: datetime, end: datetime,
                interval: str = "5m") -> pd.DataFrame:
    """
    Explicit-window OHLCV fetch for paged backfills (data_archive CLI).
    Same routing rules as fetch_data; naive start/end are taken as IST.
    yfinance path is subject to its own history caps regardless of window.
    """
    start = IST.localize(start) if start.tzinfo is None else start.astimezone(IST)
    end = IST.localize(end) if end.tzinfo is None else end.astimezone(IST)

    from indices import get_index_info
    idx_info = get_index_info(symbol)

    if idx_info is not None:
        if os.getenv("ANGEL_API_KEY"):
            try:
                return _fetch_angel_by_token(
                    exchange=idx_info["angel_exchange"],
                    token=idx_info["angel_token"],
                    period="", interval=interval, label=idx_info["display"],
                    start=start, end=end,
                )
            except Exception as e:
                logger.warning(
                    "Angel index range fetch failed for %s (%s) — yfinance fallback",
                    symbol, e,
                )
        return _fetch_yfinance_range(idx_info["yf"], start, end, interval)

    if os.getenv("ANGEL_API_KEY"):
        token, exch, _ = _resolve_token(symbol)
        df = _fetch_angel_by_token(exch, token, "", interval, symbol,
                                   start=start, end=end)
        _warn_unadjusted_split(df, symbol, ANGEL_INTERVALS.get(interval, ""))
        return df
    return _fetch_yfinance_range(symbol, start, end, interval)


def _fetch_yfinance_range(symbol: str, start: datetime, end: datetime,
                          interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(symbol).history(
        start=start, end=end, interval=interval,
        auto_adjust=True, actions=False,
    )
    if df.empty:
        raise ValueError(f"No yfinance data for {symbol} in "
                         f"{start.date()}–{end.date()}")
    return df
