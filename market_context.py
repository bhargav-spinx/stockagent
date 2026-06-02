"""
External market-context inputs for the scoring engine:

  • delivery_pct(symbol)      — NSE EOD delivery %  (free, market-wide file)
  • days_to_earnings(symbol)  — NSE results calendar (free, market-wide file)
  • news_summary(symbol)      — Marketaux live news  (API key, rate-limited)

Design / cost discipline
------------------------
Delivery % and earnings each come from ONE market-wide NSE download per trading
day, cached in memory. After the first call of the day they are pure dict
lookups, so they are safe to call per-symbol inside the universe scan.

Marketaux is per-symbol and the free tier allows 100 requests/day (3 articles
each). It must NEVER be called inside the universe loop — only to enrich the
handful of alerts that actually fire (or on-demand /score). A daily budget
counter hard-stops before the limit, and results are cached per symbol per day.

Every fetch degrades to None/empty on any failure (network, corp-TLS, NSE
markup change, missing key) so scoring NEVER breaks. The corp-TLS shim
(ssl_dev) is applied on import so requests work behind inspection.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

try:  # apply the corp-TLS shim if the project provides one (no-op otherwise)
    import ssl_dev
    ssl_dev.install_if_enabled()
except Exception:
    pass

from constants import IST

logger = logging.getLogger(__name__)

# --- tunables ----------------------------------------------------------------
HTTP_TIMEOUT = 12
MARKETAUX_DAILY_BUDGET = 90          # stop before the 100/day free-tier ceiling
MARKETAUX_ARTICLES = 3               # free tier returns up to 3 per request
_NSE_HOME = "https://www.nseindia.com"
_NSE_BHAV = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
_NSE_EVENTS = "https://www.nseindia.com/api/event-calendar"
_MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# --- daily caches (rebuilt when the IST date rolls over) ----------------------
_delivery_cache: dict = {"date": None, "data": {}}
_earnings_cache: dict = {"date": None, "data": {}}
_news_cache: dict = {"date": None, "calls": 0, "by_symbol": {}}


def _today_str() -> str:
    return datetime.now(IST).date().isoformat()


def _base_symbol(symbol: str) -> str:
    """RELIANCE.NS / RELIANCE.BO / RELIANCE -> RELIANCE (uppercased)."""
    return symbol.upper().split(".")[0].strip()


def _nse_session() -> requests.Session:
    """A requests session seeded with NSE cookies (NSE blocks cold requests)."""
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get(_NSE_HOME, timeout=HTTP_TIMEOUT)  # seed cookies; ignore the body
    except Exception:
        pass
    return s


# ----------------------------------------------------------------------------
# Delivery % — NSE security-wise EOD bhavcopy (market-wide, cached once/day)
# ----------------------------------------------------------------------------
def _load_delivery() -> dict:
    """Download the most recent available delivery bhavcopy and return
    {SYMBOL: delivery_pct}. Tries today, then walks back over weekends/holidays."""
    sess = _nse_session()
    today = datetime.now(IST).date()
    for back in range(0, 6):                       # today .. 5 days back
        d = today - timedelta(days=back)
        url = _NSE_BHAV.format(ddmmyyyy=d.strftime("%d%m%Y"))
        try:
            r = sess.get(url, timeout=HTTP_TIMEOUT)
            if r.status_code != 200 or not r.text.strip():
                continue
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]   # NSE pads names
            if "SYMBOL" not in df.columns or "DELIV_PER" not in df.columns:
                continue
            df = df[df.get("SERIES", "EQ").astype(str).str.strip() == "EQ"]
            out: dict = {}
            for sym, pct in zip(df["SYMBOL"].astype(str).str.strip(),
                                pd.to_numeric(df["DELIV_PER"], errors="coerce")):
                if pct == pct:                      # not NaN
                    out[sym] = float(pct)
            if out:
                logger.info("delivery: loaded %d symbols from %s", len(out), d.isoformat())
                return out
        except Exception as e:
            logger.warning("delivery: fetch %s failed: %s", url, e)
    return {}


def delivery_pct(symbol: str) -> float | None:
    """Delivery % for a symbol from the cached daily bhavcopy. None = unknown."""
    today = _today_str()
    if _delivery_cache["date"] != today:
        _delivery_cache["data"] = _load_delivery()
        _delivery_cache["date"] = today            # cache even on empty: don't refetch all day
    return _delivery_cache["data"].get(_base_symbol(symbol))


# ----------------------------------------------------------------------------
# Earnings proximity — NSE event calendar (market-wide, cached once/day)
# ----------------------------------------------------------------------------
def _load_earnings() -> dict:
    """Return {SYMBOL: nearest_future_results_date} from NSE's event calendar."""
    sess = _nse_session()
    try:
        r = sess.get(_NSE_EVENTS, timeout=HTTP_TIMEOUT,
                     headers={**_NSE_HEADERS, "Accept": "application/json"})
        if r.status_code != 200:
            return {}
        rows = r.json()
        if isinstance(rows, dict):                 # some responses wrap in a key
            rows = rows.get("data") or next((v for v in rows.values()
                                             if isinstance(v, list)), [])
    except Exception as e:
        logger.warning("earnings: fetch failed: %s", e)
        return {}

    today = datetime.now(IST).date()
    out: dict = {}
    for row in rows or []:
        try:
            purpose = str(row.get("purpose") or row.get("bm_desc") or "")
            if "result" not in purpose.lower():
                continue
            sym = str(row.get("symbol") or "").strip().upper()
            raw = str(row.get("date") or "").strip()
            ed = None
            for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
                try:
                    ed = datetime.strptime(raw, fmt).date()
                    break
                except ValueError:
                    continue
            if not sym or ed is None or ed < today:
                continue
            if sym not in out or ed < out[sym]:
                out[sym] = ed
        except Exception:
            continue
    if out:
        logger.info("earnings: loaded %d upcoming results dates", len(out))
    return out


def days_to_earnings(symbol: str) -> int | None:
    """Calendar days until the symbol's next scheduled results date.
    0 = today; None = none scheduled / unknown."""
    today = _today_str()
    if _earnings_cache["date"] != today:
        _earnings_cache["data"] = _load_earnings()
        _earnings_cache["date"] = today
    ed = _earnings_cache["data"].get(_base_symbol(symbol))
    if ed is None:
        return None
    return (ed - datetime.now(IST).date()).days


# ----------------------------------------------------------------------------
# Live news — Marketaux (per-symbol, rate-limited, key required)
# ----------------------------------------------------------------------------
def _marketaux_token() -> str | None:
    return os.getenv("MARKETAUX_API_TOKEN") or os.getenv("MARKETAUX_KEY")


def latest_news(symbol: str, limit: int = MARKETAUX_ARTICLES) -> list[dict] | None:
    """Up to `limit` recent articles for a symbol via Marketaux. Returns a list
    of {title, source, url, published_at}, [] if none, or None if news is
    unavailable (no key / budget spent / error). Cached per symbol per day and
    hard-capped at MARKETAUX_DAILY_BUDGET requests/day."""
    token = _marketaux_token()
    if not token:
        return None

    today = _today_str()
    if _news_cache["date"] != today:               # new day → reset budget + cache
        _news_cache.update({"date": today, "calls": 0, "by_symbol": {}})

    base = _base_symbol(symbol)
    if base in _news_cache["by_symbol"]:
        return _news_cache["by_symbol"][base]
    if _news_cache["calls"] >= MARKETAUX_DAILY_BUDGET:
        logger.info("marketaux: daily budget (%d) reached — skipping %s",
                    MARKETAUX_DAILY_BUDGET, base)
        return None

    params = {
        "api_token": token,
        "symbols": f"{base}.NS",          # Marketaux NSE suffix is .NS (not .NSE)
        "filter_entities": "true",
        "language": "en",
        "limit": limit,
    }
    try:
        _news_cache["calls"] += 1
        r = requests.get(_MARKETAUX_URL, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            logger.warning("marketaux: HTTP %s for %s", r.status_code, base)
            return None
        data = r.json().get("data", []) or []
        articles = [{
            "title": a.get("title"),
            "source": a.get("source"),
            "url": a.get("url"),
            "published_at": a.get("published_at"),
        } for a in data[:limit] if a.get("title")]
        _news_cache["by_symbol"][base] = articles
        return articles
    except Exception as e:
        logger.warning("marketaux: fetch failed for %s: %s", base, e)
        return None


def news_summary(symbol: str, limit: int = MARKETAUX_ARTICLES) -> str | None:
    """A short Markdown news block for a fired alert, or None when unavailable.
    Intended for the ≤8 alerts that fire per day, NOT the universe scan."""
    articles = latest_news(symbol, limit=limit)
    if not articles:
        return None
    lines = ["📰 *Latest news*"]
    for a in articles:
        src = f" ({a['source']})" if a.get("source") else ""
        title = (a["title"][:110] + "…") if len(a["title"]) > 110 else a["title"]
        lines.append(f"• {title}{src}")
    return "\n".join(lines)
