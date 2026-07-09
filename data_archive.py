"""
Local OHLCV candle archive (Phase 1.3).

WHY: intraday history is provider-capped (yfinance ~60 days of 5-min; Angel
serves deeper history only in paged windows), so every day of candles not
saved is an out-of-sample backtest fold lost forever. This module makes data
compound: every successful live fetch is appended here automatically (see
data_provider.fetch_data), a paged backfill CLI seeds deeper history, and
`backtest.py --local` replays from this frozen store — reproducible inputs
instead of drifting provider snapshots.

Storage: SQLite (`market_data.db`, gitignored), WAL mode, one row per candle:
    candles(symbol, interval, ts_utc ISO, open, high, low, close, volume, source)
    PRIMARY KEY (symbol, interval, ts_utc)  → INSERT OR REPLACE = idempotent
    upsert; a still-forming candle stored mid-session is overwritten by the
    settled candle on the next fetch.

Timezones: naive input indexes are assumed IST (project convention);
everything is stored as UTC ISO and loaded back as an IST-aware index.

CLI:
    python data_archive.py backfill --days 365 --universe tier1
    python data_archive.py backfill --days 400 --symbols RELIANCE TCS
    python data_archive.py coverage
Backfill pages Angel One in provider-safe chunks with pacing; on a
yfinance-only setup it still runs but is capped by yfinance's own limits
(a warning says so).
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time as time_mod
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from constants import IST

logger = logging.getLogger(__name__)

ARCHIVE_DB_PATH = Path(__file__).parent / "market_data.db"

# Angel per-request history windows vary by interval; 55 days of 5-min sits
# comfortably under every published cap. Daily pages far coarser.
_CHUNK_DAYS_INTRADAY = 55
_CHUNK_DAYS_DAILY = 365
_BACKFILL_PACING_SEC = 0.6      # ~same headroom logic as scanner pacing

_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@contextmanager
def _conn():
    conn = sqlite3.connect(ARCHIVE_DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_archive() -> None:
    with _conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol    TEXT NOT NULL,
                interval  TEXT NOT NULL,
                ts_utc    TEXT NOT NULL,
                open      REAL, high REAL, low REAL, close REAL,
                volume    REAL,
                source    TEXT,
                PRIMARY KEY (symbol, interval, ts_utc)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_candles_sym "
                  "ON candles(symbol, interval)")


def _to_utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    return idx.tz_convert("UTC")


def store_dataframe(symbol: str, interval: str, df: pd.DataFrame,
                    source: str = "unknown") -> int:
    """Upsert a fetched OHLCV frame. Returns rows written. Rows with a NaN
    Close are dropped (half-formed provider artifacts)."""
    if df is None or len(df) == 0:
        return 0
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        logger.warning("archive: %s %s missing columns %s — not stored",
                       symbol, interval, missing)
        return 0
    init_archive()
    utc_idx = _to_utc_index(df)
    rows = []
    for ts, (o, h, l, c_, v) in zip(utc_idx, df[_COLUMNS].itertuples(index=False)):
        if c_ != c_:            # NaN close
            continue
        rows.append((symbol, interval, ts.isoformat(),
                     float(o), float(h), float(l), float(c_),
                     float(v) if v == v else 0.0, source))
    if not rows:
        return 0
    with _conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO candles "
            "(symbol, interval, ts_utc, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def load(symbol: str, interval: str,
         start: datetime | None = None,
         end: datetime | None = None) -> pd.DataFrame:
    """Load archived candles as an IST-indexed OHLCV frame (fetch_data shape).
    start/end are inclusive bounds; naive datetimes are taken as IST.
    Empty archive → empty frame with the right columns."""
    init_archive()
    with _conn() as c:
        rows = c.execute(
            "SELECT ts_utc, open, high, low, close, volume FROM candles "
            "WHERE symbol = ? AND interval = ? ORDER BY ts_utc",
            (symbol, interval),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=_COLUMNS,
                            index=pd.DatetimeIndex([], tz=IST))
    idx = pd.to_datetime([r[0] for r in rows], utc=True).tz_convert(IST)
    df = pd.DataFrame(
        [(r[1], r[2], r[3], r[4], r[5]) for r in rows],
        columns=_COLUMNS, index=idx,
    ).astype(float)

    def _ist(dt: datetime) -> datetime:
        return IST.localize(dt) if dt.tzinfo is None else dt.astimezone(IST)

    if start is not None:
        df = df[df.index >= _ist(start)]
    if end is not None:
        df = df[df.index <= _ist(end)]
    return df


def coverage(symbol: str | None = None,
             interval: str | None = None) -> list[dict]:
    """Per-(symbol, interval) coverage: first/last candle (IST), row count,
    distinct session count."""
    init_archive()
    sql = ("SELECT symbol, interval, MIN(ts_utc), MAX(ts_utc), COUNT(*) "
           "FROM candles")
    where, params = [], []
    if symbol:
        where.append("symbol = ?")
        params.append(symbol)
    if interval:
        where.append("interval = ?")
        params.append(interval)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " GROUP BY symbol, interval ORDER BY symbol, interval"
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    out = []
    for sym, iv, lo, hi, n in rows:
        lo_ist = pd.Timestamp(lo).tz_convert(IST) if lo else None
        hi_ist = pd.Timestamp(hi).tz_convert(IST) if hi else None
        out.append({"symbol": sym, "interval": iv, "first": lo_ist,
                    "last": hi_ist, "rows": n})
    return out


def archive_fetch(symbol: str, interval: str, df, source: str) -> None:
    """Fire-and-forget hook used by data_provider.fetch_data on every
    successful fetch. MUST never raise into the fetch path; disable entirely
    with ARCHIVE_DISABLED=true."""
    import os
    if os.getenv("ARCHIVE_DISABLED", "").lower() in ("1", "true", "yes"):
        return
    try:
        store_dataframe(symbol, interval, df, source=source)
    except Exception as e:
        logger.warning("archive: store failed for %s %s: %s", symbol, interval, e)


# ---------------------------------------------------------------------------
# Backfill CLI
# ---------------------------------------------------------------------------

def backfill_symbol(symbol: str, days: int, interval: str = "5m",
                    pacing_sec: float = _BACKFILL_PACING_SEC) -> int:
    """Page an explicit-window fetch backwards over `days` and store every
    chunk. Returns total rows written. Chunks that fail are logged and
    skipped — a rate-limited window must not abort the whole symbol."""
    from data_provider import fetch_range

    chunk_days = _CHUNK_DAYS_DAILY if interval == "1d" else _CHUNK_DAYS_INTRADAY
    now = datetime.now(IST)
    start = now - timedelta(days=days)
    total = 0
    while start < now:
        chunk_end = min(start + timedelta(days=chunk_days), now)
        try:
            df = fetch_range(symbol, start, chunk_end, interval=interval)
            total += store_dataframe(symbol, interval, df, source="backfill")
        except Exception as e:
            logger.warning("backfill %s %s→%s: %s",
                           symbol, start.date(), chunk_end.date(), e)
        start = chunk_end
        time_mod.sleep(pacing_sec)
    return total


def _resolve_universe(name: str) -> list[str]:
    import universe
    return {
        "tier1": universe.TIER1_WATCHLIST,
        "intraday": universe.INTRADAY_UNIVERSE,
        "swing": universe.SWING_UNIVERSE,
    }[name]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Local OHLCV candle archive")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bf = sub.add_parser("backfill", help="Page provider history into the archive")
    bf.add_argument("--days", type=int, default=365)
    bf.add_argument("--interval", default="5m")
    bf.add_argument("--symbols", nargs="*", default=None)
    bf.add_argument("--universe", choices=["tier1", "intraday", "swing"],
                    default=None)
    bf.add_argument("--pacing", type=float, default=_BACKFILL_PACING_SEC)

    cov = sub.add_parser("coverage", help="Show archived coverage")
    cov.add_argument("--interval", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "coverage":
        rows = coverage(interval=args.interval)
        if not rows:
            print("Archive is empty — run a backfill or let the bot fetch.")
            return
        for r in rows:
            print(f"{r['symbol']:<16} {r['interval']:<4} "
                  f"{r['first']:%Y-%m-%d} → {r['last']:%Y-%m-%d}  "
                  f"({r['rows']} candles)")
        return

    import os
    from analyzer import normalize_symbol
    if not args.symbols and not args.universe:
        parser.error("backfill needs --symbols or --universe")
    symbols = args.symbols or _resolve_universe(args.universe)
    symbols = [normalize_symbol(s) for s in symbols]
    if not os.getenv("ANGEL_API_KEY"):
        print("NOTE: no Angel One credentials — backfill falls back to "
              "yfinance, which caps 5-min history at ~60 days regardless of "
              "--days.")
    print(f"Backfilling {len(symbols)} symbol(s), {args.days}d of "
          f"{args.interval} candles…")
    grand = 0
    for i, sym in enumerate(symbols, 1):
        n = backfill_symbol(sym, args.days, args.interval, args.pacing)
        grand += n
        print(f"[{i}/{len(symbols)}] {sym}: {n} candles")
    print(f"Done — {grand} candles stored in {ARCHIVE_DB_PATH.name}. "
          f"Use `python backtest.py --local …` to replay from the archive.")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    _cli()
