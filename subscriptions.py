"""
SQLite-backed state for auto-scan alerts.

Two tables:
- scan_subscribers: per-user opt-in to /scan_alerts
- fired_signals: dedup so the same setup doesn't alert twice on the
  same trading day. Keyed by (symbol, setup, direction, trade_date).
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timezone
from pathlib import Path

from constants import IST

DB_PATH = Path(__file__).parent / "stockagent.db"


def _ist_today() -> date:
    """Trade-date = the IST calendar date, NOT the server's local/UTC date.
    The GCP VM runs UTC; date.today() there only matches the IST trade date
    because NSE hours happen to sit inside one UTC day — a fragile accident.
    Every trade_date / dedup key must go through this instead."""
    return datetime.now(IST).date()


@contextmanager
def _conn():
    # timeout=30: wait up to 30s for a lock instead of erroring immediately.
    # 5 background loops + command handlers + the Telethon listener all write
    # this DB concurrently, so a 0s busy-timeout (the default) raises
    # "database is locked" under contention.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        # WAL: concurrent readers + a single writer without readers blocking —
        # the right mode for many background loops sharing one SQLite file.
        # Persisted in the DB header, so this runs once effectively.
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""
            CREATE TABLE IF NOT EXISTS scan_subscribers (
                user_id        INTEGER PRIMARY KEY,
                subscribed_at  DATETIME NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS swing_subscribers (
                user_id        INTEGER PRIMARY KEY,
                subscribed_at  DATETIME NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS eod_report_subscribers (
                user_id        INTEGER PRIMARY KEY,
                subscribed_at  DATETIME NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_watchlist (
                user_id   INTEGER NOT NULL,
                symbol    TEXT NOT NULL,
                added_at  DATETIME NOT NULL,
                PRIMARY KEY (user_id, symbol)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS fired_signals (
                key          TEXT PRIMARY KEY,
                fired_at     DATETIME NOT NULL,
                trade_date   DATE NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at  DATETIME NOT NULL,
                trade_date    DATE NOT NULL,
                category      TEXT NOT NULL,
                user_id       INTEGER,
                symbol        TEXT NOT NULL,
                setup         TEXT,
                direction     TEXT NOT NULL,
                entry         REAL NOT NULL,
                stop_loss     REAL NOT NULL,
                target1       REAL NOT NULL,
                target2       REAL NOT NULL
            )
        """)
        # Immutable, append-only record of every parsed channel tip AS POSTED,
        # captured at receipt BEFORE any analysis or outcome is known. This is
        # the audit trail that makes honest channel evaluation possible and
        # rules out any later lookahead (the call-time levels are frozen here).
        c.execute("""
            CREATE TABLE IF NOT EXISTS raw_tips (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at   DATETIME NOT NULL,
                channel       TEXT NOT NULL,
                msg_id        INTEGER,
                symbol        TEXT,
                action        TEXT,
                entry         REAL,
                target        REAL,
                target2       REAL,
                stop_loss     REAL,
                raw_text      TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS alert_outcomes (
                alert_id     INTEGER PRIMARY KEY,
                status       TEXT NOT NULL,
                exit_price   REAL,
                exit_time    DATETIME,
                pnl_pct      REAL,
                resolved_at  DATETIME NOT NULL,
                FOREIGN KEY (alert_id) REFERENCES alerts_log(id)
            )
        """)
        # Migration: per-outcome "notified" flag so swing completions are
        # pushed to the recipient exactly once, regardless of who resolves first.
        oc_cols = [r[1] for r in c.execute("PRAGMA table_info(alert_outcomes)").fetchall()]
        if "notified" not in oc_cols:
            c.execute("ALTER TABLE alert_outcomes ADD COLUMN notified INTEGER NOT NULL DEFAULT 0")
        # Migration (Phase 1 quant platform): excursion + duration telemetry on
        # outcomes. MFE/MAE turn each scarce resolved trade into several data
        # points (were stops too tight? targets too far? time-stop well placed?).
        for col in ("mfe_pct", "mae_pct", "duration_min"):
            if col not in oc_cols:
                c.execute(f"ALTER TABLE alert_outcomes ADD COLUMN {col} REAL")

        # Migration (Phase 1 quant platform): decision-time telemetry on alerts.
        #   score    — numeric score column (previously smuggled into `setup`
        #              as the string "score{N}"; that string is still written
        #              for backward compat with stats.py parsing).
        #   features — JSON snapshot of every engine-computed value at call
        #              time. This is the Learning Engine's training data;
        #              features CANNOT be reconstructed later from re-fetched
        #              candles (adjustments, provider caps, repaints).
        al_cols = [r[1] for r in c.execute("PRAGMA table_info(alerts_log)").fetchall()]
        if "score" not in al_cols:
            c.execute("ALTER TABLE alerts_log ADD COLUMN score REAL")
        if "features" not in al_cols:
            c.execute("ALTER TABLE alerts_log ADD COLUMN features TEXT")

        c.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON fired_signals(trade_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_watch_user ON user_watchlist(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts_log(trade_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cat_date ON alerts_log(category, trade_date)")


# ---------- intraday auto-scan subscriptions ----------

def subscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO scan_subscribers(user_id, subscribed_at) VALUES (?, ?)",
            (user_id, datetime.now(timezone.utc).isoformat()),
        )


def unsubscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM scan_subscribers WHERE user_id = ?", (user_id,))


def is_subscribed(user_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("SELECT 1 FROM scan_subscribers WHERE user_id = ?", (user_id,))
        return cur.fetchone() is not None


def get_subscribers() -> list[int]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT user_id FROM scan_subscribers").fetchall()]


# ---------- swing (end-of-day) subscriptions ----------

def swing_subscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO swing_subscribers(user_id, subscribed_at) VALUES (?, ?)",
            (user_id, datetime.now(timezone.utc).isoformat()),
        )


def swing_unsubscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM swing_subscribers WHERE user_id = ?", (user_id,))


def is_swing_subscribed(user_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("SELECT 1 FROM swing_subscribers WHERE user_id = ?", (user_id,))
        return cur.fetchone() is not None


def get_swing_subscribers() -> list[int]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT user_id FROM swing_subscribers").fetchall()]


# ---------- per-user watchlist (used by /watch + /swing_alerts) ----------

def add_to_watchlist(user_id: int, symbol: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO user_watchlist(user_id, symbol, added_at) VALUES (?, ?, ?)",
            (user_id, symbol, datetime.now(timezone.utc).isoformat()),
        )


def remove_from_watchlist(user_id: int, symbol: str) -> bool:
    """Returns True if a row was deleted."""
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM user_watchlist WHERE user_id = ? AND symbol = ?",
            (user_id, symbol),
        )
        return cur.rowcount > 0


def get_watchlist(user_id: int) -> list[str]:
    with _conn() as c:
        return [
            r[0] for r in c.execute(
                "SELECT symbol FROM user_watchlist WHERE user_id = ? ORDER BY added_at",
                (user_id,),
            ).fetchall()
        ]


# ---------- signal dedup ----------

def signal_key(symbol: str, setup: str, direction: str,
               trade_date: date | None = None) -> str:
    trade_date = trade_date or _ist_today()
    return f"{symbol}:{setup}:{direction}:{trade_date.isoformat()}"


def already_fired(key: str) -> bool:
    with _conn() as c:
        cur = c.execute("SELECT 1 FROM fired_signals WHERE key = ?", (key,))
        return cur.fetchone() is not None


def mark_fired(key: str) -> None:
    today = _ist_today().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO fired_signals(key, fired_at, trade_date) VALUES (?, ?, ?)",
            (key, datetime.now(timezone.utc).isoformat(), today),
        )


def purge_old_signals(keep_days: int = 2) -> int:
    """Drop signals older than keep_days. Returns rows deleted.

    Retention cutoffs here and in purge_old_data use SQLite's date('now'),
    which is UTC — up to 5.5h behind IST. Deliberately left as-is: against
    multi-day retention windows the skew only delays deletion slightly (the
    safe direction). Trade-date WRITES, by contrast, must be IST (_ist_today)."""
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM fired_signals "
            "WHERE trade_date < date('now', '-' || ? || ' days')",
            (keep_days,),
        )
        return cur.rowcount


def purge_old_data(alerts_keep_days: int = 730,
                   raw_tips_keep_days: int = 365) -> dict[str, int]:
    """Conservative cleanup so the DB doesn't grow unbounded. Retention is
    deliberately long — ~2y of alerts/outcomes preserves the /stats performance
    record; ~1y of raw channel tips. Returns rows deleted per table."""
    with _conn() as c:
        oc = c.execute(
            "DELETE FROM alert_outcomes WHERE alert_id IN ("
            "  SELECT id FROM alerts_log "
            "  WHERE trade_date < date('now', '-' || ? || ' days'))",
            (alerts_keep_days,),
        ).rowcount
        al = c.execute(
            "DELETE FROM alerts_log "
            "WHERE trade_date < date('now', '-' || ? || ' days')",
            (alerts_keep_days,),
        ).rowcount
        # raw_tips.received_at is an ISO timestamp; compare on its date prefix
        # (tz-suffix-agnostic) against the cutoff.
        rt = c.execute(
            "DELETE FROM raw_tips "
            "WHERE substr(received_at, 1, 10) < date('now', '-' || ? || ' days')",
            (raw_tips_keep_days,),
        ).rowcount
    return {"alert_outcomes": oc, "alerts_log": al, "raw_tips": rt}


# ---------- end-of-day report subscriptions ----------

def eod_subscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO eod_report_subscribers(user_id, subscribed_at) VALUES (?, ?)",
            (user_id, datetime.now(timezone.utc).isoformat()),
        )


def eod_unsubscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM eod_report_subscribers WHERE user_id = ?", (user_id,))


def is_eod_subscribed(user_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("SELECT 1 FROM eod_report_subscribers WHERE user_id = ?", (user_id,))
        return cur.fetchone() is not None


def get_eod_subscribers() -> list[int]:
    with _conn() as c:
        return [r[0] for r in c.execute(
            "SELECT user_id FROM eod_report_subscribers"
        ).fetchall()]


# ---------- immutable raw-tip capture (channel accountability) ----------

def log_raw_tip(
    *,
    channel: str,
    msg_id: int | None,
    symbol: str | None,
    action: str | None,
    entry: float | None,
    target: float | None,
    target2: float | None,
    stop_loss: float | None,
    raw_text: str,
    received_at: datetime | None = None,
) -> int:
    """Append a channel tip exactly as parsed, at receipt. Append-only — never
    updated — so it is an immutable call-time record. Returns the new row id."""
    received_at = received_at or datetime.now(timezone.utc)
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO raw_tips
                (received_at, channel, msg_id, symbol, action,
                 entry, target, target2, stop_loss, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (received_at.isoformat(), channel, msg_id, symbol, action,
             entry, target, target2, stop_loss, raw_text[:1000]),
        )
        return cur.lastrowid


# ---------- alerts log + outcomes ----------

def log_alert(
    *,
    category: str,           # 'scan' | 'swing_auto' | 'manual_intraday' | 'manual_swing'
    user_id: int | None,
    symbol: str,
    setup: str | None,
    direction: str,
    entry: float,
    stop_loss: float,
    target1: float,
    target2: float,
    generated_at: datetime | None = None,
    score: float | None = None,
    features: dict | None = None,
) -> int:
    """Insert an alert row. Returns new alert_id.

    `features` is the decision-time snapshot (see features.py builders) stored
    as canonical JSON; `score` is the numeric engine score. Both optional —
    older callers keep working unchanged."""
    generated_at = generated_at or datetime.now(timezone.utc)
    trade_date = _ist_today().isoformat()
    features_json = (json.dumps(features, sort_keys=True, default=str)
                     if features else None)
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO alerts_log
                (generated_at, trade_date, category, user_id, symbol, setup,
                 direction, entry, stop_loss, target1, target2, score, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (generated_at.isoformat(), trade_date, category, user_id, symbol,
             setup, direction, entry, stop_loss, target1, target2,
             score, features_json),
        )
        return cur.lastrowid


def get_alert_features(alert_id: int) -> dict | None:
    """Parsed decision-time feature snapshot for one alert (None if absent)."""
    with _conn() as c:
        row = c.execute(
            "SELECT features FROM alerts_log WHERE id = ?", (alert_id,)
        ).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return None


def get_training_rows(categories: tuple[str, ...] | None = None) -> list[dict]:
    """(features @ decision time, resolved outcome) pairs — the export the
    future probability engine trains on. Only resolved alerts WITH a feature
    snapshot qualify; rows are returned oldest-first."""
    sql = """
        SELECT a.id, a.generated_at, a.trade_date, a.category, a.symbol,
               a.setup, a.direction, a.entry, a.stop_loss, a.target1,
               a.target2, a.score, a.features,
               o.status, o.exit_price, o.exit_time, o.pnl_pct,
               o.mfe_pct, o.mae_pct, o.duration_min
        FROM alerts_log a
        JOIN alert_outcomes o ON o.alert_id = a.id
        WHERE a.features IS NOT NULL
    """
    params: list = []
    if categories:
        sql += f" AND a.category IN ({','.join('?' * len(categories))})"
        params.extend(categories)
    sql += " ORDER BY a.generated_at"
    cols = [
        "id", "generated_at", "trade_date", "category", "symbol",
        "setup", "direction", "entry", "stop_loss", "target1",
        "target2", "score", "features",
        "status", "exit_price", "exit_time", "pnl_pct",
        "mfe_pct", "mae_pct", "duration_min",
    ]
    with _conn() as c:
        rows = [dict(zip(cols, r)) for r in c.execute(sql, params).fetchall()]
    for r in rows:
        try:
            r["features"] = json.loads(r["features"]) if r["features"] else None
        except (TypeError, ValueError):
            r["features"] = None
    return rows


def get_alerts_for_date(trade_date_str: str | None = None,
                       category: str | None = None,
                       user_id: int | None = None) -> list[dict]:
    """Return all alerts (with outcomes if any) matching the filters."""
    trade_date_str = trade_date_str or _ist_today().isoformat()
    sql = """
        SELECT a.id, a.generated_at, a.trade_date, a.category, a.user_id,
               a.symbol, a.setup, a.direction, a.entry, a.stop_loss,
               a.target1, a.target2,
               o.status, o.exit_price, o.exit_time, o.pnl_pct, o.resolved_at
        FROM alerts_log a
        LEFT JOIN alert_outcomes o ON o.alert_id = a.id
        WHERE a.trade_date = ?
    """
    params: list = [trade_date_str]
    if category:
        sql += " AND a.category = ?"
        params.append(category)
    if user_id is not None:
        sql += " AND (a.user_id = ? OR a.user_id IS NULL)"
        params.append(user_id)
    sql += " ORDER BY a.generated_at"

    with _conn() as c:
        cols = [
            "id", "generated_at", "trade_date", "category", "user_id",
            "symbol", "setup", "direction", "entry", "stop_loss",
            "target1", "target2",
            "status", "exit_price", "exit_time", "pnl_pct", "resolved_at",
        ]
        return [dict(zip(cols, row)) for row in c.execute(sql, params).fetchall()]


def get_open_alerts(max_age_days: int = 30) -> list[dict]:
    """All alerts without an outcome row, capped at max_age_days. Used by resolver."""
    sql = """
        SELECT a.id, a.generated_at, a.trade_date, a.category, a.user_id,
               a.symbol, a.setup, a.direction, a.entry, a.stop_loss,
               a.target1, a.target2
        FROM alerts_log a
        LEFT JOIN alert_outcomes o ON o.alert_id = a.id
        WHERE o.alert_id IS NULL
          AND a.trade_date >= date('now', '-' || ? || ' days')
        ORDER BY a.generated_at
    """
    cols = [
        "id", "generated_at", "trade_date", "category", "user_id",
        "symbol", "setup", "direction", "entry", "stop_loss",
        "target1", "target2",
    ]
    with _conn() as c:
        return [dict(zip(cols, row)) for row in c.execute(sql, (max_age_days,)).fetchall()]


def save_outcome(
    alert_id: int,
    *,
    status: str,
    exit_price: float | None = None,
    exit_time: datetime | None = None,
    pnl_pct: float | None = None,
    mfe_pct: float | None = None,
    mae_pct: float | None = None,
    duration_min: float | None = None,
) -> None:
    with _conn() as c:
        c.execute(
            """
            INSERT OR REPLACE INTO alert_outcomes
                (alert_id, status, exit_price, exit_time, pnl_pct,
                 mfe_pct, mae_pct, duration_min, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (alert_id, status, exit_price,
             exit_time.isoformat() if exit_time else None,
             pnl_pct, mfe_pct, mae_pct, duration_min,
             datetime.now(timezone.utc).isoformat()),
        )


_ALERT_OUTCOME_COLS = [
    "id", "generated_at", "trade_date", "category", "user_id",
    "symbol", "setup", "direction", "entry", "stop_loss", "target1", "target2",
    "status", "exit_price", "exit_time", "pnl_pct", "resolved_at",
]


def get_unnotified_resolved(categories: tuple[str, ...],
                            statuses: tuple[str, ...]) -> list[dict]:
    """Resolved alerts (joined with their outcome) that have not yet been
    pushed to their recipient. Filtered to the given categories + terminal
    statuses, and to alerts that have an owning user_id. Ordered oldest-first."""
    cat_ph = ",".join("?" * len(categories))
    st_ph = ",".join("?" * len(statuses))
    sql = f"""
        SELECT a.id, a.generated_at, a.trade_date, a.category, a.user_id,
               a.symbol, a.setup, a.direction, a.entry, a.stop_loss,
               a.target1, a.target2,
               o.status, o.exit_price, o.exit_time, o.pnl_pct, o.resolved_at
        FROM alerts_log a
        JOIN alert_outcomes o ON o.alert_id = a.id
        WHERE o.notified = 0
          AND a.user_id IS NOT NULL
          AND a.category IN ({cat_ph})
          AND o.status IN ({st_ph})
        ORDER BY o.resolved_at
    """
    with _conn() as c:
        rows = c.execute(sql, (*categories, *statuses)).fetchall()
    return [dict(zip(_ALERT_OUTCOME_COLS, row)) for row in rows]


def mark_outcome_notified(alert_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE alert_outcomes SET notified = 1 WHERE alert_id = ?", (alert_id,))


def get_resolved_alerts(categories: tuple[str, ...],
                        user_id: int | None = None) -> list[dict]:
    """All resolved alerts (joined with outcome) for the given categories,
    optionally scoped to one recipient. Used for the running performance record."""
    cat_ph = ",".join("?" * len(categories))
    sql = f"""
        SELECT a.id, a.generated_at, a.trade_date, a.category, a.user_id,
               a.symbol, a.setup, a.direction, a.entry, a.stop_loss,
               a.target1, a.target2,
               o.status, o.exit_price, o.exit_time, o.pnl_pct, o.resolved_at
        FROM alerts_log a
        JOIN alert_outcomes o ON o.alert_id = a.id
        WHERE a.category IN ({cat_ph})
    """
    params: list = list(categories)
    if user_id is not None:
        sql += " AND a.user_id = ?"
        params.append(user_id)
    sql += " ORDER BY o.resolved_at"
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    return [dict(zip(_ALERT_OUTCOME_COLS, row)) for row in rows]
