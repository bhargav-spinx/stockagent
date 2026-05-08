"""
SQLite-backed state for auto-scan alerts.

Two tables:
- scan_subscribers: per-user opt-in to /scan_alerts
- fired_signals: dedup so the same setup doesn't alert twice on the
  same trading day. Keyed by (symbol, setup, direction, trade_date).
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "stockagent.db"


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON fired_signals(trade_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_watch_user ON user_watchlist(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts_log(trade_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cat_date ON alerts_log(category, trade_date)")


# ---------- intraday auto-scan subscriptions ----------

def subscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO scan_subscribers(user_id, subscribed_at) VALUES (?, ?)",
            (user_id, datetime.utcnow().isoformat()),
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
            (user_id, datetime.utcnow().isoformat()),
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
            (user_id, symbol, datetime.utcnow().isoformat()),
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
    trade_date = trade_date or date.today()
    return f"{symbol}:{setup}:{direction}:{trade_date.isoformat()}"


def already_fired(key: str) -> bool:
    with _conn() as c:
        cur = c.execute("SELECT 1 FROM fired_signals WHERE key = ?", (key,))
        return cur.fetchone() is not None


def mark_fired(key: str) -> None:
    today = date.today().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO fired_signals(key, fired_at, trade_date) VALUES (?, ?, ?)",
            (key, datetime.utcnow().isoformat(), today),
        )


def purge_old_signals(keep_days: int = 2) -> int:
    """Drop signals older than keep_days. Returns rows deleted."""
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM fired_signals "
            "WHERE trade_date < date('now', '-' || ? || ' days')",
            (keep_days,),
        )
        return cur.rowcount


# ---------- end-of-day report subscriptions ----------

def eod_subscribe(user_id: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO eod_report_subscribers(user_id, subscribed_at) VALUES (?, ?)",
            (user_id, datetime.utcnow().isoformat()),
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
) -> int:
    """Insert an alert row. Returns new alert_id."""
    generated_at = generated_at or datetime.utcnow()
    trade_date = date.today().isoformat()
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO alerts_log
                (generated_at, trade_date, category, user_id, symbol, setup,
                 direction, entry, stop_loss, target1, target2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (generated_at.isoformat(), trade_date, category, user_id, symbol,
             setup, direction, entry, stop_loss, target1, target2),
        )
        return cur.lastrowid


def get_alerts_for_date(trade_date_str: str | None = None,
                       category: str | None = None,
                       user_id: int | None = None) -> list[dict]:
    """Return all alerts (with outcomes if any) matching the filters."""
    trade_date_str = trade_date_str or date.today().isoformat()
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
) -> None:
    with _conn() as c:
        c.execute(
            """
            INSERT OR REPLACE INTO alert_outcomes
                (alert_id, status, exit_price, exit_time, pnl_pct, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (alert_id, status, exit_price,
             exit_time.isoformat() if exit_time else None,
             pnl_pct, datetime.utcnow().isoformat()),
        )
