"""
DB concurrency pragmas (Phase 10): WAL + busy_timeout, so the many background
loops sharing one SQLite file don't raise "database is locked" under contention.
CI-safe: subscriptions imports only stdlib.

    venv/Scripts/python.exe tests/test_db_pragmas.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subscriptions  # noqa: E402


def _fresh_db():
    p = Path(tempfile.gettempdir()) / "pragma_test.db"
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    subscriptions.DB_PATH = p
    subscriptions.init_db()
    return p


def test_journal_mode_is_wal():
    _fresh_db()
    with subscriptions._conn() as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_busy_timeout_is_set():
    _fresh_db()
    with subscriptions._conn() as c:
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_sequential_writes_from_separate_connections_succeed():
    # Short-lived txns (as the real code uses) must not collide.
    _fresh_db()
    subscriptions.subscribe(1)
    subscriptions.subscribe(2)
    assert set(subscriptions.get_subscribers()) == {1, 2}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1; print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
