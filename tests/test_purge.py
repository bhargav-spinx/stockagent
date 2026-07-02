"""
Conservative table pruning (#5): purge_old_data drops very old rows and keeps
recent ones (preserving the /stats record). CI-safe (subscriptions = stdlib).

    venv/Scripts/python.exe tests/test_purge.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subscriptions  # noqa: E402


def _fresh_db():
    p = Path(tempfile.gettempdir()) / "purge_test.db"
    for ext in ("", "-wal", "-shm"):
        f = Path(str(p) + ext)
        if f.exists():
            f.unlink()
    subscriptions.DB_PATH = p
    subscriptions.init_db()


def test_purge_drops_old_keeps_recent():
    _fresh_db()
    # Old alert (2020) inserted directly with an old trade_date.
    with subscriptions._conn() as c:
        c.execute(
            "INSERT INTO alerts_log (generated_at, trade_date, category, user_id, "
            "symbol, setup, direction, entry, stop_loss, target1, target2) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2020-01-01T00:00:00+00:00", "2020-01-01", "scan", 1, "OLD",
             None, "long", 100, 98, 104, 106),
        )
    # Recent alert (today) via the normal path.
    subscriptions.log_alert(category="scan", user_id=1, symbol="NEW", setup=None,
                            direction="long", entry=100, stop_loss=98,
                            target1=104, target2=106)

    res = subscriptions.purge_old_data()
    assert res["alerts_log"] >= 1, res

    # No-arg default = IST trade date, matching what log_alert wrote. Using
    # date.today() here would flake on UTC CI runners between 18:30–24:00 UTC
    # (server date one day behind IST).
    today_rows = subscriptions.get_alerts_for_date()
    symbols = {r["symbol"] for r in today_rows}
    assert "NEW" in symbols          # recent kept
    # old one is gone
    with subscriptions._conn() as c:
        n_old = c.execute(
            "SELECT COUNT(*) FROM alerts_log WHERE symbol='OLD'").fetchone()[0]
    assert n_old == 0


def test_purge_on_empty_db_is_safe():
    _fresh_db()
    res = subscriptions.purge_old_data()
    assert res == {"alert_outcomes": 0, "alerts_log": 0, "raw_tips": 0}


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
