"""
Owner-only allowlist gate (personal-use / SEBI-scope guard).

Importing bot.py pulls python-telegram-bot (+ requests), which CI deliberately
does NOT install (it runs the lean, runtime-free subset). So this test skips
when those aren't present and runs locally where they are.

    venv/Scripts/python.exe tests/test_auth_gate.py      # local
    (CI: auto-skipped)
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
    pytest.importorskip("telegram")
    pytest.importorskip("requests")
except ImportError:        # running as a plain script without pytest
    pass

import bot                                                # noqa: E402
from telegram.ext import ApplicationHandlerStop           # noqa: E402


class _FakeUser:
    def __init__(self, uid):
        self.id = uid


class _FakeMsg:
    def __init__(self):
        self.sent = []

    async def reply_text(self, text, **kw):
        self.sent.append(text)


class _FakeUpdate:
    def __init__(self, uid):
        self.effective_user = _FakeUser(uid) if uid is not None else None
        self.effective_message = _FakeMsg()


def test_unauthorized_user_is_blocked_and_notified():
    bot.AUTHORIZED_USERS = {111}
    u = _FakeUpdate(222)
    try:
        asyncio.run(bot._auth_gate(u, None))
        assert False, "expected ApplicationHandlerStop"
    except ApplicationHandlerStop:
        pass
    assert u.effective_message.sent, "user should have been told they're blocked"


def test_authorized_user_passes_silently():
    bot.AUTHORIZED_USERS = {111}
    u = _FakeUpdate(111)
    asyncio.run(bot._auth_gate(u, None))          # must NOT raise
    assert not u.effective_message.sent


def test_empty_allowlist_is_open_mode():
    bot.AUTHORIZED_USERS = set()
    u = _FakeUpdate(999)
    asyncio.run(bot._auth_gate(u, None))          # open mode: no block
    assert not u.effective_message.sent


def test_id_parsing_ignores_junk_and_includes_notify_user():
    os.environ["AUTHORIZED_USERS"] = "1, 2 ,x,3"
    os.environ["TELETHON_NOTIFY_USER_ID"] = "7"
    try:
        assert bot._authorized_user_ids() == {1, 2, 3, 7}
    finally:
        os.environ.pop("AUTHORIZED_USERS", None)
        os.environ.pop("TELETHON_NOTIFY_USER_ID", None)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
