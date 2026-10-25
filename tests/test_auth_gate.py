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


class _FakeChat:
    def __init__(self, kind):
        self.type = kind


class _FakeUpdate:
    def __init__(self, uid, chat="private", edited=False):
        self.effective_user = _FakeUser(uid) if uid is not None else None
        self.effective_message = _FakeMsg()
        self.effective_chat = _FakeChat(chat)
        self.edited_message = self.effective_message if edited else None


def _blocked(u) -> bool:
    try:
        asyncio.run(bot._auth_gate(u, None))
        return False
    except ApplicationHandlerStop:
        return True


def test_unauthorized_user_is_blocked_and_notified():
    bot.AUTHORIZED_USERS = {111}
    bot._AUTH_NOTIFIED.clear()
    u = _FakeUpdate(222)
    assert _blocked(u), "expected ApplicationHandlerStop"
    assert u.effective_message.sent, "user should have been told they're blocked"


def test_unauthorized_notice_is_sent_once_per_user():
    bot.AUTHORIZED_USERS = {111}
    bot._AUTH_NOTIFIED.clear()
    first, second = _FakeUpdate(333), _FakeUpdate(333)
    assert _blocked(first) and _blocked(second)
    assert first.effective_message.sent and not second.effective_message.sent


def test_unauthorized_in_group_is_blocked_silently():
    # Group chatter (privacy mode off for tip capture) must not get a reply
    # per message — still blocked, but silent.
    bot.AUTHORIZED_USERS = {111}
    bot._AUTH_NOTIFIED.clear()
    g = _FakeUpdate(444, chat="supergroup")
    assert _blocked(g)
    assert not g.effective_message.sent


def test_edited_messages_are_ignored():
    # update.message is None for edits; handlers would crash on it.
    bot.AUTHORIZED_USERS = {111}
    e = _FakeUpdate(111, edited=True)
    assert _blocked(e)
    assert not e.effective_message.sent


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
