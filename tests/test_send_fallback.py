"""
Broadcast delivery fallback (2026-07-22 incident): a Markdown entity error
must never cost the user an alert — retry plain, and only a double failure
counts as undelivered.

    venv/Scripts/python.exe -m pytest tests/test_send_fallback.py
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

telegram = pytest.importorskip("telegram")   # bot.py imports it at module level
import bot  # noqa: E402


class _StubBot:
    """Records send attempts; raises per the scripted behaviors."""
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)      # one entry per expected call
        self.calls = []

    async def send_message(self, chat_id, text, parse_mode=None):
        self.calls.append({"chat_id": chat_id, "parse_mode": parse_mode})
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b


def test_markdown_success_sends_once():
    stub = _StubBot([None])
    ok = asyncio.run(bot._send_md(stub, 1, "*fine*"))
    assert ok is True
    assert len(stub.calls) == 1
    assert stub.calls[0]["parse_mode"] == "Markdown"


def test_markdown_failure_falls_back_to_plain_text():
    # the incident shape: Telegram rejects the entity-broken Markdown message
    stub = _StubBot([RuntimeError("Can't parse entities: unclosed asterisk"),
                     None])
    ok = asyncio.run(bot._send_md(stub, 1, "news with a stray * asterisk"))
    assert ok is True                          # user still got the alert
    assert len(stub.calls) == 2
    assert stub.calls[0]["parse_mode"] == "Markdown"
    assert stub.calls[1]["parse_mode"] is None  # plain-text retry


def test_double_failure_reports_undelivered():
    stub = _StubBot([RuntimeError("parse"), RuntimeError("network down")])
    ok = asyncio.run(bot._send_md(stub, 1, "msg"))
    assert ok is False                         # caller must not count/mark it
    assert len(stub.calls) == 2
