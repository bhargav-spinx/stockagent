"""
Regression tests for the Telethon channel listener startup contract.

Would have caught:
 - H1: a relative session path (CWD-dependent → phantom unauthorized session).
 - a NewMessage handler not actually being registered for a configured channel.

The live pieces (entity resolution, loop running, authorization) need a real
session + network and are covered by the structured startup logging instead.

Importing telethon_listener needs python-telegram-bot (telegram.ext), which CI's
lean install omits — so skip there; runs locally.

    venv/Scripts/python.exe tests/test_telethon_listener.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
    pytest.importorskip("telegram")
except ImportError:        # plain-script run without pytest
    pass

import telethon_listener as tl  # noqa: E402


def test_session_path_is_absolute_and_pinned_to_module_dir():
    # H1 guard: a relative session name would resolve against the process CWD
    # and silently create an unauthorized session on restart.
    assert os.path.isabs(tl.SESSION_NAME), tl.SESSION_NAME
    assert os.path.basename(tl.SESSION_NAME) == "telethon_session"
    expected_dir = os.path.dirname(os.path.abspath(tl.__file__))
    assert os.path.dirname(tl.SESSION_NAME) == expected_dir


def test_register_channel_registers_a_newmessage_handler():
    listener = tl.TelethonListener(bot_app=object())

    class _FakeClient:
        def __init__(self):
            self.registered = []

        def on(self, event):
            def deco(fn):
                self.registered.append((event, fn))
                return fn
            return deco

    class _FakeNewMessage:
        def __init__(self, chats=None):
            self.chats = chats

    class _FakeEvents:
        NewMessage = _FakeNewMessage

    listener.client = _FakeClient()
    listener._register_channel("STOCKGAINERSS", _FakeEvents)

    assert len(listener.client.registered) == 1, "handler was not registered"
    event, fn = listener.client.registered[0]
    assert event.chats == "STOCKGAINERSS"
    assert callable(fn)


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
