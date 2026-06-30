"""
Regression guard for the CRITICAL finding: a Telethon USER-ACCOUNT session
(telethon_session.session) had been committed to the repo. This test fails the
build if ANY credential-class file is tracked by git, so it can't happen again.

CI-safe: git is available in GitHub Actions. Skips if not a git checkout.

    venv/Scripts/python.exe tests/test_no_secrets_tracked.py
"""
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=_REPO,
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None                      # git unavailable → skip
    if out.returncode != 0:
        return None
    return out.stdout.splitlines()


def test_no_credential_files_are_tracked():
    files = _tracked_files()
    if files is None:
        print("git unavailable — skipping")
        return
    banned = []
    for f in files:
        base = os.path.basename(f)
        if f.endswith((".session", ".session-journal")):
            banned.append(f)
        elif base == ".env" or (base.startswith(".env.") and base != "env.example.txt"):
            banned.append(f)
    assert not banned, f"credential/secret files are git-tracked: {banned}"


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
