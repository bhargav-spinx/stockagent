"""
Local-dev SSL workaround for corp TLS inspection.

When DISABLE_SSL_VERIFY=true, monkey-patches requests.Session so the
Angel One SDK and any other `requests`-based callers skip cert
verification. Imported (with side effect) by entry points that need it:
bot.py, scanner.py, and any standalone scripts.

Do NOT enable in production.
"""
import os


def install_if_enabled() -> bool:
    """Apply the patch if DISABLE_SSL_VERIFY=true. Returns True if applied."""
    if os.getenv("DISABLE_SSL_VERIFY", "").lower() not in ("1", "true", "yes"):
        return False

    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if getattr(requests.Session.__init__, "_ssl_dev_patched", False):
        return True  # already patched, idempotent

    _orig = requests.Session.__init__

    def _patched(self, *a, **kw):
        _orig(self, *a, **kw)
        self.verify = False

    _patched._ssl_dev_patched = True
    requests.Session.__init__ = _patched
    return True
