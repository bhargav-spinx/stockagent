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

    # Patch `requests.Session` — used by Angel SDK, ipify, NSE bhavcopy.
    if not getattr(requests.Session.__init__, "_ssl_dev_patched", False):
        _orig = requests.Session.__init__

        def _patched(self, *a, **kw):
            _orig(self, *a, **kw)
            self.verify = False

        _patched._ssl_dev_patched = True
        requests.Session.__init__ = _patched

    # Patch `curl_cffi.requests.Session` — used by yfinance (recent versions).
    try:
        import curl_cffi.requests as ccr

        if not getattr(ccr.Session.__init__, "_ssl_dev_patched", False):
            _orig_cc = ccr.Session.__init__

            def _patched_cc(self, *a, **kw):
                kw.setdefault("verify", False)
                _orig_cc(self, *a, **kw)

            _patched_cc._ssl_dev_patched = True
            ccr.Session.__init__ = _patched_cc
    except ImportError:
        pass  # curl_cffi not installed — fine

    return True
