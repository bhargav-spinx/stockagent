"""
Intraday scanner for STRATEGY.md setups.

Phase A (this file): polling-based via existing data_provider, Setup A only.
Phase B: WebSocket streaming + Setups B/C + bhavcopy delivery%.

CLI:
    python scanner.py RELIANCE
    python scanner.py RELIANCE TCS INFY
    python scanner.py --watchlist           # tier-1 from STRATEGY.md
    python scanner.py --watchlist --skip-time-filter   # off-hours testing

The scanner is import-safe — bot.py can call scan_one() / scan_many()
directly without the CLI.
"""
import argparse
import logging
import sys
import time
from typing import Iterable

from dotenv import load_dotenv

# Must load env + apply SSL workaround BEFORE importing data_provider,
# so Angel SDK sees ANGEL_API_KEY / DISABLE_SSL_VERIFY at construction time.
load_dotenv()
import ssl_dev  # noqa: E402
ssl_dev.install_if_enabled()

from data_provider import fetch_data  # noqa: E402
from scanner_filters import apply_universal_filters  # noqa: E402
from scanner_setups import ALL_DETECTORS, Signal  # noqa: E402
from universe import (  # noqa: E402
    INTRADAY_UNIVERSE,
    SWING_UNIVERSE,
    TIER1_WATCHLIST,
)

logger = logging.getLogger(__name__)


def _normalize(symbol: str) -> str:
    s = symbol.upper().strip()
    return s if "." in s else f"{s}.NS"


def scan_one(symbol: str, check_time: bool = True) -> dict:
    """
    Scan a single symbol. Returns one of:
        {"symbol", "status": "signal",   "signal": Signal}
        {"symbol", "status": "skip",     "reason": str}      # filter rejected
        {"symbol", "status": "no_setup"}                     # filters passed, no setup matched
        {"symbol", "status": "error",    "reason": str}
    """
    sym = _normalize(symbol)
    try:
        df = fetch_data(sym, period="5d", interval="5m")
    except Exception as e:
        return {"symbol": sym, "status": "error", "reason": str(e)}

    if len(df) < 30:
        return {"symbol": sym, "status": "skip",
                "reason": f"Insufficient data ({len(df)} candles)"}

    # First pass: check universal filters with neutral 'long' bias.
    # Direction-specific re-check happens after a setup matches.
    f = apply_universal_filters(df, direction="long", check_time=check_time)
    if not f.passed:
        return {"symbol": sym, "status": "skip", "reason": f.reason}

    for detector in ALL_DETECTORS:
        sig = detector(df, sym)
        if sig is None:
            continue
        # Re-check direction-specific filters (e.g. round-number filter
        # depends on whether we're going long or short).
        f2 = apply_universal_filters(df, direction=sig.direction, check_time=check_time)
        if not f2.passed:
            return {"symbol": sym, "status": "skip",
                    "reason": f"Setup {sig.setup} matched but {f2.reason}"}
        return {"symbol": sym, "status": "signal", "signal": sig}

    return {"symbol": sym, "status": "no_setup"}


# Angel historical-data rate limit ~3 req/sec. Pace at 0.5s between calls
# to leave headroom and avoid the cascade of 429 retries we hit at full speed.
SCAN_PACING_SEC = 0.5


def scan_many(symbols: Iterable[str], check_time: bool = True,
              pacing_sec: float = SCAN_PACING_SEC) -> list[dict]:
    """Scan symbols sequentially, sleeping briefly between calls to respect Angel rate limits."""
    results = []
    syms = list(symbols)
    for i, sym in enumerate(syms):
        results.append(scan_one(sym, check_time=check_time))
        if pacing_sec > 0 and i < len(syms) - 1:
            time.sleep(pacing_sec)
    return results


def format_signal(sig: Signal) -> str:
    """Plain-text format for terminal output."""
    risk = abs(sig.entry - sig.stop_loss)
    rr1 = abs(sig.target1 - sig.entry) / risk if risk > 0 else 0
    rr2 = abs(sig.target2 - sig.entry) / risk if risk > 0 else 0
    arrow = "🟢 LONG" if sig.direction == "long" else "🔴 SHORT"
    sl_pct = ((sig.stop_loss - sig.entry) / sig.entry) * 100
    return (
        f"\n📈 SETUP {sig.setup} — {sig.symbol}  {arrow}\n"
        f"   Entry: ₹{sig.entry:.2f}\n"
        f"   SL:    ₹{sig.stop_loss:.2f}  ({sl_pct:+.2f}%)\n"
        f"   T1:    ₹{sig.target1:.2f}  RR 1:{rr1:.2f}\n"
        f"   T2:    ₹{sig.target2:.2f}  RR 1:{rr2:.2f}\n"
        f"   Confluences:\n"
        + "\n".join(f"     • {c}" for c in sig.confluences)
        + f"\n   {sig.notes}\n"
    )


def format_signal_telegram(sig: Signal) -> str:
    """Markdown format for Telegram messages."""
    risk = abs(sig.entry - sig.stop_loss)
    rr1 = abs(sig.target1 - sig.entry) / risk if risk > 0 else 0
    rr2 = abs(sig.target2 - sig.entry) / risk if risk > 0 else 0
    arrow = "🟢 *LONG*" if sig.direction == "long" else "🔴 *SHORT*"
    sl_pct = ((sig.stop_loss - sig.entry) / sig.entry) * 100
    confluences = "\n".join(f"• {c}" for c in sig.confluences)
    return (
        f"📈 *Setup {sig.setup}* — `{sig.symbol}` {arrow}\n\n"
        f"🎯 Entry: ₹{sig.entry:.2f}\n"
        f"🛑 SL:    ₹{sig.stop_loss:.2f} ({sl_pct:+.2f}%)\n"
        f"🥇 T1:    ₹{sig.target1:.2f}  RR 1:{rr1:.2f}\n"
        f"🥈 T2:    ₹{sig.target2:.2f}  RR 1:{rr2:.2f}\n\n"
        f"*Confluences:*\n{confluences}\n\n"
        f"_{sig.notes}_"
    )


def _cli():
    # UTF-8 stdout for Windows so emojis don't crash the script
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Intraday scanner (STRATEGY.md)")
    parser.add_argument("symbols", nargs="*", help="Symbols to scan (e.g. RELIANCE TCS)")
    parser.add_argument("--watchlist", action="store_true",
                        help="Scan tier-1 watchlist from STRATEGY.md")
    parser.add_argument("--skip-time-filter", action="store_true",
                        help="Skip the time-of-day filter (for off-hours testing)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.watchlist:
        symbols = TIER1_WATCHLIST
    elif args.symbols:
        symbols = args.symbols
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Scanning {len(symbols)} symbol(s)...\n")
    signals = []
    for result in scan_many(symbols, check_time=not args.skip_time_filter):
        if result["status"] == "signal":
            print(format_signal(result["signal"]))
            signals.append(result["signal"])
        elif result["status"] == "skip":
            print(f"⏸  {result['symbol']}: {result['reason']}")
        elif result["status"] == "error":
            print(f"❌ {result['symbol']}: {result['reason']}")
        else:
            print(f"⚪ {result['symbol']}: no setup")

    print(f"\n{len(signals)} signal(s) fired.")


if __name__ == "__main__":
    _cli()
