"""
Backtest harness for STRATEGY.md setups.

Walks historical 5-min OHLCV forward candle-by-candle, runs the SAME
universal filters + setup detectors used in production, simulates trade
outcomes per §7 partial-exit rules, and reports honest stats.

CLI:
    python backtest.py RELIANCE --days 90
    python backtest.py --watchlist --days 90
    python backtest.py RELIANCE TCS INFY --days 60 --setup A

The backtest applies §7 partial-exit logic (50% at T1 → trail SL to entry → 50% at T2)
and the §8 cost model (~0.13% per round trip). It does NOT model live spread
widening or slippage on stops — assume the data shows actual fills at the
candle's exact OHLC range. Real-world results will be slightly worse.
"""
import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from typing import Iterable

import pandas as pd

from dotenv import load_dotenv
load_dotenv()
import ssl_dev  # noqa: E402
ssl_dev.install_if_enabled()

from data_provider import fetch_data  # noqa: E402
from scanner_filters import apply_universal_filters, is_intraday_entry_window  # noqa: E402
from scanner_setups import detect_setup_a, detect_setup_b, detect_setup_c, Signal  # noqa: E402
from universe import TIER1_WATCHLIST  # noqa: E402
import eod_report  # noqa: E402
from eod_report import COST_PER_TRADE_PCT  # noqa: E402
from constants import IST  # noqa: E402

logger = logging.getLogger(__name__)

# Min candles required before any setup can fire (indicator warm-up)
MIN_WARMUP_CANDLES = 30


# ---------- data structures ----------

@dataclass
class Trade:
    symbol: str
    setup: str
    direction: str
    entry_time: datetime
    entry: float
    stop_loss: float
    target1: float
    target2: float
    status: str = "open"
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl_gross_pct: float = 0.0


@dataclass
class BacktestStats:
    symbol: str
    period_start: datetime
    period_end: datetime
    trades: list[Trade] = field(default_factory=list)
    candles_in_window: int = 0
    filter_rejections: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_gross_pct > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl_gross_pct < 0)

    @property
    def break_even(self) -> int:
        return sum(1 for t in self.trades if t.pnl_gross_pct == 0)

    @property
    def hit_rate(self) -> float:
        return (self.wins / self.n * 100) if self.n else 0.0

    @property
    def gross_pnl_total(self) -> float:
        return sum(t.pnl_gross_pct for t in self.trades)

    @property
    def gross_pnl_avg(self) -> float:
        return (self.gross_pnl_total / self.n) if self.n else 0.0

    @property
    def net_pnl_avg(self) -> float:
        return self.gross_pnl_avg - COST_PER_TRADE_PCT

    @property
    def net_pnl_total(self) -> float:
        return self.net_pnl_avg * self.n

    @property
    def avg_win(self) -> float:
        ws = [t.pnl_gross_pct for t in self.trades if t.pnl_gross_pct > 0]
        return sum(ws) / len(ws) if ws else 0.0

    @property
    def avg_loss(self) -> float:
        ls = [t.pnl_gross_pct for t in self.trades if t.pnl_gross_pct < 0]
        return sum(ls) / len(ls) if ls else 0.0

    @property
    def expectancy(self) -> float:
        """Per-trade expectancy: avg net P&L."""
        return self.net_pnl_avg

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough drawdown of cumulative net equity (as %)."""
        if not self.trades:
            return 0.0
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in sorted(self.trades, key=lambda x: x.entry_time):
            equity += t.pnl_gross_pct - COST_PER_TRADE_PCT
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)
        return max_dd

    def status_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.trades:
            out[t.status] = out.get(t.status, 0) + 1
        return out


# ---------- core walk-forward ----------

def _filters_with_atr_override(df, direction: str, atr_lo: float, atr_hi: float):
    """Run universal filters but with a custom ATR band. Used by backtest tuning."""
    from scanner_filters import (
        FilterResult, atr_bounds_filter, trigger_volume_filter,
        vwap_slope_filter, round_number_filter,
    )
    for f in [
        atr_bounds_filter(df, lo=atr_lo, hi=atr_hi),
        trigger_volume_filter(df),
        vwap_slope_filter(df),
        round_number_filter(df, direction),
    ]:
        if not f.passed:
            return f
    return FilterResult(True)


def _normalize_ist(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        return df.tz_localize(IST)
    if str(df.index.tz) != str(IST):
        return df.tz_convert(IST)
    return df


def _resolve_outcome(sig: Signal, sig_time: datetime, df: pd.DataFrame) -> dict:
    """Use eod_report's resolver against a pre-fetched df."""
    alert = {
        "symbol": sig.symbol,
        "entry": sig.entry,
        "stop_loss": sig.stop_loss,
        "target1": sig.target1,
        "target2": sig.target2,
        "direction": sig.direction,
        "generated_at": sig_time,
    }
    return eod_report.resolve_intraday(alert, df=df)


def backtest_symbol(symbol: str, days: int = 90,
                   setup_filter: str | None = None,
                   atr_lo: float | None = None,
                   atr_hi: float | None = None) -> BacktestStats:
    """
    Run backtest on one symbol over the last `days` calendar days.

    setup_filter='A' to test only Setup A. None = all available setups.
    atr_lo / atr_hi to override the universal ATR-bounds filter (defaults
    are 0.004 / 0.015 from STRATEGY.md §5).
    """
    sym = symbol.upper().strip()
    if "." not in sym:
        sym = f"{sym}.NS"

    period_str = f"{days}d" if days <= 60 else f"{(days + 29) // 30}mo"
    df = fetch_data(sym, period=period_str, interval="5m")
    df = _normalize_ist(df)

    if len(df) < MIN_WARMUP_CANDLES + 10:
        logger.warning("%s: insufficient history (%d candles)", sym, len(df))
        return BacktestStats(sym, df.index[0] if len(df) else datetime.now(IST),
                            df.index[-1] if len(df) else datetime.now(IST))

    detectors = [detect_setup_a, detect_setup_b, detect_setup_c]
    if setup_filter:
        detectors = [d for d in detectors
                    if d.__name__ == f"detect_setup_{setup_filter.lower()}"]

    stats = BacktestStats(symbol=sym, period_start=df.index[0], period_end=df.index[-1])
    fired_today: set[tuple[str, str]] = set()  # (symbol, setup) — no re-entry same day
    last_session_date = None
    filter_rejections: dict[str, int] = {}
    candles_in_window = 0

    for i in range(MIN_WARMUP_CANDLES, len(df)):
        slice_df = df.iloc[: i + 1]
        ts = slice_df.index[-1].to_pydatetime()

        # Reset same-day re-entry tracking on new session
        cur_date = ts.date()
        if cur_date != last_session_date:
            fired_today.clear()
            last_session_date = cur_date

        if not is_intraday_entry_window(ts):
            continue
        candles_in_window += 1

        # Universal filters first (skip time-window check; we pre-filtered above).
        # When custom ATR bounds are passed, swap the default filter for one
        # using the override values.
        if atr_lo is not None or atr_hi is not None:
            f = _filters_with_atr_override(
                slice_df, direction="long",
                atr_lo=atr_lo if atr_lo is not None else 0.004,
                atr_hi=atr_hi if atr_hi is not None else 0.015,
            )
        else:
            f = apply_universal_filters(slice_df, direction="long", check_time=False)
        if not f.passed:
            # Coarse-bucket the rejection reason
            bucket = f.reason.split("(")[0].split("<")[0].split(">")[0].strip()[:40]
            filter_rejections[bucket] = filter_rejections.get(bucket, 0) + 1
            continue

        for detector in detectors:
            sig = detector(slice_df, sym)
            if sig is None:
                continue
            # Re-check filters with direction-specific gates
            if atr_lo is not None or atr_hi is not None:
                f2 = _filters_with_atr_override(
                    slice_df, direction=sig.direction,
                    atr_lo=atr_lo if atr_lo is not None else 0.004,
                    atr_hi=atr_hi if atr_hi is not None else 0.015,
                )
            else:
                f2 = apply_universal_filters(slice_df, direction=sig.direction, check_time=False)
            if not f2.passed:
                continue
            # Same-day re-entry guard
            key = (sig.symbol, sig.setup)
            if key in fired_today:
                continue
            fired_today.add(key)

            # Resolve outcome using forward bars (rest of df after ts)
            outcome = _resolve_outcome(sig, ts, df)
            if outcome.get("status") == "no_data":
                continue
            trade = Trade(
                symbol=sig.symbol,
                setup=sig.setup,
                direction=sig.direction,
                entry_time=ts,
                entry=sig.entry,
                stop_loss=sig.stop_loss,
                target1=sig.target1,
                target2=sig.target2,
                status=outcome.get("status", "open"),
                exit_time=outcome.get("exit_time"),
                exit_price=outcome.get("exit_price"),
                pnl_gross_pct=outcome.get("pnl_pct") or 0.0,
            )
            stats.trades.append(trade)
            break  # one trade per candle; move to next candle

    stats.candles_in_window = candles_in_window
    stats.filter_rejections = filter_rejections
    return stats


def backtest_symbol_score(symbol: str, days: int = 90,
                          min_score: int = 90) -> BacktestStats:
    """Walk-forward backtest of the intraday_score engine (#1).

    At each in-window candle, score the history-so-far; the first candle of a
    session whose score ≥ min_score with a direction opens a trade, resolved
    with the same §7 partial-exit model used in production. One trade per
    symbol per day (matches the live autoscan dedup). The market-index regime
    gate is NOT applied here — index history isn't reconstructed per-candle —
    so live results will be slightly stricter than this backtest."""
    import intraday_score

    sym = symbol.upper().strip()
    if "." not in sym:
        sym = f"{sym}.NS"

    period_str = f"{days}d" if days <= 60 else f"{(days + 29) // 30}mo"
    df = fetch_data(sym, period=period_str, interval="5m")
    df = _normalize_ist(df)

    stats = BacktestStats(
        symbol=sym,
        period_start=df.index[0] if len(df) else datetime.now(IST),
        period_end=df.index[-1] if len(df) else datetime.now(IST),
    )
    if len(df) < MIN_WARMUP_CANDLES + 10:
        logger.warning("%s: insufficient history (%d candles)", sym, len(df))
        return stats

    fired_days: set = set()      # session dates already traded
    candles_in_window = 0
    for i in range(MIN_WARMUP_CANDLES, len(df)):
        slice_df = df.iloc[: i + 1]
        ts = slice_df.index[-1].to_pydatetime()
        if not is_intraday_entry_window(ts):
            continue
        candles_in_window += 1
        if ts.date() in fired_days:
            continue
        try:
            # skip_external: no live NSE delivery/earnings calls — keep the
            # backtest hermetic (and those inputs are today's, not historical).
            card = intraday_score.score_stock(slice_df, sym, skip_external=True)
        except Exception:
            continue
        if card.direction == "none" or card.score < min_score:
            continue
        alert = {
            "symbol": sym, "entry": card.entry, "stop_loss": card.stop_loss,
            "target1": card.target1, "target2": card.target2,
            "direction": card.direction, "generated_at": ts,
        }
        outcome = eod_report.resolve_intraday(alert, df=df)
        if outcome.get("status") in (None, "open", "no_data"):
            continue  # last candle of data — no forward bars to resolve against
        fired_days.add(ts.date())
        stats.trades.append(Trade(
            symbol=sym, setup=f"score{card.score}", direction=card.direction,
            entry_time=ts, entry=card.entry, stop_loss=card.stop_loss,
            target1=card.target1, target2=card.target2,
            status=outcome.get("status", "open"),
            exit_time=outcome.get("exit_time"),
            exit_price=outcome.get("exit_price"),
            pnl_gross_pct=outcome.get("pnl_pct") or 0.0,
        ))
    stats.candles_in_window = candles_in_window
    return stats


def backtest_many(symbols: Iterable[str], days: int = 90,
                 setup_filter: str | None = None,
                 atr_lo: float | None = None,
                 atr_hi: float | None = None,
                 score_mode: bool = False,
                 min_score: int = 90) -> list[BacktestStats]:
    results = []
    for sym in symbols:
        try:
            if score_mode:
                results.append(backtest_symbol_score(sym, days, min_score=min_score))
            else:
                results.append(backtest_symbol(
                    sym, days, setup_filter, atr_lo=atr_lo, atr_hi=atr_hi
                ))
        except Exception as e:
            logger.error("backtest %s failed: %s", sym, e)
    return results


# ---------- reporting ----------

def format_stats(stats: BacktestStats) -> str:
    if stats.n == 0:
        lines = [
            f"\n=== BACKTEST: {stats.symbol} ===",
            f"Period: {stats.period_start.date()} → {stats.period_end.date()}",
            f"No trades — universal filters or setup detector never matched.",
        ]
        if stats.candles_in_window:
            lines.append(f"Candles in trading window: {stats.candles_in_window}")
            lines.append("Top filter rejections:")
            for reason, count in sorted(
                stats.filter_rejections.items(), key=lambda x: -x[1]
            )[:5]:
                pct = count / stats.candles_in_window * 100
                lines.append(f"  {reason:<40} {count:>6}  ({pct:5.1f}%)")
        return "\n".join(lines) + "\n"

    breakdown = stats.status_breakdown()
    lines = [
        f"\n=== BACKTEST: {stats.symbol} ===",
        f"Period: {stats.period_start.date()} → {stats.period_end.date()}",
        f"",
        f"Trades:           {stats.n}",
        f"Hit rate:         {stats.hit_rate:.1f}%  ({stats.wins} wins / {stats.losses} losses / {stats.break_even} BE)",
        f"Avg gross P&L:    {stats.gross_pnl_avg:+.3f}% per trade",
        f"Net (after {COST_PER_TRADE_PCT}% costs): {stats.net_pnl_avg:+.3f}% per trade",
        f"Total net P&L:    {stats.net_pnl_total:+.2f}% over {stats.n} trades",
        f"Avg win:          {stats.avg_win:+.2f}%",
        f"Avg loss:         {stats.avg_loss:+.2f}%",
        f"Expectancy:       {stats.expectancy:+.3f}% per trade",
        f"Max drawdown:     {stats.max_drawdown:.2f}%",
        f"",
        f"Outcome breakdown:",
    ]
    for status in ("t2_hit", "t1_then_squareoff", "t1_then_breakeven",
                  "sl_hit", "time_stop", "squareoff_no_t1"):
        n = breakdown.get(status, 0)
        pct = (n / stats.n * 100) if stats.n else 0
        lines.append(f"  {status:<22}  {n:>3}  ({pct:5.1f}%)")
    return "\n".join(lines) + "\n"


def format_summary(all_stats: list[BacktestStats]) -> str:
    """Aggregate summary across all backtested symbols."""
    total_trades = sum(s.n for s in all_stats)
    if total_trades == 0:
        return "\n=== AGGREGATE: no trades fired across all symbols ===\n"

    total_wins = sum(s.wins for s in all_stats)
    total_gross = sum(s.gross_pnl_total for s in all_stats)
    total_net = sum(s.net_pnl_total for s in all_stats)

    lines = [
        "\n" + "=" * 60,
        f"AGGREGATE SUMMARY  ({len(all_stats)} symbols)",
        "=" * 60,
        f"Total trades:     {total_trades}",
        f"Hit rate:         {(total_wins / total_trades * 100):.1f}%",
        f"Total gross P&L:  {total_gross:+.2f}%",
        f"Total net P&L:    {total_net:+.2f}%  (after {COST_PER_TRADE_PCT}%/trade costs)",
        f"Avg net per trade: {(total_net / total_trades):+.3f}%",
        "",
        "Per-symbol summary:",
        f"{'Symbol':<14} {'Trades':>7} {'HitRate':>9} {'Net P&L':>10} {'Expect':>9}",
    ]
    for s in sorted(all_stats, key=lambda x: x.net_pnl_total, reverse=True):
        if s.n == 0:
            continue
        lines.append(
            f"{s.symbol:<14} {s.n:>7} {s.hit_rate:>8.1f}% "
            f"{s.net_pnl_total:>+9.2f}% {s.expectancy:>+8.3f}%"
        )
    return "\n".join(lines) + "\n"


def _cli():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Backtest STRATEGY.md setups")
    parser.add_argument("symbols", nargs="*", help="Symbols (e.g. RELIANCE TCS)")
    parser.add_argument("--watchlist", action="store_true",
                        help="Backtest tier-1 watchlist")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of history (default: 90)")
    parser.add_argument("--setup", choices=["A", "B", "C"], default=None,
                        help="Restrict to one setup (default: all)")
    parser.add_argument("--score", action="store_true",
                        help="Backtest the intraday_score engine instead of setups A/B/C")
    parser.add_argument("--min-score", type=int, default=90,
                        help="Score gate for --score mode (default: 90)")
    parser.add_argument("--atr-lo", type=float, default=None,
                        help="Override ATR lower bound (default 0.004 = 0.4%%)")
    parser.add_argument("--atr-hi", type=float, default=None,
                        help="Override ATR upper bound (default 0.015 = 1.5%%)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.watchlist:
        symbols = TIER1_WATCHLIST
    elif args.symbols:
        symbols = args.symbols
    else:
        parser.print_help()
        sys.exit(1)

    atr_note = ""
    if args.atr_lo is not None or args.atr_hi is not None:
        atr_note = (f"  ATR override: lo={args.atr_lo or 0.004:.4f} "
                    f"hi={args.atr_hi or 0.015:.4f}")
    mode_note = f"score≥{args.min_score}" if args.score else f"setup={args.setup or 'all'}"
    print(f"Backtesting {len(symbols)} symbol(s) over {args.days} days "
          f"({mode_note}){atr_note}\n")

    all_stats = backtest_many(
        symbols, days=args.days, setup_filter=args.setup,
        atr_lo=args.atr_lo, atr_hi=args.atr_hi,
        score_mode=args.score, min_score=args.min_score,
    )
    for s in all_stats:
        print(format_stats(s))
    if len(all_stats) > 1:
        print(format_summary(all_stats))


if __name__ == "__main__":
    _cli()
