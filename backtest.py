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
import riskmetrics  # noqa: E402
from scanner_filters import apply_universal_filters, is_intraday_entry_window  # noqa: E402
from scanner_setups import detect_setup_a, detect_setup_b, detect_setup_c, Signal  # noqa: E402
import universe  # noqa: E402
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
    def net_returns(self) -> list[float]:
        """Per-trade NET % returns (gross minus round-trip costs) — the series
        risk metrics operate on."""
        return [t.pnl_gross_pct - COST_PER_TRADE_PCT for t in self.trades]

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


def _period_str(days: int) -> str:
    return f"{days}d" if days <= 60 else f"{(days + 29) // 30}mo"


def _fetch_5m(symbol: str, days: int) -> tuple[str, pd.DataFrame]:
    """Normalize symbol, fetch 5m history, IST-localize. Returns (sym, df)."""
    sym = symbol.upper().strip()
    if "." not in sym:
        sym = f"{sym}.NS"
    df = fetch_data(sym, period=_period_str(days), interval="5m")
    df = _normalize_ist(df)
    # MEDIUM-7: requested vs actual coverage. Intraday history is provider-
    # capped (yfinance ~60d of 5m), so "--days 90" can silently cover far less.
    if len(df):
        span = (df.index[-1].date() - df.index[0].date()).days + 1
        if span < days * 0.6:
            logger.warning(
                "%s: requested %dd of 5m history but only ~%dd returned "
                "(provider cap — your backtest window is shorter than asked).",
                sym, days, span)
    return sym, df


def _simulate_setups(df: pd.DataFrame, sym: str,
                     setup_filter: str | None = None,
                     atr_lo: float | None = None,
                     atr_hi: float | None = None,
                     entry_dates: set | None = None,
                     ) -> tuple[list[Trade], int, dict[str, int]]:
    """Pure in-memory candle walk for setups A/B/C over a pre-fetched df.

    `entry_dates` (a set of dates) restricts which sessions may OPEN trades —
    indicator context and outcome resolution still use the full df, so a fold's
    test window gets historical context without being able to peek forward.
    Returns (trades, candles_in_window, filter_rejections)."""
    detectors = [detect_setup_a, detect_setup_b, detect_setup_c]
    if setup_filter:
        detectors = [d for d in detectors
                     if d.__name__ == f"detect_setup_{setup_filter.lower()}"]

    trades: list[Trade] = []
    fired_today: set[tuple[str, str]] = set()
    last_session_date = None
    filter_rejections: dict[str, int] = {}
    candles_in_window = 0

    for i in range(MIN_WARMUP_CANDLES, len(df)):
        slice_df = df.iloc[: i + 1]
        ts = slice_df.index[-1].to_pydatetime()

        cur_date = ts.date()
        if cur_date != last_session_date:
            fired_today.clear()
            last_session_date = cur_date

        if not is_intraday_entry_window(ts):
            continue
        if entry_dates is not None and cur_date not in entry_dates:
            continue
        candles_in_window += 1

        if atr_lo is not None or atr_hi is not None:
            f = _filters_with_atr_override(
                slice_df, direction="long",
                atr_lo=atr_lo if atr_lo is not None else 0.004,
                atr_hi=atr_hi if atr_hi is not None else 0.015,
            )
        else:
            f = apply_universal_filters(slice_df, direction="long", check_time=False)
        if not f.passed:
            bucket = f.reason.split("(")[0].split("<")[0].split(">")[0].strip()[:40]
            filter_rejections[bucket] = filter_rejections.get(bucket, 0) + 1
            continue

        for detector in detectors:
            sig = detector(slice_df, sym)
            if sig is None:
                continue
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
            key = (sig.symbol, sig.setup)
            if key in fired_today:
                continue
            fired_today.add(key)

            outcome = _resolve_outcome(sig, ts, df)
            if outcome.get("status") == "no_data":
                continue
            trades.append(Trade(
                symbol=sig.symbol, setup=sig.setup, direction=sig.direction,
                entry_time=ts, entry=sig.entry, stop_loss=sig.stop_loss,
                target1=sig.target1, target2=sig.target2,
                status=outcome.get("status", "open"),
                exit_time=outcome.get("exit_time"),
                exit_price=outcome.get("exit_price"),
                pnl_gross_pct=outcome.get("pnl_pct") or 0.0,
            ))
            break  # one trade per candle

    return trades, candles_in_window, filter_rejections


def _simulate_score(df: pd.DataFrame, sym: str, min_score: int = 90,
                    entry_dates: set | None = None) -> tuple[list[Trade], int]:
    """Pure in-memory candle walk for the intraday_score engine over a
    pre-fetched df. `entry_dates` restricts which sessions may open trades."""
    import intraday_score

    trades: list[Trade] = []
    fired_days: set = set()
    candles_in_window = 0
    for i in range(MIN_WARMUP_CANDLES, len(df)):
        slice_df = df.iloc[: i + 1]
        ts = slice_df.index[-1].to_pydatetime()
        if not is_intraday_entry_window(ts):
            continue
        d = ts.date()
        if entry_dates is not None and d not in entry_dates:
            continue
        candles_in_window += 1
        if d in fired_days:
            continue
        try:
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
            continue
        fired_days.add(d)
        trades.append(Trade(
            symbol=sym, setup=f"score{card.score}", direction=card.direction,
            entry_time=ts, entry=card.entry, stop_loss=card.stop_loss,
            target1=card.target1, target2=card.target2,
            status=outcome.get("status", "open"),
            exit_time=outcome.get("exit_time"),
            exit_price=outcome.get("exit_price"),
            pnl_gross_pct=outcome.get("pnl_pct") or 0.0,
        ))
    return trades, candles_in_window


def backtest_symbol(symbol: str, days: int = 90,
                   setup_filter: str | None = None,
                   atr_lo: float | None = None,
                   atr_hi: float | None = None) -> BacktestStats:
    """In-sample backtest on one symbol over the last `days` calendar days.

    NOTE: this fits and reports on the SAME window — use it for exploration, not
    as evidence of edge. For an honest out-of-sample read use walk_forward().
    setup_filter='A' restricts to Setup A; atr_lo/atr_hi override the §5 ATR band.
    """
    sym, df = _fetch_5m(symbol, days)
    if len(df) < MIN_WARMUP_CANDLES + 10:
        logger.warning("%s: insufficient history (%d candles)", sym, len(df))
        return BacktestStats(sym, df.index[0] if len(df) else datetime.now(IST),
                            df.index[-1] if len(df) else datetime.now(IST))

    trades, ciw, rej = _simulate_setups(df, sym, setup_filter, atr_lo, atr_hi)
    stats = BacktestStats(symbol=sym, period_start=df.index[0], period_end=df.index[-1])
    stats.trades = trades
    stats.candles_in_window = ciw
    stats.filter_rejections = rej
    return stats


def backtest_symbol_score(symbol: str, days: int = 90,
                          min_score: int = 90) -> BacktestStats:
    """In-sample backtest of the intraday_score engine (#1) on one symbol.

    One trade per symbol per day (matches the live autoscan dedup). The
    market-index regime gate is NOT applied here, so live results will be
    slightly stricter. In-sample — see walk_forward() for an OOS read."""
    sym, df = _fetch_5m(symbol, days)
    stats = BacktestStats(
        symbol=sym,
        period_start=df.index[0] if len(df) else datetime.now(IST),
        period_end=df.index[-1] if len(df) else datetime.now(IST),
    )
    if len(df) < MIN_WARMUP_CANDLES + 10:
        logger.warning("%s: insufficient history (%d candles)", sym, len(df))
        return stats
    trades, ciw = _simulate_score(df, sym, min_score=min_score)
    stats.trades = trades
    stats.candles_in_window = ciw
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


# ---------- walk-forward (out-of-sample) ----------

def _chunks(seq: list, n: int) -> list[list]:
    """Split a list into n contiguous near-equal chunks."""
    n = max(1, min(n, len(seq)))
    size = len(seq) / n
    return [seq[round(i * size):round((i + 1) * size)] for i in range(n)]


def _slice_with_context(df: pd.DataFrame, dates: set, context_days: int) -> pd.DataFrame:
    """All candles from (min(dates) − context_days) through max(dates).
    Gives the entry window indicator/prior-session context WITHOUT any forward
    leakage (nothing after max(dates) is included)."""
    lo, hi = min(dates), max(dates)
    start = lo - timedelta(days=context_days)
    d = df.index.date
    mask = (d >= start) & (d <= hi)
    return df[mask]


def _sim_param(df: pd.DataFrame, sym: str, param, score_mode: bool,
               entry_dates: set) -> list[Trade]:
    if score_mode:
        trades, _ = _simulate_score(df, sym, min_score=param, entry_dates=entry_dates)
    else:
        lo, hi = param
        trades, _, _ = _simulate_setups(df, sym, atr_lo=lo, atr_hi=hi,
                                        entry_dates=entry_dates)
    return trades


def _mean_net(trades: list[Trade]) -> float:
    if not trades:
        return float("-inf")          # a param that never trades can't be "best"
    return sum(t.pnl_gross_pct - COST_PER_TRADE_PCT for t in trades) / len(trades)


# Default parameter grids searched per training window.
_SCORE_GRID = [80, 85, 90, 95]
_ATR_GRID = [(0.004, 0.015), (0.003, 0.020), (0.005, 0.012), (0.004, 0.025)]


def walk_forward(symbols: Iterable[str], days: int = 120,
                 score_mode: bool = False, n_folds: int = 4,
                 train_folds: int = 2, context_days: int = 10,
                 grid: list | None = None) -> dict | None:
    """Anchored walk-forward optimization — the only mode here that yields an
    OUT-OF-SAMPLE read.

    Splits the calendar into `n_folds` contiguous folds. For each test fold, the
    parameter (score gate, or ATR band) is optimized on the preceding
    `train_folds` folds, then applied UNSEEN to the test fold. Only test-fold
    trades are reported. Entries fire only on in-window sessions; indicator
    context comes from `context_days` of prior history with no forward peeking.

    Returns a dict with oos_trades, per-fold log, and the number of parameter
    evaluations (for multiple-testing context), or None if data is insufficient.
    """
    grid = grid or (_SCORE_GRID if score_mode else _ATR_GRID)

    # Fetch each symbol once; walk-forward is then fully in-memory.
    dfs: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            sym, df = _fetch_5m(s, days)
            if len(df) >= MIN_WARMUP_CANDLES + 10:
                dfs[sym] = df
        except Exception as e:
            logger.error("walk_forward fetch %s failed: %s", s, e)
    if not dfs:
        return None

    all_dates = sorted({d for df in dfs.values() for d in set(df.index.date)})
    folds = _chunks(all_dates, n_folds)
    if len(folds) <= train_folds:
        logger.warning("walk_forward: not enough sessions (%d) for %d folds",
                       len(all_dates), n_folds)
        return None

    oos: list[Trade] = []
    optimizations = 0
    fold_log: list[dict] = []

    for f in range(train_folds, len(folds)):
        train_dates = {d for c in folds[f - train_folds:f] for d in c}
        test_dates = set(folds[f])
        if not train_dates or not test_dates:
            continue

        best_param, best_metric = None, float("-inf")
        for param in grid:
            optimizations += 1
            tr: list[Trade] = []
            for sym, df in dfs.items():
                sub = _slice_with_context(df, train_dates, context_days)
                tr += _sim_param(sub, sym, param, score_mode, train_dates)
            m = _mean_net(tr)
            if m > best_metric:
                best_metric, best_param = m, param

        if best_param is None:
            fold_log.append({"fold": f, "param": None, "train_metric": None,
                             "test_trades": 0, "test_net": 0.0})
            continue

        test_tr: list[Trade] = []
        for sym, df in dfs.items():
            sub = _slice_with_context(df, test_dates, context_days)
            test_tr += _sim_param(sub, sym, best_param, score_mode, test_dates)
        oos += test_tr
        fold_log.append({
            "fold": f, "param": best_param,
            "train_metric": best_metric,
            "test_trades": len(test_tr),
            "test_net": sum(t.pnl_gross_pct - COST_PER_TRADE_PCT for t in test_tr),
        })

    return {
        "oos_trades": oos, "folds": fold_log, "optimizations": optimizations,
        "n_symbols": len(dfs), "grid_size": len(grid),
        "period_start": all_dates[0], "period_end": all_dates[-1],
    }


def format_walkforward(res: dict | None, score_mode: bool) -> str:
    if not res or not res["folds"]:
        return ("\n=== WALK-FORWARD: insufficient data for an out-of-sample "
                "run (need several folds of intraday history) ===\n")

    oos = res["oos_trades"]
    returns = [t.pnl_gross_pct - COST_PER_TRADE_PCT for t in oos]
    n = len(oos)
    wins = sum(1 for t in oos if t.pnl_gross_pct > 0)
    net = sum(returns)
    param_label = "min_score" if score_mode else "ATR band"

    lines = [
        "\n" + "=" * 64,
        f"WALK-FORWARD (OUT-OF-SAMPLE)  ·  {res['n_symbols']} symbols  ·  "
        f"{res['period_start']} → {res['period_end']}",
        "=" * 64,
        f"Optimised {param_label} on each training window, then applied UNSEEN "
        f"to the next fold.",
        f"Parameter evaluations: {res['optimizations']} "
        f"(grid {res['grid_size']} × {len(res['folds'])} folds) — treat any "
        f"single 'good' result with multiple-testing skepticism.",
        "",
        "Per test fold:",
        f"{'Fold':<5} {'Param':<14} {'TestTrades':>10} {'TestNet%':>10}",
    ]
    for fl in res["folds"]:
        p = fl["param"]
        p_str = (str(p) if p is not None else "—")[:14]
        lines.append(f"{fl['fold']:<5} {p_str:<14} {fl['test_trades']:>10} "
                     f"{fl['test_net']:>+10.2f}")

    lines += [
        "",
        f"OOS trades:       {n}",
        f"OOS hit rate:     {(wins / n * 100):.1f}%" if n else "OOS hit rate:     n/a",
        f"OOS total net:    {net:+.2f}%",
        f"OOS avg/trade:    {(net / n):+.3f}%" if n else "OOS avg/trade:    n/a",
        riskmetrics.format_line(returns, "OOS risk-adj"),
        "",
        "_This is the honest number. If the OOS 95% CI straddles 0, the "
        "strategy has no demonstrable edge on unseen data — regardless of how "
        "good the in-sample backtest looked._",
    ]
    return "\n".join(lines) + "\n"


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
        f"{riskmetrics.format_line(stats.net_returns, 'Risk-adj')}",
        f"",
        f"Outcome breakdown:",
    ]
    for status in ("t2_hit", "t1_then_squareoff", "t1_then_breakeven",
                  "sl_hit", "time_stop", "squareoff_no_t1"):
        n = breakdown.get(status, 0)
        pct = (n / stats.n * 100) if stats.n else 0
        lines.append(f"  {status:<22}  {n:>3}  ({pct:5.1f}%)")
    return "\n".join(lines) + "\n"


def benchmark_return(period_start: datetime, period_end: datetime,
                     alias: str = "NIFTY") -> float | None:
    """Buy-and-hold % return of the index over [period_start, period_end].
    Best-effort: returns None if data can't be fetched. Used to answer the
    only question that matters — did the strategy beat simply holding the index
    over the same window?"""
    try:
        df = fetch_data(alias, period="1y", interval="1d")
        df = _normalize_ist(df)
        window = df[(df.index >= period_start) & (df.index <= period_end)]
        if len(window) < 2:
            return None
        first = float(window["Close"].iloc[0])
        last = float(window["Close"].iloc[-1])
        if first == 0:
            return None
        return (last - first) / first * 100
    except Exception as e:
        logger.warning("benchmark fetch failed: %s", e)
        return None


def format_summary(all_stats: list[BacktestStats]) -> str:
    """Aggregate summary across all backtested symbols."""
    total_trades = sum(s.n for s in all_stats)
    if total_trades == 0:
        return "\n=== AGGREGATE: no trades fired across all symbols ===\n"

    total_wins = sum(s.wins for s in all_stats)
    total_gross = sum(s.gross_pnl_total for s in all_stats)
    total_net = sum(s.net_pnl_total for s in all_stats)
    all_returns = [r for s in all_stats for r in s.net_returns]

    # Benchmark: NIFTY buy-and-hold over the full backtest window.
    starts = [s.period_start for s in all_stats if s.n]
    ends = [s.period_end for s in all_stats if s.n]
    bench = benchmark_return(min(starts), max(ends)) if starts else None
    bench_line = (f"NIFTY buy&hold:   {bench:+.2f}% over same window "
                  f"(strategy total net {total_net:+.2f}%)"
                  if bench is not None else
                  "NIFTY buy&hold:   n/a (benchmark fetch failed)")

    lines = [
        "\n" + "=" * 60,
        f"AGGREGATE SUMMARY  ({len(all_stats)} symbols)",
        "=" * 60,
        f"Total trades:     {total_trades}",
        f"Hit rate:         {(total_wins / total_trades * 100):.1f}%",
        f"Total gross P&L:  {total_gross:+.2f}%",
        f"Total net P&L:    {total_net:+.2f}%  (after {COST_PER_TRADE_PCT}%/trade costs)",
        f"Avg net per trade: {(total_net / total_trades):+.3f}%",
        riskmetrics.format_line(all_returns, "Risk-adj"),
        bench_line,
        "_Benchmark caveat: this is an in-sample, survivorship-biased universe "
        "(see universe.py / backtest --walkforward). Beating it here is NOT "
        "evidence of a live edge._",
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
    parser.add_argument("--walkforward", action="store_true",
                        help="Out-of-sample walk-forward optimisation — the only "
                             "honest evaluation. Optimises params per training "
                             "window, reports unseen test folds only.")
    parser.add_argument("--folds", type=int, default=4,
                        help="Walk-forward folds (default: 4)")
    parser.add_argument("--train-folds", type=int, default=2,
                        help="Training folds preceding each test fold (default: 2)")
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

    if not universe.has_point_in_time_data():
        print(universe.SURVIVORSHIP_NOTE + "\n")

    if args.walkforward:
        print(f"Walk-forward (OOS) on {len(symbols)} symbol(s) over {args.days} "
              f"days, {args.folds} folds ({'score' if args.score else 'setups'})\n")
        res = walk_forward(symbols, days=args.days, score_mode=args.score,
                           n_folds=args.folds, train_folds=args.train_folds)
        print(format_walkforward(res, args.score))
        return

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
