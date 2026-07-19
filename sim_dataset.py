"""
Simulated-trade training dataset (Phase 4.1).

Replays archived 5-min candles through the SAME machinery the live bot uses —
score engine, entry window, canonical levels, outcome resolver, feature
builders — to produce labeled (features @ decision time, outcome) rows at
scale. This is what the probability model TRAINS on; live alerts (a few per
week) are reserved for the calibration sanity check, and the gap between the
two IS the live-vs-sim optimism estimate.

HONESTY CONTRACT — every consumer of this table must know:
- Rows are SIMULATION-DERIVED. Fills are optimistic (exact-touch, no
  gap-through, no spread, no partial fills) — same assumptions as the
  resolver/backtest, inherited deliberately so sim and live labels share one
  definition of "outcome".
- External point-in-time context CANNOT be reconstructed: delivery %,
  earnings proximity, live index regime and news are absent (score_stock runs
  skip_external=True; regime/idx features are omitted rather than faked).
  Every row carries features["sim"]=1 so the trainer can see the gap.
- Sampling frame: the FIRST candle per (symbol, session) whose score clears
  --min-score (default 60 = the "Watchlist" floor) inside the live entry
  window, mirroring the live fired_signals one-per-day dedup. Sub-60 setups
  are deliberately absent — the model ranks candidates the platform would
  consider, not the whole universe; that truncation is part of the model card.
- No lookahead: the daily frame passed for 52-week context is sliced STRICTLY
  before the trade date; the index frame for relative strength is sliced to
  candles at/before the decision timestamp; the score sees only candles up to
  the decision candle. The resolver may read the full session AFTER the
  decision — that is the outcome, not a feature.

CLI:
    python sim_dataset.py generate --universe tier1 --min-score 60
    python sim_dataset.py generate --symbols RELIANCE TCS --days 120
    python sim_dataset.py stats
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from dotenv import load_dotenv
load_dotenv()
import ssl_dev  # noqa: E402
ssl_dev.install_if_enabled()

import data_archive  # noqa: E402
import eod_report  # noqa: E402
import intraday_score  # noqa: E402
from backtest import MIN_WARMUP_CANDLES  # noqa: E402  (single source)
from features import FEATURE_SCHEMA_VERSION, features_from_scorecard, phase3_features  # noqa: E402
from scanner_filters import is_intraday_entry_window  # noqa: E402

logger = logging.getLogger(__name__)

# Rows live next to the candles they were derived from (market_data.db is
# already gitignored — research data never enters the repo).
DATASET_VERSION = 1

_PASS = eod_report.PASS_STATUSES
_FAIL = eod_report.FAIL_STATUSES


def _init_table() -> None:
    data_archive.init_archive()
    with data_archive._conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sim_trades (
                symbol        TEXT NOT NULL,
                ts            TEXT NOT NULL,      -- decision candle, ISO IST
                trade_date    TEXT NOT NULL,
                direction     TEXT NOT NULL,
                score         INTEGER,
                entry         REAL, stop_loss REAL, target1 REAL, target2 REAL,
                status        TEXT,
                pnl_pct       REAL,
                mfe_pct       REAL, mae_pct REAL, duration_min REAL,
                r_multiple    REAL,
                label_win     INTEGER,            -- T1 before SL (PASS statuses)
                label_decisive INTEGER,           -- PASS or FAIL (excl. timeouts)
                features      TEXT,               -- decision-time JSON snapshot
                dataset_v     INTEGER,
                run_id        TEXT,
                PRIMARY KEY (symbol, ts, direction)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_sim_date ON sim_trades(trade_date)")


def _r_multiple(entry: float, stop: float, pnl_pct: float | None) -> float | None:
    if pnl_pct is None or not entry:
        return None
    risk_pct = abs(entry - stop) / entry * 100
    return round(pnl_pct / risk_pct, 4) if risk_pct > 0 else None


def _daily_before(daily: pd.DataFrame, d) -> pd.DataFrame | None:
    """Daily context sliced STRICTLY before the trade date — a same-day daily
    bar would leak the session's own high/low into the 52-week note."""
    if daily is None or len(daily) == 0:
        return None
    sliced = daily[daily.index.date < d]
    return sliced if len(sliced) else None


def generate_symbol(symbol: str, *, min_score: int = 60,
                    days: int | None = None,
                    index_df: pd.DataFrame | None = None,
                    run_id: str = "manual") -> int:
    """Replay one symbol's archived 5-min history; insert labeled rows.
    Returns rows written. Idempotent: (symbol, ts, direction) upserts."""
    _init_table()
    df = data_archive.load(symbol, "5m")
    if len(df) < MIN_WARMUP_CANDLES + 10:
        logger.info("%s: archive too thin (%d candles) — skipped", symbol, len(df))
        return 0
    daily = data_archive.load(symbol, "1d")

    sessions = sorted(set(df.index.date))
    if days:
        sessions = sessions[-days:]
    allowed = set(sessions)

    rows: list[tuple] = []
    fired_days: set = set()
    for i in range(MIN_WARMUP_CANDLES, len(df)):
        ts = df.index[i].to_pydatetime()
        d = ts.date()
        if d not in allowed or d in fired_days:
            continue
        if not is_intraday_entry_window(ts):
            continue
        slice_df = df.iloc[: i + 1]
        try:
            card = intraday_score.score_stock(
                slice_df, symbol, skip_external=True,
                daily_df=_daily_before(daily, d))
        except Exception as e:
            logger.debug("%s @ %s: score failed: %s", symbol, ts, e)
            continue
        if card.direction == "none" or card.score < min_score:
            continue

        fired_days.add(d)
        alert = {"symbol": symbol, "entry": card.entry,
                 "stop_loss": card.stop_loss, "target1": card.target1,
                 "target2": card.target2, "direction": card.direction,
                 "generated_at": ts}
        outcome = eod_report.resolve_intraday(alert, df=df)
        status = (outcome or {}).get("status")
        if status in (None, "open", "no_data"):
            continue

        feats = features_from_scorecard(card, now=ts)
        idx_slice = (index_df[index_df.index <= df.index[i]]
                     if index_df is not None and len(index_df) else None)
        feats.update(phase3_features(slice_df, index_df=idx_slice))
        feats["sim"] = 1

        pnl = outcome.get("pnl_pct")
        rows.append((
            symbol, ts.isoformat(), d.isoformat(), card.direction,
            int(card.score), card.entry, card.stop_loss, card.target1,
            card.target2, status, pnl,
            outcome.get("mfe_pct"), outcome.get("mae_pct"),
            outcome.get("duration_min"),
            _r_multiple(card.entry, card.stop_loss, pnl),
            int(status in _PASS), int(status in _PASS or status in _FAIL),
            json.dumps(feats, sort_keys=True, default=str),
            DATASET_VERSION, run_id,
        ))

    if rows:
        with data_archive._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO sim_trades VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def generate(symbols: Iterable[str], *, min_score: int = 60,
             days: int | None = None, run_id: str | None = None) -> dict:
    """Generate over many symbols. The NIFTY frame (if archived) feeds the
    relative-strength features, sliced point-in-time per decision candle."""
    _init_table()
    run_id = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    index_df = data_archive.load("NIFTY", "5m")
    if len(index_df) == 0:
        index_df = None
        logger.info("no archived NIFTY 5m — relative-strength features omitted")
    total, per_symbol = 0, {}
    for sym in symbols:
        n = generate_symbol(sym, min_score=min_score, days=days,
                            index_df=index_df, run_id=run_id)
        per_symbol[sym] = n
        total += n
    return {"run_id": run_id, "rows": total, "symbols": per_symbol}


def load_rows(min_score: int | None = None,
              decisive_only: bool = False) -> list[dict[str, Any]]:
    """Training rows, oldest-first (time order matters for purged CV).
    features come back parsed."""
    _init_table()
    sql = "SELECT * FROM sim_trades"
    clauses, params = [], []
    if min_score is not None:
        clauses.append("score >= ?")
        params.append(min_score)
    if decisive_only:
        clauses.append("label_decisive = 1")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY ts"
    with data_archive._conn() as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute(sql, params).fetchall()]
    for r in rows:
        try:
            r["features"] = json.loads(r["features"]) if r["features"] else {}
        except (TypeError, ValueError):
            r["features"] = {}
    return rows


def dataset_stats() -> dict:
    _init_table()
    with data_archive._conn() as c:
        n, wins, decisive = c.execute(
            "SELECT count(*), coalesce(sum(label_win),0), "
            "coalesce(sum(label_decisive),0) FROM sim_trades").fetchone()
        span = c.execute(
            "SELECT min(trade_date), max(trade_date) FROM sim_trades").fetchone()
        by_bucket = c.execute(
            "SELECT CASE WHEN score>=90 THEN '90+' WHEN score>=80 THEN '80-89' "
            "WHEN score>=70 THEN '70-79' ELSE '60-69' END b, count(*), "
            "coalesce(sum(label_win),0) FROM sim_trades GROUP BY b ORDER BY b"
        ).fetchall()
    return {"rows": n, "wins": wins, "decisive": decisive,
            "date_range": span, "by_score_bucket": by_bucket}


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Simulated-trade training dataset (replay archive through "
                    "the live engines)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--symbols", nargs="*", default=None)
    gen.add_argument("--universe", choices=["tier1", "intraday"], default=None)
    gen.add_argument("--min-score", type=int, default=60,
                     help="Sampling floor (default 60 = Watchlist)")
    gen.add_argument("--days", type=int, default=None,
                     help="Only replay the last N archived sessions")
    sub.add_parser("stats")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "stats":
        s = dataset_stats()
        print(f"sim_trades: {s['rows']} rows "
              f"({s['decisive']} decisive, {s['wins']} wins), "
              f"dates {s['date_range'][0]} → {s['date_range'][1]}")
        for b, n, w in s["by_score_bucket"]:
            wr = (w / n * 100) if n else 0.0
            print(f"  score {b:>6}: n={n:<6} win {wr:.0f}% (sim-optimistic)")
        return

    from analyzer import normalize_symbol
    if not args.symbols and not args.universe:
        parser.error("generate needs --symbols or --universe")
    if args.symbols:
        symbols = [normalize_symbol(s) for s in args.symbols]
    else:
        import universe as universe_mod
        symbols = (universe_mod.TIER1_WATCHLIST if args.universe == "tier1"
                   else universe_mod.INTRADAY_UNIVERSE)
        symbols = [normalize_symbol(s) for s in symbols]
    out = generate(symbols, min_score=args.min_score, days=args.days)
    thin = sum(1 for n in out["symbols"].values() if n == 0)
    print(f"{out['run_id']}: {out['rows']} rows from "
          f"{len(out['symbols'])} symbol(s) ({thin} skipped/empty). "
          f"Run `python sim_dataset.py stats` for the breakdown.")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    _cli()
