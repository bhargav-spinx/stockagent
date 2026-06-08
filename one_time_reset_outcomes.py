"""
One-time maintenance: quarantine pre-fix trade outcomes.

WHY
    Before the BLOCKER-1 fix, winning trades stored a hardcoded P&L (+1.5% /
    +0.5%) that had nothing to do with the actual targets. Those rows are still
    in `alert_outcomes`. `resolve_pending` only ever resolves OPEN alerts, so it
    will NEVER recompute them — meaning /stats silently blends old-fabricated
    and new-correct numbers, which is worse than an obviously-broken stat.

WHAT THIS DOES
    Deletes every row in `alert_outcomes`. It does NOT touch `alerts_log`, so
    each alert simply reverts to "open" and will be re-resolved correctly by the
    fixed resolver on the next `resolve_pending` run — BUT:
      • swing / channel alerts still inside the daily-history window re-resolve
        with correct P&L;
      • old INTRADAY alerts cannot be recovered (5-min history is provider-
        capped, the candles are gone) — they just drop out of the stats, which
        is the honest outcome. Do not try to "recompute" them; that would
        fabricate again.

SAFETY
    • BACK UP stockagent.db first (copy the file).
    • Dry-run by default: prints the row count and exits.
    • Pass --yes to actually delete.

    python one_time_reset_outcomes.py            # dry run, shows what it'd do
    python one_time_reset_outcomes.py --yes      # perform the reset
"""
import sys

import subscriptions


def main() -> int:
    confirm = "--yes" in sys.argv[1:]

    with subscriptions._conn() as c:
        total = c.execute("SELECT COUNT(*) FROM alert_outcomes").fetchone()[0]
        by_status = c.execute(
            "SELECT status, COUNT(*) FROM alert_outcomes GROUP BY status "
            "ORDER BY COUNT(*) DESC"
        ).fetchall()

    print(f"alert_outcomes rows currently stored: {total}")
    for status, n in by_status:
        print(f"  {status:<22} {n}")

    if total == 0:
        print("\nNothing to do — table is already empty.")
        return 0

    if not confirm:
        print(
            "\nDRY RUN — no changes made.\n"
            "These outcomes include pre-fix fabricated win P&L. Deleting them\n"
            "lets the fixed resolver recompute what it still can (swing/channel\n"
            "within the daily window) and drops unrecoverable old intraday rows.\n\n"
            "BACK UP stockagent.db, then re-run with --yes to perform the reset."
        )
        return 0

    with subscriptions._conn() as c:
        deleted = c.execute("DELETE FROM alert_outcomes").rowcount

    print(f"\nDeleted {deleted} outcome rows. Alerts are now 'open' again.")
    print("Next resolve_pending (EOD job, /stats, or eod_report.resolve_pending())")
    print("will re-resolve everything still inside the data window with correct P&L.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
