"""
One-time maintenance: repair alerts logged with BUY/SELL directions.

WHY
    Manual (/swing, /intraday) and swing_auto alerts were logged with the
    analyzer's "BUY"/"SELL" as `direction`. The resolver only recognised
    "long", so every BUY was resolved AS A SHORT — P&L sign, SL/target sides
    and MFE/MAE all flipped (a quiet day booked "sl_hit +3.00%"). log_alert now
    normalises to long/short, but rows already stored keep the raw value, and
    `resolve_pending` never recomputes an alert that already has an outcome.

WHAT THIS DOES
    1. Rewrites alerts_log.direction BUY→long, SELL→short (SELL rows were
       resolved correctly by accident; they are normalised for consistency).
    2. Deletes the outcome rows of the former-BUY alerts only, so they revert
       to "open" and the fixed resolver re-resolves what it still can on its
       next run (swing within the ~3-month daily window; old intraday rows
       can't be recovered — 5-min history is gone — and simply drop out).

SAFETY
    • BACK UP stockagent.db first (copy the file).
    • Dry-run by default: prints what it would change and exits.
    • Pass --yes to apply.

    python one_time_fix_directions.py            # dry run
    python one_time_fix_directions.py --yes      # apply
"""
import sys

import subscriptions


def main() -> int:
    confirm = "--yes" in sys.argv[1:]

    with subscriptions._conn() as c:
        by_dir = c.execute(
            "SELECT category, direction, COUNT(*) FROM alerts_log "
            "WHERE upper(direction) IN ('BUY', 'SELL') "
            "GROUP BY category, direction ORDER BY category").fetchall()
        buy_outcomes = c.execute(
            "SELECT COUNT(*) FROM alert_outcomes o JOIN alerts_log a "
            "ON a.id = o.alert_id WHERE upper(a.direction) = 'BUY'").fetchone()[0]

    if not by_dir:
        print("Nothing to do — no alerts with BUY/SELL directions.")
        return 0

    print("alerts_log rows with BUY/SELL directions:")
    for cat, d, n in by_dir:
        print(f"  {cat:<18} {d:<5} {n}")
    print(f"outcome rows on former-BUY alerts (resolved as shorts): {buy_outcomes}")

    if not confirm:
        print("\nDRY RUN — no changes made.\n"
              "BACK UP stockagent.db, then re-run with --yes to apply.")
        return 0

    with subscriptions._conn() as c:
        deleted = c.execute(
            "DELETE FROM alert_outcomes WHERE alert_id IN "
            "(SELECT id FROM alerts_log WHERE upper(direction) = 'BUY')").rowcount
        longs = c.execute("UPDATE alerts_log SET direction = 'long' "
                          "WHERE upper(direction) = 'BUY'").rowcount
        shorts = c.execute("UPDATE alerts_log SET direction = 'short' "
                           "WHERE upper(direction) = 'SELL'").rowcount

    print(f"\nNormalised {longs} BUY→long and {shorts} SELL→short; deleted "
          f"{deleted} wrong outcome rows (those alerts are 'open' again).")
    print("The next resolve_pending (EOD job or /stats) re-resolves what is "
          "still inside the data window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
