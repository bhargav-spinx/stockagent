"""
Telegram bot for Indian stock recommendations.
Run: python bot.py
"""

import asyncio
import os
import logging
from datetime import datetime, time as dt_time

import pytz
from dotenv import load_dotenv

IST = pytz.timezone("Asia/Kolkata")

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")

# Local-dev escape hatch for corp TLS inspection. Patches `requests.Session`
# so Angel SDK / ipify / scrip-master downloads skip verification.
# Activated only when DISABLE_SSL_VERIFY=true. Do NOT enable in production.
if DISABLE_SSL_VERIFY:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_session_init = requests.Session.__init__
    def _patched_session_init(self, *a, **kw):
        _orig_session_init(self, *a, **kw)
        self.verify = False
    requests.Session.__init__ = _patched_session_init

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from telegram.request import HTTPXRequest

from analyzer import analyze, format_report, normalize_symbol
from data_provider import force_angel_login, angel_session_active, get_provider_name
from scanner import scan_many, format_signal_telegram, TIER1_WATCHLIST
import subscriptions
import eod_report

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Watchlists are persisted in SQLite via subscriptions.user_watchlist.
# See add_to_watchlist / remove_from_watchlist / get_watchlist.


# ---------- Command Handlers ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "👋 *Welcome to the Indian Stock Signal Bot*\n\n"
        "I analyze NSE/BSE stocks using technical indicators "
        "(RSI, MACD, SMA, Bollinger Bands) and give BUY/SELL/HOLD signals.\n\n"
        "*Analysis modes:*\n"
        "🟦 *Swing / Positional* — daily candles, multi-day to weeks\n"
        "🟥 *Intraday* — 5-min candles, same-session trades\n\n"
        "*Commands:*\n"
        "/swing SYMBOL — full swing analysis (daily candles)\n"
        "/intraday SYMBOL — full intraday analysis (5-min candles)\n"
        "/analyze SYMBOL — alias for /swing\n"
        "/quick SYMBOL — one-line swing signal\n"
        "/quickintra SYMBOL — one-line intraday signal\n"
        "/scan — scan tier-1 watchlist for ORB / VWAP / range setups\n"
        "/scan SYMBOL — scan one or more symbols\n"
        "`/scan_alerts on` — intraday auto-scan every 5 min, ping on setups\n"
        "`/swing_alerts on` — end-of-day BUY/SELL alerts (15:45 IST, your watchlist)\n"
        "`/eod_report on` — daily summary of alerts + outcomes (15:35 IST)\n"
        "/today — on-demand EOD report right now\n"
        "/watch SYMBOL — add to watchlist\n"
        "/unwatch SYMBOL — remove from watchlist\n"
        "/mywatch — swing-analyze entire watchlist\n"
        "`/angel_status` — show data source & session status\n"
        "`/angel_login` — force a fresh Angel One login\n"
        "/help — show this message\n\n"
        "💡 *Tip:* tap the `/` icon below to see all commands in a tap-to-pick menu.\n\n"
        "_Use NSE tickers (RELIANCE, TCS, INFY) — `.NS` is added automatically._\n"
        "_For BSE, append `.BO` (e.g. `RELIANCE.BO`)._\n\n"
        "⚠️ *Educational tool only. Yahoo data is ~15 min delayed. "
        "Not SEBI-registered investment advice.*"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def _full_analysis(update: Update, symbol: str, mode: str):
    label = "Intraday" if mode == "intraday" else "Swing"
    await update.message.reply_text(f"🔍 {label} analysis: {symbol.upper()}...")
    try:
        result = analyze(symbol, mode=mode)
        report = format_report(result)
        keyboard = [[
            InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh:{mode}:{symbol}"),
            InlineKeyboardButton("👁 Watch", callback_data=f"watch:{symbol}"),
        ]]
        await update.message.reply_text(
            report,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        # Log to alerts journal if signal is actionable
        if result["signal"] in ("BUY", "SELL"):
            ts = result.get("trade_setup", {})
            if ts.get("action") in ("BUY", "SELL"):
                subscriptions.log_alert(
                    category=("manual_intraday" if mode == "intraday" else "manual_swing"),
                    user_id=update.effective_user.id,
                    symbol=result["symbol"],
                    setup=None,
                    direction=result["signal"],
                    entry=ts["entry"],
                    stop_loss=ts["stop_loss"],
                    target1=ts["target1"],
                    target2=ts["target2"],
                )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/analyze RELIANCE`", parse_mode="Markdown")
        return
    await _full_analysis(update, context.args[0], mode="swing")


async def swing_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/swing RELIANCE`", parse_mode="Markdown")
        return
    await _full_analysis(update, context.args[0], mode="swing")


async def intraday_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/intraday RELIANCE`", parse_mode="Markdown")
        return
    await _full_analysis(update, context.args[0], mode="intraday")


async def _quick(update: Update, symbol: str, mode: str):
    try:
        r = analyze(symbol, mode=mode)
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[r["signal"]]
        tag = "INTRA" if mode == "intraday" else "SWING"
        msg = (
            f"{emoji} *{r['symbol']}* → *{r['signal']}* `[{tag}]`\n"
            f"₹{r['price']} ({r['change_pct']:+.2f}% {r['change_label']}) | "
            f"Confidence: {r['confidence']}%"
        )
        if mode == "intraday" and not r.get("market_open"):
            msg += "\n⚠️ _NSE closed — data stale_"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def quick_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/quick RELIANCE`", parse_mode="Markdown")
        return
    await _quick(update, context.args[0], mode="swing")


async def quickintra_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/quickintra RELIANCE`", parse_mode="Markdown")
        return
    await _quick(update, context.args[0], mode="intraday")


async def watch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/watch TCS`", parse_mode="Markdown")
        return

    user_id = update.effective_user.id
    symbol = normalize_symbol(context.args[0])
    subscriptions.add_to_watchlist(user_id, symbol)
    await update.message.reply_text(f"✅ Added {symbol} to your watchlist.")


async def unwatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/unwatch TCS`", parse_mode="Markdown")
        return

    user_id = update.effective_user.id
    symbol = normalize_symbol(context.args[0])
    if subscriptions.remove_from_watchlist(user_id, symbol):
        await update.message.reply_text(f"🗑 Removed {symbol}.")
    else:
        await update.message.reply_text("Not in your watchlist.")


async def angel_login_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force a fresh Angel One login. Useful after session expiry or TOTP drift."""
    if not os.getenv("ANGEL_API_KEY"):
        await update.message.reply_text(
            "Angel One is not configured. Currently using "
            f"{get_provider_name()}.\n"
            "Add ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PASSWORD, "
            "and ANGEL_TOTP_SECRET to .env to enable."
        )
        return

    await update.message.reply_text("🔐 Logging in to Angel One...")
    try:
        info = force_angel_login()
        await update.message.reply_text(
            f"✅ Angel One session active\nClient: {info.get('client_code')}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Login failed: {e}")


async def angel_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Report whether Angel One session is active and which provider is in use."""
    provider = get_provider_name()
    active = angel_session_active()
    if not os.getenv("ANGEL_API_KEY"):
        status = "Angel One not configured (using fallback)"
    elif active:
        status = "Angel One session: active ✅"
    else:
        status = "Angel One session: not initialized (will log in on first request)"
    await update.message.reply_text(
        f"📡 Data source: {provider}\n{status}\n\n"
        "Use /angel_login to force a fresh session."
    )


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Run the intraday scanner. Without args, scans tier-1 watchlist.
    With args, scans the given symbols (max 5 per request).
    """
    if context.args:
        symbols = [a.upper() for a in context.args[:5]]
        await update.message.reply_text(
            f"🔍 Scanning {len(symbols)} symbol(s)...",
        )
    else:
        symbols = TIER1_WATCHLIST
        await update.message.reply_text(
            f"🔍 Scanning tier-1 watchlist ({len(symbols)} stocks). "
            "Takes ~10 sec — Angel rate-limit pacing."
        )

    results = await asyncio.to_thread(scan_many, symbols)

    signals = [r for r in results if r["status"] == "signal"]
    skips = [r for r in results if r["status"] == "skip"]
    no_setup = [r for r in results if r["status"] == "no_setup"]
    errors = [r for r in results if r["status"] == "error"]

    for r in signals:
        await update.message.reply_text(
            format_signal_telegram(r["signal"]),
            parse_mode="Markdown",
        )

    lines = [
        "*Scan complete*",
        f"📈 Signals fired: {len(signals)}",
        f"⏸ Filtered out: {len(skips)}",
        f"⚪ No setup: {len(no_setup)}",
    ]
    if errors:
        lines.append(f"❌ Errors: {len(errors)}")
    if not signals:
        lines.append("")
        lines.append("_No qualifying setups right now._")

    # First few skip reasons help debugging filters
    if skips and not signals:
        lines.append("")
        lines.append("*Top filter rejections:*")
        for r in skips[:5]:
            lines.append(f"• `{r['symbol']}` — {r['reason']}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def mywatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    watchlist = subscriptions.get_watchlist(user_id)

    if not watchlist:
        await update.message.reply_text("Your watchlist is empty. Use /watch SYMBOL.")
        return

    await update.message.reply_text(f"📋 Analyzing {len(watchlist)} stock(s)...")
    for symbol in watchlist:
        try:
            r = analyze(symbol)
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[r["signal"]]
            msg = (
                f"{emoji} *{r['symbol']}* → *{r['signal']}*\n"
                f"₹{r['price']} ({r['change_pct']:+.2f}%) | "
                f"RSI: {r['rsi']} | Conf: {r['confidence']}%"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ {symbol}: {e}")


# ---------- Inline Button Handler ----------

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    parts = query.data.split(":", 2)
    action = parts[0]

    if action == "refresh":
        # callback_data format: refresh:<mode>:<symbol>
        mode, symbol = parts[1], parts[2]
        try:
            result = analyze(symbol, mode=mode)
            report = format_report(result)
            keyboard = [[
                InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh:{mode}:{symbol}"),
                InlineKeyboardButton("👁 Watch", callback_data=f"watch:{symbol}"),
            ]]
            await query.edit_message_text(
                report,
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {e}")

    elif action == "watch":
        symbol = parts[1]
        user_id = query.from_user.id
        sym = normalize_symbol(symbol)
        subscriptions.add_to_watchlist(user_id, sym)
        await query.message.reply_text(f"✅ Added {sym} to watchlist.")


# ---------- Plain text handler (treat any message as symbol) ----------

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip()
    if " " in symbol or len(symbol) > 20:
        await update.message.reply_text("Send a single ticker like `RELIANCE`, or use /help.")
        return

    try:
        r = analyze(symbol)
        report = format_report(r)
        await update.message.reply_text(report, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ {e}\n\nTry /help for commands.")


# ---------- Auto-scan alerts ----------

# Auto-scan loop config
AUTOSCAN_INTERVAL_SEC = 300   # run every 5 min
AUTOSCAN_FIRST_DELAY_SEC = 30  # wait this long after bot start before first run


async def scan_alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle auto-scan alerts for the calling user. /scan_alerts on|off (no arg = status)."""
    user_id = update.effective_user.id
    arg = (context.args[0].lower() if context.args else "")

    if arg not in ("on", "off"):
        sub = subscriptions.is_subscribed(user_id)
        await update.message.reply_text(
            f"📡 Auto-scan alerts: *{'ON' if sub else 'OFF'}*\n\n"
            "`/scan_alerts on` — get pinged when a setup fires (5-min checks)\n"
            "`/scan_alerts off` — stop alerts",
            parse_mode="Markdown",
        )
        return

    if arg == "on":
        subscriptions.subscribe(user_id)
        await update.message.reply_text(
            "✅ Auto-scan alerts *ON*.\n\n"
            "I'll scan the tier-1 watchlist every 5 minutes during NSE hours "
            "(09:30–14:30 IST, excluding 12:00–13:30 lunch) and ping you when a setup fires.\n\n"
            "Same setup on same stock won't be re-sent the same day (dedup).\n"
            "`/scan_alerts off` to stop.",
            parse_mode="Markdown",
        )
    else:
        subscriptions.unsubscribe(user_id)
        await update.message.reply_text("🔕 Auto-scan alerts *OFF*.", parse_mode="Markdown")


def is_market_active(now: datetime | None = None) -> bool:
    """NSE intraday entry window per STRATEGY.md §5: 09:30–14:30, no lunch (12:00–13:30)."""
    now = now or datetime.now(IST)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    t = now.time()
    if t < dt_time(9, 30) or t >= dt_time(14, 30):
        return False
    if dt_time(12, 0) <= t < dt_time(13, 30):
        return False
    return True


async def _autoscan_tick(app: Application) -> None:
    """One pass of the auto-scan loop. Safe to call any time; no-ops outside market hours."""
    if not is_market_active():
        return

    subs = subscriptions.get_subscribers()
    if not subs:
        return

    logger.info("autoscan: scanning %d symbols for %d subscriber(s)",
                len(TIER1_WATCHLIST), len(subs))
    try:
        results = await asyncio.to_thread(scan_many, TIER1_WATCHLIST)
    except Exception as e:
        logger.exception("autoscan: scan_many crashed: %s", e)
        return

    new_signals = 0
    for r in results:
        if r["status"] != "signal":
            continue
        sig = r["signal"]
        key = subscriptions.signal_key(sig.symbol, sig.setup, sig.direction)
        if subscriptions.already_fired(key):
            continue
        subscriptions.mark_fired(key)
        new_signals += 1

        # Log once (system-level alert; not per-user — outcome is identical)
        subscriptions.log_alert(
            category="scan",
            user_id=None,
            symbol=sig.symbol,
            setup=sig.setup,
            direction=sig.direction,
            entry=sig.entry,
            stop_loss=sig.stop_loss,
            target1=sig.target1,
            target2=sig.target2,
        )

        msg = "🔔 *Auto-Signal*\n\n" + format_signal_telegram(sig)
        for uid in subs:
            try:
                await app.bot.send_message(uid, msg, parse_mode="Markdown")
            except Exception as e:
                logger.warning("autoscan: send to %s failed: %s", uid, e)

    if new_signals:
        logger.info("autoscan: fired %d new signal(s) to %d sub(s)", new_signals, len(subs))


async def _autoscan_loop(app: Application) -> None:
    """Run forever: tick every AUTOSCAN_INTERVAL_SEC."""
    logger.info("autoscan loop started (every %ds)", AUTOSCAN_INTERVAL_SEC)
    await asyncio.sleep(AUTOSCAN_FIRST_DELAY_SEC)
    while True:
        try:
            await _autoscan_tick(app)
        except Exception as e:
            logger.exception("autoscan loop tick failed: %s", e)
        # Daily dedup-table prune (cheap, just runs every 5 min)
        try:
            subscriptions.purge_old_signals(keep_days=2)
        except Exception:
            pass
        await asyncio.sleep(AUTOSCAN_INTERVAL_SEC)


# ---------- Swing alerts (end-of-day) ----------

# Run once per trading day in this window
SWING_RUN_WINDOW = (dt_time(15, 45), dt_time(16, 15))
SWING_LOOP_TICK_SEC = 300  # check every 5 min if it's run-window time


async def swing_alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle end-of-day swing alerts. /swing_alerts on|off (no arg = status)."""
    user_id = update.effective_user.id
    arg = (context.args[0].lower() if context.args else "")

    if arg not in ("on", "off"):
        sub = subscriptions.is_swing_subscribed(user_id)
        wl = subscriptions.get_watchlist(user_id)
        await update.message.reply_text(
            f"📡 Swing alerts: *{'ON' if sub else 'OFF'}*\n"
            f"📋 Your watchlist: {len(wl)} stock(s)\n\n"
            "`/swing_alerts on` — daily BUY/SELL signals after market close (15:45 IST)\n"
            "`/swing_alerts off` — stop alerts\n"
            "`/watch SYMBOL` — add to watchlist (alerts run on these)",
            parse_mode="Markdown",
        )
        return

    if arg == "on":
        subscriptions.swing_subscribe(user_id)
        wl_count = len(subscriptions.get_watchlist(user_id))
        msg = (
            "✅ Swing alerts *ON*.\n\n"
            "Every trading day around 15:45 IST (after market close), "
            "I'll run swing analysis on your watchlist and send BUY/SELL signals "
            "with entry, stop-loss, and targets.\n"
            "_HOLD signals are silenced — only actionable BUY/SELL trigger an alert._\n\n"
        )
        if wl_count == 0:
            msg += "⚠️ Your watchlist is empty — add stocks via `/watch RELIANCE` first."
        else:
            msg += f"📋 Currently watching {wl_count} stock(s)."
        await update.message.reply_text(msg, parse_mode="Markdown")
    else:
        subscriptions.swing_unsubscribe(user_id)
        await update.message.reply_text("🔕 Swing alerts *OFF*.", parse_mode="Markdown")


def _is_swing_run_window(now: datetime | None = None) -> bool:
    """Are we in the daily swing-run window? Mon–Fri only."""
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return SWING_RUN_WINDOW[0] <= now.time() < SWING_RUN_WINDOW[1]


def _swing_run_key(d: datetime | None = None) -> str:
    d = d or datetime.now(IST)
    return f"swing_run:{d.date().isoformat()}"


async def _swing_alert_tick(app: Application) -> None:
    """One pass: if in run window and not yet ran today, scan all swing subs' watchlists."""
    if not _is_swing_run_window():
        return
    key = _swing_run_key()
    if subscriptions.already_fired(key):
        return  # already ran today

    subs = subscriptions.get_swing_subscribers()
    if not subs:
        subscriptions.mark_fired(key)
        return

    logger.info("swing alerts: running for %d subscriber(s)", len(subs))
    subscriptions.mark_fired(key)  # mark first so a crash mid-run doesn't double-send

    for uid in subs:
        watchlist = subscriptions.get_watchlist(uid)
        if not watchlist:
            try:
                await app.bot.send_message(
                    uid,
                    "🌅 *End-of-Day Swing Run*\n\n"
                    "Your watchlist is empty — nothing to scan.\n"
                    "Add stocks via `/watch RELIANCE` to get alerts tomorrow.",
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning("swing intro send to %s failed: %s", uid, e)
            continue

        sent = 0
        for symbol in watchlist:
            try:
                result = await asyncio.to_thread(analyze, symbol, "swing")
            except Exception as e:
                logger.warning("swing analyze failed for %s/%s: %s", uid, symbol, e)
                await asyncio.sleep(0.5)
                continue
            if result["signal"] in ("BUY", "SELL"):
                ts = result.get("trade_setup", {})
                if ts.get("action") in ("BUY", "SELL"):
                    subscriptions.log_alert(
                        category="swing_auto",
                        user_id=uid,
                        symbol=result["symbol"],
                        setup=None,
                        direction=result["signal"],
                        entry=ts["entry"],
                        stop_loss=ts["stop_loss"],
                        target1=ts["target1"],
                        target2=ts["target2"],
                    )
                msg = "🌅 *End-of-Day Swing Signal*\n\n" + format_report(result)
                try:
                    await app.bot.send_message(uid, msg, parse_mode="Markdown")
                    sent += 1
                except Exception as e:
                    logger.warning("swing signal send to %s failed: %s", uid, e)
            await asyncio.sleep(0.5)  # Angel rate-limit pacing

        # Send a summary so the user knows the run completed even with no signals
        try:
            await app.bot.send_message(
                uid,
                f"📋 Swing run complete — scanned {len(watchlist)}, "
                f"sent {sent} BUY/SELL alert(s). HOLDs silenced.",
            )
        except Exception:
            pass


async def _swing_alert_loop(app: Application) -> None:
    """Wakes every 5 min, runs the daily swing alert at most once per day."""
    logger.info("swing alert loop started (window %s–%s IST)",
                SWING_RUN_WINDOW[0], SWING_RUN_WINDOW[1])
    while True:
        try:
            await _swing_alert_tick(app)
        except Exception as e:
            logger.exception("swing alert tick failed: %s", e)
        await asyncio.sleep(SWING_LOOP_TICK_SEC)


# ---------- End-of-day report ----------

# Daily report fires once in this window (15:35–16:05 IST)
EOD_RUN_WINDOW = (dt_time(15, 35), dt_time(16, 5))
EOD_LOOP_TICK_SEC = 300  # check every 5 min


async def eod_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle automatic EOD report. /eod_report on|off (no arg = status)."""
    user_id = update.effective_user.id
    arg = (context.args[0].lower() if context.args else "")

    if arg not in ("on", "off"):
        sub = subscriptions.is_eod_subscribed(user_id)
        await update.message.reply_text(
            f"📡 EOD report: *{'ON' if sub else 'OFF'}*\n\n"
            "`/eod_report on`  — receive a daily summary at 15:35 IST\n"
            "`/eod_report off` — stop\n"
            "`/today`          — on-demand report right now",
            parse_mode="Markdown",
        )
        return

    if arg == "on":
        subscriptions.eod_subscribe(user_id)
        await update.message.reply_text(
            "✅ EOD report *ON*.\n\n"
            "After NSE close (15:35 IST) you'll get a summary of every alert "
            "fired today + outcome (T1/T2/SL/expired) + hypothetical P&L.\n\n"
            "Run `/today` anytime to fetch it on demand.",
            parse_mode="Markdown",
        )
    else:
        subscriptions.eod_unsubscribe(user_id)
        await update.message.reply_text("🔕 EOD report *OFF*.", parse_mode="Markdown")


async def today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """On-demand EOD report for the calling user."""
    await update.message.reply_text("📊 Building report — resolving outcomes…")
    try:
        report = await asyncio.to_thread(
            eod_report.build_report, update.effective_user.id
        )
    except Exception as e:
        logger.exception("today_cmd build_report failed: %s", e)
        await update.message.reply_text(f"❌ Report build failed: {e}")
        return
    await update.message.reply_text(report, parse_mode="Markdown")


def _is_eod_run_window(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return EOD_RUN_WINDOW[0] <= now.time() < EOD_RUN_WINDOW[1]


def _eod_run_key(d: datetime | None = None) -> str:
    d = d or datetime.now(IST)
    return f"eod_run:{d.date().isoformat()}"


async def _eod_report_tick(app: Application) -> None:
    if not _is_eod_run_window():
        return
    key = _eod_run_key()
    if subscriptions.already_fired(key):
        return  # already ran today

    subs = subscriptions.get_eod_subscribers()
    if not subs:
        subscriptions.mark_fired(key)
        return

    logger.info("EOD report: running for %d subscriber(s)", len(subs))
    subscriptions.mark_fired(key)

    for uid in subs:
        try:
            report = await asyncio.to_thread(eod_report.build_report, uid)
            await app.bot.send_message(uid, report, parse_mode="Markdown")
        except Exception as e:
            logger.warning("EOD report send to %s failed: %s", uid, e)


async def _eod_report_loop(app: Application) -> None:
    logger.info("EOD report loop started (window %s–%s IST)",
                EOD_RUN_WINDOW[0], EOD_RUN_WINDOW[1])
    while True:
        try:
            await _eod_report_tick(app)
        except Exception as e:
            logger.exception("EOD report tick failed: %s", e)
        await asyncio.sleep(EOD_LOOP_TICK_SEC)


# Telegram's /-menu. Shown when user taps the / icon or hovers the bot.
# Set once in _post_init via bot.set_my_commands().
COMMAND_MENU = [
    BotCommand("start", "Welcome + command list"),
    BotCommand("help", "Same as /start"),
    # Analysis
    BotCommand("swing", "Swing analysis  (e.g. /swing RELIANCE)"),
    BotCommand("intraday", "Intraday analysis  (e.g. /intraday TCS)"),
    BotCommand("quick", "One-line swing signal"),
    BotCommand("quickintra", "One-line intraday signal"),
    # Scanner
    BotCommand("scan", "Scan tier-1 watchlist for setups"),
    BotCommand("scan_alerts", "Auto-scan during NSE hours (on/off)"),
    # Swing
    BotCommand("swing_alerts", "End-of-day BUY/SELL alerts (on/off)"),
    # EOD report
    BotCommand("eod_report", "Daily summary report (on/off)"),
    BotCommand("today", "On-demand EOD report"),
    # Watchlist
    BotCommand("watch", "Add to watchlist  (e.g. /watch INFY)"),
    BotCommand("unwatch", "Remove from watchlist"),
    BotCommand("mywatch", "Run swing on entire watchlist"),
    # Diagnostics
    BotCommand("angel_status", "Data-source status"),
    BotCommand("angel_login", "Force fresh Angel login"),
]


async def _post_init(app: Application) -> None:
    """Initialize SQLite + register command menu + start background tasks."""
    subscriptions.init_db()
    try:
        await app.bot.set_my_commands(COMMAND_MENU)
        logger.info("Registered %d commands in Telegram menu", len(COMMAND_MENU))
    except Exception as e:
        logger.warning("Failed to set bot commands menu: %s", e)
    app.bot_data["autoscan_task"] = asyncio.create_task(_autoscan_loop(app))
    app.bot_data["swing_alert_task"] = asyncio.create_task(_swing_alert_loop(app))
    app.bot_data["eod_report_task"] = asyncio.create_task(_eod_report_loop(app))


# ---------- Main ----------

def main():
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env file")

    builder = Application.builder().token(TOKEN).post_init(_post_init)

    # Local-dev: skip TLS verification for python-telegram-bot's httpx client
    # when behind a corp SSL-inspecting proxy. Only when DISABLE_SSL_VERIFY=true.
    if DISABLE_SSL_VERIFY:
        req = HTTPXRequest(httpx_kwargs={"verify": False})
        get_updates_req = HTTPXRequest(httpx_kwargs={"verify": False})
        builder = builder.request(req).get_updates_request(get_updates_req)

    app = builder.build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("swing", swing_cmd))
    app.add_handler(CommandHandler("intraday", intraday_cmd))
    app.add_handler(CommandHandler("quick", quick_cmd))
    app.add_handler(CommandHandler("quickintra", quickintra_cmd))
    app.add_handler(CommandHandler("watch", watch_cmd))
    app.add_handler(CommandHandler("unwatch", unwatch_cmd))
    app.add_handler(CommandHandler("mywatch", mywatch_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("scan_alerts", scan_alerts_cmd))
    app.add_handler(CommandHandler("swing_alerts", swing_alerts_cmd))
    app.add_handler(CommandHandler("eod_report", eod_report_cmd))
    app.add_handler(CommandHandler("today", today_cmd))
    app.add_handler(CommandHandler("angel_login", angel_login_cmd))
    app.add_handler(CommandHandler("angel_status", angel_status_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Webhook mode (Cloud Run / any HTTPS host) when WEBHOOK_URL is set.
    # PORT comes from the platform; defaults to 8080 (Cloud Run convention).
    # WEBHOOK_SECRET is the URL path segment + Telegram secret_token — keeps
    # randoms from spamming the endpoint.
    webhook_url = os.getenv("WEBHOOK_URL")
    if webhook_url:
        port = int(os.getenv("PORT", "8080"))
        secret = os.getenv("WEBHOOK_SECRET", "tg-webhook")
        logger.info("Bot starting in webhook mode on port %d", port)
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=secret,
            secret_token=secret,
            webhook_url=f"{webhook_url.rstrip('/')}/{secret}",
        )
    else:
        logger.info("Bot starting in polling mode")
        app.run_polling()


if __name__ == "__main__":
    main()
