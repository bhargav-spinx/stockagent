"""
Telegram bot for Indian stock recommendations.
Run: python bot.py
"""

import os
import logging

from dotenv import load_dotenv

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

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from telegram.request import HTTPXRequest

from analyzer import analyze, format_report, normalize_symbol
from data_provider import force_angel_login, angel_session_active, get_provider_name

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# In-memory watchlist (use a DB for production)
WATCHLISTS: dict[int, set[str]] = {}


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
        "/watch SYMBOL — add to watchlist\n"
        "/unwatch SYMBOL — remove from watchlist\n"
        "/mywatch — swing-analyze entire watchlist\n"
        "/angel_status — show data source & session status\n"
        "/angel_login — force a fresh Angel One login\n"
        "/help — show this message\n\n"
        "_Use NSE tickers (RELIANCE, TCS, INFY) — `.NS` is added automatically._\n"
        "_For BSE, append `.BO` (e.g. `RELIANCE.BO`)._\n\n"
        "⚠️ *Educational tool only. Yahoo data is ~15 min delayed. "
        "Not SEBI-registered investment advice.*"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


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
    WATCHLISTS.setdefault(user_id, set()).add(symbol)
    await update.message.reply_text(f"✅ Added {symbol} to your watchlist.")


async def unwatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/unwatch TCS`", parse_mode="Markdown")
        return

    user_id = update.effective_user.id
    symbol = normalize_symbol(context.args[0])
    if user_id in WATCHLISTS and symbol in WATCHLISTS[user_id]:
        WATCHLISTS[user_id].remove(symbol)
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


async def mywatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    watchlist = WATCHLISTS.get(user_id, set())

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
        WATCHLISTS.setdefault(user_id, set()).add(sym)
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


# ---------- Main ----------

def main():
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env file")

    builder = Application.builder().token(TOKEN)

    # Local-dev: skip TLS verification for python-telegram-bot's httpx client
    # when behind a corp SSL-inspecting proxy. Only when DISABLE_SSL_VERIFY=true.
    if DISABLE_SSL_VERIFY:
        req = HTTPXRequest(httpx_kwargs={"verify": False})
        get_updates_req = HTTPXRequest(httpx_kwargs={"verify": False})
        builder = builder.request(req).get_updates_request(get_updates_req)

    app = builder.build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("swing", swing_cmd))
    app.add_handler(CommandHandler("intraday", intraday_cmd))
    app.add_handler(CommandHandler("quick", quick_cmd))
    app.add_handler(CommandHandler("quickintra", quickintra_cmd))
    app.add_handler(CommandHandler("watch", watch_cmd))
    app.add_handler(CommandHandler("unwatch", unwatch_cmd))
    app.add_handler(CommandHandler("mywatch", mywatch_cmd))
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
