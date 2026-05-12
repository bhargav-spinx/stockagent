"""
Telethon listener — subscribes to public Telegram channels via the user's
personal account and pipes tip-shaped posts into the bot's analysis pipeline.

This is intentionally a SECONDARY data source — it runs alongside the Bot API,
not in place of it. The bot owns Telegram conversations with users; Telethon
only reads channel broadcasts.

Required `.env` keys:
    TELETHON_API_ID       int — from my.telegram.org
    TELETHON_API_HASH     32-char hex — from my.telegram.org
    TELETHON_PHONE        e.g. +919876543210 (with country code)
    TELETHON_CHANNELS     comma-separated @usernames, e.g. STOCKGAINERSS,NSEResearch
    TELETHON_NOTIFY_USER_ID  Telegram user ID where analyses are delivered

First run requires a one-time SMS OTP — see auth_once.py for the interactive
flow. After auth, a `telethon_session.session` file is created and reused.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import pytz
from telegram.ext import Application

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

SESSION_NAME = "telethon_session"


def _enabled() -> bool:
    return bool(
        os.getenv("TELETHON_API_ID")
        and os.getenv("TELETHON_API_HASH")
        and os.getenv("TELETHON_PHONE")
        and os.getenv("TELETHON_CHANNELS")
        and os.getenv("TELETHON_NOTIFY_USER_ID")
    )


def _channels() -> list[str]:
    raw = os.getenv("TELETHON_CHANNELS", "")
    return [c.strip().lstrip("@") for c in raw.split(",") if c.strip()]


def _notify_user_id() -> int:
    return int(os.getenv("TELETHON_NOTIFY_USER_ID", "0"))


class TelethonListener:
    """Listens to configured channels; on each post extracts a tip and
    pushes the analysis to the bot owner via the Bot API."""

    def __init__(self, bot_app: Application):
        self.bot_app = bot_app
        self.client = None
        self.task: Optional[asyncio.Task] = None
        self.channels = _channels()
        self.notify_user_id = _notify_user_id()
        self.message_count = 0
        self.tip_count = 0
        self.started_at: Optional[datetime] = None

    async def start(self) -> None:
        if not _enabled():
            logger.info("Telethon disabled — set TELETHON_* env vars to enable")
            return

        # Late import so a missing telethon package doesn't break bot startup
        from telethon import TelegramClient, events

        api_id = int(os.getenv("TELETHON_API_ID"))
        api_hash = os.getenv("TELETHON_API_HASH")
        phone = os.getenv("TELETHON_PHONE")

        self.client = TelegramClient(SESSION_NAME, api_id, api_hash)

        try:
            await self.client.start(phone=phone)
        except Exception as e:
            logger.exception("Telethon login failed: %s", e)
            logger.error(
                "If this is the first run, you need to authenticate "
                "interactively. Run: python auth_once.py"
            )
            return

        me = await self.client.get_me()
        logger.info("Telethon connected as %s (user_id=%s)", me.username or me.first_name, me.id)

        # Subscribe to each configured channel
        for ch in self.channels:
            self._register_channel(ch, events)

        self.started_at = datetime.now(IST)
        logger.info("Telethon listening on %d channel(s): %s", len(self.channels), self.channels)

        # Run forever
        await self.client.run_until_disconnected()

    def _register_channel(self, channel: str, events_module) -> None:
        @self.client.on(events_module.NewMessage(chats=channel))
        async def _handler(event, _ch=channel):
            await self._on_channel_post(event, _ch)

    async def _on_channel_post(self, event, channel: str) -> None:
        self.message_count += 1
        text = event.message.message or ""
        try:
            await self._process_post(event, channel, text)
        except Exception as e:
            logger.exception("Failed processing post from @%s: %s", channel, e)

    async def _process_post(self, event, channel: str, text: str) -> None:
        """Extract a tip from the post (text or photo) and deliver analysis."""
        import tip_parser
        from universe import INTRADAY_UNIVERSE

        known = set(INTRADAY_UNIVERSE)

        # Try text extraction first
        tip = tip_parser.extract_tip(text, known) if text.strip() else None

        # If no text tip but post has a photo, try OCR
        if not tip and event.message.photo:
            try:
                image_bytes = await event.message.download_media(file=bytes)
                if image_bytes:
                    tip = await asyncio.to_thread(
                        tip_parser.extract_tip_from_image,
                        image_bytes,
                        known,
                    )
            except RuntimeError as e:
                logger.warning("OCR unavailable: %s", e)
            except Exception as e:
                logger.warning("OCR failed for @%s post: %s", channel, e)

        if not tip:
            return  # silent — not every channel post is a tip

        self.tip_count += 1
        await self._deliver(channel, tip)

    async def _deliver(self, channel: str, tip: dict) -> None:
        """Send tip extraction + own analysis to the notify user via Bot API."""
        from analyzer import analyze, format_report

        arrow = "🟢" if tip["action"] == "BUY" else "🔴"
        lines = [
            f"📡 *Channel tip from @{channel}*\n",
            f"{arrow} *{tip['symbol']}*  ·  *{tip['action']}*",
        ]
        if tip.get("entry"):
            lines.append(f"🎯 Entry:  ₹{tip['entry']:,.2f}")
        if tip.get("target"):
            lines.append(f"🥇 Target: ₹{tip['target']:,.2f}")
        if tip.get("sl"):
            lines.append(f"🛑 SL:     ₹{tip['sl']:,.2f}")
        summary = "\n".join(lines)

        try:
            await self.bot_app.bot.send_message(
                self.notify_user_id, summary, parse_mode="Markdown"
            )
        except Exception as e:
            logger.warning("Telethon: send to user %s failed: %s",
                          self.notify_user_id, e)
            return

        # Run own analysis
        try:
            result = await asyncio.to_thread(analyze, tip["symbol"], "swing")
        except Exception as e:
            await self.bot_app.bot.send_message(
                self.notify_user_id,
                f"❌ Couldn't analyze {tip['symbol']}: {e}",
            )
            return

        bot_sig = result["signal"]
        if tip["action"] == bot_sig:
            verdict = "✅ *Agreement.* My analysis matches the channel tip."
        elif bot_sig == "HOLD":
            verdict = "🟡 *Neutral.* My indicators say HOLD."
        else:
            verdict = "⚠️ *Conflict.* Channel and my indicators disagree."

        msg = (
            f"🔍 *My analysis of {result['symbol']}*\n\n"
            f"Signal: *{bot_sig}*  ({result['confidence']}% confidence)\n"
            f"RSI: {result['rsi']}\n\n"
            + "\n".join(
                f"{'🟢' if s == 'BUY' else '🔴' if s == 'SELL' else '🟡'} {n}: {r}"
                for n, s, r in result["indicators"][:4]
            )
            + f"\n\n{verdict}\n\n_Educational only._"
        )
        await self.bot_app.bot.send_message(
            self.notify_user_id, msg, parse_mode="Markdown"
        )

    async def stop(self) -> None:
        if self.client:
            await self.client.disconnect()
            logger.info("Telethon disconnected")


async def run_listener(bot_app: Application) -> None:
    """Entry point for bot.py's post_init — launches the Telethon listener."""
    listener = TelethonListener(bot_app)
    bot_app.bot_data["telethon_listener"] = listener
    await listener.start()
