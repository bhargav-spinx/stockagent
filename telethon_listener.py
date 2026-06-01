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

from telegram.ext import Application

from constants import IST

logger = logging.getLogger(__name__)

SESSION_NAME = "telethon_session"

# --- "Our standard" gate (STRATEGY.md) ---------------------------------------
# A parsed channel tip is forwarded only if it passes BOTH gates:
#   1. Swing confidence ≥ SWING_MIN_CONFIDENCE (3+ of 4 daily indicators agree)
#   2. The §5 universal filters (ATR band, volume, VWAP slope, round-number)
# Universal filters are intraday entry-mechanics, so we run them with
# check_time=False — the tip's arrival time is not our entry time. Flip
# APPLY_UNIVERSAL_FILTERS to False to gate on swing confidence alone.
SWING_MIN_CONFIDENCE = 75
APPLY_UNIVERSAL_FILTERS = True
SUMMARY_EVERY = 10  # emit a pass/drop digest after this many analyzed tips


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
        # Standard-gate tallies (cumulative; window resets after each digest)
        self.passed_count = 0
        self.dropped_confidence = 0
        self.dropped_filters = 0
        self.dropped_error = 0
        self._tips_since_summary = 0
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
        await self._evaluate_and_deliver(channel, tip)

    async def _evaluate_and_deliver(self, channel: str, tip: dict) -> None:
        """Apply 'our standard' to a parsed tip; deliver only if it passes.

        Gate 1 — swing confidence ≥ SWING_MIN_CONFIDENCE with a BUY/SELL signal.
        Gate 2 — §5 universal filters (optional, check_time=False).
        Dropped tips are silenced but counted into a periodic digest.
        """
        from analyzer import analyze, normalize_symbol

        sym = tip["symbol"]
        direction = "long" if tip["action"] == "BUY" else "short"

        # Gate 1: swing analysis (daily candles).
        try:
            result = await asyncio.to_thread(analyze, sym, "swing")
        except Exception as e:
            logger.warning("standard gate: analyze failed for %s: %s", sym, e)
            self.dropped_error += 1
            await self._maybe_send_digest(channel)
            return

        conf = result.get("confidence", 0)
        swing_ok = result["signal"] in ("BUY", "SELL") and conf >= SWING_MIN_CONFIDENCE
        if not swing_ok:
            logger.info("standard gate: %s dropped — swing %s @ %s%% (<%s)",
                        sym, result["signal"], conf, SWING_MIN_CONFIDENCE)
            self.dropped_confidence += 1
            await self._maybe_send_digest(channel)
            return

        # Gate 2: §5 universal filters. check_time=False — the tip's arrival
        # time is not our entry time, so we validate setup structure only.
        if APPLY_UNIVERSAL_FILTERS:
            from data_provider import fetch_data
            from scanner_filters import apply_universal_filters

            filt_reason = ""
            filt_ok = False
            try:
                df = await asyncio.to_thread(
                    fetch_data, normalize_symbol(sym), "5d", "5m"
                )
                if len(df) < 30:
                    filt_reason = f"insufficient 5m data ({len(df)} candles)"
                else:
                    f = apply_universal_filters(df, direction=direction,
                                                check_time=False)
                    filt_ok, filt_reason = f.passed, f.reason
            except Exception as e:
                filt_reason = str(e)

            if not filt_ok:
                logger.info("standard gate: %s dropped — universal filter: %s",
                            sym, filt_reason)
                self.dropped_filters += 1
                await self._maybe_send_digest(channel)
                return

        # Passed both gates.
        self.passed_count += 1
        await self._deliver(channel, tip, result)
        await self._maybe_send_digest(channel)

    async def _maybe_send_digest(self, channel: str) -> None:
        """After every SUMMARY_EVERY analyzed tips, send a one-line digest of
        how the standard gate is performing, then reset the window."""
        self._tips_since_summary += 1
        if self._tips_since_summary < SUMMARY_EVERY:
            return
        window = self._tips_since_summary
        self._tips_since_summary = 0
        drops = []
        if self.dropped_confidence:
            drops.append(f"{self.dropped_confidence} low-confidence")
        if self.dropped_filters:
            drops.append(f"{self.dropped_filters} failed §5 filters")
        if self.dropped_error:
            drops.append(f"{self.dropped_error} errors")
        detail = f" ({'; '.join(drops)})" if drops else ""
        try:
            await self.bot_app.bot.send_message(
                self.notify_user_id,
                f"📋 Standard-gate digest — last {window} tips analyzed, "
                f"{self.passed_count} forwarded, "
                f"{self.dropped_confidence + self.dropped_filters + self.dropped_error} "
                f"dropped{detail}. (cumulative since start)",
            )
        except Exception as e:
            logger.warning("digest send failed: %s", e)

    async def _deliver(self, channel: str, tip: dict, result: dict) -> None:
        """Send tip extraction + own analysis to the notify user via Bot API.

        `result` is the swing analysis already computed by the standard gate,
        passed in to avoid re-analyzing.
        """
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

        # `result` is the swing analysis from the standard gate (BUY/SELL,
        # ≥ SWING_MIN_CONFIDENCE) — no need to re-analyze.
        from analyzer import signal_agreement, format_indicator_lines

        bot_sig = result["signal"]
        emoji, label = signal_agreement(tip["action"], bot_sig)
        detail = {
            "Agreement": "My analysis matches the channel tip.",
            "Neutral": "My indicators say HOLD.",
            "Conflict": "Channel and my indicators disagree.",
        }[label]

        msg = (
            f"🔍 *My analysis of {result['symbol']}*\n\n"
            f"Signal: *{bot_sig}*  ({result['confidence']}% confidence)\n"
            f"RSI: {result['rsi']}\n\n"
            + format_indicator_lines(result)
            + f"\n\n{emoji} *{label}.* {detail}\n\n_Educational only._"
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
