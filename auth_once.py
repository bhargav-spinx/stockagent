"""
One-time Telethon authentication.

Run this ONCE before starting the bot. It will:
1. Connect using TELETHON_API_ID / TELETHON_API_HASH / TELETHON_PHONE from .env
2. Telegram sends an OTP to your phone via SMS or in-app
3. You enter the code at the prompt
4. A `telethon_session.session` file is created — used for all future logins

After this works once, the bot.py background task will reuse the session
without prompting. Re-run this script only if the session expires (rare).

Usage:
    python auth_once.py
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    api_id = os.getenv("TELETHON_API_ID")
    api_hash = os.getenv("TELETHON_API_HASH")
    phone = os.getenv("TELETHON_PHONE")

    missing = []
    if not api_id: missing.append("TELETHON_API_ID")
    if not api_hash: missing.append("TELETHON_API_HASH")
    if not phone: missing.append("TELETHON_PHONE")

    if missing:
        print("❌ Missing .env keys:", ", ".join(missing))
        print()
        print("Get these from https://my.telegram.org (API development tools).")
        sys.exit(1)

    from telethon import TelegramClient

    print(f"Connecting as {phone}…")
    client = TelegramClient("telethon_session", int(api_id), api_hash)

    await client.start(phone=phone)

    me = await client.get_me()
    print()
    print(f"✅ Authenticated as: {me.first_name} {me.last_name or ''}".rstrip())
    print(f"   Username: @{me.username or '(none)'}")
    print(f"   User ID:  {me.id}")
    print()
    print("Session saved to telethon_session.session — keep this file private!")
    print("You can now start the bot — it will reuse this session without OTP.")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
