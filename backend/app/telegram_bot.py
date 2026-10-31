"""Telegram delivery + two-way voice Q&A.

Linking : the dashboard shows t.me/<bot>?start=<field_id>; /start <field_id> links that chat to the field.
Alerts  : text (local language + numbered sources) and an OGG/Opus voice note.
Questions: voice or text -> Sarvam STT -> agent 'ask' run -> spoken, cited answer.
Transport: webhook when PUBLIC_BASE_URL is set (production), long-polling otherwise (local dev).
"""
from __future__ import annotations

import asyncio
import logging

from app import config, db
from app.tools import voice
from app.tools.base import client

log = logging.getLogger("khetrakshak.telegram")
API = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{config.TELEGRAM_BOT_TOKEN}"
bot_username: str | None = None


async def call(method: str, **kwargs) -> dict:
    resp = await client().post(f"{API}/{method}", timeout=60.0, **kwargs)
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method}: {body.get('description')}")
    return body["result"]


def _sources_text(final: dict) -> str:
    lines = []
    for c in final.get("citations", []):
        if c["type"] == "document":
            lines.append(f"[{c['n']}] {c['title']} - {c['publisher']}, p.{c['page']}")
        elif c["type"] == "rule":
            crit = f" ({c['criterion']}, p.{c['page']})" if c.get("criterion") else f" ({c.get('basis')})"
            lines.append(f"[{c['n']}] {c['title']}{crit}")
        else:
            cached = " (cached)" if c.get("cached") else ""
            lines.append(f"[{c['n']}] {c['title']}{cached}, {str(c.get('fetched_at', ''))[:16]} UTC")
    return "\n".join(lines)


async def send_text(chat_id: int, text: str) -> None:
    await call("sendMessage", json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True})


async def send_voice(chat_id: int, wav: bytes) -> None:
    ogg = await asyncio.to_thread(voice.wav_to_ogg_opus, wav)
    await call("sendVoice", data={"chat_id": str(chat_id)}, files={"voice": ("alert.ogg", ogg, "audio/ogg")})


async def send_alert(chat_id: int, final: dict, audio: bytes | None) -> bool:
    if not config.TELEGRAM_BOT_TOKEN:
        return False
    try:
        body = final.get("text_local") or final.get("text_en", "")
        src = _sources_text(final)
        await send_text(chat_id, f"KhetRakshak\n\n{body}" + (f"\n\nSources:\n{src}" if src else ""))
        if audio:
            await send_voice(chat_id, audio)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("telegram send failed: %s", exc)
        return False


async def _answer(chat_id: int, field: dict, question: str) -> None:
    from app.agents.graph import start_run
    from app.agents.events import artifacts

    rid, final = await start_run(field, "ask", question=question, background=False)
    if not final or final.get("error"):
        await send_text(chat_id, "Sorry, something went wrong. Please try again in a minute.")
        return
    await send_alert(chat_id, final, artifacts.get(rid, {}).get("audio"))


async def handle_update(update: dict) -> None:
    msg = update.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    if not chat_id:
        return
    text = (msg.get("text") or "").strip()
    try:
        if text.startswith("/start"):
            parts = text.split(maxsplit=1)
            field = await db.link_telegram(parts[1].strip(), chat_id) if len(parts) == 2 else None
            if field:
                await send_text(chat_id, f"Linked to field '{field['name']}' ({field['crop']}). You will get "
                                         "alerts here. Send a voice note or text to ask about your crop.")
            else:
                await send_text(chat_id, "Welcome to KhetRakshak. Open your field on the dashboard and tap "
                                         "'Link Telegram' to connect this chat.")
            return
        field = await db.field_for_chat(chat_id)
        if not field:
            await send_text(chat_id, "This chat is not linked to a field yet. Use 'Link Telegram' on the dashboard.")
            return
        if msg.get("voice") or msg.get("audio"):
            fid = (msg.get("voice") or msg.get("audio"))["file_id"]
            info = await call("getFile", json={"file_id": fid})
            audio = (await client().get(f"{FILE_API}/{info['file_path']}", timeout=60.0)).content
            heard = await voice.stt(audio)
            if not heard["transcript"]:
                await send_text(chat_id, "Sorry, I could not hear that clearly. Please send the voice note again.")
                return
            await send_text(chat_id, f"You asked: {heard['transcript']}")
            question = heard["transcript"]
        elif text:
            question = text
        else:
            return
        asyncio.create_task(_answer(chat_id, field, question))
    except Exception as exc:  # noqa: BLE001 - one bad update must not stop the bot
        log.warning("telegram update failed: %s", exc)
        try:
            await send_text(chat_id, "Sorry, something went wrong. Please try again.")
        except Exception:  # noqa: BLE001
            pass


async def start() -> str:
    """Webhook in production, long-polling locally. Returns the mode for /health."""
    global bot_username
    if not config.TELEGRAM_BOT_TOKEN:
        return "disabled (no token)"
    try:
        bot_username = (await call("getMe"))["username"]
        if config.PUBLIC_BASE_URL:
            await call("setWebhook", json={"url": f"{config.PUBLIC_BASE_URL}/api/telegram/webhook",
                                           "secret_token": config.TELEGRAM_WEBHOOK_SECRET or None,
                                           "allowed_updates": ["message"]})
            return "webhook"
        await call("deleteWebhook")
        asyncio.create_task(_poll())
        return "polling"
    except Exception as exc:  # noqa: BLE001
        log.warning("telegram start failed: %s", exc)
        return f"error: {exc}"


async def _poll() -> None:
    offset = 0
    while True:
        try:
            updates = await call("getUpdates", json={"offset": offset, "timeout": 25, "allowed_updates": ["message"]})
            for u in updates:
                offset = u["update_id"] + 1
                asyncio.create_task(handle_update(u))
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram poll error: %s", exc)
            await asyncio.sleep(5)
