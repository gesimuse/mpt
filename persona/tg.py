"""Telegram for the persona bot. Plain requests, no Wan2GP, so cli.py can use it too.

Its own bot (PERSONA_BOT_TOKEN), for the same reason local/wan2gp_bot.py has one:
the mpt bot's updates go to the Cloudflare Worker by webhook, and Telegram refuses
getUpdates on a bot with a webhook. If PERSONA_BOT_TOKEN is unset it falls back to
LOCAL_BOT_TOKEN. That works because the two daemons cannot run together anyway
(one GPU, one 30GB of RAM), but each keeps its own update offset, so stop one
before starting the other.

Channels:
  PERSONA_CHAT_ID         review channel: casting, refs, social-lane scenes, commands
  PERSONA_FANVUE_CHAT_ID  Fanvue-lane scenes (default: PERSONA_CHAT_ID). Make it a
                          separate private channel if anyone else can see the first.
"""
import json
import time
from pathlib import Path

import requests

from . import config

API = "https://api.telegram.org"


class TelegramError(RuntimeError):
    pass


def token():
    t = config.env("PERSONA_BOT_TOKEN") or config.env("LOCAL_BOT_TOKEN")
    if not t:
        raise TelegramError("set PERSONA_BOT_TOKEN (a bot from @BotFather, admin in the persona channels)")
    if t == config.env("TELEGRAM_BOT_TOKEN"):
        raise TelegramError("PERSONA_BOT_TOKEN is the mpt bot's token -- its webhook belongs to the "
                            "Cloudflare Worker; use a separate bot")
    return t


def chat(lane="social"):
    review = config.env("PERSONA_CHAT_ID")
    if not review:
        raise TelegramError("set PERSONA_CHAT_ID (the private review channel)")
    return config.env("PERSONA_FANVUE_CHAT_ID") or review if lane == "fanvue" else review


def redact(text):
    text = str(text)
    try:
        t = token()
    except TelegramError:
        return text
    return text.replace(t, "<TOKEN>").replace(t.split(":", 1)[-1], "<TOKEN>")


def call(method, http_timeout=60, files=None, attempts=3, **params):
    """Retries network errors (an SSL EOF dropped one of the first cloud posts).
    Uploads are retried too: the file objects are rewound before each try."""
    for attempt in range(attempts):
        try:
            return _call_once(method, http_timeout, files, **params)
        except TelegramError as e:
            if "unreachable" not in str(e) or attempt == attempts - 1:
                raise
            for f in (files or {}).values():
                f.seek(0)
            time.sleep(3 * (attempt + 1))


def _call_once(method, http_timeout=60, files=None, **params):
    url = f"{API}/bot{token()}/{method}"
    try:
        if files:
            data = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                    for k, v in params.items() if v is not None}
            r = requests.post(url, data=data, files=files, timeout=http_timeout)
        else:
            r = requests.post(url, json={k: v for k, v in params.items() if v is not None},
                              timeout=http_timeout)
    except requests.RequestException as e:
        raise TelegramError(f"{method} unreachable: {redact(e)[:200]}") from None
    body = r.json()
    if not body.get("ok"):
        raise TelegramError(f"{method} rejected: {redact(body)[:300]}")
    return body["result"]


def text(chat_id, msg, reply_to=None, buttons=None):
    return call("sendMessage", chat_id=chat_id, text=msg[:4096], reply_to_message_id=reply_to,
                reply_markup=buttons, disable_web_page_preview=True)


def photo(chat_id, path, caption="", buttons=None, reply_to=None):
    with open(path, "rb") as f:
        return call("sendPhoto", http_timeout=120, files={"photo": f}, chat_id=chat_id,
                    caption=caption[:1024], reply_markup=buttons, reply_to_message_id=reply_to)


def video(chat_id, path, caption="", buttons=None, reply_to=None):
    with open(path, "rb") as f:
        try:
            return call("sendVideo", http_timeout=300, files={"video": f}, chat_id=chat_id,
                        caption=caption[:1024], reply_markup=buttons, supports_streaming=True,
                        reply_to_message_id=reply_to)
        except TelegramError:
            pass
    with open(path, "rb") as f:
        return call("sendDocument", http_timeout=300, files={"document": f}, chat_id=chat_id,
                    caption=caption[:1024], reply_markup=buttons, reply_to_message_id=reply_to)


def album(chat_id, paths, caption=""):
    paths = [Path(p) for p in paths][:10]
    files = {f"f{i}": open(p, "rb") for i, p in enumerate(paths)}
    try:
        media = [{"type": "photo", "media": f"attach://f{i}", **({"caption": caption[:1024]} if i == 0 else {})}
                 for i in range(len(paths))]
        return call("sendMediaGroup", http_timeout=180, files=files, chat_id=chat_id, media=media)
    finally:
        for f in files.values():
            f.close()


def answer(callback_id, msg=""):
    try:
        call("answerCallbackQuery", callback_query_id=callback_id, text=msg[:200])
    except TelegramError:
        pass   # a stale callback (>15 min) cannot be answered; the action still ran


def set_buttons(chat_id, message_id, buttons):
    try:
        call("editMessageReplyMarkup", chat_id=chat_id, message_id=message_id, reply_markup=buttons)
    except TelegramError:
        pass


def kb(*rows):
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows if row]}


def download(file_id, dest):
    """A file someone posted (or the bot posted) in a chat, by its file_id."""
    info = call("getFile", file_id=file_id)
    r = requests.get(f"{API}/file/bot{token()}/{info['file_path']}", timeout=120)
    r.raise_for_status()
    Path(dest).write_bytes(r.content)
    return Path(dest)
