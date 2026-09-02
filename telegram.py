"""Post generated media to a Telegram channel with inline action buttons.

The review surface, replacing the picker page that used to live on gh-pages. That
branch is NOT going anywhere regardless: it hosts the image and video files TikTok's
PULL_FROM_URL fetches, and that a Space fetches when animating a still, so it is
load-bearing for publishing rather than just for a UI.

Division of labour, which is the whole design:
  * This module (running inside a GitHub Actions job) SENDS. One message per generated
    image, each carrying its own buttons.
  * A Cloudflare Worker (worker/) RECEIVES the button presses over Telegram's webhook
    and acts on them immediately -- editing or deleting the message, dispatching the
    video workflow, and writing posted.json through GitHub's Contents API.

That split exists because there is no always-on server here. A button press has to
reach something within seconds, and a cron-driven job cannot do that.

An image that is neither skipped nor sent to video generation simply stays in the
channel. The channel IS the backlog; nothing expires it.

Two channels, not one. Images accumulate as a working backlog and videos accumulate
as output; interleaving them in a single channel makes both unreadable once there are
more than a handful of each.

  TELEGRAM_CHAT_ID        outstanding photos -- the backlog you work from
  TELEGRAM_VIDEO_CHAT_ID  generated videos -- output, with download/retry
                          (falls back to TELEGRAM_CHAT_ID when unset, so a
                          single-channel setup keeps working unchanged)

Env:
  TELEGRAM_BOT_TOKEN      from @BotFather
  TELEGRAM_CHAT_ID        e.g. -1001234567890
  TELEGRAM_VIDEO_CHAT_ID  optional second channel
"""
import os

import requests

API = "https://api.telegram.org"
TIMEOUT = int(os.environ.get("TELEGRAM_TIMEOUT", "30"))


def log(msg): print(f"[telegram] {msg}", flush=True)


def enabled():
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")
                and os.environ.get("TELEGRAM_CHAT_ID"))


def _call(method, payload):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    r = requests.post(f"{API}/bot{token}/{method}", json=payload, timeout=TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"telegram {method} failed: {r.status_code} {r.text[:300]}")
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"telegram {method} rejected: {str(body)[:300]}")
    return body["result"]


# Callback payloads are capped at 64 BYTES by Telegram, which is far too little for a
# URL. So the button carries only what identifies the image -- the batch timestamp and
# the index within that batch -- and the Worker looks the URL up in posted.json. That
# also means a button stays correct if the file is later re-hosted.
def _callback(action, ts, index):
    return f"{action}|{ts}|{index}"


def video_chat_id():
    """The videos channel, or the photos one when no separate channel is configured."""
    return (os.environ.get("TELEGRAM_VIDEO_CHAT_ID", "").strip()
            or os.environ.get("TELEGRAM_CHAT_ID", "").strip())


def _image_keyboard(ts, index):
    """Three actions, and the important one is what "Make video" does NOT do.

    Making a video leaves the image in the channel with its buttons intact, because
    one still is worth several attempts -- a different motion prompt on the same photo
    is a normal thing to want, and an image that vanished the moment it was used made
    that impossible. Only Done and Skip remove it.

    Done and Skip both remove, and differ in what they record: Done means "used this,
    finished with it", Skip means "didn't want it". The Worker writes that to
    owner_verdict, which is exactly the signal imageslides._owner_theme_rates biases
    theme and subject selection with -- so the learning loop comes back for free, off
    buttons that had to exist anyway."""
    return {"inline_keyboard": [[
        {"text": "🎬 Make video", "callback_data": _callback("vid", ts, index)},
    ], [
        {"text": "✅ Done", "callback_data": _callback("done", ts, index)},
        {"text": "🗑 Skip", "callback_data": _callback("skip", ts, index)},
    ]]}


def send_image(image_url, ts, index, caption=None, motion_prompt=None, chat_id=None):
    """One reviewable image in the channel. Returns Telegram's message_id.

    The motion prompt is shown in the caption rather than hidden: it is what the video
    model would actually be told to do, and the point of showing it is that the owner
    can reply to this message with a different one before pressing Make video."""
    lines = [caption] if caption else []
    if motion_prompt:
        lines.append(f"🎬 {motion_prompt}")
    lines.append("Reply with a different prompt to make another video from this "
                 "photo. Done or Skip removes it.")
    return _call("sendPhoto", {
        "chat_id": chat_id or os.environ["TELEGRAM_CHAT_ID"],
        "photo": image_url,
        "caption": "\n\n".join(lines)[:1024],
        "reply_markup": _image_keyboard(ts, index),
    })["message_id"]


def _video_keyboard(video_url, ts, failed):
    row = [{"text": "⬇ Download", "url": video_url}]
    if failed:
        # Republishes the SAME mp4 rather than regenerating -- a TikTok-side rejection
        # (frame rate, fetch failure) says nothing about the video itself, and
        # regenerating would burn ZeroGPU quota to produce an equivalent file.
        row.append({"text": "🔄 Retry post", "callback_data": _callback("retry", ts, 0)})
    return {"inline_keyboard": [row]}


def send_video(video_url, ts, caption=None, failed=False, chat_id=None):
    """A generated clip, with a download link and -- when TikTok rejected it -- a
    retry button. Sent as a document, not a video: Telegram re-encodes videos for
    streaming, and the whole point of the download button is to get the exact file
    that was hosted and posted."""
    return _call("sendDocument", {
        "chat_id": chat_id or video_chat_id(),
        "document": video_url,
        "caption": (caption or "")[:1024],
        "reply_markup": _video_keyboard(video_url, ts, failed),
    })["message_id"]


def send_text(text, chat_id=None):
    return _call("sendMessage", {
        "chat_id": chat_id or os.environ["TELEGRAM_CHAT_ID"],
        "text": text[:4096],
        "disable_web_page_preview": True,
    })["message_id"]


def post_batch(image_urls, ts, caption=None, motion_prompts=None, chat_id=None):
    """Every image from one batch, newest batch last. Never raises: Telegram being
    unreachable must not fail a run whose images are already hosted and whose TikTok
    draft is already queued -- the review surface is downstream of all of that.
    Returns the message ids it managed to send."""
    ids = []
    for i, url in enumerate(image_urls or []):
        try:
            prompt = (motion_prompts or [None] * len(image_urls))[i]
        except IndexError:
            prompt = None
        try:
            ids.append(send_image(url, ts, i, caption=caption if i == 0 else None,
                                  motion_prompt=prompt, chat_id=chat_id))
        except Exception as e:
            log(f"could not post image {i} ({type(e).__name__}: {str(e)[:150]})")
    log(f"posted {len(ids)}/{len(image_urls or [])} images to the channel")
    return ids
