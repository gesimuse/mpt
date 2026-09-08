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
  TELEGRAM_FANVUE_CHAT_ID the Fanvue approve/disapprove queue (see
                          send_fanvue_photo) -- also falls back to
                          TELEGRAM_CHAT_ID when unset

Env:
  TELEGRAM_BOT_TOKEN      from @BotFather
  TELEGRAM_CHAT_ID        e.g. -1001234567890
  TELEGRAM_VIDEO_CHAT_ID  optional second channel
  TELEGRAM_FANVUE_CHAT_ID optional third channel
"""
import json
import os

import requests

API = "https://api.telegram.org"
TIMEOUT = int(os.environ.get("TELEGRAM_TIMEOUT", "30"))


def redact(text):
    """Strip the bot token out of anything about to be logged or raised.

    Telegram has no header auth -- the token is a path segment, so every request URL
    contains it. requests puts that URL into its own exception messages verbatim
    ("Max retries exceeded with url: /bot<TOKEN>/sendPhoto"), and callers log
    str(e) into GitHub Actions logs, which are PUBLIC on this repo. That is a real
    credential disclosure from an ordinary network blip, not a hypothetical.

    Redacting at the boundary rather than asking every call site to remember: the
    ones that forgot are exactly the ones that would leak."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    text = str(text)
    if token:
        text = text.replace(token, "<TELEGRAM_BOT_TOKEN>")
        # The numeric bot id before the colon is not secret, but the secret half alone
        # is still enough to act as the bot, so scrub it even if the full token was
        # split or truncated across a log boundary.
        secret = token.split(":", 1)[-1]
        if len(secret) > 8:
            text = text.replace(secret, "<TELEGRAM_BOT_TOKEN>")
    return text


def log(msg): print(f"[telegram] {redact(msg)}", flush=True)


def enabled():
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")
                and os.environ.get("TELEGRAM_CHAT_ID"))


def _call(method, payload):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    try:
        r = requests.post(f"{API}/bot{token}/{method}", json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        # Re-raised redacted, because requests embeds the full request URL -- token
        # included -- in its own message, and every caller logs that.
        raise RuntimeError(f"telegram {method} unreachable: {redact(e)[:300]}") from None
    if not r.ok:
        raise RuntimeError(
            f"telegram {method} failed: {r.status_code} {redact(r.text)[:300]}")
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"telegram {method} rejected: {redact(body)[:300]}")
    return body["result"]


def _call_multipart(method, payload, file_field, file_path):
    """Same as _call, but uploads a LOCAL file's raw bytes instead of passing a URL
    in the JSON body -- what send_fanvue_photo() needs: Fanvue-track images must
    never touch gh-pages/GitHub at all (an explicit requirement, not an optimization),
    so there is no URL to hand Telegram in the first place, only a local temp file."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    try:
        with open(file_path, "rb") as f:
            r = requests.post(f"{API}/bot{token}/{method}", data=payload,
                             files={file_field: f}, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise RuntimeError(f"telegram {method} unreachable: {redact(e)[:300]}") from None
    if not r.ok:
        raise RuntimeError(
            f"telegram {method} failed: {r.status_code} {redact(r.text)[:300]}")
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"telegram {method} rejected: {redact(body)[:300]}")
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
    """Four actions, and the important one is what "Make video" does NOT do.

    Making a video leaves the image in the channel with its buttons intact, because
    one still is worth several attempts -- a different motion prompt on the same photo
    is a normal thing to want, and an image that vanished the moment it was used made
    that impossible. Only Done, Skip, Good and Bad remove it.

    Done and Skip both remove, and differ in what they record: Done means "used this,
    finished with it", Skip means "didn't want it". The Worker writes that to
    owner_verdict, which is exactly the signal imageslides._owner_theme_rates biases
    theme and subject selection with -- so the learning loop comes back for free, off
    buttons that had to exist anyway.

    Good/Bad are a separate rating, of the CHECKPOINT + SD PROMPT that produced the
    image (send_image's model_name/image_prompt), not of whether the post got used --
    a photo can be a bad TikTok fit (wrong vibe, Skip) while still being exactly what
    that model+prompt combo reliably renders (Good), and the two must not be
    conflated. The Worker writes both to model_leaderboard.json: Good is +1 to the
    model's score and to that prompt's own score under it, Bad is -1 to both -- a net
    score, so a combo that keeps getting rated bad actually sorts below an unrated
    one instead of both reading as 0."""
    return {"inline_keyboard": [[
        {"text": "🎬 Make video", "callback_data": _callback("vid", ts, index)},
    ], [
        {"text": "✅ Done", "callback_data": _callback("done", ts, index)},
        {"text": "🗑 Skip", "callback_data": _callback("skip", ts, index)},
    ], [
        {"text": "👍 Good", "callback_data": _callback("good", ts, index)},
        {"text": "👎 Not good", "callback_data": _callback("bad", ts, index)},
    ]]}


def send_image(image_url, ts, index, caption=None, motion_prompt=None,
               model_name=None, image_prompt=None, chat_id=None):
    """One reviewable image in the channel. Returns Telegram's message_id.

    model_name and image_prompt (the checkpoint and the actual SD prompt that
    rendered this specific photo) are shown so Good/Bad is a rating of something the
    owner can actually read, not a blind button. The motion prompt is shown for the
    same reason as always: it is what the video model would actually be told to do,
    and the owner can reply with a different one before pressing Make video."""
    lines = [caption] if caption else []
    if model_name:
        lines.append(f"🧪 {model_name}")
    if image_prompt:
        lines.append(f"📝 {image_prompt}")
    if motion_prompt:
        lines.append(f"🎬 {motion_prompt}")
    lines.append("👍/👎 rates this model + prompt. Reply with a different prompt to "
                 "make another video from this photo. Done or Skip removes it.")
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
    # The video is already hosted on gh-pages (TikTok's PULL_FROM_URL requires it) --
    # unlike the Fanvue image queue, there is no "must not touch GitHub" constraint on
    # video, so this button just hands the Worker the same URL Retry post already
    # uses. One click; posts the exact file already generated, no regeneration.
    row.append({"text": "📮 Post to Fanvue", "callback_data": _callback("fvvid", ts, 0)})
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


def fanvue_chat_id():
    """The Fanvue review queue's own channel, or the photos one when unset -- same
    fallback shape as video_chat_id(), so a single-channel setup keeps working."""
    return (os.environ.get("TELEGRAM_FANVUE_CHAT_ID", "").strip()
            or os.environ.get("TELEGRAM_CHAT_ID", "").strip())


def _fanvue_keyboard():
    """Approve/Disapprove only -- no Make video, no Good/Bad. Deliberately carries no
    ts/index: unlike every other button here, this image was never hosted anywhere
    (send_fanvue_photo uploads local bytes directly, no gh-pages URL exists to look
    up), so there is nothing in posted.json for a callback to reference. The Worker
    reads the photo straight off cq.message.photo instead -- the message IS the only
    copy of this image that exists anywhere."""
    return {"inline_keyboard": [[
        {"text": "✅ Approve -> post to Fanvue", "callback_data": "fvpost"},
        {"text": "🗑 Disapprove", "callback_data": "fvskip"},
    ]]}


def send_fanvue_photo(image_path, prompt=None, model_name=None, chat_id=None):
    """One Fanvue-queue candidate, uploaded as raw bytes (never a gh-pages URL -- see
    _fanvue_keyboard's docstring: this image must never touch GitHub at all). The
    "💜 FANVUE" header is the visual distinction from the TikTok queue's photos the
    account owner asked for -- same channel by default (fanvue_chat_id() falls back to
    TELEGRAM_CHAT_ID), different-looking message, different buttons.

    Approve triggers the Worker's onFanvuePost, which downloads this exact message's
    photo from Telegram, posts it to Fanvue, and ONLY THEN deletes the message -- so a
    failed Fanvue post leaves the candidate in the channel to retry, never silently
    drops it."""
    lines = ["💜 FANVUE -- pending approval"]
    if model_name:
        lines.append(f"🧪 {model_name}")
    if prompt:
        lines.append(f"📝 {prompt}")
    return _call_multipart("sendPhoto", {
        "chat_id": chat_id or fanvue_chat_id(),
        "caption": "\n\n".join(lines)[:1024],
        "reply_markup": json.dumps(_fanvue_keyboard()),
    }, "photo", image_path)["message_id"]


def post_fanvue_batch(image_paths, prompts=None, model_name=None, chat_id=None):
    """Every image from one Fanvue-track batch. Never raises, same reasoning as
    post_batch(): one failed upload must not cost the rest of an already-generated
    batch. Returns the message ids it managed to send."""
    ids = []
    for i, path in enumerate(image_paths or []):
        try:
            prompt = (prompts or [None] * len(image_paths))[i]
        except IndexError:
            prompt = None
        try:
            ids.append(send_fanvue_photo(path, prompt=prompt, model_name=model_name,
                                         chat_id=chat_id))
        except Exception as e:
            log(f"could not post fanvue image {i} "
                f"({type(e).__name__}: {redact(e)[:150]})")
    log(f"posted {len(ids)}/{len(image_paths or [])} images to the fanvue queue")
    return ids


def send_text(text, chat_id=None):
    return _call("sendMessage", {
        "chat_id": chat_id or os.environ["TELEGRAM_CHAT_ID"],
        "text": text[:4096],
        "disable_web_page_preview": True,
    })["message_id"]


def post_batch(image_urls, ts, caption=None, motion_prompts=None, image_prompts=None,
               model_name=None, chat_id=None):
    """Every image from one batch, newest batch last. Never raises: Telegram being
    unreachable must not fail a run whose images are already hosted and whose TikTok
    draft is already queued -- the review surface is downstream of all of that.
    Returns the message ids it managed to send.

    model_name is the one checkpoint the whole batch was generated from (imageslides.
    generate()'s model_info); image_prompts is the per-image SD prompt, same order as
    image_urls. Both shown on every image so Good/Bad has something to rate."""
    ids = []
    for i, url in enumerate(image_urls or []):
        try:
            prompt = (motion_prompts or [None] * len(image_urls))[i]
        except IndexError:
            prompt = None
        try:
            img_prompt = (image_prompts or [None] * len(image_urls))[i]
        except IndexError:
            img_prompt = None
        try:
            ids.append(send_image(url, ts, i, caption=caption if i == 0 else None,
                                  motion_prompt=prompt, model_name=model_name,
                                  image_prompt=img_prompt, chat_id=chat_id))
        except Exception as e:
            log(f"could not post image {i} ({type(e).__name__}: {redact(e)[:150]})")
    log(f"posted {len(ids)}/{len(image_urls or [])} images to the channel")
    return ids
