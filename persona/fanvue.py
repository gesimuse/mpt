"""Fanvue posting, from this machine.

The Python twin of worker/src/fanvue.js, with the same caveat: UNVERIFIED AGAINST A
LIVE ACCOUNT. As of 2026-09-08, Fanvue's creator API keys are waitlisted
(help.fanvue.com, "Creator settings: request API keys"). See fanvue.js's
docstring for the documented upload flow and the one path most likely to need
fixing (START_UPLOAD_PATH):
  1. POST START_UPLOAD_PATH           -> {mediaUuid, uploadId, uploadUrl}
  2. PUT  <uploadUrl>, raw bytes       -> keep the ETag
  3. POST /posts {image|video: {...}, text, audience}

Without FANVUE_API_TOKEN, nothing fails: the file and its caption go into
PERSONA_HOME/<slug>/outbox/fanvue/ for a manual upload in Fanvue's web app. Once a
token arrives, the same button posts directly.

The Fanvue lane's images are never hosted anywhere in between. Bytes go from this
disk to Fanvue's signed upload URL, and nowhere else.

Fanvue rules to keep in mind (check their current creator terms; they change):
AI-generated content must be disclosed as AI on the account and posts, and the
account itself is verified with YOUR identity as the creator.
"""
import mimetypes
import shutil
import time
from pathlib import Path

import requests

from . import config

API = "https://api.fanvue.com"
API_VERSION = "2025-06-26"
START_UPLOAD_PATH = "/media/images"   # best guess, see worker/src/fanvue.js


class FanvueError(RuntimeError):
    pass


def enabled():
    return bool(config.env("FANVUE_API_TOKEN"))


def _headers(extra=None):
    return {"Authorization": f"Bearer {config.env('FANVUE_API_TOKEN')}",
            "X-Fanvue-API-Version": API_VERSION, **(extra or {})}


def _post_api(path, media_path, caption):
    media_path = Path(media_path)
    ctype = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
    r = requests.post(f"{API}{START_UPLOAD_PATH}", headers=_headers({"content-type": "application/json"}),
                      timeout=60)
    if not r.ok:
        raise FanvueError(f"start-upload failed: {r.status_code} {r.text[:300]}")
    slot = r.json()
    if not all(slot.get(k) for k in ("mediaUuid", "uploadId", "uploadUrl")):
        raise FanvueError(f"start-upload: unexpected response {str(slot)[:200]}")
    # No Authorization header: uploadUrl is a signed URL on a storage host, and the
    # bearer token must not be sent to a third party.
    with open(media_path, "rb") as f:
        put = requests.put(slot["uploadUrl"], data=f, headers={"content-type": ctype}, timeout=600)
    if not put.ok:
        raise FanvueError(f"upload PUT failed: {put.status_code} {put.text[:300]}")
    etag = put.headers.get("etag")
    if not etag:
        raise FanvueError("upload PUT returned no ETag")
    kind = "video" if ctype.startswith("video/") else "image"
    body = {"text": (caption or "")[:5000], "audience": config.env("FANVUE_AUDIENCE", "subscribers"),
            kind: {"mediaUuid": slot["mediaUuid"], "uploadId": slot["uploadId"], "etag": etag}}
    r = requests.post(f"{API}/posts", headers=_headers({"content-type": "application/json"}),
                      json=body, timeout=60)
    if not r.ok:
        raise FanvueError(f"create-post failed: {r.status_code} {r.text[:300]}")
    return r.json()


def to_outbox(char, media_path, caption, item_id):
    out = char.dir / "outbox" / "fanvue"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    media_path = Path(media_path)
    dest = out / f"{stamp}-{item_id}{media_path.suffix}"
    shutil.copyfile(media_path, dest)
    dest.with_suffix(".txt").write_text((caption or "").strip() + "\n")
    return dest


def post(char, media_path, caption, item_id):
    """Post to Fanvue if a token is configured, otherwise drop it in the outbox.
    Returns ("posted", response) or ("outbox", path)."""
    if enabled():
        return "posted", _post_api(media_path, caption)
    return "outbox", to_outbox(char, media_path, caption, item_id)
