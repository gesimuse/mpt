"""TikTok drafts and the Instagram outbox for the SOCIAL lane only.

TikTok: reuses tiktok.py, i.e. the same gh-pages hosting + MEDIA_UPLOAD inbox
draft as the aibeauty account, authenticated as the persona's OWN TikTok account:
TIKTOK_REFRESH_TOKEN_<PERSONA_TIKTOK_ACCOUNT> (run get_tiktok_token.py while
logged into her account and save it under that name). Photos need at least two
per carousel, so 📱 on a photo queues it, and the draft goes out once
PERSONA_TIKTOK_BATCH (default 3) are queued. A video goes out on its own.

gh-pages is public, which is fine for the social lane (it is going on TikTok
anyway) and is exactly why a Fanvue-lane item can never reach this module:
push() refuses anything not marked lane=social.

Instagram has no draft API: approved items are copied to outbox/instagram/ with
their caption, ready to upload from the phone or Meta Business Suite.

Both platforms want AI content labelled: TikTok's "AI-generated" toggle in the
draft, Instagram's "AI info" label.
"""
import json
import shutil
import sys
import time
from pathlib import Path

from . import config

if str(config.REPO) not in sys.path:
    sys.path.insert(0, str(config.REPO))


class SocialError(RuntimeError):
    pass


def account():
    return config.env("PERSONA_TIKTOK_ACCOUNT") or ""


def _queue_path(char):
    return char.dir / "tiktok_queue.json"


def _guard(item):
    if item.get("lane") != "social":
        raise SocialError("only social-lane items can go to TikTok or Instagram")


def caption_for(char, item):
    tags = " ".join("#" + t.replace(" ", "") for t in item.get("tags", [])[:4])
    base = item.get("caption") or ""
    return f"{base}\n\n{tags} #aigenerated".strip()


def tiktok(char, item, media_path):
    """Returns a human-readable status line."""
    _guard(item)
    import tiktok as tt
    acct = account()
    if not acct or not tt.enabled(acct):
        raise SocialError("no TikTok account for the persona: set PERSONA_TIKTOK_ACCOUNT=<name> and "
                          "TIKTOK_REFRESH_TOKEN_<NAME> (get_tiktok_token.py, logged into her account)")
    media_path = Path(media_path)
    if media_path.suffix.lower() == ".mp4":
        pid = tt.publish_video_draft(str(media_path), acct, caption=caption_for(char, item))
        return f"TikTok video draft queued ({pid})"
    queue = json.loads(_queue_path(char).read_text()) if _queue_path(char).exists() else []
    if item["id"] not in [q["id"] for q in queue]:
        queue.append({"id": item["id"], "path": str(media_path)})
    batch = int(config.env("PERSONA_TIKTOK_BATCH", "3"))
    if len(queue) < batch:
        _queue_path(char).write_text(json.dumps(queue))
        return f"queued for a TikTok carousel ({len(queue)}/{batch})"
    first = char.item(queue[0]["id"]) or item
    pid = tt.publish_photos_draft([q["path"] for q in queue], acct, caption=caption_for(char, first))
    _queue_path(char).write_text("[]")
    return f"TikTok carousel draft of {len(queue)} queued ({pid})"


def instagram(char, item, media_path):
    _guard(item)
    out = char.dir / "outbox" / "instagram"
    out.mkdir(parents=True, exist_ok=True)
    media_path = Path(media_path)
    dest = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{item['id']}{media_path.suffix}"
    shutil.copyfile(media_path, dest)
    dest.with_suffix(".txt").write_text(caption_for(char, item) + "\n")
    return f"copied to Instagram outbox: {dest.name}"
