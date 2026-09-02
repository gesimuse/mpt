#!/usr/bin/env python3
"""Post every outstanding image (and generated video) into the Telegram channel, so
the bot holds what picker.html used to show and the backlog can be worked from there.

Run once when switching from the page to the bot:

    python3 backfill_telegram.py            # report only, posts nothing
    python3 backfill_telegram.py --post     # actually post
    python3 backfill_telegram.py --post --prune-dead

"Outstanding" means the same thing the picker meant: an aibeauty image that has not
already been SUCCESSFULLY animated. A failed video attempt does not consume its source
image -- that is deliberate (autopilot._pick_source_image_url has the same rule), since
the point of a retriable failure is that the image is still fair game.

Two things this has to handle that a naive loop would get wrong:

DEAD LINKS. tiktok.KEEP_MEDIA prunes gh-pages by oldest-mtime, so roughly half of the
historical entries point at files that no longer exist. sendPhoto on a dead URL fails,
and posting 20 failures into a channel is worse than posting nothing -- so every URL is
HEAD-checked first. --prune-dead additionally drops those entries from posted.json,
which is real cleanup: the picker was rendering them as broken thumbnails.

RATE LIMITS. Telegram throttles channel posts hard (sustained, well under a message a
second). A burst gets 429s, and a 429 that is retried immediately just earns another.
Posts are spaced and any retry_after Telegram asks for is honoured exactly.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "posted.json"

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import telegram  # noqa: E402  (after .env is loaded, so enabled() is meaningful)

# Comfortably under Telegram's channel throughput. The whole backfill is a one-off, so
# there is nothing to gain by pushing this and a 429 storm to lose.
DELAY_S = float(os.environ.get("BACKFILL_DELAY_S", "3"))


def log(msg): print(f"[backfill] {msg}", flush=True)


def load():
    return json.loads(STATE.read_text())


def animated_urls(state):
    """Source images consumed by a SUCCESSFUL video, and therefore not outstanding."""
    return {u.get("motionforge_source_url") for u in state.get("uploads", [])
            if u.get("niche") == "aibeautyvideo" and u.get("tiktok")
            and u.get("motionforge_source_url")}


def outstanding(state):
    used = animated_urls(state)
    posted = set()
    for u in state.get("uploads", []):
        for mid_url in (u.get("telegram_posted") or []):
            posted.add(mid_url)
    out = []
    for u in state.get("uploads", []):
        if u.get("niche") != "aibeauty" or not u.get("tiktok"):
            continue
        for i, url in enumerate(u.get("image_urls") or []):
            if url in used or url in posted:
                continue
            prompts = u.get("motion_prompts") or []
            fallback = u.get("image_prompts") or []
            prompt = (prompts[i] if i < len(prompts) else None) \
                or (fallback[i] if i < len(fallback) else None)
            out.append({"ts": u["ts"], "index": i, "url": url, "prompt": prompt})
    return out


def videos(state):
    seen = {u.get("video_url") for u in state.get("uploads", [])
            if u.get("telegram_posted") and u.get("video_url")
            and u["video_url"] in u["telegram_posted"]}
    return [{"ts": u["ts"], "url": u["video_url"], "failed": not u.get("tiktok"),
             "title": u.get("title") or ""}
            for u in state.get("uploads", [])
            if u.get("video_url") and u["video_url"] not in seen]


def is_live(url):
    try:
        return requests.head(url, timeout=15, allow_redirects=False).status_code == 200
    except requests.RequestException:
        return False


def mark_posted(state, ts, url):
    for u in state.get("uploads", []):
        if u.get("ts") == ts:
            u.setdefault("telegram_posted", []).append(url)
            return


def send_with_backoff(fn, *args, **kwargs):
    """Honour Telegram's own retry_after rather than guessing. Retrying a 429
    immediately is what turns one throttle into a cascade of them."""
    for attempt in range(4):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg and attempt < 3:
                wait = 30
                for token in msg.replace('"', " ").replace(":", " ").split():
                    if token.isdigit() and 0 < int(token) <= 300:
                        wait = int(token)
                        break
                log(f"rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            raise
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true", help="actually post (default: report)")
    ap.add_argument("--prune-dead", action="store_true",
                    help="also drop entries whose files are gone from gh-pages")
    ap.add_argument("--limit", type=int, default=0, help="cap how many images to post")
    args = ap.parse_args()

    if args.post and not telegram.enabled():
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not configured")

    state = load()
    items = outstanding(state)
    vids = videos(state)
    log(f"{len(items)} outstanding image(s), {len(vids)} generated video(s)")

    log("checking which files still exist on gh-pages...")
    live = [i for i in items if is_live(i["url"])]
    dead = [i for i in items if i not in live]
    live_v = [v for v in vids if is_live(v["url"])]
    log(f"images: {len(live)} live, {len(dead)} pruned")
    log(f"videos: {len(live_v)} live, {len(vids) - len(live_v)} pruned")

    if args.limit:
        live = live[:args.limit]

    if not args.post:
        log("dry run -- nothing posted. Re-run with --post.")
        return

    sent = 0
    for item in live:
        try:
            send_with_backoff(telegram.send_image, item["url"], item["ts"], item["index"],
                              motion_prompt=item["prompt"])
            mark_posted(state, item["ts"], item["url"])
            sent += 1
            STATE.write_text(json.dumps(state, indent=2))  # checkpoint every message
        except Exception as e:
            log(f"could not post {item['url'][-28:]} ({type(e).__name__}: {str(e)[:120]})")
        time.sleep(DELAY_S)
    log(f"posted {sent}/{len(live)} images")

    for v in live_v:
        try:
            send_with_backoff(telegram.send_video, v["url"], v["ts"],
                              caption=v["title"], failed=v["failed"])
            mark_posted(state, v["ts"], v["url"])
            STATE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            log(f"could not post video ({type(e).__name__}: {str(e)[:120]})")
        time.sleep(DELAY_S)

    if args.prune_dead and dead:
        gone = {d["url"] for d in dead}
        for u in state.get("uploads", []):
            urls = u.get("image_urls") or []
            for i in range(len(urls) - 1, -1, -1):
                if urls[i] in gone:
                    urls.pop(i)
                    for field in ("image_prompts", "motion_prompts"):
                        if u.get(field) and i < len(u[field]):
                            u[field].pop(i)
        state["uploads"] = [u for u in state["uploads"]
                            if u.get("image_urls") or not u.get("image_urls") == []]
        STATE.write_text(json.dumps(state, indent=2))
        log(f"pruned {len(gone)} dead image reference(s) from posted.json")

    log("done -- commit and push posted.json so the buttons resolve against it")


if __name__ == "__main__":
    main()
