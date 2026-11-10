"""Her TikTok account's numbers and real viewers' comments, sent to the Worker every few
hours (POST /persona/admin/laptop/stats -> laptopStats in worker/src/persona/index.js),
which turns them into a Part 2 of the best talk, her answer to a comment, and a daily
report in the channel (2026-11-10). yt-dlp only, free: TikTok's listing carries every
video's views/likes/comments/shares/saves; TikTok comments themselves can't be read that
way, so the comments come from viral YouTube Shorts on the girls' topics."""
import json
import random
import subprocess
import sys
import time

from . import config
from .talk import _worker

EVERY_S = 6 * 3600
QUESTIONS = ["why men pull away relationship advice", "how to get over him breakup advice", "red flags in a man dating advice",
             "situationship advice for women", "marriage truths wives", "glow up confidence women advice",
             "who should pay on a date", "signs he is not serious about you"]


def _json(args, timeout=180):
    out = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", *args], capture_output=True, text=True, timeout=timeout)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else {}


def videos(handle):
    d = _json(["--flat-playlist", "--playlist-end", "60", "-J", f"https://www.tiktok.com/@{handle}"])
    out = []
    for e in d.get("entries") or []:
        if not e or not e.get("id"):
            continue
        out.append({"id": e["id"], "ts": int(e["id"]) >> 32,  # a TikTok id starts with its unix time
                    "views": e.get("view_count") or 0, "likes": e.get("like_count") or 0,
                    "comments_n": e.get("comment_count") or 0, "shares": e.get("repost_count") or 0,
                    "saves": e.get("save_count") or 0, "title": (e.get("description") or e.get("title") or "")[:300]})
    return out


def comments(n_shorts=2):
    """Top comments of viral YouTube Shorts on the girls' topics: [{cid, text, likes}]."""
    found = []
    for q in random.sample(QUESTIONS, 2):
        for e in (_json(["--flat-playlist", "--playlist-end", str(n_shorts), "-J", f"ytsearch{n_shorts}:{q} #shorts"]).get("entries") or [])[:n_shorts]:
            d = _json(["--skip-download", "--write-comments", "--extractor-args", "youtube:max_comments=40,all,0;comment_sort=top",
                       "-J", f"https://www.youtube.com/shorts/{e['id']}"], timeout=200)
            for c in d.get("comments") or []:
                found.append({"cid": f"yt:{c.get('id')}", "text": c.get("text") or "", "likes": c.get("like_count") or 0})
    return found


def run(log):
    handle = config.env("PERSONA_TT_HANDLE") or "aibeatuy"
    vids = videos(handle)
    if not vids:
        log(f"stats: no videos read from @{handle}")
        return
    try:
        cs = comments()
    except Exception as e:
        log(f"stats: comments failed ({e})")
        cs = []
    r = _worker("/persona/admin/laptop/stats", {"videos": vids, "comments": cs})
    log(f"stats: {len(vids)} videos, {len(cs)} comments -> {r}")


def due(stamp):
    try:
        return time.time() - float(stamp.read_text()) >= EVERY_S
    except Exception:
        return True
