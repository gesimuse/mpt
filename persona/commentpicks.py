"""Daily comment picks (2026-11-28): fresh videos from big dating/relationship TikTok accounts
that are pulling comments fast, sent to the Worker, which writes one short funny comment for
each in a girl's voice and posts them in MPT's channel. The user posts them by hand from the
TikTok app -- no bot ever comments (TikTok's rules on automated activity).

    python -m persona.commentpicks        # once, now
"""
import json
import time
import urllib.request

from . import config, ttstats

# Active, busy accounts (2026-11-28 check: median views / comments per video).
ACCOUNTS = ["drewafualo", "jayshetty", "melrobbins", "jessicaos"]


def due(stamp):
    """Once a day, after 08:00 local."""
    try:
        last = float(stamp.read_text())
    except (OSError, ValueError):
        last = 0
    now = time.localtime()
    return now.tm_hour >= 8 and time.strftime("%Y%m%d", time.localtime(last)) != time.strftime("%Y%m%d", now)


def picks(n=3, max_age_h=72):
    """The n videos getting comments fastest, posted in the last 3 days: early enough
    that a new comment can still be seen near the top."""
    now, cands = time.time(), []
    for h in config.env("PERSONA_COMMENT_ACCOUNTS", ",".join(ACCOUNTS)).split(","):
        for v in ttstats.videos(h.strip()):
            age = (now - v["ts"]) / 3600
            if 1 <= age <= max_age_h and v["comments_n"] >= 20:
                cands.append({**v, "account": h.strip(), "age_h": round(age, 1),
                              "url": f"https://www.tiktok.com/@{h.strip()}/video/{v['id']}",
                              "pace": v["comments_n"] / age})
    cands.sort(key=lambda c: -c["pace"])
    out, seen = [], set()
    for c in cands:  # one per account
        if c["account"] not in seen:
            seen.add(c["account"])
            out.append(c)
    return out[:n]


def run(log=print):
    vids = picks()
    if not vids:
        log("comment picks: no fresh busy videos")
        return
    body = {"slug": "mpt", "videos": [{k: v[k] for k in ("url", "account", "title", "views", "comments_n", "age_h")} for v in vids]}
    req = urllib.request.Request(config.env("PERSONA_WORKER_URL").rstrip("/") + "/persona/admin/laptop/commentpicks",
                                 method="POST", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {config.env('PERSONA_ADMIN_SECRET')}",
                                          "Content-Type": "application/json", "User-Agent": "mpt-persona-agent/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        log(f"comment picks: {len(vids)} sent -> {r.read()[:200]!r}")


if __name__ == "__main__":
    config.load_env()
    run()
