"""Post an aibeauty carousel as a native TikTok inbox draft -- no Buffer involved.

TikTok's Content Posting API, in MEDIA_UPLOAD mode, drops the carousel into the
account's own TikTok app inbox as an unpublished draft: no sound, but the caption
IS pre-filled (checked live against TikTok's current docs -- MEDIA_UPLOAD now accepts
post_info the same as DIRECT_POST, which this module's original design predated and
assumed it didn't). The account owner opens the app, adds trending audio, and posts
by hand -- that's the one manual step left, not the caption too. This is the
deliberate tradeoff of an unaudited app: TikTok only grants video.upload/video.publish
without a review, direct posting is SELF_ONLY-only even with that scope, and the
inbox draft is the one path that ends in a real public post without waiting on an
app audit.

Photos only accept PULL_FROM_URL as their source, not FILE_UPLOAD (raw-byte upload,
the way video posting works) -- this is a real TikTok API constraint, not a choice
made here, and it also matches how Buffer's own carousel implementation only ever
took image URLs, never raw files, for the same underlying endpoint. PULL_FROM_URL
requires the fetch domain to be verified for this TikTok app under
developers.tiktok.com -> Content Posting API -> URL properties -- a one-time manual
step; without it, every call here fails with a domain-verification error TikTok's own
API returns, not something this code can detect in advance.

Verified live end to end against the real aibeauty account: status/fetch on a real
publish_id returned SEND_TO_USER_INBOX, not just a 200 from the init call (which only
means the job was accepted, not that it succeeded -- two real failures,
photo_pull_failed and file_format_check_failed, both only showed up at that later,
async stage). Two things that broke it before this passed: GitHub Pages build/deploy
latency racing TikTok's fetch (fixed by _wait_until_live below), and TikTok's photo
endpoint rejecting PNG (fixed in sdgen.py -- JPEG only, see its own comment).

Images are hosted on GitHub Pages (the repo's gh-pages branch, media/ folder), not
GitHub Releases -- a live check found release-asset download URLs 302-redirect to a
signed, ~1-hour-expiring release-assets.githubusercontent.com URL, and TikTok's own
docs explicitly disallow PULL_FROM_URL redirecting. Pages serves files directly, no
redirect, stable URL, confirmed live. The gh-pages branch is kept as a git worktree
(orphan branch, shares objects with the main checkout) specifically so the noisy
per-image commits stay out of main's source history.

Env:
  TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET   from the TikTok developer app
  TIKTOK_REFRESH_TOKEN_<NICHEID>             per-account OAuth refresh token
  PAGES_BASE_URL                             e.g. https://gesimuse.github.io/mpt
"""
import os, subprocess, time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent
PAGES_WORKTREE = REPO_ROOT / ".gh-pages-worktree"
PAGES_BRANCH = "gh-pages"
# picker.html's own display window alone needs 4 batches x 5 images (20) + 10
# videos (10) = 30 files alive at once with zero slack -- a live run showed
# exactly this: files still referenced by picker's "last 4 batches" got pruned
# out from under it (404s in the grid) once a day's actual run volume (several
# image batches + video attempts + the odd manual "+upload your own") pushed
# past the old cap of 30. This is still a blunt oldest-by-mtime cutoff, not
# reference-aware -- raised well past picker's own minimum need for headroom.
KEEP_MEDIA = 60


def log(msg): print(f"[tiktok] {msg}", flush=True)


def enabled(niche_id):
    ck = os.environ.get("TIKTOK_CLIENT_KEY")
    cs = os.environ.get("TIKTOK_CLIENT_SECRET")
    refresh = os.environ.get(f"TIKTOK_REFRESH_TOKEN_{niche_id.upper()}")
    return bool(ck and cs and refresh)


# ---------- hosting (GitHub Pages, so PULL_FROM_URL has a stable, non-redirecting URL) ----------
def _git(*args, cwd):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr[:300]}")
    return r.stdout


def _pages_worktree():
    """A local checkout of gh-pages, used purely as static asset storage -- created
    once and reused (a worktree, not a second clone, so it shares objects/history
    with the main checkout instead of a second full network fetch)."""
    if PAGES_WORKTREE.exists():
        return PAGES_WORKTREE
    _git("fetch", "origin", PAGES_BRANCH, cwd=REPO_ROOT)
    _git("worktree", "add", str(PAGES_WORKTREE), PAGES_BRANCH, cwd=REPO_ROOT)
    _git("config", "user.name", "autopilot-bot", cwd=PAGES_WORKTREE)
    _git("config", "user.email", "actions@github.com", cwd=PAGES_WORKTREE)
    return PAGES_WORKTREE


def _prune_media(worktree, keep=None):
    # keep's default is looked up here, not bound as a default argument -- a default
    # of keep=KEEP_MEDIA would capture that value at function-definition time, so
    # patching the module-level KEEP_MEDIA later (tests, or a future env var override)
    # would silently have no effect. Caught by this file's own test suite.
    keep = KEEP_MEDIA if keep is None else keep
    media_dir = worktree / "media"
    files = sorted(media_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    stale = files[:-keep] if len(files) > keep else []
    for f in stale:
        f.unlink(missing_ok=True)
    return stale


def host_file(path, base_url=None):
    """Commit the file into gh-pages's media/ folder and return its Pages URL."""
    base_url = base_url or (os.environ.get("PAGES_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError("PAGES_BASE_URL is needed to host images "
                           "(e.g. https://gesimuse.github.io/mpt)")
    path = Path(path)
    worktree = _pages_worktree()
    name = f"{int(time.time())}-{path.name}".replace(" ", "_")
    media_dir = worktree / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    dest = media_dir / name
    dest.write_bytes(path.read_bytes())
    stale = _prune_media(worktree)

    _git("add", "-A", cwd=worktree)
    _git("commit", "-m", f"media: {name}" + (f" (pruned {len(stale)})" if stale else ""),
        cwd=worktree)
    last = None
    for _attempt in range(3):
        try:
            _git("push", "origin", PAGES_BRANCH, cwd=worktree)
            break
        except RuntimeError as e:
            last = e
            _git("pull", "--rebase", "origin", PAGES_BRANCH, cwd=worktree)
    else:
        raise RuntimeError(f"could not push {name} to {PAGES_BRANCH} after 3 attempts: {last}")

    url = f"{base_url.rstrip('/')}/media/{name}"
    _wait_until_live(url)
    log(f"hosted {name} ({dest.stat().st_size // 1024}KB) at {url}")
    return url


def _wait_until_live(url, timeout=120, interval=3):
    """A live check found TikTok's own PULL_FROM_URL fetch failing with
    photo_pull_failed right after a successful push -- git push succeeding only means
    GitHub has the commit, not that Pages has finished building and deploying it
    (confirmed separately: Pages build/propagate latency is real, not instant). TikTok
    likely tries to fetch within seconds of our init call, well before that finishes.
    Poll our own URL until it's actually serving before handing it to TikTok at all."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.head(url, timeout=10, allow_redirects=False)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise RuntimeError(f"{url} did not go live within {timeout}s of pushing it")


# ---------- TikTok OAuth ----------
def _access_token(ck, cs, refresh):
    r = requests.post(
        "https://open.tiktokapis.com/v2/oauth/token/",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"client_key": ck, "client_secret": cs, "grant_type": "refresh_token",
              "refresh_token": refresh},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ---------- publishing ----------
def publish_photos_draft(image_paths, niche_id, image_urls=None, caption=None, title=None):
    """Queue a photo carousel as a TikTok inbox draft. Returns the publish_id, or None
    if this niche has no TikTok credentials configured.

    caption/title pre-fill the draft's description/title -- TikTok's docs (checked
    live, since this module's original design predated it) confirm MEDIA_UPLOAD now
    accepts post_info same as DIRECT_POST: "it will be reflected in the editing flow
    once user clicks on the inbox notification." title caps at 90 UTF-16 runes,
    description at 4000; title defaults to caption's first line when not given
    explicitly, matching the cap already used elsewhere for the same purpose."""
    ck = os.environ.get("TIKTOK_CLIENT_KEY")
    cs = os.environ.get("TIKTOK_CLIENT_SECRET")
    refresh = os.environ.get(f"TIKTOK_REFRESH_TOKEN_{niche_id.upper()}")
    if not (ck and cs and refresh):
        return None
    urls = image_urls or [host_file(p) for p in image_paths]
    if len(urls) < 2:
        raise RuntimeError(f"a TikTok carousel needs at least 2 images, got {len(urls)}")

    body = {
        "source_info": {
            "source": "PULL_FROM_URL",
            "photo_cover_index": 0,
            "photo_images": urls,
        },
        "post_mode": "MEDIA_UPLOAD",
        "media_type": "PHOTO",
    }
    if caption:
        post_title = title or caption.splitlines()[0]
        body["post_info"] = {"title": post_title[:90], "description": caption[:4000]}

    access = _access_token(ck, cs, refresh)
    r = requests.post(
        "https://open.tiktokapis.com/v2/post/publish/content/init/",
        headers={"Authorization": f"Bearer {access}", "Content-Type": "application/json; charset=UTF-8"},
        json=body,
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"TikTok photo draft init failed: {r.status_code} {r.text[:300]}")
    d = r.json()["data"]
    log(f"queued to inbox as draft, publish_id={d['publish_id']}"
        + (" (with caption)" if caption else ""))
    return d["publish_id"]


def publish_video_draft(video_path, niche_id, video_url=None, caption=None, title=None,
                       token_niche=None):
    """Queue a video as a TikTok inbox draft via PULL_FROM_URL. Returns the
    publish_id, or None if this niche has no TikTok credentials configured.

    token_niche picks which TIKTOK_REFRESH_TOKEN_<X> to authenticate with. Defaults to
    niche_id -- the video niche (aibeautyvideo) points this at "aibeauty" to reuse
    the same channel's token instead of needing a second OAuth setup.

    PULL_FROM_URL for video was confirmed via TikTok's Content Posting API reference
    (developers.tiktok.com/doc/content-posting-api-reference-upload-video) -- inbox
    videos accept the same source shape as inbox photos. Same verified domain used
    for photos works for video too; no separate portal verification step needed."""
    token_niche = token_niche or niche_id
    ck = os.environ.get("TIKTOK_CLIENT_KEY")
    cs = os.environ.get("TIKTOK_CLIENT_SECRET")
    refresh = os.environ.get(f"TIKTOK_REFRESH_TOKEN_{token_niche.upper()}")
    if not (ck and cs and refresh):
        return None
    url = video_url or host_file(video_path)

    body = {"source_info": {"source": "PULL_FROM_URL", "video_url": url}}
    if caption:
        post_title = title or caption.splitlines()[0]
        body["post_info"] = {"title": post_title[:90], "description": caption[:4000]}

    access = _access_token(ck, cs, refresh)
    r = requests.post(
        "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/",
        headers={"Authorization": f"Bearer {access}", "Content-Type": "application/json; charset=UTF-8"},
        json=body,
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"TikTok video draft init failed: {r.status_code} {r.text[:300]}")
    d = r.json()["data"]
    log(f"queued video to inbox as draft, publish_id={d['publish_id']}"
        + (" (with caption)" if caption else ""))
    return d["publish_id"]


# Terminal states from /v2/post/publish/status/fetch/ -- anything else (PROCESSING_
# UPLOAD, PROCESSING_DOWNLOAD, ...) means still in progress.
_TERMINAL_STATUSES = {"SEND_TO_USER_INBOX", "PUBLISH_COMPLETE", "FAILED"}


def check_publish_status(publish_id, niche_id, timeout=180, interval=5, token_niche=None):
    """Poll TikTok for what actually happened to a publish_id after init returned it.

    init's 200 OK only means the job was ACCEPTED, not that it succeeded -- caught
    live: a real run's init calls all returned a publish_id and got recorded as
    successful uploads, but 4 of 8 had actually failed downstream
    (photo_pull_failed/file_format_check_failed, a GitHub Pages build-latency race and
    a since-fixed PNG-vs-JPEG bug) and never reached the inbox at all. Nothing was
    ever polling this endpoint in production, only ever checked by hand during
    development -- posted.json's tiktok:true was trusting the init call alone.

    Returns (status, fail_reason) -- status is one of _TERMINAL_STATUSES, or
    "TIMEOUT" if it never reached one within `timeout` seconds (treated as a failure
    by the caller, not a success -- an unconfirmed draft must not count toward the
    pending-drafts cap either)."""
    ck = os.environ.get("TIKTOK_CLIENT_KEY")
    cs = os.environ.get("TIKTOK_CLIENT_SECRET")
    refresh = os.environ.get(f"TIKTOK_REFRESH_TOKEN_{(token_niche or niche_id).upper()}")
    access = _access_token(ck, cs, refresh)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
            headers={"Authorization": f"Bearer {access}", "Content-Type": "application/json; charset=UTF-8"},
            json={"publish_id": publish_id},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        status = data.get("status")
        if status in _TERMINAL_STATUSES:
            return status, data.get("fail_reason")
        time.sleep(interval)
    return "TIMEOUT", None
