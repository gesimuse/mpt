"""Post an aibeauty carousel as a native TikTok inbox draft -- no Buffer involved.

TikTok's Content Posting API, in MEDIA_UPLOAD mode, drops the carousel into the
account's own TikTok app inbox as an unpublished draft: no caption, no sound -- the
account owner opens the app, adds trending audio and a caption, then posts it by hand.
This is the deliberate tradeoff of an unaudited app: TikTok only grants
video.upload/video.publish without a review, direct posting is SELF_ONLY-only even
with that scope, and the inbox draft is the one path that ends in a real public post
without waiting on an app audit.

Photos only accept PULL_FROM_URL as their source, not FILE_UPLOAD (raw-byte upload,
the way video posting works) -- this is a real TikTok API constraint, not a choice
made here, and it also matches how Buffer's own carousel implementation only ever
took image URLs, never raw files, for the same underlying endpoint. PULL_FROM_URL
requires the fetch domain to be verified for this TikTok app under
developers.tiktok.com -> Content Posting API -> URL properties -- a one-time manual
step; without it, every call here fails with a domain-verification error TikTok's own
API returns, not something this code can detect in advance.

This has not been verified against a real TikTok account yet -- unlike the rest of
this pipeline, which was live-tested before being trusted. Treat the first real run as
a test: check that the draft actually lands in the aibeauty account's inbox before
relying on this unattended.

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
KEEP_MEDIA = 30


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
    log(f"hosted {name} ({dest.stat().st_size // 1024}KB) at {url}")
    return url


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
def publish_photos_draft(image_paths, niche_id, image_urls=None):
    """Queue a photo carousel as a TikTok inbox draft. Returns the publish_id, or None
    if this niche has no TikTok credentials configured."""
    ck = os.environ.get("TIKTOK_CLIENT_KEY")
    cs = os.environ.get("TIKTOK_CLIENT_SECRET")
    refresh = os.environ.get(f"TIKTOK_REFRESH_TOKEN_{niche_id.upper()}")
    if not (ck and cs and refresh):
        return None
    urls = image_urls or [host_file(p) for p in image_paths]
    if len(urls) < 2:
        raise RuntimeError(f"a TikTok carousel needs at least 2 images, got {len(urls)}")

    access = _access_token(ck, cs, refresh)
    r = requests.post(
        "https://open.tiktokapis.com/v2/post/publish/content/init/",
        headers={"Authorization": f"Bearer {access}", "Content-Type": "application/json; charset=UTF-8"},
        json={
            "source_info": {
                "source": "PULL_FROM_URL",
                "photo_cover_index": 0,
                "photo_images": urls,
            },
            "post_mode": "MEDIA_UPLOAD",
            "media_type": "PHOTO",
        },
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"TikTok photo draft init failed: {r.status_code} {r.text[:300]}")
    d = r.json()["data"]
    log(f"queued to inbox as draft, publish_id={d['publish_id']}")
    return d["publish_id"]
