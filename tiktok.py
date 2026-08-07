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

Images are hosted as GitHub release assets (public repo, stable URL, stays out of git
history) purely so TikTok's servers have something to fetch from.

Env:
  TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET   from the TikTok developer app
  TIKTOK_REFRESH_TOKEN_<NICHEID>             per-account OAuth refresh token
  GITHUB_TOKEN / GITHUB_REPOSITORY           for hosting the images; Actions provides both
"""
import mimetypes, os, time
from pathlib import Path

import requests

MEDIA_TAG = "autopilot-media"
KEEP_ASSETS = 20


def log(msg): print(f"[tiktok] {msg}", flush=True)


def enabled(niche_id):
    ck = os.environ.get("TIKTOK_CLIENT_KEY")
    cs = os.environ.get("TIKTOK_CLIENT_SECRET")
    refresh = os.environ.get(f"TIKTOK_REFRESH_TOKEN_{niche_id.upper()}")
    return bool(ck and cs and refresh)


# ---------- hosting (GitHub release asset, so PULL_FROM_URL has something to fetch) ----------
def _gh_api(method, url, token, **kw):
    r = requests.request(method, url, timeout=180, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        **kw.pop("headers", {}),
    }, **kw)
    if not r.ok:
        raise RuntimeError(f"github {method} {url.split('/')[-1]}: {r.status_code} {r.text[:200]}")
    return r


def _release(repo, token):
    """The rolling release that holds image assets, created on first use."""
    base = f"https://api.github.com/repos/{repo}"
    r = requests.get(f"{base}/releases/tags/{MEDIA_TAG}", timeout=60, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    if r.status_code == 200:
        return r.json()
    return _gh_api("POST", f"{base}/releases", token, json={
        "tag_name": MEDIA_TAG,
        "name": "Autopilot media",
        "body": "Generated images hosted for TikTok to fetch. Managed automatically.",
        "prerelease": True,
    }).json()


def _prune(repo, token, release, keep=KEEP_ASSETS):
    assets = sorted(release.get("assets", []), key=lambda a: a.get("created_at", ""))
    for asset in assets[:-keep] if len(assets) > keep else []:
        try:
            _gh_api("DELETE", f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}", token)
        except Exception as e:
            log(f"could not delete old asset {asset.get('name')}: {str(e)[:80]}")


def host_file(path, token=None, repo=None):
    """Upload a file as a release asset and return its public URL."""
    token = token or (os.environ.get("GITHUB_TOKEN") or "").strip()
    repo = repo or (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if not token or not repo:
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPOSITORY are needed to host the images")
    path = Path(path)
    release = _release(repo, token)
    name = f"{int(time.time())}-{path.name}".replace(" ", "_")
    upload = release["upload_url"].split("{")[0]
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        asset = _gh_api("POST", f"{upload}?name={name}", token,
                        headers={"Content-Type": ctype}, data=f).json()
    _prune(repo, token, release)
    log(f"hosted {name} ({path.stat().st_size // 1024}KB)")
    return asset["browser_download_url"]


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
