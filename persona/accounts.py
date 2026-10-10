"""Link a persona's own channel, Instagram and TikTok to the Cloudflare Worker,
which publishes on 👍 (worker/src/persona/social.js).

    python -m persona.cli channel <slug> <telegram chat id>
    python -m persona.cli accounts <slug> tiktok [--direct]
    python -m persona.cli accounts <slug> instagram <access token>
    python -m persona.cli accounts <slug> tiktok-verify <file name> <file contents>
    python -m persona.cli accounts                      (what is linked)

TikTok: opens TikTok's consent page (get_tiktok_token.py); log in as HER account
first. --direct asks for video.publish too -- only after the app has passed
TikTok's audit, then videos with sound go out public instead of as drafts.

Instagram: her account must be a professional (creator/business) account added
to the Meta app as an Instagram tester; the token comes from the app dashboard
(Instagram > API setup with Instagram login > Generate token).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from . import config


def _worker(path, method="POST", **kw):
    url, secret = config.env("PERSONA_WORKER_URL"), config.env("PERSONA_ADMIN_SECRET")
    if not (url and secret):
        sys.exit("PERSONA_WORKER_URL and PERSONA_ADMIN_SECRET must be set in .env")
    r = requests.request(method, f"{url}{path}", headers={"Authorization": f"Bearer {secret}"}, timeout=60, **kw)
    r.raise_for_status()
    return r


def channel(slug, chat_id, tiktok_direct=None):
    entry = next((p for p in _worker("/persona/admin/personas", "GET").json() if p["slug"] == slug), {"slug": slug})
    entry = {k: v for k, v in entry.items() if k != "accounts"}
    entry["chat_id"] = str(chat_id)
    if tiktok_direct is not None:
        entry["tiktok_direct"] = tiktok_direct
    _worker("/persona/admin/personas", json=entry)
    return f"{slug} -> channel {chat_id}"


def tiktok(slug, direct=False):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "token"
        env = dict(os.environ, TIKTOK_TOKEN_OUT=str(out))
        if direct:
            env["TIKTOK_SCOPES"] = "user.info.basic,video.upload,video.publish"
        subprocess.run([sys.executable, str(config.REPO / "get_tiktok_token.py")], env=env, check=True)
        token = out.read_text().strip()
    _worker("/persona/admin/account", params={"slug": slug}, json={"platform": "tiktok", "refresh_token": token})
    if direct:
        p = next(p for p in _worker("/persona/admin/personas", "GET").json() if p["slug"] == slug)
        channel(slug, p["chat_id"], tiktok_direct=True)
    return f"TikTok linked for {slug}" + (" (direct posting)" if direct else " (inbox drafts)")


def instagram(slug, token):
    me = requests.get("https://graph.instagram.com/v23.0/me", params={"fields": "user_id,username",
                                                                     "access_token": token}, timeout=30).json()
    if "user_id" not in me:
        sys.exit(f"Instagram did not accept the token: {me}")
    _worker("/persona/admin/account", params={"slug": slug},
            json={"platform": "instagram", "token": token, "user_id": me["user_id"]})
    return f"Instagram @{me.get('username')} linked for {slug}"


def tiktok_verify(name, body):
    _worker("/persona/admin/account", json={"platform": "tiktok_verify", "name": name, "body": body})
    url = config.env("PERSONA_WORKER_URL")
    return f"served at {url}/persona/media/{name}"


def show():
    lines = []
    for p in _worker("/persona/admin/personas", "GET").json():
        a = p.get("accounts", {})
        lines.append(f"{p['slug']}: channel {p.get('chat_id')} · Instagram {'✓' if a.get('instagram') else '—'} · "
                     f"TikTok {'✓' if a.get('tiktok') else '—'}{' (direct)' if p.get('tiktok_direct') else ''}")
    return "\n".join(lines)
