"""Link a persona's own Telegram channel and her Buffer channels (Instagram,
TikTok) to the Cloudflare Worker, which publishes on 👍 through Buffer
(worker/src/persona/social.js).

    python -m persona.cli channel <slug> <telegram chat id>
    python -m persona.cli accounts buffer                     (Buffer's channels)
    python -m persona.cli accounts <slug> buffer <instagram channel> <tiktok channel>
    python -m persona.cli accounts                            (what is linked)

Connect her Instagram (a professional account) and TikTok in Buffer first
(publish.buffer.com, the account BUFFER_ACCESS_TOKEN belongs to); channels can be
named by their Buffer name or id, "-" for none.
"""
import sys

import requests

from . import config

BUFFER = "https://api.buffer.com"


def _worker(path, method="POST", **kw):
    url, secret = config.env("PERSONA_WORKER_URL"), config.env("PERSONA_ADMIN_SECRET")
    if not (url and secret):
        sys.exit("PERSONA_WORKER_URL and PERSONA_ADMIN_SECRET must be set in .env")
    r = requests.request(method, f"{url}{path}", headers={"Authorization": f"Bearer {secret}"}, timeout=60, **kw)
    r.raise_for_status()
    return r


def _buffer(query, variables=None):
    r = requests.post(BUFFER, json={"query": query, "variables": variables or {}}, timeout=30,
                      headers={"Authorization": f"Bearer {config.env('BUFFER_ACCESS_TOKEN')}"})
    d = r.json()
    if d.get("errors"):
        sys.exit(f"Buffer: {d['errors']}")
    return d["data"]


def buffer_channels():
    orgs = _buffer("{ account { organizations { id name } } }")["account"]["organizations"]
    out = []
    for org in orgs:
        out += _buffer("query($o: OrganizationId!) { channels(input: {organizationId: $o}) "
                       "{ id service name isDisconnected } }", {"o": org["id"]})["channels"]
    return out


def _entry(slug):
    entry = next((p for p in _worker("/persona/admin/personas", "GET").json() if p["slug"] == slug), {"slug": slug})
    return {k: v for k, v in entry.items() if k != "accounts"}


def channel(slug, chat_id):
    entry = _entry(slug)
    entry["chat_id"] = str(chat_id)
    _worker("/persona/admin/personas", json=entry)
    return f"{slug} -> Telegram channel {chat_id}"


def link_buffer(slug, instagram, tiktok):
    chans = buffer_channels()

    def find(name, service):
        if name in ("-", "", None):
            return None
        for c in chans:
            if c["service"] == service and name in (c["id"], c["name"]):
                return c["id"]
        sys.exit(f"no {service} channel {name!r} in Buffer; have: "
                 + ", ".join(f"{c['service']}:{c['name']}" for c in chans))
    entry = _entry(slug)
    entry["buffer"] = {"instagram": find(instagram, "instagram"), "tiktok": find(tiktok, "tiktok")}
    _worker("/persona/admin/personas", json=entry)
    return f"{slug}: Instagram {instagram} · TikTok {tiktok} (Buffer)"


def show():
    lines = []
    for p in _worker("/persona/admin/personas", "GET").json():
        a = p.get("accounts", {})
        lines.append(f"{p['slug']}: channel {p.get('chat_id')} · Instagram {'✓' if a.get('instagram') else '—'} · "
                     f"TikTok {'✓' if a.get('tiktok') else '—'}")
    return "\n".join(lines)
