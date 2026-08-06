# TikTok for Developers — app submission copy

Paste-ready answers for the MPT app on developers.tiktok.com.

## Basic information

**App icon:** `docs/app-icon.png` (1024×1024 PNG)

**App name:** `MPT`

**Category:** Education

**Description** (≤120 chars):

```
Generates short educational programming videos and sends them to the owner's own TikTok inbox as drafts.
```

**Terms of Service URL:** `https://codeaz-org.github.io/mpt/terms.html`

**Privacy Policy URL:** `https://codeaz-org.github.io/mpt/privacy.html`

**Platforms:** Desktop (the app is a self-hosted Python program; it has no end-user website)

**Web/Desktop URL:** `https://codeaz-org.github.io/mpt/`

**Redirect URI** (Desktop — TikTok requires host `localhost:<port>` or `127.0.0.1:<port>`):

```
http://localhost:8723/callback
```

`get_tiktok_token.py` starts a loopback HTTP server on that port, opens the consent page, and captures `?code=` automatically. PKCE is mandatory — TikTok rejects the authorize call with `code_challenge` otherwise, and its challenge is a **hex** SHA-256 digest, not the base64url of RFC 7636. Must match byte-for-byte in the portal, the authorize URL, and the token exchange. To use a different port, set `TIKTOK_REDIRECT_URI` in `.env` and register the same value.

## Products

- Login Kit — one-time OAuth so the operator authorizes their own TikTok account
- Content Posting API — inbox upload (draft), `/v2/post/publish/inbox/video/init/`

Do **not** add Share Kit, Display API, Research API, or Embed Videos. Unused products delay review.

## Scopes

- `user.info.basic` — required by Login Kit; identifies the authorizing account
- `video.upload` — sends the rendered MP4 to that account's inbox as a draft

Do not request `video.publish` — the app never posts publicly, the account owner publishes manually from the inbox.

## "Explain how each product and scope works" (≤1000 chars)

```
MPT is a self-hosted desktop tool run by one creator for their own TikTok account. No sign-ups, not multi-tenant.

Login Kit (user.info.basic): the operator runs get_tiktok_token.py once, is sent to TikTok's authorization page, signs into their own account and grants access. The returned code is exchanged for an access/refresh token stored locally in .env. user.info.basic is the scope Login Kit requires to identify which account authorized the app; no profile fields are read or stored.

Content Posting API (video.upload): on each scheduled run the app generates a short educational video about a programming concept (LLM topic, stock footage, text-to-speech), refreshes the access token, calls /v2/post/publish/inbox/video/init/ with FILE_UPLOAD, then PUTs the MP4 to the returned upload_url. The video lands in that account's inbox as a draft. Nothing publishes automatically: the owner opens TikTok, reviews the draft, writes the caption, and posts manually. No caption is sent via API.
```

## Demo video checklist

Record one screen capture, mp4/mov, ≤50MB, showing the whole loop:

1. Terminal on the desktop, project folder visible.
2. `python get_tiktok_token.py` → TikTok authorization page opens → sign in → tap Authorize → redirect URL with `?code=` → paste code → refresh token printed (blur/crop the token).
3. `python autopilot.py` → log lines through video render → `TikTok: queued to inbox as draft, publish_id=...`.
4. Switch to the phone/TikTok app → Inbox → the new draft appears → open it → show the caption being written → the post screen. Show the manual publish step, since that is the whole reason no `video.publish` scope is requested.

Notes: use the sandbox on the Developer Portal (app is not yet approved). Show the app being launched at the start. Every scope requested must appear in the video.
