#!/usr/bin/env python3
"""One-time (per channel) helper: prints a TikTok refresh token.
Sandbox / unaudited apps can only post to the user inbox (as drafts).
Scopes: user.info.basic, video.upload.

Setup once:
  - developers.tiktok.com -> create app
  - Add products: "Login Kit" + "Content Posting API"
  - Platform: Desktop. TikTok requires the Redirect URI host to be
    localhost:<port> or 127.0.0.1:<port>, so register exactly:
        http://localhost:8723/callback
  - Copy Client Key / Client Secret into .env as TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET
  - (optional) TIKTOK_REDIRECT_URI if you registered a different port/path

This script starts a loopback web server on that port, opens the TikTok consent
page, and captures the ?code= automatically -- no copy/paste.
"""
import os, sys, hashlib, secrets, threading, urllib.parse, webbrowser, requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

CLIENT_KEY = os.environ["TIKTOK_CLIENT_KEY"]
CLIENT_SECRET = os.environ["TIKTOK_CLIENT_SECRET"]
REDIRECT = os.environ.get("TIKTOK_REDIRECT_URI", "http://localhost:8723/callback")

parsed = urllib.parse.urlparse(REDIRECT)
if parsed.hostname not in ("localhost", "127.0.0.1"):
    sys.exit(f"TIKTOK_REDIRECT_URI host must be localhost or 127.0.0.1, got: {REDIRECT}")
PORT = parsed.port or 80
PATH = parsed.path or "/"
STATE = secrets.token_urlsafe(16)

# PKCE. TikTok requires it for Desktop/Web apps and, unlike RFC 7636, expects the
# challenge as a lowercase hex SHA-256 digest rather than base64url.
CODE_VERIFIER = secrets.token_hex(48)  # 96 chars, within the 43-128 range
CODE_CHALLENGE = hashlib.sha256(CODE_VERIFIER.encode()).hexdigest()

result = {}
done = threading.Event()

PAGE = """<!doctype html><meta charset="utf-8">
<title>MPT Autopilot</title>
<body style="font:16px system-ui;max-width:34rem;margin:4rem auto;padding:0 1rem">
<h1>{h}</h1><p>{p}</p></body>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != PATH:
            self.send_error(404)
            return
        q = urllib.parse.parse_qs(u.query)
        result["code"] = (q.get("code") or [None])[0]
        result["state"] = (q.get("state") or [None])[0]
        result["error"] = (q.get("error_description") or q.get("error") or [None])[0]
        ok = result["code"] and result["state"] == STATE
        body = PAGE.format(
            h="Authorized" if ok else "Authorization failed",
            p="You can close this tab and return to the terminal."
            if ok
            else (result["error"] or "State mismatch -- possible CSRF. Re-run the script."),
        )
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())
        done.set()

    def log_message(self, *a):
        pass


auth_url = "https://www.tiktok.com/v2/auth/authorize/?" + urllib.parse.urlencode({
    "client_key": CLIENT_KEY,
    "scope": "user.info.basic,video.upload",
    "response_type": "code",
    "redirect_uri": REDIRECT,
    "state": STATE,
    "code_challenge": CODE_CHALLENGE,
    "code_challenge_method": "S256",
})

try:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
except OSError as e:
    sys.exit(f"Cannot bind {REDIRECT}: {e}\nFree the port or set TIKTOK_REDIRECT_URI to another one.")

threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"\nListening on {REDIRECT}")
print("\nOpen this URL in the browser signed into the aibeauty TikTok account,")
print("then grant access:\n")
print(f"   {auth_url}\n")
webbrowser.open(auth_url)

print("Waiting for the redirect (5 min timeout)...")
if not done.wait(300):
    sys.exit("Timed out waiting for TikTok to redirect back.")
server.shutdown()

if result.get("error"):
    sys.exit(f"Authorization failed: {result['error']}")
if result.get("state") != STATE:
    sys.exit("State mismatch -- discarding the response. Re-run the script.")
code = result.get("code")
if not code:
    sys.exit("No 'code' in the redirect.")

r = requests.post(
    "https://open.tiktokapis.com/v2/oauth/token/",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    data={
        "client_key": CLIENT_KEY,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT,
        "code_verifier": CODE_VERIFIER,
    },
    timeout=30,
)
if not r.ok:
    sys.exit(f"Token exchange failed: {r.status_code} {r.text}")
data = r.json()
if "refresh_token" not in data:
    sys.exit(f"No refresh_token in response: {data}")
print(f"\nREFRESH TOKEN:\n{data['refresh_token']}")
print("\nSave as TIKTOK_REFRESH_TOKEN_AIBEAUTY in .env (local) and GitHub secret.")
