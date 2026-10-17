"""The laptop's side of the persona studio: while it is open, on-demand jobs from
the channel (🎬 🔁 🎭) run here on its GPU instead of on Kaggle.

    python -m persona.agent          (systemd: mpt-persona-agent.service)

Every 30s it checks in with the Worker (POST /persona/admin/laptop/claim). The
check-in also collects the bot's taps (while the bot has no webhook), so buttons
answer in seconds instead of at the next 15-minute cron. When a job is handed
out it runs in its own process (python -m persona.jobs) -- the models' RAM is
freed after each job -- and the finished photo/video goes back to the Worker
(POST /persona/admin/laptop/result), which posts it like a Kaggle result.

Kaggle stays the fallback: when the laptop has not checked in for a few minutes
(lid closed, asleep, off), the Worker sends jobs to Kaggle as before. The daily
render and the post slots stay on Kaggle and the Worker's cron.

Needs PERSONA_WORKER_URL and PERSONA_ADMIN_SECRET in .env.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

from . import config

POLL_S = 30


def log(msg):
    print(f"[persona.agent] {msg}", flush=True)


def _call(path, body=None, data=None, content_type="application/json", timeout=60):
    url = config.env("PERSONA_WORKER_URL").rstrip("/") + path
    req = urllib.request.Request(url, method="POST", data=data if data is not None else json.dumps(body or {}).encode(),
                                 headers={"Authorization": f"Bearer {config.env('PERSONA_ADMIN_SECRET')}",
                                          "Content-Type": content_type, "User-Agent": "mpt-persona-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return raw.decode(errors="replace")  # /laptop/result answers in plain text


def _multipart(fields, files):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, path in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{Path(path).name}"\r\n'
                     f'Content-Type: application/octet-stream\r\n\r\n'.encode() + Path(path).read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def heartbeat_while(proc, job_id):
    """Keep checking in while a job runs (no new job is taken): the Worker hands a
    job back to Kaggle only when the laptop goes quiet."""
    while proc.poll() is None:
        try:
            _call("/persona/admin/laptop/claim", {"busy": job_id})
        except Exception as e:
            log(f"check-in failed: {e}")
        for _ in range(POLL_S):
            if proc.poll() is not None:
                return
            time.sleep(1)


UNSENT = Path.home() / ".cache" / "mpt-persona" / "unsent"


def _telegram(method, fields, files=None):
    """The persona bot, called from the laptop (its token is in .env here)."""
    token = config.env("PERSONA_BOT_TOKEN") or config.env("LOCAL_BOT_TOKEN")
    data, ctype = _multipart(fields, files or {})
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", data=data, method="POST",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    if not body.get("ok"):
        raise RuntimeError(f"telegram {method}: {str(body)[:200]}")
    return body["result"]


def deliver(work, r):
    """Post one result. Photos/videos go to Telegram from here (caption and buttons
    from the Worker), then the Worker is told what was sent; large videos pushed
    through the Worker failed with HTTP 500."""
    item = None
    if r.get("ok"):
        item = json.loads((work / "items" / f"{r['id']}.json").read_text())
    if r.get("kind") == "render-done":
        return _call("/persona/admin/laptop/result", {"result": r})
    if r.get("render"):
        # A daily-render photo/Reel, kept by the Worker for the day's post slots as a
        # Telegram file reference: uploaded silently to her channel and deleted at
        # once (the file stays valid), so no large file goes through the Worker.
        file_id = None
        if item:
            chat = r.get("chat") or config.env("PERSONA_CHAT_ID")
            video = item["path"].endswith(".mp4")
            msg = _telegram("sendVideo" if video else "sendPhoto", {"chat_id": chat, "disable_notification": "true"},
                            {"video" if video else "photo": work / "items" / item["path"]})
            _telegram("deleteMessage", {"chat_id": chat, "message_id": msg["message_id"]})
            file_id = (msg.get("video") or (msg.get("photo") or [{}])[-1]).get("file_id")
        return _call("/persona/admin/laptop/render-item", {"result": r, "item": item, "file_id": file_id})
    if not item or not (config.env("PERSONA_BOT_TOKEN") or config.env("LOCAL_BOT_TOKEN")):
        files = {} if not item else {"item": work / "items" / f"{r['id']}.json", "file": work / "items" / item["path"]}
        data, ctype = _multipart({"result": json.dumps(r)}, files)
        return _call("/persona/admin/laptop/result", data=data, content_type=ctype, timeout=300)
    m = _call("/persona/admin/laptop/prepare", {"result": r, "item": item})
    if m.get("edit"):
        # 🔁: the redrawn photo replaces the old one in its own message.
        try:
            sent = _telegram("editMessageMedia", {
                "chat_id": m["chat"], "message_id": m["edit"], "reply_markup": json.dumps(m["buttons"]),
                "media": json.dumps({"type": "photo", "media": "attach://file", "caption": m["caption"][:1024]})},
                {"file": work / "items" / item["path"]})
            return _call("/persona/admin/laptop/result", {"result": r, "item": item, "sent": sent})
        except Exception as e:
            log(f"in-place replace failed ({e}); posting it instead")
    fields = {"chat_id": m["chat"], "caption": m["caption"][:1024], "reply_markup": json.dumps(m["buttons"])}
    if m.get("replyTo"):
        # Posted anyway if the message it answers was deleted (Telegram: "message to
        # be replied not found", 2026-10-16 -- the finished video was re-rendered twice).
        fields["reply_parameters"] = json.dumps({"message_id": m["replyTo"], "allow_sending_without_reply": True})
    if m["kind"] == "video":
        fields["supports_streaming"] = "true"
    sent = _telegram("sendVideo" if m["kind"] == "video" else "sendPhoto", fields,
                     {m["kind"]: work / "items" / item["path"]})
    return _call("/persona/admin/laptop/result", {"result": r, "item": item, "sent": sent})


def deliver_all(work):
    """Every result of a finished job; True when all went out. Kept on disk until then."""
    results = json.loads((work / "results.json").read_text())
    left = []
    for r in results:
        try:
            log(f"{r['request']}: {'ok' if r.get('ok') else r.get('reason')} -> {deliver(work, r)}")
        except Exception as e:
            log(f"{r['request']}: delivery failed ({e}); kept to retry")
            left.append(r)
    (work / "results.json").write_text(json.dumps(left))
    return not left


def retry_unsent():
    for work in sorted(UNSENT.glob("*")) if UNSENT.exists() else []:
        if (work / "results.json").exists() and deliver_all(work):
            shutil.rmtree(work, ignore_errors=True)


def run_job(job):
    UNSENT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{job['id']}-", dir=UNSENT))
    (work / "job.json").write_text(json.dumps(job))
    log(f"job {job['id']} ({job.get('kind') or 'video'})")
    t = time.time()
    # No sleep while a job runs (the old laptop bot once slept mid-job for 3 days).
    inhibit = None
    try:
        inhibit = subprocess.Popen(["systemd-inhibit", "--what=sleep:idle", "--who=mpt-persona-agent",
                                    f"--why=rendering {job['id']}", "sleep", "infinity"])
    except Exception as e:
        log(f"could not inhibit sleep ({e})")
    try:
        proc = subprocess.Popen([sys.executable, "-m", "persona.jobs", str(work / "job.json"), str(work)],
                                cwd=str(config.REPO))
        heartbeat_while(proc, job["id"])
        results_file = work / "results.json"
        if not results_file.exists():
            results_file.write_text(json.dumps([
                {"request": job["id"], "ok": False, "kind": "image" if job.get("kind") == "redo" else "video",
                 "recreate": job.get("kind") == "recreate", "reason": f"laptop job exited {proc.returncode}",
                 "chat": job.get("chat"), "message_id": job.get("message_id")}]))
        results = json.loads(results_file.read_text())
        for r in results:
            r["slug"] = job.get("slug", "")
            r["persona"] = job.get("persona") or job.get("slug", "")
        results_file.write_text(json.dumps(results))
        # A failed delivery keeps the finished work here, so it is sent again rather
        # than rendered again.
        if deliver_all(work):
            shutil.rmtree(work, ignore_errors=True)
        log(f"job {job['id']} done in {int(time.time() - t)}s")
    finally:
        if inhibit:
            inhibit.terminate()


def main():
    config.load_env()
    if not config.env("PERSONA_WORKER_URL") or not config.env("PERSONA_ADMIN_SECRET"):
        sys.exit("PERSONA_WORKER_URL and PERSONA_ADMIN_SECRET must be set in .env")
    log("checking in with the Worker every 30s")
    while True:
        try:
            retry_unsent()
        except Exception as e:
            log(f"retrying unsent results failed: {e}")
        try:
            r = _call("/persona/admin/laptop/claim", {})
            job = r.get("job")
            # Already rendered, only its delivery is pending (retry_unsent): never again.
            if job and any((d / "results.json").exists() for d in UNSENT.glob(f"{job['id']}-*")):
                job = None
            if job:
                run_job(job)
                continue
        except Exception as e:
            log(f"check-in failed: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
