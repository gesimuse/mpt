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


def run_job(job):
    work = Path(tempfile.mkdtemp(prefix=f"persona-job-{job['id']}-"))
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
        results = json.loads(results_file.read_text()) if results_file.exists() else [
            {"request": job["id"], "ok": False, "kind": "image" if job.get("kind") == "redo" else "video",
             "recreate": job.get("kind") == "recreate", "reason": f"laptop job exited {proc.returncode}",
             "chat": job.get("chat"), "message_id": job.get("message_id")}]
        for r in results:
            files = {}
            if r.get("ok"):
                item = json.loads((work / "items" / f"{r['id']}.json").read_text())
                files = {"item": work / "items" / f"{r['id']}.json", "file": work / "items" / item["path"]}
            data, ctype = _multipart({"result": json.dumps(r)}, files)
            for attempt in range(5):
                try:
                    log(f"{r['request']}: {'ok' if r.get('ok') else r.get('reason')} -> "
                        f"{_call('/persona/admin/laptop/result', data=data, content_type=ctype, timeout=300)}")
                    break
                except Exception as e:
                    log(f"upload failed ({e}), retrying")
                    time.sleep(30 * (attempt + 1))
        log(f"job {job['id']} done in {int(time.time() - t)}s")
    finally:
        if inhibit:
            inhibit.terminate()
        shutil.rmtree(work, ignore_errors=True)


def main():
    config.load_env()
    if not config.env("PERSONA_WORKER_URL") or not config.env("PERSONA_ADMIN_SECRET"):
        sys.exit("PERSONA_WORKER_URL and PERSONA_ADMIN_SECRET must be set in .env")
    log("checking in with the Worker every 30s")
    while True:
        try:
            r = _call("/persona/admin/laptop/claim", {})
            if r.get("job"):
                run_job(r["job"])
                continue
        except Exception as e:
            log(f"check-in failed: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
