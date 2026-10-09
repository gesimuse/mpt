"""On-demand jobs from the Telegram channel, the same code on Kaggle and the laptop:

    recreate   🎭 a TikTok/Facebook link or an uploaded video, redone with her
    redo       🔁 the same scene, a new seed, the same anatomy check
    (video)    🎬 / a reply with a motion prompt: a photo animated

Each returns a result dict ({"request", "id", "ok", "kind", ...}) and leaves the
finished file and its item JSON in `out`. persona/kaggle_kernel.py runs them on
Kaggle; persona/agent.py on the laptop, one job per process:

    python -m persona.jobs <job.json> <out_dir>      writes <out_dir>/results.json
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import config


def log(m):
    print(f"[persona.jobs] {m}", flush=True)


def fetch(url, dest, timeout=300):
    """Download from a signed Worker link. Cloudflare answers 403 to Python's
    default User-Agent ("Python-urllib"), so send our own."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "mpt-persona-kernel/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        Path(dest).write_bytes(r.read())


def _reply(job):
    return {"chat": job.get("chat"), "message_id": job.get("message_id")}


def download_clip(job, dest):
    """The clip a 🎭 job names: an uploaded file (signed Worker link) or a link."""
    if job.get("video_url"):
        fetch(job["video_url"], dest)
        return
    # Kept current: TikTok and Facebook break old versions. The laptop's Wan2GP venv
    # has no pip (uv-made), Kaggle has no uv.
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=True)
    else:
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        subprocess.run([uv, "pip", "install", "-q", "-U", "--python", sys.executable, "yt-dlp"], check=True)
    # Facebook serves reels as separate video and audio streams: merge them (ffmpeg)
    # into one mp4. TikTok's single mp4 still matches.
    dl = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
                         "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
                         "--merge-output-format", "mp4", "-o", str(dest), job["url"]],
                        capture_output=True, text=True)
    if dl.returncode == 0:
        return
    err = (dl.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
    if "Log in for access" in err[0] or "comfortable for some audiences" in err[0]:
        raise RuntimeError("TikTok marks this video age-restricted and only shows it to logged-in users. "
                           "Save it on your phone and post the video file here instead.")
    if "facebook" in job["url"] or "fb.watch" in job["url"]:
        raise RuntimeError("could not download the Facebook video (private, or only shown to logged-in "
                           f"users?). Save it on your phone and post the video file here instead. ({err[0][:150]})")
    raise RuntimeError(f"could not download the TikTok: {err[0][:200]}")


def recreate(job, char, out, inp, work_root, code_dir):
    """🎭 in its own process (persona/recreate.py runs its two models in separate
    processes too), so the caller's RAM stays free for them."""
    try:
        src = inp / f"{job['id']}.mp4"
        download_clip(job, src)
        work = work_root / job["id"]
        cmd = [sys.executable, "-m", "persona.recreate", str(src), str(work)]
        if job.get("seconds"):
            cmd += ["--seconds", str(job["seconds"])]
        subprocess.run(cmd, cwd=str(code_dir), check=True)
        made = next(work.glob("recreated.*"))
        vid = char.new_item(kind="video", lane="social", model="viggle_animate", recreate=True,
                            source=job.get("url", ""))
        dest = out / f"{vid['id']}{made.suffix}"
        shutil.copyfile(made, dest)
        vid = char.update_item(vid["id"], path=dest.name, request=job["id"])
        (out / f"{vid['id']}.json").write_text(json.dumps(vid, ensure_ascii=False))
        log(f"recreate {vid['id']} for request {job['id']}")
        return {"request": job["id"], "id": vid["id"], "ok": True, "kind": "video", "recreate": True, **_reply(job)}
    except Exception as e:
        log(f"recreate {job['id']} failed: {e}")
        return {"request": job["id"], "ok": False, "kind": "video", "recreate": True,
                "reason": (str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")[:300], **_reply(job)}


def redo(job, char, studio, out):
    results = []
    for item in studio.render(char, lane="social", scene_list=[job["scene"]]):
        if item.get("ok"):
            dest = out / Path(item["path"]).name
            shutil.copyfile(item["path"], dest)
            item["path"] = dest.name
            (out / f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False))
        results.append({"request": job["id"], "id": item["id"], "ok": bool(item.get("ok")), "kind": "image",
                        "reason": item.get("reason"), **_reply(job)})
        log(f"redo {item['id']} for request {job['id']}: {'ok' if item.get('ok') else item.get('reason')}")
    return results


def video(job, char, studio, out, inp):
    src = inp / f"{job['id']}.jpg"
    if job.get("image_url"):
        # A signed, short-lived link served by the Worker, which fetches the photo
        # from Telegram itself -- the bot token never leaves the Worker.
        fetch(job["image_url"], src, timeout=120)
    else:
        src.write_bytes(base64.b64decode(job["image_b64"]))
    photo = char.new_item(kind="image", lane="social", path=str(src), model="telegram",
                          scene={"motion": job.get("motion") or ""}, tags=job.get("tags", []))
    try:
        vid = studio.animate(char, photo, job.get("motion") or None)
    except Exception as e:
        log(f"video {job['id']} failed: {e}")
        return {"request": job["id"], "ok": False, "reason": f"{type(e).__name__}: {e}"[:300], **_reply(job)}
    dest = out / Path(vid["path"]).name
    shutil.copyfile(vid["path"], dest)
    vid.update({"path": dest.name, "request": job["id"]})
    (out / f"{vid['id']}.json").write_text(json.dumps(vid, ensure_ascii=False))
    log(f"video {vid['id']} for request {job['id']}")
    return {"request": job["id"], "id": vid["id"], "ok": True, "kind": "video", "motion": job.get("motion", ""),
            **_reply(job)}


def run(jobs, char, studio_factory, out, inp, work_root, code_dir):
    """All jobs, 🎭 first: they run in child processes, and this process has not
    loaded a model yet, so its RAM is still free for them."""
    results = []
    for job in [j for j in jobs if j.get("kind") == "recreate"]:
        results.append(recreate(job, char, out, inp, work_root, code_dir))
    rest = [j for j in jobs if j.get("kind") != "recreate"]
    studio = studio_factory() if rest else None
    for job in rest:
        if job.get("kind") == "redo":
            results.extend(redo(job, char, studio, out))
        else:
            results.append(video(job, char, studio, out, inp))
    return results


def main():
    config.load_env()
    job_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    job = json.loads(job_file.read_text())
    from . import character
    from .engine import Engine
    from .studio import Studio
    out = out_dir / "items"
    inp = out_dir / "in"
    for d in (out, inp):
        d.mkdir(parents=True, exist_ok=True)
    try:
        results = run([job], character.active(), lambda: Studio(Engine()), out, inp, out_dir / "recreate", config.REPO)
    except Exception as e:
        results = [{"request": job["id"], "ok": False, "kind": "image" if job.get("kind") == "redo" else "video",
                    "recreate": job.get("kind") == "recreate", "reason": f"{type(e).__name__}: {e}"[:300],
                    **_reply(job)}]
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
