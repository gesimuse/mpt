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


def has_audio(path):
    """Whether a video carries sound: a 🎭 recreate keeps the original's audio, a
    🎬 video has none. Decides how it goes to TikTok (posted with its sound, or a
    draft to add a song to)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


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
        # Her, not whichever persona is active on this machine: the recreate's stages
        # run as child processes and read the slug from the environment.
        import os
        os.environ["PERSONA_RECREATE_SLUG"] = char.slug
        cmd = [sys.executable, "-m", "persona.recreate", str(src), str(work)]
        if job.get("seconds"):
            cmd += ["--seconds", str(job["seconds"])]
        subprocess.run(cmd, cwd=str(code_dir), check=True)
        made = next(work.glob("recreated.*"))
        report = work / "swapped.json"  # persona/faceswap.py's check of her face
        face = json.loads(report.read_text()) if report.exists() else None
        vid = char.new_item(kind="video", lane="social", model="viggle_animate", recreate=True,
                            source=job.get("url", ""), face=face, audio=has_audio(made))
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
                        "redo": True, "reason": item.get("reason"), **_reply(job)})
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
    vid.update({"path": dest.name, "request": job["id"], "audio": has_audio(dest),
                "caption": vid.get("caption") or job.get("caption", "")})
    (out / f"{vid['id']}.json").write_text(json.dumps(vid, ensure_ascii=False))
    log(f"video {vid['id']} for request {job['id']}")
    return {"request": job["id"], "id": vid["id"], "ok": True, "kind": "video", "motion": job.get("motion", ""),
            **_reply(job)}


def render(job, char, studio, out):
    """The daily render on the laptop (the Worker gives it to the laptop when it is
    open at render time, Kaggle otherwise): the day's scenes, each photo a result."""
    results = []
    for item in studio.render(char, lane="social", scene_list=job["scenes"]):
        if item.get("ok"):
            dest = out / Path(item["path"]).name
            shutil.copyfile(item["path"], dest)
            item["path"] = dest.name
            (out / f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False))
        results.append({"request": job["id"], "id": item["id"], "ok": bool(item.get("ok")), "kind": "image",
                        "render": job.get("date", ""), "reason": item.get("reason"), "chat": job.get("chat")})
        log(f"render {item['id']}: {'ok' if item.get('ok') else item.get('reason')}")
    # The day's Reels: the first N photos that passed, animated (Reels reach more new
    # people than photos for a small account).
    for r in [r for r in results if r["ok"]][: int(job.get("videos", 0))]:
        try:
            vid = studio.animate(char, char.item(r["id"]))
        except Exception as e:
            log(f"reel of {r['id']} failed: {e}")
            continue
        dest = out / Path(vid["path"]).name
        shutil.copyfile(vid["path"], dest)
        vid.update({"path": dest.name, "audio": has_audio(dest), "caption": char.item(r["id"]).get("caption", "")})
        (out / f"{vid['id']}.json").write_text(json.dumps(vid, ensure_ascii=False))
        results.append({"request": job["id"], "id": vid["id"], "ok": True, "kind": "video", "parent": r["id"],
                        "render": job.get("date", ""), "chat": job.get("chat")})
        log(f"reel {vid['id']} of {r['id']}")
    # Tells the Worker the render is over (its laptop lane is freed).
    results.append({"request": job["id"], "ok": True, "kind": "render-done", "render": job.get("date", "")})
    return results


# Carousel moves: what changes between the photos of one shoot.
CAROUSEL_POSES = [
    "sitting down on something nearby, legs crossed",
    "leaning back against the wall or a railing behind her",
    "standing turned sideways to the camera, one hand in her hair",
    "taking a slow step toward the camera",
    "standing with both hands on her hips, weight on one leg",
    "looking back over her shoulder, body turned slightly away",
    "with one hand touching her necklace or collar, relaxed",
    "sitting on the edge of a ledge or table, leaning forward slightly",
]
CAROUSEL_SHOTS = [
    "closer, from the chest up",
    "wider, showing more of the place around her, full body",
    "from slightly lower, looking up at her",
    "from a little to her side, three-quarter view of the scene",
    "waist-up, the camera moved to the other side of her",
]


def carousel(job, char, studio, out, inp):
    """🖼: the same moment from other angles, for an Instagram carousel with the
    original photo (carousels earn the most saves and engagement per view).

    Each angle is an EDIT of the original photo -- same woman, outfit, place and
    light, a new pose and head angle. Drawn from the scene text instead (the first
    version), every angle came out in a different outfit and body (2026-10-16)."""
    import random
    from . import faceid, qa, scenes
    src = inp / f"{job['id']}-original.jpg"
    if job.get("image_url"):
        fetch(job["image_url"], src, timeout=120)
    else:
        src.write_bytes(base64.b64decode(job["image_b64"]))
    base = job.get("scene") or {}
    angles = [a for a in scenes.ANGLES if a[1] and a[1] != base.get("angle_sheet")]
    random.shuffle(angles)
    picked, seen = [], set()
    for a in angles:
        if a[1] not in seen:
            seen.add(a[1])
            picked.append(a)
    exprs = [e for e in scenes.EXPRESSIONS if e[0] != base.get("expression")]
    random.shuffle(exprs)
    model = studio.model("edit", lane="social")
    poses = random.sample(CAROUSEL_POSES, len(CAROUSEL_POSES))
    shots = random.sample(CAROUSEL_SHOTS, len(CAROUSEL_SHOTS))
    front = [p for p in char.refs() if p.name.startswith("01-")][:1]
    emb = faceid.reference_embedding(char.refs())
    results = []
    for a, e in list(zip(picked, exprs))[: int(job.get("count", 2))]:
        item = char.new_item(kind="image", lane="social", model=model["id"],
                             scene=dict(base, angle=a[0], expression=e[0], angle_sheet=a[1]),
                             tags=base.get("tags", []), caption=base.get("caption", ""))
        pose, shot = poses.pop(), shots.pop()
        # Pose and camera change too: told to keep "body and framing", the edit only
        # turned her head and the rest of the photo stayed identical (2026-10-16).
        prompt = ("A new photo from the same photo shoot as image 1: the same woman, wearing exactly the same "
                  "outfit (same garments, colours and fit), in the same location with the same lighting. "
                  f"This time she is {pose}, {a[0]}, {e[0]}. Camera: {shot}. Her pose, the camera position and "
                  "the framing are different from image 1; her face, hair, outfit and the place stay the same. "
                  "She is the only person in the photo.")
        path, why = None, ""
        for attempt in range(3):
            path = studio.engine.image(model, prompt, char.dir / "items", item["id"],
                                       refs=[str(src)] + [str(f) for f in front],
                                       negative=scenes.negative(char, "social"))
            good, why, _ = qa.check(path)
            if good:
                ok, info = faceid.check(path, emb, min_sim=0.5)
                if ok:
                    break
                why = info.get("reason", "face check")
            path.unlink(missing_ok=True)
            path = None
        if path:
            dest = out / path.name
            shutil.copyfile(path, dest)
            item = char.update_item(item["id"], status="review", ok=True, path=dest.name)
            (out / f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False))
        results.append({"request": job["id"], "id": item["id"], "ok": bool(path), "kind": "image",
                        "carousel": job["item"], "replace": job.get("replace"), "reason": None if path else why,
                        **_reply(job)})
        log(f"carousel angle {item['id']} of {job['item']}: {'ok' if path else why}")
    return results


def run(jobs, char, studio_factory, out, inp, work_root, code_dir):
    """All jobs, 🎭 first: they run in child processes, and this process has not
    loaded a model yet, so its RAM is still free for them."""
    results = []
    for job in [j for j in jobs if j.get("kind") == "recreate"]:
        results.append(recreate(job, char, out, inp, work_root, code_dir))
    rest = [j for j in jobs if j.get("kind") != "recreate"]
    studio = studio_factory() if rest else None
    for job in rest:
        if job.get("kind") == "render":
            results.extend(render(job, char, studio, out))
        elif job.get("kind") == "carousel":
            results.extend(carousel(job, char, studio, out, inp))
        elif job.get("kind") == "redo":
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
        char = character.Character(job["slug"]) if job.get("slug") else character.active()
        results = run([job], char, lambda: Studio(Engine()), out, inp, out_dir / "recreate", config.REPO)
    except Exception as e:
        results = [{"request": job["id"], "ok": False, "kind": "image" if job.get("kind") == "redo" else "video",
                    "recreate": job.get("kind") == "recreate", "reason": f"{type(e).__name__}: {e}"[:300],
                    **_reply(job)}]
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
