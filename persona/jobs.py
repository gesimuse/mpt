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
import os
import re
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
    # character: which roster girl a result belongs to (MPT's shared channel).
    out = {"chat": job.get("chat"), "message_id": job.get("message_id"), "character": job.get("character")}
    # A redo of a shoot photo comes back like a shoot photo (a Telegram file, no
    # post of its own): the Worker puts it in the shoot and posts the album again.
    if job.get("shoot"):
        out.update(shoot=job["shoot"], replaces=job.get("replaces"), render=job.get("render") or "redo")
    return out


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
    """🔁: look at the photo, name what is wrong (extra hands, a third leg, fused
    fingers...), and fix THAT in an edit of the same photo -- same girl, outfit, place
    and pose -- verified afterwards; 3 failed fixes fall back to a fresh redraw of the
    scene (2026-10-20: a redo used to redraw from scratch, and could bring new defects)."""
    results = []
    # A note about the clothes means the old photo's outfit is wrong: an edit keeps
    # what is there, so it is drawn again instead (Alina's gown, 2026-10-20). Same for
    # "cover the boob": three edits in a row kept the bare breast (Chloe, 2026-10-22).
    clothes = re.search(r"\b(dress|gown|outfit|wear|wearing|skirt|trousers|pants|jeans|top|bikini|shirt|shorts|"
                        r"clothes|heels|shoes|bra|bandeau|cover|covered|boobs?|breasts?|nipples?|tits?|chest)\b", (job.get("scene") or {}).get("note", ""), re.I)
    # A note asking for a different pose or a new photo: an edit keeps the pose, and
    # 3-4 redos of one MPT photo came back the same (2026-10-23, "legs are in weird
    # position. Change the position of the legs").
    change = re.search(r"\b(change|changed|different|differently|completely|totally|new|another|instead|pose|position|"
                       r"posture|sit|sitting|stand|standing|lying|kneel|kneeling|turn|turned|angle|background|place|"
                       r"from scratch|again)\b", (job.get("scene") or {}).get("note", ""), re.I)
    fixed = _redo_fix(job, char, studio, out) if job.get("image_url") and not clothes and not change else None
    if fixed:
        results.append(fixed)
        return results
    scene = dict(job["scene"])
    for item in studio.render(char, lane="social", scene_list=[scene]):
        if item.get("ok"):
            dest = out / Path(item["path"]).name
            shutil.copyfile(item["path"], dest)
            item["path"] = dest.name
            (out / f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False))
        results.append({"request": job["id"], "id": item["id"], "ok": bool(item.get("ok")), "kind": "image",
                        "redo": True, "reason": item.get("reason"), "qa": item.get("qa"), "how": "redrawn",
                        "fixed": job.get("seen_problems") or [], **_reply(job)})
        log(f"redo {item['id']} for request {job['id']}: {'ok' if item.get('ok') else item.get('reason')}")
    return results


def _redo_fix(job, char, studio, out):
    from . import faceid, proportions, qa, scenes
    src = out / f"{job['id']}-original.jpg"
    try:
        fetch(job["image_url"], src, timeout=120)
    except Exception as e:
        log(f"redo {job['id']}: couldn't fetch the photo ({e}); redrawing instead")
        return None
    scene = dict(job.get("scene") or {})
    problems = qa.diagnose(src)
    seen = bool(problems)
    note = scene.get("note", "")
    log(f"redo {job['id']}: problems seen {problems or 'none'}{f'; note: {note}' if note else ''}")
    if not problems and not note:
        problems = ["hands, arms and legs that don't look natural"]
    model = studio.model("edit", lane="social")
    front = [p for p in char.refs() if p.name.startswith("01-")][:1]
    emb = faceid.reference_embedding(char.refs())
    floor = min(0.5, scenes.ANGLE_FLOOR.get(scene.get("angle_sheet", ""), 0.5)) - 0.05
    # The outfit is the SHOOT's, from its scene text -- "the same outfit as image 1"
    # kept a wrong outfit (Alina's trousers instead of her gown) even when the note
    # asked for the gown (2026-10-20). The note comes first: it is what you asked.
    outfit = scene.get("outfit") or "the same outfit as in image 1"
    prompt = ((f"MOST IMPORTANT: {note}. " if note else "")
              + "Image 1 is a photo of this woman. Make this photo again -- the same woman in the same place, "
                f"light, pose and framing, wearing {outfit} -- "
              + (f"and fix these problems so she looks like a real human: {'; '.join(problems)}. " if problems else "")
              + "She has exactly two arms, two hands with five natural fingers each, and two legs, all correctly "
                "attached and in natural positions. Keep everything else as in image 1. She is the only "
                "person in the photo.")
    item = char.new_item(kind="image", lane="social", model=model["id"], scene=scene, prompt=prompt,
                         tags=scene.get("tags", []), caption=scene.get("caption", ""))
    fails = []
    for attempt in range(3):
        path = studio.engine.image(model, prompt, char.dir / "items", item["id"], refs=[str(src)] + [str(f) for f in front],
                                   negative=scenes.negative(char, "social", scene))
        # An edit that changed next to nothing is no redo: straight to a redraw.
        if _barely_changed(src, path):
            log(f"redo {job['id']}: the edit came back almost the same photo; redrawing instead")
            Path(path).unlink(missing_ok=True)
            fails.append("barely changed")
            break
        good, why, _ = qa.check(path)
        if good and scene.get("view") != "back":
            good, why, _ = proportions.check(path, ref=char.refs_dir / "04-full-body.jpg")
        if good and qa.still_has(path, problems):
            good, why = False, "the problem is still there"
        info = {}
        if good and scene.get("view") != "back":
            good, info = faceid.check(path, emb, min_sim=floor)
            why = why if good else info.get("reason", "face check")
        if good:
            item = char.update_item(item["id"], status="review", ok=True, path=str(path), check=info,
                                    fixed=problems, qa={"keys": scenes.learn_keys(scene), "fails": fails, "ok": True})
            dest = out / Path(path).name
            shutil.copyfile(path, dest)
            item["path"] = dest.name
            (out / f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False))
            log(f"redo {item['id']} for request {job['id']}: fixed by edit (attempt {attempt + 1})")
            return {"request": job["id"], "id": item["id"], "ok": True, "kind": "image", "redo": True,
                    "qa": item.get("qa"), "fixed": problems if seen else [], "how": "fixed", **_reply(job)}
        fails.append(str(why)[:80])
        log(f"redo {job['id']}: fix attempt {attempt + 1} failed ({why})")
        Path(path).unlink(missing_ok=True)
    # Redrawn instead -- told what went wrong before.
    if problems:
        job["scene"] = dict(scene, note="; ".join(filter(None, [note, "avoid: " + "; ".join(problems)])))
    job["seen_problems"] = problems if seen else []
    return None


def _barely_changed(a, b, limit=14.0):
    """Mean pixel difference (0-255) of the two photos at 256px grey, under `limit`: Gia's
    redos that kept the pose measured 9-10, different photos 25+ (2026-10-23)."""
    try:
        import numpy as np
        from PIL import Image
        x, y = (np.asarray(Image.open(f).convert("L").resize((256, 256)), dtype=np.float32) for f in (a, b))
        return float(np.abs(x - y).mean()) < limit
    except Exception:
        return False


def video(job, char, studio, out, inp):
    src = inp / f"{job['id']}.jpg"
    if job.get("image_url"):
        # A signed, short-lived link served by the Worker, which fetches the photo
        # from Telegram itself -- the bot token never leaves the Worker.
        fetch(job["image_url"], src, timeout=120)
    else:
        src.write_bytes(base64.b64decode(job["image_b64"]))
    photo = char.new_item(kind="image", lane="social", path=str(src), model="telegram",
                          scene={"motion": job.get("motion") or "", "hook": job.get("hook") or ""}, tags=job.get("tags", []))
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
                        "render": job.get("date", ""), "reason": item.get("reason"), "chat": job.get("chat"),
                        "character": job.get("character"), "shoot": job.get("shoot"), "qa": item.get("qa")})
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
                        "render": job.get("date", ""), "chat": job.get("chat"),
                        "character": job.get("character"), "shoot": job.get("shoot"), "qa": item.get("qa")})
        log(f"reel {vid['id']} of {r['id']}")
    # Tells the Worker the render is over (its laptop lane is freed).
    results.append({"request": job["id"], "ok": True, "kind": "render-done", "render": job.get("date", ""),
                    "character": job.get("character"), "shoot": job.get("shoot"), "chat": job.get("chat")})
    return results


# Carousel moves: what changes between the photos of one shoot.
CAROUSEL_POSES = [
    "one hand in her hair, weight shifted onto one hip",
    "arms loosely crossed, relaxed",
    "laughing with her head tilted back a little, one hand at her collarbone",
    "turned side-on to the camera, chin slightly down, looking toward the lens",
    "both hands lifting her hair",
    "hands clasped loosely in front of her, shoulders relaxed",
    "one hand on her hip, the other relaxed at her side",
    "touching her earring, looking off to the side",
]
# Camera moves that stay in the same spot. A wide full-body shot stretched her
# proportions and a tight close-up was a near copy of the original (2026-10-17).
CAROUSEL_SHOTS = [
    "from a step to her left, waist-up",
    "from a step to her right, waist-up",
    "slightly lower camera, head to knees",
    "slightly higher camera, waist-up",
]


def _background_drift(a, b):
    """0 = same surroundings, 1 = different: colour histograms of the image border
    (top and side strips, where the background is), so a carousel shot that moved
    her somewhere else is redrawn."""
    import numpy as np
    from PIL import Image
    def border(p):
        im = np.asarray(Image.open(p).convert("HSV").resize((128, 160)), dtype=np.float32)
        strips = [im[:32], im[:, :20].reshape(-1, 20, 3), im[:, -20:].reshape(-1, 20, 3)]
        px = np.concatenate([x.reshape(-1, 3) for x in strips])
        h, _ = np.histogramdd(px[:, :2], bins=(18, 8), range=((0, 256), (0, 256)))
        return h.ravel() / max(1, h.sum())
    ha, hb = border(a), border(b)
    return float(1 - np.minimum(ha, hb).sum())


def carousel(job, char, studio, out, inp):
    """🖼: the same moment from other angles, for an Instagram carousel with the
    original photo (carousels earn the most saves and engagement per view).

    Each angle is an EDIT of the original photo -- same woman, outfit, place and
    light, a new pose and head angle. Drawn from the scene text instead (the first
    version), every angle came out in a different outfit and body (2026-10-16)."""
    import random
    from . import faceid, proportions, qa, scenes
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
    front = [p for p in char.refs() if p.name.startswith("01-")][:1]
    emb = faceid.reference_embedding(char.refs())
    results, failed = [], []
    # A failed angle is replaced by the next one, so the carousel gets its `count`
    # (2026-10-22: asked for 2, one usually failed the head-size check -> only 1).
    want = int(job.get("count", 2))
    for tried, (a, e) in enumerate(list(zip(picked, exprs))[: want + 2]):
        if len(results) >= want:
            break
        item = char.new_item(kind="image", lane="social", model=model["id"],
                             scene=dict(base, angle=a[0], expression=e[0], angle_sheet=a[1]),
                             tags=base.get("tags", []), caption=base.get("caption", ""))
        pose = poses.pop()
        # The first extra shot of a carousel is full body (asked for, 2026-10-17):
        # zoomed out from the same spot, natural proportions; the drift check still
        # applies, so a full-body shot that invents a new place is redrawn.
        full = tried == 0 and job.get("full_body", True)
        # Pose and camera change too: told to keep "body and framing", the edit only
        # turned her head and the rest of the photo stayed identical (2026-10-16).
        # The camera stays where it was: asked to move it ("a step to her left"), the
        # model re-imagined the place (a sunflower field became a room, 2026-10-17).
        prompt = ("The next photo from the same photo shoot as image 1, taken from the SAME camera position "
                  "with the same framing: the same woman, wearing exactly the same outfit (same garments, colours "
                  "and fit), in exactly the same place with the same background and lighting. Only she moves: "
                  f"this time she is {pose}, {a[0]}, {e[0]}. Everything around her stays exactly as in image 1. "
                  "She is the only person in the photo.")
        if full:
            prompt = ("The next photo from the same photo shoot as image 1: the same woman, wearing exactly the same "
                      "outfit (same garments, colours and fit), in exactly the same place with the same lighting, "
                      "now a FULL-BODY shot from head to toe -- the camera at the same spot, zoomed out, so more of "
                      "the same surroundings is visible around her. Natural, realistic body proportions, normal leg "
                      "length, wearing shoes that fit the outfit and the place (never barefoot unless at the beach or "
                      f"pool), feet on the ground. She is {pose}, {e[0]}. She is the only person in the photo.")
        path, why = None, ""
        spare = list(poses)   # retries take another pose: the same one came back the same
        # Her face sheet portrait at this angle as the face ref: with only her front
        # ref the edit kept the original head and pose (2026-10-20, a near copy).
        stamp = char.refs_dir / "sheet" / f"{a[1]}.jpg"
        face_refs = [stamp] if (stamp.exists() and not full) else front
        # A turned head scores lower against her front-heavy refs (scenes.ANGLE_FLOOR):
        # a real profile was dropped at 0.38 < 0.5.
        floor = min(0.5, scenes.ANGLE_FLOOR.get(a[1], 0.5)) - 0.05
        for attempt in range(3):
            if attempt and why == "same pose as the original" and spare:
                prompt = prompt.replace(pose, spare[-1]) + (" Her pose, body angle and head direction are clearly "
                                                            "different from image 1.")
                pose = spare.pop()
            path = studio.engine.image(model, prompt, char.dir / "items", item["id"],
                                       refs=[str(src)] + [str(f) for f in face_refs],
                                       negative=scenes.negative(char, "social"))
            good, why, _ = qa.check(path)
            if good:
                good, why, _ = proportions.check(path, ref=char.refs_dir / "04-full-body.jpg")
            if good and _background_drift(src, path) > 0.5:
                good, why = False, "background changed"
            if good and proportions.too_similar(src, path):
                good, why = False, "same pose as the original"
            if good:
                ok, info = faceid.check(path, emb, min_sim=floor)
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
        (results if path else failed).append({"request": job["id"], "id": item["id"], "ok": bool(path), "kind": "image",
                                              "carousel": job["item"], "replace": job.get("replace"),
                                              "reason": None if path else why, "of": want, **_reply(job)})
        log(f"carousel angle {item['id']} of {job['item']}: {'ok' if path else why}")
    # Failures are only reported when nothing came out. `of`: how many this set has,
    # so the Worker offers the carousel once all of them are posted.
    for r in results:
        r["of"] = len(results)
    return results or failed[:1]


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
            # You asked for this one: more tries before it gives up (a back shot failed
            # its twist check twice and the redraw came back with nothing, 2026-10-20).
            prev = os.environ.get("PERSONA_QA_RETRIES")
            os.environ["PERSONA_QA_RETRIES"] = "3"
            try:
                results.extend(redo(job, char, studio, out))
            finally:
                if prev is None:
                    os.environ.pop("PERSONA_QA_RETRIES", None)
                else:
                    os.environ["PERSONA_QA_RETRIES"] = prev
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
