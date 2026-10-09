"""Kaggle kernel: render pre-written social-lane scenes of a persona on a T4.

Template, filled in by persona/kaggle.py at push time:
  __PAYLOAD_B64__   {"slug", "scenes", "roles", "wan2gp_commit", "face_min", "min_age",
                     "videos", "date"}
  __PACKAGE_B64__   the persona/ package as a tar.gz, so the kernel runs the exact
                    same studio/engine code as the laptop
The persona herself (bible, refs, storyline) comes from the private dataset
<user>/mpt-persona-<slug>, attached as a dataset source.

Scenes arrive already written: the scene LLM needs an HF token, and Kaggle kernels
pushed through the API get no secrets. Only social-lane, kaggle_ok models run here;
registry.check enforces it, since Kaggle's terms forbid explicit generation.

Writes /kaggle/working/out/items/<id>.jpg|.json and status.json. Same install
recipe and profile 5 as kaggle/video_pipeline.py (see there for why).
"""
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

PAYLOAD_B64 = "__PAYLOAD_B64__"
PACKAGE_B64 = "__PACKAGE_B64__"

WORK = Path("/kaggle/working")
CODE = Path("/kaggle/tmp/code")
WAN2GP_DIR = Path("/kaggle/tmp/Wan2GP")
HOME = Path("/kaggle/tmp/persona-home")
STATUS = WORK / "status.json"


def fetch(url, dest, timeout=300):
    """Download from a signed Worker link. Cloudflare answers 403 to Python's
    default User-Agent ("Python-urllib"), so send our own."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "mpt-persona-kernel/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        Path(dest).write_bytes(r.read())


def log(m):
    print(f"[persona-kernel] {m}", flush=True)


def write_status(stage, ok, extra=None):
    STATUS.write_text(json.dumps({"stage": stage, "ok": ok, **(extra or {})}))


def main():
    write_status("start", False)
    stage = "start"
    try:
        payload = json.loads(base64.b64decode(PAYLOAD_B64).decode())
        os.environ["USE_TF"] = "0"

        stage = "unpack"
        CODE.mkdir(parents=True, exist_ok=True)
        if PACKAGE_B64.startswith("__"):
            # Pushed by the Cloudflare Worker: no bundled code, take persona/ from the
            # public repo at the commit the Worker names (or main).
            repo = Path("/kaggle/tmp/mpt-repo")
            subprocess.run(["git", "clone", "-q", "https://github.com/gesimuse/mpt", str(repo)], check=True)
            if payload.get("repo_ref"):
                subprocess.run(["git", "checkout", "-q", payload["repo_ref"]], cwd=repo, check=True)
            shutil.copytree(repo / "persona", CODE / "persona")
            shutil.copyfile(repo / "llm.py", CODE / "llm.py")
        else:
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKAGE_B64)), mode="r:gz") as tar:
                tar.extractall(CODE)
        # The mount path has changed between Kaggle images (/kaggle/input/<slug> vs
        # deeper owner/slug layouts), so find her bible wherever it landed.
        found = sorted(Path("/kaggle/input").rglob("bible.json"))
        if not found:
            tree = [str(p) for p in Path("/kaggle/input").rglob("*")][:40]
            raise RuntimeError(f"no bible.json under /kaggle/input; saw {tree}")
        dataset = found[0].parent
        char_dir = HOME / payload["slug"]
        shutil.copytree(dataset, char_dir)
        sys.path.insert(0, str(CODE))
        from persona.kaggle import restore_layout
        restore_layout(char_dir)
        (HOME / "settings.json").write_text(json.dumps({"active": payload["slug"], "roles": payload["roles"]}))

        stage = "install"
        log("cloning Wan2GP...")
        subprocess.run(["git", "clone", "https://github.com/deepbeepmeep/Wan2GP", str(WAN2GP_DIR)], check=True)
        if payload.get("wan2gp_commit"):
            subprocess.run(["git", "checkout", "-q", payload["wan2gp_commit"]], cwd=WAN2GP_DIR, check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.10.0", "torchvision==0.25.0",
                        "torchaudio==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu130"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
                       cwd=WAN2GP_DIR, check=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
        # The anatomy check (persona/qa.py, Qwen2.5-VL-7B in 4-bit) gets its own venv:
        # tested 2026-10-07, it fails on the T4 under Wan2GP's torch 2.10+cu130 with a
        # CUDA JIT error, and runs fine under torch 2.11+cu128 (Kaggle's stock build).
        stage = "install-qa"
        qa_env = Path("/kaggle/tmp/qaenv")
        try:
            # `python -m venv` fails on Kaggle's image (no ensurepip); uv builds the venv
            # without it and installs much faster.
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
            subprocess.run([sys.executable, "-m", "uv", "venv", "-q", "--python", sys.executable, str(qa_env)], check=True)
            qpip = [sys.executable, "-m", "uv", "pip", "install", "-q", "--python", str(qa_env / "bin" / "python")]
            # torchvision too: transformers' Qwen2-VL processor refuses to load without it.
            subprocess.run(qpip + ["torch==2.11.0", "torchvision==0.26.0",
                                   "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
            subprocess.run(qpip + ["transformers", "accelerate", "bitsandbytes", "pillow", "numpy"], check=True)
            os.environ.update({"PERSONA_QA_SUBPROCESS": "1", "PERSONA_QA_PYTHON": str(qa_env / "bin" / "python")})
        except Exception as e:
            log(f"QA venv failed ({e}); images will post without the anatomy check")
            os.environ["PERSONA_VISION_QA"] = "0"

        stage = "render"
        os.environ.update({"PERSONA_HOME": str(HOME), "WAN2GP_DIR": str(WAN2GP_DIR), "WAN2GP_PROFILE": "5",
                           "WAN2GP_ATTENTION": "sdpa", "PERSONA_FACE_MIN": str(payload["face_min"]),
                           "PERSONA_MIN_AGE": str(payload["min_age"])})
        from persona import character
        from persona.engine import Engine
        from persona.studio import Studio

        char = character.Character(payload["slug"])
        studio = Studio(Engine(), kaggle=True)
        out = WORK / "out" / "items"
        out.mkdir(parents=True, exist_ok=True)
        results = []
        if payload.get("mode") == "video":
            # On-demand videos (🎬 / a reply with a motion prompt): the photo comes in
            # the payload, straight from Telegram.
            inp = WORK / "in"
            inp.mkdir(exist_ok=True)
            # 🎭 recreate jobs first, each in its own process (persona/recreate.py runs
            # its two models in separate processes): this process has not loaded any
            # model yet, so its RAM is still free for them.
            jobs = sorted(payload["jobs"], key=lambda j: j.get("kind") != "recreate")
            for job in [j for j in jobs if j.get("kind") == "recreate"]:
                try:
                    src = inp / f"{job['id']}.mp4"
                    if job.get("video_url"):
                        fetch(job["video_url"], src)
                    else:
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=True)
                        dl = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "mp4/best",
                                             "-o", str(src), job["url"]], capture_output=True, text=True)
                        if dl.returncode != 0:
                            err = (dl.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
                            if "Log in for access" in err[0] or "comfortable for some audiences" in err[0]:
                                raise RuntimeError("TikTok marks this video age-restricted and only shows it to "
                                                   "logged-in users. Save it on your phone and post the video "
                                                   "file here instead.")
                            raise RuntimeError(f"could not download the TikTok: {err[0][:200]}")
                    work = WORK / "recreate" / job["id"]
                    cmd = [sys.executable, "-m", "persona.recreate", str(src), str(work)]
                    if job.get("seconds"):
                        cmd += ["--seconds", str(job["seconds"])]
                    subprocess.run(cmd, cwd=str(CODE), check=True)
                    made = next(work.glob("recreated.*"))
                    vid = char.new_item(kind="video", lane="social", model="viggle_animate", recreate=True,
                                        source=job.get("url", ""))
                    dest = out / f"{vid['id']}{made.suffix}"
                    shutil.copyfile(made, dest)
                    vid = char.update_item(vid["id"], path=dest.name, request=job["id"])
                    (out / f"{vid['id']}.json").write_text(json.dumps(vid, ensure_ascii=False))
                    results.append({"request": job["id"], "id": vid["id"], "ok": True, "kind": "video",
                                    "recreate": True, "chat": job.get("chat"), "message_id": job.get("message_id")})
                    log(f"recreate {vid['id']} for request {job['id']}")
                except Exception as e:
                    results.append({"request": job["id"], "ok": False, "kind": "video", "recreate": True,
                                    "reason": (str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")[:300],
                                    "chat": job.get("chat"), "message_id": job.get("message_id")})
                    log(f"recreate {job['id']} failed: {e}")
            for job in [j for j in jobs if j.get("kind") != "recreate"]:
                if job.get("kind") == "redo":
                    # 🔁: the same scene, a new seed, the same anatomy check.
                    for item in studio.render(char, lane="social", scene_list=[job["scene"]]):
                        if item.get("ok"):
                            dest = out / Path(item["path"]).name
                            shutil.copyfile(item["path"], dest)
                            item["path"] = dest.name
                            (out / f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False))
                        results.append({"request": job["id"], "id": item["id"], "ok": bool(item.get("ok")),
                                        "kind": "image", "reason": item.get("reason"),
                                        "chat": job.get("chat"), "message_id": job.get("message_id")})
                        log(f"redo {item['id']} for request {job['id']}: {'ok' if item.get('ok') else item.get('reason')}")
                    continue
                src = inp / f"{job['id']}.jpg"
                if job.get("image_url"):
                    # A signed, short-lived link served by the Worker, which fetches the
                    # photo from Telegram itself -- the bot token never comes here.
                    fetch(job["image_url"], src, timeout=120)
                else:
                    src.write_bytes(base64.b64decode(job["image_b64"]))
                photo = char.new_item(kind="image", lane="social", path=str(src), model="telegram",
                                      scene={"motion": job.get("motion") or ""}, tags=job.get("tags", []))
                try:
                    vid = studio.animate(char, photo, job.get("motion") or None)
                except Exception as e:
                    results.append({"request": job["id"], "ok": False, "reason": f"{type(e).__name__}: {e}"[:300],
                                    "chat": job.get("chat"), "message_id": job.get("message_id")})
                    log(f"video {job['id']} failed: {e}")
                    continue
                dest = out / Path(vid["path"]).name
                shutil.copyfile(vid["path"], dest)
                vid.update({"path": dest.name, "request": job["id"]})
                (out / f"{vid['id']}.json").write_text(json.dumps(vid, ensure_ascii=False))
                results.append({"request": job["id"], "id": vid["id"], "ok": True, "kind": "video",
                                "chat": job.get("chat"), "message_id": job.get("message_id"),
                                "motion": job.get("motion", "")})
                log(f"video {vid['id']} for request {job['id']}")
            write_status("done", True, {"items": results, "mode": "video",
                                        "requests": [j["id"] for j in payload["jobs"]]})
            return
        for item in studio.render(char, lane="social", scene_list=payload["scenes"]):
            if item.get("path"):
                dest = out / Path(item["path"]).name
                shutil.copyfile(item["path"], dest)
                item["path"] = dest.name
            (out / f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False))
            results.append({"id": item["id"], "ok": item.get("ok"), "reason": item.get("reason"), "kind": "image"})
            log(f"item {item['id']}: {'ok' if item.get('ok') else item.get('reason')}")
        # Videos of the first passing images. The image model's weights are deleted
        # first: Qwen 2.1 plus Wan 2.2 14B do not both fit Kaggle's disk.
        n_videos = int(payload.get("videos", 0))
        if n_videos:
            for f in list((WAN2GP_DIR / "ckpts").glob("qwen_image_21*")) + \
                     list((WAN2GP_DIR / "ckpts").glob("Qwen3-VL*")):
                shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
            write_status("video", False, {"items": results})
            for r in [r for r in results if r["ok"]][:n_videos]:
                try:
                    vid = studio.animate(char, char.item(r["id"]))
                except Exception as e:
                    log(f"video of {r['id']} failed: {type(e).__name__}: {e}")
                    continue
                dest = out / Path(vid["path"]).name
                shutil.copyfile(vid["path"], dest)
                vid["path"] = dest.name
                (out / f"{vid['id']}.json").write_text(json.dumps(vid, ensure_ascii=False))
                results.append({"id": vid["id"], "ok": True, "reason": None, "kind": "video", "parent": r["id"]})
                log(f"video {vid['id']} of {r['id']}")
        write_status("done", True, {"items": results, "date": payload.get("date", "")})
    except Exception as e:
        write_status(stage, False, {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]})
        raise


if __name__ == "__main__":
    main()
