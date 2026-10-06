"""Render social-lane scenes on Kaggle's T4 instead of the laptop.

  sync(char)   uploads bible.json, refs/, storyline.jsonl and votes.json as the
               PRIVATE dataset <KAGGLE_USERNAME>/mpt-persona-<slug>. Social-lane
               data only: items, outboxes and anything from the Fanvue lane stay
               on this machine.
  run(char)    writes the scenes here (LLM needs HF_TOKEN, which a pushed kernel
               cannot have), pushes persona/kaggle_kernel.py with them, polls,
               downloads the output and imports the items into her local store,
               keeping their ids so the Telegram buttons work on them.

Only models with kaggle_ok and not local_only are allowed (registry.check), so the
uncensored model and the Fanvue lane can never be selected here.

Uses the `kaggle` CLI (pip install kaggle) and KAGGLE_USERNAME + KAGGLE_API_TOKEN,
the same credentials as kaggle_imagegen.py / kaggle_videogen.py.
"""
import base64
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from . import config, registry, scenes

ACCELERATOR = config.env("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")
POLL_TIMEOUT = int(config.env("PERSONA_KAGGLE_TIMEOUT", "7200"))


def log(msg):
    print(f"[persona.kaggle] {msg}", flush=True)


def _env():
    env = os.environ.copy()
    if "KAGGLE_KEY" not in env and env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]
    if not env.get("KAGGLE_USERNAME") or not env.get("KAGGLE_KEY"):
        raise RuntimeError("set KAGGLE_USERNAME and KAGGLE_API_TOKEN")
    if not shutil.which("kaggle"):
        raise RuntimeError("the kaggle CLI is not installed (pip install kaggle)")
    return env


def _kaggle(*args, env, check=True):
    r = subprocess.run(["kaggle", *args], env=env, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:2])} failed: {out[-600:]}")
    return out


def dataset_id(char_or_slug):
    slug = getattr(char_or_slug, "slug", char_or_slug)
    return f"{os.environ['KAGGLE_USERNAME'].strip()}/mpt-persona-{slug}"


STATE_FILES = ("storyline.jsonl", "votes.json", "posted.json", "recent_scenes.jsonl", "videos.json")


def sync(char):
    env = _env()
    ds = dataset_id(char)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # A new version replaces every file. In cloud mode the dataset holds the
        # live votes/storyline/posted state, so a laptop sync must not wipe what
        # it does not have locally: take those files from the current version.
        if any(not (char.dir / n).exists() for n in STATE_FILES):
            mine = _kaggle("datasets", "list", "--mine", "-s", f"mpt-persona-{char.slug}", env=env, check=False)
            if ds in mine:
                cur = tmp / "_current"
                _kaggle("datasets", "download", ds, "--unzip", "-p", str(cur), env=env, check=False)
                for n in STATE_FILES:
                    if not (char.dir / n).exists() and (cur / n).exists():
                        shutil.copyfile(cur / n, char.dir / n)
                shutil.rmtree(cur, ignore_errors=True)
        shutil.copyfile(char.dir / "bible.json", tmp / "bible.json")
        for name in STATE_FILES:
            if (char.dir / name).exists():
                shutil.copyfile(char.dir / name, tmp / name)
        # Flat files, not a refs/ folder: Kaggle zips subfolders of a dataset, and a
        # zip inside a download is one more thing to unpack differently in each place.
        for ref in char.refs():
            shutil.copyfile(ref, tmp / f"refs__{ref.name}")
        for ref in sorted((char.refs_dir / "sheet").glob("*.jpg")):
            shutil.copyfile(ref, tmp / f"refs__sheet__{ref.name}")
        (tmp / "dataset-metadata.json").write_text(json.dumps(
            {"title": f"mpt-persona-{char.slug}", "id": ds, "licenses": [{"name": "other"}]}))
        # A dataset that does not exist answers `status` with 403, same as one we
        # cannot read, so ask for our own list instead.
        mine = _kaggle("datasets", "list", "--mine", "-s", f"mpt-persona-{char.slug}", env=env, check=False)
        if ds in mine:
            _kaggle("datasets", "version", "-p", str(tmp), "-m", time.strftime("%Y-%m-%d %H:%M"), env=env)
        else:
            # `datasets create` makes it private unless --public is passed.
            _kaggle("datasets", "create", "-p", str(tmp), env=env)
    # A new version takes a moment to process; a kernel pushed before it is ready
    # mounts the previous one.
    for _ in range(30):
        if "ready" in _kaggle("datasets", "status", ds, env=env, check=False).lower():
            break
        time.sleep(10)
    return f"synced {char.name} to private dataset {ds}"


def restore_layout(char_dir):
    """Undo sync()'s flattening: refs__<name> files back into refs/<name>."""
    char_dir = Path(char_dir)
    (char_dir / "refs").mkdir(parents=True, exist_ok=True)
    (char_dir / "refs" / "sheet").mkdir(exist_ok=True)
    for f in char_dir.glob("refs__*"):
        f.rename(char_dir / "refs" / f.name[len("refs__"):].replace("sheet__", "sheet/", 1))
    (char_dir / "dataset-metadata.json").unlink(missing_ok=True)


def _package_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(config.PACKAGE, arcname="persona",
                filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add(config.REPO / "llm.py", arcname="llm.py")
    return base64.b64encode(buf.getvalue()).decode()


def _wan2gp_commit():
    wan = Path(config.env("WAN2GP_DIR") or Path.home() / "apps/Wan2GP").expanduser()
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wan, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def push(char, scene_list, videos=0, date=""):
    """Push the render kernel for these scenes (and `videos` clips of the first
    passing images). Returns immediately; the kernel runs on Kaggle."""
    settings = config.load_settings()
    model = registry.for_role("edit", settings=settings, lane="social", kaggle=True)
    registry.for_role("video", settings=settings, lane="social", kaggle=True)
    _push(char, kernel_id(char), {"scenes": scene_list, "roles": {**settings["roles"], "edit": model["id"]},
                                  "videos": int(videos), "date": date},
          f"{len(scene_list)} scenes, {videos} videos")


def video_kernel_id(char):
    return f"{os.environ['KAGGLE_USERNAME'].strip()}/mpt-persona-video-{char.slug}"


def push_videos(char, jobs):
    """Push the on-demand video kernel: jobs are {id, image_b64, motion}. A
    separate kernel from the daily render, so a 🎬 never waits for (or clobbers)
    the render's output."""
    settings = config.load_settings()
    registry.for_role("video", settings=settings, lane="social", kaggle=True)
    _push(char, video_kernel_id(char), {"mode": "video", "jobs": jobs, "roles": settings["roles"]},
          f"{len(jobs)} videos on demand")


def _push(char, slug, extra, what):
    env = _env()
    payload = {"slug": char.slug, "wan2gp_commit": _wan2gp_commit(),
               "face_min": config.env("PERSONA_FACE_MIN", "0.45"),
               "min_age": config.env("PERSONA_MIN_AGE", "21"), **extra}
    src = (config.PACKAGE / "kaggle_kernel.py").read_text()
    # Replace the assignment lines, not the first occurrence: the template's own
    # docstring names both placeholders too.
    for name, value in (("PAYLOAD_B64", base64.b64encode(json.dumps(payload).encode()).decode()),
                        ("PACKAGE_B64", _package_b64())):
        line = f'{name} = "__{name}__"'
        if line not in src:
            raise RuntimeError(f"kaggle_kernel.py has no {line!r}")
        src = src.replace(line, f'{name} = "{value}"')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "persona_render.py").write_text(src)
        (tmp / "kernel-metadata.json").write_text(json.dumps({
            "id": slug, "title": slug.split("/", 1)[1], "code_file": "persona_render.py",
            "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_internet": "true", "dataset_sources": [dataset_id(char)], "competition_sources": [],
            "kernel_sources": []}))
        log(f"pushing {slug} ({ACCELERATOR}), {what}...")
        _kaggle("kernels", "push", "-p", str(tmp), "--accelerator", ACCELERATOR, env=env)


def wait(char):
    env = _env()
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        out = _kaggle("kernels", "status", kernel_id(char), env=env, check=False)
        if "COMPLETE" in out or "ERROR" in out:
            return out
        time.sleep(30)
    raise RuntimeError(f"Kaggle kernel did not finish within {POLL_TIMEOUT}s")


def run(char, n=3, hint="", do_sync=True, videos=0):
    """Laptop/CLI path: write scenes, sync, push, wait, import."""
    scene_list = scenes.write(char, lane="social", hint=hint, n=n)
    if do_sync:
        log(sync(char))
    push(char, scene_list, videos=videos)
    wait(char)
    return import_output(char)


def status(slug):
    return _kaggle("kernels", "status", slug, env=_env(), check=False)


def download_output(char, dest, slug=None):
    """A kernel's latest COMPLETED output into dest (the daily render unless `slug`
    is given). Returns its status.json dict, or None if there is none."""
    env = _env()
    _kaggle("kernels", "output", slug or kernel_id(char), "-p", str(dest), env=env)
    status = Path(dest) / "status.json"
    return json.loads(status.read_text()) if status.exists() else None


def kernel_id(char):
    return f"{os.environ['KAGGLE_USERNAME'].strip()}/mpt-persona-render-{char.slug}"


def import_output(char):
    """Download the render kernel's latest output and add its items to her local
    store, keeping their ids. Safe to repeat: an already-imported item is only
    refreshed. The bot calls this when a button names an item it does not have,
    which is what happens to batches the GitHub Actions cron rendered."""
    env = _env()
    slug = kernel_id(char)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _kaggle("kernels", "output", slug, "-p", str(tmp), env=env)
        status_file = tmp / "status.json"
        if not status_file.exists():
            logs = next(tmp.glob("*.log"), None)
            tail = logs.read_text(errors="replace")[-1500:] if logs else "no log"
            raise RuntimeError(f"kernel wrote no status.json (killed?). Log tail:\n{tail}")
        status = json.loads(status_file.read_text())
        if not status.get("ok"):
            raise RuntimeError(f"kernel failed at {status.get('stage')}: {status.get('error')}\n"
                               f"{status.get('traceback', '')[-1500:]}")
        items = []
        for meta_file in sorted((tmp / "out" / "items").glob("*.json")):
            item = json.loads(meta_file.read_text())
            if item.get("path"):
                dest = char.dir / "items" / item["path"]
                shutil.copyfile(meta_file.parent / item["path"], dest)
                item["path"] = str(dest)
            item["rendered_on"] = "kaggle"
            known = char.item(item["id"]) or {}
            items.append(char.update_item(item["id"], **{**item, **{k: known[k] for k in ("done", "status", "message")
                                                                     if k in known}}))
        return items
