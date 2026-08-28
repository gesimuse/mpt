"""aibeauty image generation on Kaggle's own GPU -- a faster alternative to
imageslides.generate()'s default local (GH Actions CPU) path. Never the only
path: autopilot.py's run_niche() tries this first when Kaggle credentials are
configured and falls back to the unchanged local path on ANY failure here
(missing credentials, push/poll error, kernel crash, timeout) -- a GPU speed
win is not worth risking the pipeline that already works.

Push/poll/output follows the same pattern as motionforge's old Kaggle bridge
(videogen.py, before it was removed there for the video niche -- see that
module's docstring). The difference here: Kaggle actually does real compute
for this one. Standard SD1.5/SDXL fp16 inference (what sdgen.py already runs)
has none of the exotic-quantized-kernel / accelerator-selection problems that
made the Wan2.2 video attempt a dead end -- it's run fine on P100-class GPUs
for years, so Kaggle's inability to request a specific accelerator via its
API doesn't block this the way it did there.

Needs env: KAGGLE_USERNAME, KAGGLE_API_TOKEN (or KAGGLE_KEY -- aliased same
as videogen.py used to)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL_SLUG = "mpt-image-gen-worker"


def log(msg): print(f"[kaggle_imagegen] {msg}", flush=True)


def available():
    """True if Kaggle credentials are configured at all -- callers use this to
    decide whether attempting Kaggle is even worth the round trip."""
    env = os.environ
    has_key = bool(env.get("KAGGLE_KEY") or env.get("KAGGLE_API_TOKEN"))
    return bool(env.get("KAGGLE_USERNAME")) and has_key


def _kaggle_env():
    env = os.environ.copy()
    if "KAGGLE_KEY" not in env and env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]
    return env


def _poll(slug, env, timeout=1800, interval=15):
    """Poll kernel status until COMPLETE or ERROR. No official Kaggle API
    field for "how much longer" -- just polls at a fixed interval like
    motionforge's own poll_kaggle.py did."""
    deadline = time.time() + timeout
    last_seen = None
    while time.time() < deadline:
        r = subprocess.run(["kaggle", "kernels", "status", slug], env=env,
                           capture_output=True, text=True)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        if "KernelWorkerStatus.COMPLETE" in out:
            return "complete"
        if "KernelWorkerStatus.ERROR" in out:
            return "error"
        if out != last_seen:
            log(f"status: {out[-200:]}")
            last_seen = out
        time.sleep(interval)
    raise RuntimeError(f"kernel did not reach a terminal state within {timeout}s")


def generate(niche, state=None, out_dir=None):
    """Push a Kaggle kernel that clones mpt and runs imageslides.generate()
    for this niche on Kaggle's GPU, poll for completion, copy the images back.
    Returns (image_paths, vibe, image_prompts) -- same 3-tuple contract as
    imageslides.generate() itself. Raises on any failure; caller (autopilot.py)
    treats that as "fall back to the local path", not a hard stop."""
    if not available():
        raise RuntimeError("Kaggle credentials not configured "
                           "(KAGGLE_USERNAME + KAGGLE_API_TOKEN/KAGGLE_KEY)")
    username = os.environ["KAGGLE_USERNAME"].strip()
    env = _kaggle_env()
    env["NICHE_ID"] = niche["id"]

    log(f"preparing kernel for niche {niche['id']!r}...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_image_kernel.py")],
                   cwd=str(ROOT), env=env, check=True)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT / "kernel_build_imagegen")],
                   cwd=str(ROOT), env=env, check=True)

    log("polling Kaggle...")
    slug = f"{username}/{KERNEL_SLUG}"
    _poll(slug, env)

    out_root = Path(tempfile.mkdtemp(prefix="kaggle_imagegen_out_"))
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(out_root)],
                   env=env, check=True)

    status_file = out_root / "status.json"
    if not status_file.exists():
        # The Wan2.2 experiment hit this exact shape of failure repeatedly: a
        # hard kill (OOM or similar) below Python's own exception handling,
        # so neither a traceback nor status.json ever gets written. Standard
        # SD1.5/SDXL fp16 shouldn't hit that (see module docstring), but if it
        # ever does, there's no diagnostic to offer beyond this.
        raise RuntimeError("kernel finished but wrote no status.json (a silent "
                           "hard-kill, not a catchable error -- no diagnosis "
                           "possible from here)")
    status = json.loads(status_file.read_text())
    if not status.get("ok"):
        raise RuntimeError(f"kernel reported failure: {status.get('error')}")

    images_dir = out_root / "images"
    names = status.get("images") or []
    image_paths = [images_dir / n for n in names]
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing or not image_paths:
        raise RuntimeError(f"kernel reported success but images missing: {missing or 'none listed'}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        moved = []
        for p in image_paths:
            dest = out_dir / p.name
            shutil.copy(p, dest)
            moved.append(dest)
        image_paths = moved

    log(f"{len(image_paths)} images from Kaggle")
    image_prompts = status.get("image_prompts") or [None] * len(image_paths)
    return [str(p) for p in image_paths], status.get("vibe"), image_prompts
