"""Image-to-video on Kaggle's own T4, via Wan2GP (deepbeepmeep/Wan2GP) -- a second
attempt at a fallback under videogen.py's HF ZeroGPU ladder. See kernel_build_
videogen/video_pipeline.py's docstring for the full reasoning: a prior Kaggle-T4
video path (LTX-Video, removed in 31a738f) worked but was rejected for visibly worse
quality than a smaller substitute model; Wan2GP runs the actual Wan 2.2 model family
quantized for low VRAM instead of substituting a different, weaker model.

Push/poll/output follows kaggle_imagegen.py's pattern exactly (base64 payload
substitution at kernel-push time, status.json + kernel-log-tail diagnostics, never
letting an OOM hard-kill read as success) -- read that module for the shared design.

STATUS: prototype, not wired into videogen.py's ladder yet. Exists to prove out (or
disprove) the Wan2GP-on-Kaggle approach with a real test run before any integration
decision -- the account owner needs to judge the output quality first, same gate the
removed LTX attempt was held to.

Needs env: KAGGLE_USERNAME, KAGGLE_API_TOKEN (or KAGGLE_KEY -- aliased the same as
kaggle_imagegen.py)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL_SLUG = "mpt-video-gen-worker"
ACCELERATOR = os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")
# A cold run downloads Wan2GP's own dependencies plus several GB of quantized model
# weights before it generates anything -- a much higher ceiling than the image
# kernel's (which reuses Kaggle's own preinstalled torch/diffusers most of the time).
POLL_TIMEOUT = int(os.environ.get("KAGGLE_VIDEO_TIMEOUT", "5700"))


def log(msg): print(f"[kaggle_videogen] {msg}", flush=True)


def available():
    env = os.environ
    has_key = bool(env.get("KAGGLE_KEY") or env.get("KAGGLE_API_TOKEN"))
    return bool(env.get("KAGGLE_USERNAME")) and has_key


def _kaggle_env():
    env = os.environ.copy()
    if "KAGGLE_KEY" not in env and env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]
    return env


def _poll(slug, env, timeout, interval=20):
    """Same shape as kaggle_imagegen._poll -- see there for why the ERROR verdict is
    used, not discarded."""
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


def _kernel_log_tail(out_root, n_chars=4000):
    log_file = out_root / f"{KERNEL_SLUG}.log"
    if not log_file.exists():
        return None
    return log_file.read_text(errors="replace")[-n_chars:]


def generate_from_url(image_url, prompt, video_length=81, resolution="512x896",
                      seed=-1, dest=None):
    """Push one kernel bearing an already-hosted image URL and a prompt, poll it,
    and copy the resulting mp4 to `dest` (or a temp path if not given). Returns the
    local Path. Raises on any failure -- no multi-round retry the way imageslides.
    generate() has; this is a single, real, expensive Kaggle round, and the caller
    (once this is wired into a ladder) is expected to treat a failure as "fall
    through to whatever comes next", the same shape as every other rung."""
    if not available():
        raise RuntimeError("Kaggle credentials not configured "
                           "(KAGGLE_USERNAME + KAGGLE_API_TOKEN/KAGGLE_KEY)")
    username = os.environ["KAGGLE_USERNAME"].strip()
    env = _kaggle_env()

    payload = {"image_url": image_url, "prompt": prompt, "video_length": video_length,
              "resolution": resolution, "seed": seed}
    env["VIDEOGEN_PAYLOAD_JSON"] = json.dumps(payload)

    log("preparing kernel...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_video_kernel.py")],
                   cwd=str(ROOT), env=env, check=True)

    log(f"pushing kernel (accelerator={ACCELERATOR})...")
    subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT / "kernel_build_videogen"),
                    "--accelerator", ACCELERATOR],
                   cwd=str(ROOT), env=env, check=True)

    log("polling Kaggle (a cold run installs Wan2GP + downloads model weights "
        "before generating anything -- this can take a while)...")
    slug = f"{username}/{KERNEL_SLUG}"
    if _poll(slug, env, POLL_TIMEOUT) == "error":
        log("Kaggle marked this kernel ERROR; fetching its log for the reason")

    out_root = Path(tempfile.mkdtemp(prefix="kaggle_videogen_out_"))
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(out_root)],
                   env=env, check=True)

    status_file = out_root / "status.json"
    if not status_file.exists():
        tail = _kernel_log_tail(out_root)
        raise RuntimeError(
            "kernel finished but wrote no status.json (a silent hard-kill, not "
            "a catchable error)" + (f" -- kernel log tail:\n{tail}" if tail else
                                    " -- no kernel log either, no diagnosis possible"))
    status = json.loads(status_file.read_text())
    if not status.get("ok"):
        parts = [f"kernel reported failure at stage {status.get('stage')!r}: "
                f"{status.get('error')}"]
        if status.get("traceback"):
            parts.append(f"traceback:\n{status['traceback']}")
        if status.get("stdout_tail"):
            parts.append(f"wgp.py stdout tail:\n{status['stdout_tail']}")
        if status.get("stderr_tail"):
            parts.append(f"wgp.py stderr tail:\n{status['stderr_tail']}")
        tail = _kernel_log_tail(out_root)
        if tail:
            parts.append(f"kernel log tail:\n{tail}")
        raise RuntimeError("\n".join(parts))

    src = out_root / status["video"]
    if not src.exists():
        raise RuntimeError(f"status.json reported success but {src} is missing")
    dest = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="kaggle_videogen_")) / "final.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    log(f"wrote {dest} ({dest.stat().st_size} bytes)")
    return dest
