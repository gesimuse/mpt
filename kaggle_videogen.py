"""Image-to-video on Kaggle's own T4, as the fallback under videogen.py's ZeroGPU
ladder.

The ceiling this lifts: HF ZeroGPU gives ~5 minutes of GPU per HF account per DAY.
HF_TOKENS multiplies that by however many accounts exist, and videogen.py now walks
three Spaces, but the total is still small and fixed. Kaggle gives ~30 GPU-hours a
week on a real T4. So this is not a speed win -- ZeroGPU is much faster end to end --
it is a "the day's quota is gone and a video still needs to exist" path.

Push/poll/output follows kaggle_imagegen.py exactly; read that module for the shared
design (base64 payload substitution, status.json + kernel-log-tail diagnostics, and
why nothing on the Kaggle side is ever allowed to talk to civitai.com). The kernel
itself is kaggle/video_pipeline.py.

Needs env: KAGGLE_USERNAME, KAGGLE_API_TOKEN (or KAGGLE_KEY).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import videogen

ROOT = Path(__file__).resolve().parent
KERNEL_SLUG = "mpt-video-gen-worker"
# Same reasoning as kaggle_imagegen.ACCELERATOR: Kaggle's default P100 is sm_60, which
# its own preinstalled torch has no kernels for. T4 x2 is UI-only; this is the single
# T4, which is what matters.
ACCELERATOR = os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")
# A cold run downloads ~10GB of LTX weights before it generates anything, so the
# ceiling is much higher than the image kernel's.
POLL_TIMEOUT = int(os.environ.get("KAGGLE_VIDEO_TIMEOUT", "2700"))


def log(msg): print(f"[kaggle_videogen] {msg}", flush=True)


def available():
    """True if Kaggle credentials are configured at all -- callers use this to decide
    whether attempting Kaggle is even worth the round trip."""
    env = os.environ
    has_key = bool(env.get("KAGGLE_KEY") or env.get("KAGGLE_API_TOKEN"))
    return bool(env.get("KAGGLE_USERNAME")) and has_key


def _kaggle_env():
    env = os.environ.copy()
    if "KAGGLE_KEY" not in env and env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]
    return env


def _poll(slug, env, timeout=None, interval=20):
    timeout = POLL_TIMEOUT if timeout is None else timeout
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


def generate(image_url, prompt, length_s=5.0, steps=8, negative_prompt=None,
             out_dir=None):
    """Animate image_url on Kaggle's GPU and return a local mp4 path, already
    normalized for TikTok. Raises on any failure -- the caller treats that the same
    way it treats videogen.generate() failing: a skipped run, not a crash."""
    if not available():
        raise RuntimeError("Kaggle credentials not configured "
                           "(KAGGLE_USERNAME + KAGGLE_API_TOKEN/KAGGLE_KEY)")
    username = os.environ["KAGGLE_USERNAME"].strip()
    env = _kaggle_env()
    env["VIDEOGEN_PAYLOAD_JSON"] = json.dumps({
        "image_url": image_url, "prompt": prompt,
        "length_s": float(length_s), "steps": int(steps),
        # Same string the ZeroGPU path uses -- the kernel has its own fallback, but
        # sending it explicitly keeps the two paths from drifting apart.
        "negative_prompt": negative_prompt or videogen.VIDEO_NEGATIVE,
    })

    log("preparing kernel...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_video_kernel.py")],
                   cwd=str(ROOT), env=env, check=True)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT / "kernel_build_videogen"),
                    "--accelerator", ACCELERATOR],
                   cwd=str(ROOT), env=env, check=True)

    log("polling Kaggle (a cold run downloads ~10GB of weights first)...")
    slug = f"{username}/{KERNEL_SLUG}"
    _poll(slug, env)

    out_root = Path(tempfile.mkdtemp(prefix="kaggle_videogen_out_"))
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(out_root)],
                   env=env, check=True)

    status_file = out_root / "status.json"
    if not status_file.exists():
        tail = _kernel_log_tail(out_root)
        raise RuntimeError(
            "kernel finished but wrote no status.json (a silent hard-kill, not a "
            "catchable error)" + (f" -- kernel log tail:\n{tail}" if tail else
                                  " -- no kernel log either, no diagnosis possible"))
    status = json.loads(status_file.read_text())
    if not status.get("ok"):
        parts = [f"kernel reported failure: {status.get('error')}"]
        if status.get("traceback"):
            parts.append(f"traceback:\n{status['traceback']}")
        tail = _kernel_log_tail(out_root)
        if tail:
            parts.append(f"kernel log tail:\n{tail}")
        raise RuntimeError("\n".join(parts))

    raw = out_root / (status.get("video") or "video.mp4")
    if not raw.exists():
        raise RuntimeError(f"kernel reported success but {raw.name} is not in its output")

    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="kaggle_videogen_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "final.mp4"
    # The exact same re-encode the ZeroGPU path gets: TikTok rejected a real draft
    # with frame_rate_check_failed on raw model output, and this kernel's export is
    # no more TikTok-ready than a Space's was.
    videogen.normalize_for_tiktok(raw, dest)
    log(f"wrote {dest} ({dest.stat().st_size // 1024}KB)")
    return str(dest)
