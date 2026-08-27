"""Wan 2.2 I2V rCM via the motionforge Kaggle bridge.

Motionforge (gesimuse/motionforge) IS the pipeline: prepare a Kaggle kernel with
image_url + motion prompt baked in, push it, poll until Kaggle finishes, download
final.mp4. Kaggle does NO compute itself; the kernel proxies to the linoyts/
wan2-2-i2v-rCM HF ZeroGPU Space. All the timing-tuned polling/retry code lives in
motionforge already -- rather than re-implement it here (or vendor its files into
this repo), the workflow clones motionforge next to this checkout and we shell out
to its own scripts with our env vars.

Repo path resolution: MOTIONFORGE_DIR (env) > $GITHUB_WORKSPACE/motionforge_ext >
../motionforge (sibling checkout, matches local dev layout). Fails loud if none
exist, since a silent skip here would look like "video niche silently produced
nothing" in the state file.

Needs env: KAGGLE_USERNAME, KAGGLE_API_TOKEN (motionforge's run_local.sh names it
KAGGLE_API_TOKEN even though Kaggle's own env var is KAGGLE_KEY; we set both
below), HF_TOKEN. Motionforge's scripts read IMAGE_URL/PROMPT/LENGTH_S/STEPS from
env, which we set per call."""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path


def log(msg): print(f"[videogen] {msg}", flush=True)


def _motionforge_dir():
    env = os.environ.get("MOTIONFORGE_DIR")
    if env:
        return Path(env)
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        candidate = Path(workspace).parent / "motionforge_ext"
        if candidate.exists():
            return candidate
        candidate = Path(workspace) / "motionforge_ext"
        if candidate.exists():
            return candidate
    sibling = Path(__file__).resolve().parent.parent / "motionforge"
    if sibling.exists():
        return sibling
    raise RuntimeError(
        "motionforge checkout not found -- set MOTIONFORGE_DIR, or clone "
        "gesimuse/motionforge as a sibling directory / into "
        "$GITHUB_WORKSPACE/motionforge_ext")


def generate(image_url, prompt, length_s=5.0, steps=4, seed=None,
             negative_prompt=None, out_dir=None):
    """Kick off motionforge's Kaggle push + poll, return the local mp4 path.
    Raises on Kaggle failure/timeout -- caller treats it as a skipped run."""
    mf = _motionforge_dir()
    prepare = mf / "scripts" / "prepare_kernel.py"
    poll = mf / "scripts" / "poll_kaggle.py"
    for f in (prepare, poll):
        if not f.exists():
            raise RuntimeError(f"motionforge script missing: {f}")

    env = os.environ.copy()
    env["IMAGE_URL"] = image_url
    env["PROMPT"] = prompt
    env["LENGTH_S"] = str(length_s)
    env["STEPS"] = str(steps)
    if seed is not None:
        env["IMAGE_SEED"] = str(seed)
    if negative_prompt:
        env["NEGATIVE_PROMPT"] = negative_prompt
    # motionforge's kaggle CLI wrapper reads KAGGLE_KEY; run_local.sh names the
    # secret KAGGLE_API_TOKEN. Set both so this works whichever name is provided.
    if "KAGGLE_KEY" not in env and env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]

    log(f"prepare_kernel: image_url={image_url[:80]}, prompt={prompt[:60]!r}, "
        f"length_s={length_s}, steps={steps}")
    subprocess.run([sys.executable, str(prepare)], cwd=str(mf), env=env, check=True)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(mf / "kernel_build")],
                   cwd=str(mf), env=env, check=True)
    log("polling Kaggle (motionforge/scripts/poll_kaggle.py handles timing + retries)")
    subprocess.run([sys.executable, str(poll)], cwd=str(mf), env=env, check=True)

    src = mf / "kaggle_output" / "final.mp4"
    if not src.exists():
        raise RuntimeError(f"motionforge finished but {src} is missing")
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="videogen_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "final.mp4"
    shutil.copy(src, dest)
    log(f"wrote {dest} ({dest.stat().st_size // 1024}KB)")
    return str(dest)
