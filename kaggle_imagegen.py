"""aibeauty image generation on Kaggle's own GPU -- a faster alternative to
imageslides.generate()'s default local (GH Actions CPU) path. Never the only
path: autopilot.py's run_niche() tries this first when Kaggle credentials are
configured and falls back to the unchanged local path on ANY failure here
(missing credentials, push/poll error, kernel crash, timeout, not enough
images passing review) -- a GPU speed win is not worth risking the pipeline
that already works.

Split from a single "clone mpt, run imageslides.generate() on Kaggle" kernel
(the original design here) because civitai.com's own domain 451s every
request from Kaggle's network -- confirmed live, not an auth issue -- while
the actual model files live on a separate Cloudflare R2 domain that IS
reachable from Kaggle. So CivitAI stays entirely in this process (unblocked,
running in GH Actions): decide_reference(), build the prompt batch, resolve
the checkpoint's final R2 download link (civitai.resolve_final_url), and only
hand Kaggle that already-resolved link plus the prompts -- the kernel never
talks to civitai.com at all, it just downloads from R2 and runs sdgen.

Deliberately ONE round, not imageslides.generate()'s full multi-round
re-decide/retry loop: a Kaggle round is a full kernel push+poll+download (real
wall-clock cost), and retrying here would duplicate logic the local path
already owns. Any shortfall -- bad checkpoint, too few approved, Kaggle infra
hiccup -- just raises, and autopilot.py's existing except falls back to a
fresh, full local imageslides.generate() run (its own multi-round retry,
supervisor-broken fallback, etc. all still apply there, unchanged).

Push/poll/output follows the same pattern as motionforge's old Kaggle bridge
(videogen.py, before it was removed there for the video niche -- see that
module's docstring). The difference here: Kaggle actually does real compute
for this one. Standard SD1.5/SDXL fp16 inference (what sdgen.py already runs)
has none of the exotic-quantized-kernel problems that made the earlier Wan2.2
video attempt a dead end. Accelerator choice was the other claimed blocker and
it turned out not to be one: `kaggle kernels push --accelerator NvidiaTeslaT4`
works on the CLI this pins (1.8.4), which is what removes the P100/Pascal
torch-pinning mess the kernel used to carry unconditionally.

Needs env: KAGGLE_USERNAME, KAGGLE_API_TOKEN (or KAGGLE_KEY -- aliased same
as videogen.py used to)."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import civitai
import imageslides
import supervisor

ROOT = Path(__file__).resolve().parent
KERNEL_SLUG = "mpt-image-gen-worker"
# Overridable so a Kaggle-side outage of one accelerator type doesn't need a
# code change: any id from `kaggle kernels push --help` works.
ACCELERATOR = os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")


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


def _kernel_log_tail(out_root, n_chars=4000):
    """`kaggle kernels output` also downloads the kernel's own stdout/stderr as
    <slug>.log. sdgen.py's per-image handler truncates its own log lines to 100
    chars (see its own docstring on why), but the FULL traceback/reason is still
    in this file even then -- surface its tail on any failure so the real cause
    shows up directly in the GH Actions log instead of just a generic
    "no images were generated"/status error with no way to dig further from here."""
    log_file = out_root / f"{KERNEL_SLUG}.log"
    if not log_file.exists():
        return None
    text = log_file.read_text(errors="replace")
    return text[-n_chars:]


def _generate_batch_on_kaggle(resolved, prompts, negatives, adopted, workdir):
    """Push one kernel bearing an already-R2-resolved checkpoint link plus this
    round's prompts, poll it, and copy the resulting images into `workdir`.
    Returns a list of local Paths, named sd_<i>.jpg same as sdgen.generate_batch
    itself -- so the caller's own prompt_by_path regex match works unchanged."""
    username = os.environ["KAGGLE_USERNAME"].strip()
    env = _kaggle_env()

    r2_url = civitai.resolve_final_url(resolved["url"])
    payload = {
        "resolved": {**resolved, "url": r2_url},
        "prompts": prompts,
        "negatives": negatives,
        "adopted": adopted,
    }
    env["IMAGEGEN_PAYLOAD_JSON"] = json.dumps(payload)

    log("preparing kernel...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_image_kernel.py")],
                   cwd=str(ROOT), env=env, check=True)
    # --accelerator picks the GPU. Kaggle's default is the P100, whose Pascal (sm_60)
    # architecture Kaggle's own preinstalled torch no longer has kernels for -- the
    # cause of "CUDA error: no kernel image is available for execution on the device"
    # on every image of a live run, and of the whole torch==2.6.0/cu124 pin dance in
    # kaggle/image_pipeline.py. The T4 is sm_75 and needs none of it.
    #
    # This file (and videogen.py) used to assert as fact that Kaggle's API "has no way
    # to request a specific accelerator". Not true of the CLI this workflow pins
    # (kaggle<2.0 resolves to 1.8.4, whose `kernels push` takes --accelerator).
    #
    # VERIFIED on a real probe kernel pushed with exactly this flag, rather than
    # assumed: Kaggle honoured it and handed back TWO Tesla T4s (device_count 2,
    # 15360 MiB each, sm_75) running Kaggle's own stock torch 2.10.0+cu128 -- the
    # very build that has no Pascal kernels -- with fp32 and fp16 matmuls both
    # executing cleanly. So the T4 request works, and "T4 x2 is UI-only" (which an
    # earlier version of this comment stated) is wrong: the CLI gets both cards.
    #
    # Still not load-bearing. If Kaggle ever can't honour it and falls back to a
    # P100, the kernel detects sm_60 at runtime and installs the cu124 torch itself.
    subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT / "kernel_build_imagegen"),
                    "--accelerator", ACCELERATOR],
                   cwd=str(ROOT), env=env, check=True)

    log("polling Kaggle...")
    slug = f"{username}/{KERNEL_SLUG}"
    # See kaggle_videogen for why the verdict is used rather than discarded.
    if _poll(slug, env) == "error":
        log("Kaggle marked this kernel ERROR; fetching its log for the reason")

    out_root = Path(tempfile.mkdtemp(prefix="kaggle_imagegen_out_"))
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(out_root)],
                   env=env, check=True)

    status_file = out_root / "status.json"
    if not status_file.exists():
        # The Wan2.2 experiment hit this exact shape of failure repeatedly: a
        # hard kill (OOM or similar) below Python's own exception handling,
        # so neither a traceback nor status.json ever gets written.
        tail = _kernel_log_tail(out_root)
        raise RuntimeError(
            "kernel finished but wrote no status.json (a silent hard-kill, not "
            "a catchable error)" + (f" -- kernel log tail:\n{tail}" if tail else
                                    " -- no kernel log either, no diagnosis possible"))
    status = json.loads(status_file.read_text())
    if not status.get("ok"):
        # The diagnostic import block in image_pipeline.py writes a full
        # traceback into status.json itself (confirmed live to be far more useful
        # than the kernel's own stdout/stderr log, which by the time we read it is
        # buried under pip's dependency-conflict warnings and Kaggle's own
        # trailing notebook-conversion noise, appended after the script exits).
        parts = [f"kernel reported failure: {status.get('error')}"]
        if status.get("traceback"):
            parts.append(f"traceback:\n{status['traceback']}")
        tail = _kernel_log_tail(out_root)
        if tail:
            parts.append(f"kernel log tail:\n{tail}")
        raise RuntimeError("\n".join(parts))

    images_dir = out_root / "images"
    names = status.get("images") or []
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in names:
        src = images_dir / n
        if not src.exists():
            continue
        dest = workdir / n
        shutil.copy(src, dest)
        paths.append(dest)
    return paths


def generate(niche, count=None, workdir=None, state=None):
    """One Kaggle round: decide a CivitAI checkpoint + reference prompt (same
    logic imageslides.generate() uses), generate `count` camera variations on
    Kaggle's GPU, keep what passes supervisor.py review. Returns (image_paths,
    vibe, image_prompts) -- same 3-tuple contract as imageslides.generate().
    Raises on any shortfall; caller (autopilot.py) treats that as "fall back
    to the local path", not a hard stop."""
    if not available():
        raise RuntimeError("Kaggle credentials not configured "
                           "(KAGGLE_USERNAME + KAGGLE_API_TOKEN/KAGGLE_KEY)")
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="kaggle_imagegen_"))
    workdir.mkdir(parents=True, exist_ok=True)

    count = count or int(niche.get("images_per_video", 10))
    min_images = int(niche.get("min_images", 3))
    max_images = int(niche.get("max_images", niche.get("images_per_video", 5)))

    resolved, reference = imageslides.decide_reference(niche, state=state)
    civitai_spec = f"{resolved['model_id']}:{resolved['version_id']}"
    prefix, vibe = imageslides._build_prefix(niche, reference, state=state)
    base_negative = ", ".join(
        x for x in (imageslides.NEGATIVE_HARD, reference["negative_prompt"],
                    imageslides.NEGATIVE_QUALITY) if x)
    log(f"{resolved['name']!r} | prefix: {prefix} | reference: {reference['prompt'][:120]}")

    prompts, negatives = imageslides.build_variations(
        prefix, reference["prompt"], base_negative, count, niche)
    adopted = imageslides._adopted_settings(reference)
    if adopted:
        log(f"using the checkpoint creator's own posted settings where safe: {adopted}")

    generated = _generate_batch_on_kaggle(resolved, prompts, negatives, adopted, workdir)

    prompt_by_path = {}
    for path in generated:
        m = re.search(r"sd_(\d+)\.[^.]+$", str(path))
        if m and int(m.group(1)) < len(prompts):
            prompt_by_path[str(path)] = prompts[int(m.group(1))]

    supervisor_on = os.environ.get("SUPERVISOR_ENABLED", "1").strip().lower() not in (
        "0", "false", "no")
    approved = list(supervisor.filter_images(generated)) if supervisor_on else generated
    imageslides._record_model_result(state, civitai_spec, resolved["name"],
                                     len(generated), len(approved))
    imageslides._record_theme_result(state, vibe, len(generated), len(approved))
    log(f"{len(approved)}/{len(generated)} passed, {len(approved)}/{min_images} needed")
    if len(approved) < min_images:
        # Not the multi-round supervisor-broken fallback imageslides.generate() has --
        # any shortfall here just raises, and the local fallback path (which DOES have
        # that handling, and its own fresh multi-round retry) takes over instead of
        # duplicating that logic on top of an already-spent Kaggle round.
        raise RuntimeError(
            f"only {len(approved)} of {len(generated)} images passed review "
            f"(need at least {min_images}); not using this Kaggle round")

    kept = approved[:max_images]
    return kept, vibe, [prompt_by_path.get(str(p)) for p in kept]
