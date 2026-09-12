"""mpt image-to-video -- Kaggle worker, running Wan2GP (deepbeepmeep/Wan2GP), a
low-VRAM runtime for the real Wan 2.2 model family -- not a smaller substitute model.
Animates one already-hosted still into a short mp4 on Kaggle's own T4, then writes it
plus a status.json back for kaggle_videogen.py to collect.

Why this is a second attempt, not the first: a prior Kaggle-T4 video path existed
(kaggle_videogen.py, kaggle/video_pipeline.py, removed in 31a738f) using LTX-Video
directly via diffusers. It worked end to end, but the account owner rejected the
output outright -- LTX's largest T4-fitting variant (2B transformer) is not in the
same league as Wan 2.2 on HF ZeroGPU (13B/14B), so "it worked" still shipped visibly
worse video than the primary path. That removal's own conclusion: "nothing which
fits a 15 GB T4 is in the same league as Wan 2.2 on ZeroGPU" -- true for a smaller
SUBSTITUTE model, which is what LTX-2B was.

Wan2GP is a different bet on the same hardware: it runs the ACTUAL Wan 2.2 14B i2v
weights (same model family as ZeroGPU), fitted into low VRAM via quantization
(fp8/int8/nvfp4) and aggressive offloading, purpose-built for exactly this
"GPU-poor" case, rather than substituting a smaller/weaker model. Whether quantized
Wan 2.2 clears the same quality bar unquantized Wan 2.2 does is the actual open
question this first run is for -- expect to judge the output by eye, same as last
time, not just "did it produce an mp4".

Model choice: i2v_2_2_Enhanced_Lightning_v2 (defaults/i2v_2_2_Enhanced_Lightning_v2.
json in the Wan2GP repo) -- fp8-quantized Wan 2.2 14B i2v, Lightning-distilled to 4
inference steps. Picked over the plain i2v_2_2 default specifically for wall-clock:
a full-schedule (20-40 step) 14B model on a single free T4 risks running past
Kaggle's session limits before producing anything, and this repo's own experience
with LCM-distilled SD1.5 (sdgen.py) already showed a 4-8 step distilled model is what
actually fits a constrained free runner's time budget.

VRAM/RAM profile: 5 (Kaggle's own docs/CLI.md: "VerylowRAM_LowVRAM (Fail safe): at
least 24 GB of RAM and 10 GB of VRAM") -- deliberately the most conservative option,
not the "Recommended" Profile 4 (32 GB RAM floor), because the prior LTX attempt's
actual failure mode was a HOST RAM OOM (not VRAM) loading a large text encoder, and
Kaggle's real host RAM ceiling sits close to Profile 4's stated floor. Profile 5's
whole purpose is exactly this situation.

Placeholders below are substituted by scripts/prepare_video_kernel.py at push time.
"""
import base64
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

PAYLOAD_B64 = "__PAYLOAD_B64__"

WORK = Path("/kaggle/working")
WAN2GP_DIR = Path("/kaggle/tmp/Wan2GP")
WAN2GP_DIR.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR = WORK / "out"
STATUS = WORK / "status.json"


def log(m: str) -> None:
    print(f"[kaggle_videogen] {m}", flush=True)


def write_status(stage: str, ok: bool, extra: dict | None = None) -> None:
    payload = {"stage": stage, "ok": ok, "ts": time.time()}
    if extra:
        payload.update(extra)
    STATUS.write_text(json.dumps(payload, indent=2))


def main() -> None:
    # False until the run actually finishes -- see kaggle_imagegen's kernel for why
    # (an OOM hard-kill happens below Python's own exception handling; a status.json
    # left saying {"stage": "start", "ok": true} from before that point reads as a
    # false success instead of surfacing the kill).
    write_status("start", False)
    try:
        payload = json.loads(base64.b64decode(PAYLOAD_B64).decode())
        image_url = payload["image_url"]
        prompt = payload["prompt"]
        video_length = int(payload.get("video_length", 81))  # ~5.06s @ 16fps, Wan's default
        resolution = payload.get("resolution", "512x896")
        seed = int(payload.get("seed", -1))

        # Same Kaggle-base-image gotcha the image kernel hit: transformers imports
        # TensorFlow unconditionally for one CLIP loader path we never use.
        os.environ["USE_TF"] = "0"

        capability = None
        try:
            import torch as _preinstalled_torch
            if _preinstalled_torch.cuda.is_available():
                capability = _preinstalled_torch.cuda.get_device_capability()
                log(f"accelerator: {_preinstalled_torch.cuda.get_device_name()} "
                    f"(sm_{capability[0]}{capability[1]}), "
                    f"preinstalled torch {_preinstalled_torch.__version__}")
        except Exception as e:
            log(f"could not probe preinstalled torch ({type(e).__name__}: {e}) -- "
                "continuing, Wan2GP's own install below picks a torch build regardless")

        log("cloning Wan2GP...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/deepbeepmeep/Wan2GP", str(WAN2GP_DIR)],
            check=True)

        # Wan2GP's own README, RTX 20XX+ install (a T4 is Turing/sm_75, squarely in
        # that bracket) -- installed explicitly rather than trusting Kaggle's
        # preinstalled torch, since Wan2GP documents this exact pin as required, not
        # optional the way the image kernel's Pascal-only reinstall is.
        log("installing Wan2GP's own pinned torch (RTX 20XX+ build)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0",
             "--index-url", "https://download.pytorch.org/whl/cu130"],
            check=True)

        log("installing Wan2GP's requirements.txt...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            cwd=str(WAN2GP_DIR), check=True)

        # Same fix the image kernel needed, same reason: transformers/peft both probe
        # is_torchao_available() and skip cleanly when it's absent; Kaggle's own
        # preinstalled torchao build satisfies neither's version floor once a specific
        # torch build is pinned above it.
        log("removing Kaggle's preinstalled torchao...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
            check=True)

        log(f"downloading input image from {image_url[:80]}...")
        img_path = WORK / "input.jpg"
        urllib.request.urlretrieve(image_url, img_path)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        settings = {
            "model_type": "i2v_2_2_Enhanced_Lightning_v2",
            "prompt": prompt,
            "image_start": str(img_path),
            "video_length": video_length,
            "resolution": resolution,
            "seed": seed,
        }
        settings_path = WORK / "task.json"
        settings_path.write_text(json.dumps(settings))
        log(f"task: {json.dumps(settings)[:300]}")

        log("running wgp.py --process (this downloads the model on a cold run, "
            "expect several minutes before generation even starts)...")
        r = subprocess.run(
            [sys.executable, "wgp.py", "--i2v-14B",
             "--process", str(settings_path),
             "--output-dir", str(OUT_DIR),
             "--profile", "5",
             "--attention", "sdpa"],
            cwd=str(WAN2GP_DIR), capture_output=True, text=True,
            timeout=int(os.environ.get("WAN2GP_TIMEOUT", "5400")))
        log(f"wgp.py exited {r.returncode}")
        stdout_tail, stderr_tail = r.stdout[-4000:], r.stderr[-4000:]

        if r.returncode != 0:
            write_status("wgp_failed", False, {
                "error": f"wgp.py exited {r.returncode}",
                "stdout_tail": stdout_tail, "stderr_tail": stderr_tail,
            })
            return

        videos = sorted(OUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not videos:
            write_status("no_output", False, {
                "error": "wgp.py exited 0 but wrote no .mp4",
                "stdout_tail": stdout_tail, "stderr_tail": stderr_tail,
            })
            return

        final = WORK / "final.mp4"
        final.write_bytes(videos[-1].read_bytes())
        write_status("done", True, {"video": final.name,
                                    "size_bytes": final.stat().st_size,
                                    "stdout_tail": stdout_tail[-1000:]})
        log(f"wrote {final} ({final.stat().st_size} bytes)")
    except subprocess.CalledProcessError as e:
        write_status("failed", False, {
            "error": f"{e.cmd} exited {e.returncode}",
            "stdout_tail": (e.stdout or "")[-2000:] if hasattr(e, "stdout") else None,
            "stderr_tail": (e.stderr or "")[-2000:] if hasattr(e, "stderr") else None,
            "traceback": traceback.format_exc(),
        })
        raise
    except Exception as e:
        write_status("failed", False, {"error": str(e), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
